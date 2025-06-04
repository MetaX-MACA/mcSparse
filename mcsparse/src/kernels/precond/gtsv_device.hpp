#ifndef KERNELS_PRECOND_GTSV_DEVICE_HPP__
#define KERNELS_PRECOND_GTSV_DEVICE_HPP__

#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_general_utility.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

constexpr double epslion_gtsv_device = 1e-7;

template <typename valType>
__device__ void applyThreshold(valType *x, valType *y) {
    if (std::abs(*x) <= 0) {
        *x = epslion_gtsv_device;
    }
    if (std::abs(*y) <= 0) {
        *y = 0;
    }
}

template <unsigned int PARTITION_SIZE, typename idxType, typename valType>
__global__ void mcspGtsvThomasKernel(idxType m, const valType *dl, const valType *d, valType *du, valType *B) {
    const idxType col = blockIdx.y;
    dl += col * m;
    d += col * m;
    du += col * m;
    B += col * m;
    valType du_tmp[PARTITION_SIZE];
    du_tmp[0] = du[0] / THRESHOLD(d[0], epslion_gtsv_device);
    B[0] = B[0] / THRESHOLD(d[0], epslion_gtsv_device);
    valType coef;
    for (int i = 1; i < m; ++i) {
        coef = d[i] - dl[i] * du_tmp[i - 1];
        coef = 1.0f / THRESHOLD(coef, epslion_gtsv_device);
        du_tmp[i] = du[i] * coef;
        B[i] = (B[i] - dl[i] * B[i - 1]) * coef;
    }
    for (int i = m - 2; i >= 0; --i) {
        B[i] = B[i] - du_tmp[i] * B[i + 1];
    }
}

template <unsigned int PARTITION_SIZE, typename idxType, typename valType>
__global__ void mcspGtsvBatchGivensQrKernel(idxType m, valType *dl, valType *d, valType *du, valType *B) {
    const idxType col = blockIdx.y;
    dl += col * m;
    d += col * m;
    du += col * m;
    B += col * m;
    valType r2[PARTITION_SIZE];

    for (int row = 0; row < m - 1; ++row) {
        valType dlp_1 = dl[row + 1];
        valType dp_0 = d[row];
        valType dp_1 = d[row + 1];
        valType dup_0 = du[row];
        valType dup_1 = du[row + 1];
        valType Bp_0 = B[row];
        valType Bp_1 = B[row + 1];

        valType r = sqrt(std::abs(dp_0 * mcsp_conj(dp_0) + dlp_1 * mcsp_conj(dlp_1)));
        valType c = mcsp_conj(dp_0) / THRESHOLD(r, epslion_gtsv_device);
        valType s = mcsp_conj(dlp_1) / THRESHOLD(r, epslion_gtsv_device);

        d[row] = dp_0 * c + dlp_1 * s;
        d[row + 1] = dup_0 * -mcsp_conj(s) + dp_1 * mcsp_conj(c);

        du[row] = dup_0 * c + dp_1 * s;
        du[row + 1] = dup_1 * mcsp_conj(c);
        r2[row] = dup_1 * s;

        B[row] = Bp_0 * c + Bp_1 * s;
        B[row + 1] = Bp_0 * -mcsp_conj(s) + Bp_1 * mcsp_conj(c);
    }

    B[m - 1] = B[m - 1] / THRESHOLD(d[m - 1], epslion_gtsv_device);
    B[m - 2] = (B[m - 2] - du[m - 2] * B[m - 1]) / THRESHOLD(d[m - 2], epslion_gtsv_device);

    for (int row = m - 3; row >= 0; --row) {
        B[row] = (B[row] - du[row] * B[row + 1] - r2[row] * B[row + 2]) / THRESHOLD(d[row], epslion_gtsv_device);
    }
}

template <unsigned int PARTITION_SIZE, typename idxType, typename valType>
__global__ void eliminateBandKernel(idxType t_num, idxType m, idxType m_next, const valType *dl, const valType *d,
                                    const valType *du, const valType *B, valType *dl_next, valType *d_next,
                                    valType *du_next, valType *B_next) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType idx = threadIdx.x;
    const idxType col = blockIdx.y;
    idxType M = PARTITION_SIZE;
    if (tid >= t_num) {
        return;
    } else if (tid == t_num - 1) {
        M = m - tid * PARTITION_SIZE < PARTITION_SIZE ? m - tid * PARTITION_SIZE : PARTITION_SIZE;
    }

    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *sp = reinterpret_cast<valType *>(smem);
    sp += idx * 4;

    dl += tid * PARTITION_SIZE + col * m;
    d += tid * PARTITION_SIZE + col * m;
    du += tid * PARTITION_SIZE + col * m;
    B += tid * PARTITION_SIZE + col * m;

    valType rp, rc;
    valType sc1;
    valType tmp_val = B[1];

    sp[0] = dl[1];
    sp[1] = d[1];
    sp[2] = du[1];
    sp[3] = 0;

    for (int j = 2; j < M; ++j) {
        sc1 = dl[j];
        applyThreshold(sp + 1, &sc1);
        if (std::abs(sc1) <= std::abs(sp[1])) {
            rp = -sc1 / sp[1];
            rc = 1;
        } else {
            rp = 1;
            rc = -sp[1] / sc1;
        }
        sp[0] = rp * sp[0];
        sp[1] = rp * sp[2] + rc * d[j];
        sp[2] = rp * sp[3] + rc * du[j];
        tmp_val = rp * tmp_val + rc * B[j];
        sp[3] = 0;
    }
    dl_next[tid * 2 + 1 + col * m_next] = sp[0];
    d_next[tid * 2 + 1 + col * m_next] = sp[1];
    du_next[tid * 2 + 1 + col * m_next] = sp[2];
    B_next[(tid * 2 + 1) + col * m_next] = tmp_val;
}

template <unsigned int PARTITION_SIZE, typename idxType, typename valType>
__global__ void eliminateBandReverseKernel(idxType t_num, idxType m, idxType m_next, const valType *dl,
                                           const valType *d, const valType *du, const valType *B, valType *dl_next,
                                           valType *d_next, valType *du_next, valType *B_next) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType idx = threadIdx.x;
    const idxType col = blockIdx.y;
    idxType M = PARTITION_SIZE;
    if (tid >= t_num) {
        return;
    } else if (tid == t_num - 1) {
        M = m - tid * PARTITION_SIZE < PARTITION_SIZE ? m - tid * PARTITION_SIZE : PARTITION_SIZE;
    }
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *sp = reinterpret_cast<valType *>(smem);
    sp += idx * 4;

    dl += tid * PARTITION_SIZE + col * m;
    d += tid * PARTITION_SIZE + col * m;
    du += tid * PARTITION_SIZE + col * m;
    B += tid * PARTITION_SIZE + col * m;

    valType rp, rc;
    valType sc1;
    valType tmp_val = B[M - 2];

    sp[0] = du[M - 2];
    sp[1] = d[M - 2];
    sp[2] = dl[M - 2];
    sp[3] = 0;

    for (int j = M - 3; j >= 0; --j) {
        sc1 = du[j];
        applyThreshold(sp + 1, &sc1);
        if (std::abs(sc1) <= std::abs(sp[1])) {
            rp = -sc1 / sp[1];
            rc = 1;
        } else {
            rp = 1;
            rc = -sp[1] / sc1;
        }
        sp[0] = rp * sp[0];
        sp[1] = rp * sp[2] + rc * d[j];
        sp[2] = rp * sp[3] + rc * dl[j];
        tmp_val = rp * tmp_val + rc * B[j];
        sp[3] = 0;
    }
    dl_next[tid * 2 + col * m_next] = sp[2];
    d_next[tid * 2 + col * m_next] = sp[1];
    du_next[tid * 2 + col * m_next] = sp[0];
    B_next[(tid * 2) + col * m_next] = tmp_val;
}

template <unsigned int PARTITION_SIZE, typename idxType, typename valType>
__global__ void copyBoundaryKernel(idxType t_num, idxType m, idxType m_next, const valType *x_next, valType *x) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType col = blockIdx.y;
    idxType M = PARTITION_SIZE;
    if (tid >= t_num) {
        return;
    } else if (tid == t_num - 1) {
        M = m - tid * PARTITION_SIZE < PARTITION_SIZE ? m - tid * PARTITION_SIZE : PARTITION_SIZE;
    }

    x += tid * PARTITION_SIZE + col * (m + 2);
    x[0] = x_next[tid * 2 + col * (m_next + 2)];
    x[M - 1] = x_next[(tid * 2 + 1) + col * (m_next + 2)];
}

template <unsigned int PARTITION_SIZE, typename idxType, typename valType>
__global__ void substitutionKernel(idxType t_num, idxType m, idxType m_next, const valType *x_next, valType *dl,
                                   valType *d, valType *du, valType *B, valType *x) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType idx = threadIdx.x;
    const idxType col = blockIdx.y;
    idxType M = PARTITION_SIZE;
    if (tid >= t_num) {
        return;
    } else if (tid == t_num - 1) {
        M = m - tid * PARTITION_SIZE < PARTITION_SIZE ? m - tid * PARTITION_SIZE : PARTITION_SIZE;
    }
    extern __shared__ __align__(sizeof(short)) unsigned char smem[];
    valType *sp = reinterpret_cast<valType *>(smem + idx * (sizeof(valType) * 4 + sizeof(short) * PARTITION_SIZE));
    short *indexs = reinterpret_cast<short *>(sp + 4);

    dl += tid * PARTITION_SIZE + col * m;
    d += tid * PARTITION_SIZE + col * m;
    du += tid * PARTITION_SIZE + col * m;
    B += tid * PARTITION_SIZE + col * m;
    x += tid * PARTITION_SIZE + col * (m + 2);

    B[M - 2] = B[M - 2] - du[M - 2] * x[M - 1];
    du[M - 2] = 0;

    valType rp, rc;
    valType sc1;
    valType tmp_val = B[1] - dl[1] * x[0];

    sp[1] = d[1];
    sp[2] = du[1];
    sp[3] = 0;
    indexs[0] = 1;

    for (int j = 2; j < M - 1; ++j) {
        sc1 = dl[j];
        applyThreshold(sp + 1, &sc1);
        if (std::abs(sc1) <= std::abs(sp[1])) {
            dl[indexs[0]] = sp[1];
            d[indexs[0]] = sp[2];
            du[indexs[0]] = 0;
            B[indexs[0]] = tmp_val;
            indexs[j - 1] = indexs[0];
            rp = -sc1 / sp[1];
            rc = 1;
            indexs[0] = j;
        } else {
            indexs[j - 1] = j;
            rp = 1;
            rc = -sp[1] / sc1;
        }
        sp[2] = rp * sp[2] + rc * d[j];
        sp[3] = rp * sp[3] + rc * du[j];
        tmp_val = rp * tmp_val + rc * B[j];
        sp[1] = sp[2];
        sp[2] = sp[3];
        sp[3] = 0;
    }

    if (std::abs(sp[1]) >= std::abs(dl[M - 1])) {
        x[M - 2] = tmp_val / sp[1];
    } else {
        x[M - 2] = (B[M - 1] - d[M - 1] * x[M - 1] - du[M - 1] * x[M]) / dl[M - 1];
    }
    for (int j = M - 3; j > 1; --j) {
        int k = indexs[j];
        x[j] = (B[k] - d[k] * x[j + 1] - du[k] * x[j + 2]) / dl[k];
    }
    int k = indexs[1];
    if (std::abs(dl[k]) >= std::abs(du[0])) {
        x[1] = (B[k] - d[k] * x[2] - du[k] * x[3]) / dl[k];
    } else {
        x[1] = (B[0] - d[0] * x[0] - dl[0] * x[-1]) / du[0];
    }
}

#endif