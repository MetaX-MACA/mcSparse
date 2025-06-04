#ifndef GTSV_INTERLEAVED_BATCH_DEVICE_H_
#define GTSV_INTERLEAVED_BATCH_DEVICE_H_

#include "mcsp_config.h"
#include "mcsp_general_utility.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

constexpr double epslion_gtsv_batch_device = 1e-7;

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspBatchedGtsvLUnoPivotKernel(idxType row_num, idxType batch_stride, idxType batch_count, valType* dl,
                                               valType* d, valType* du, valType* x) {
    idxType tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= batch_count) {
        return;
    }

    // LU decomposition
    for (idxType row = 0; row < row_num - 1; ++row) {
        idxType idx_cur = batch_stride * row + tid;
        idxType idx_next = batch_stride * (row + 1) + tid;

        valType d_cur = d[idx_cur];
        valType dl_next = dl[idx_next];
        valType new_dl_next = dl_next / THRESHOLD(d_cur, epslion_gtsv_batch_device);
        dl[idx_next] = new_dl_next;
        d[idx_next] = d[idx_next] - new_dl_next * du[idx_cur];
    }

    // Forward elimination
    for (idxType row = 1; row < row_num; ++row) {
        idxType index = batch_stride * row + tid;
        x[index] = x[index] - x[index - batch_stride] * dl[index];
    }

    // backward substitution
    x[batch_stride * (row_num - 1) + tid] = x[batch_stride * (row_num - 1) + tid] /
                                            THRESHOLD(d[batch_stride * (row_num - 1) + tid], epslion_gtsv_batch_device);

    for (int row = row_num - 2; row >= 0; --row) {
        idxType index = batch_stride * row + tid;
        x[index] = (x[index] - du[index] * x[index + batch_stride]) / THRESHOLD(d[index], epslion_gtsv_batch_device);
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspBatchedGtsvLUKernel(idxType row_num, idxType batch_stride, idxType batch_count, valType* dl,
                                        valType* d, valType* du, valType* x, valType* u2, idxType* p) {
    idxType tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= batch_count) {
        return;
    }

    // pivoting index
    p[tid] = 0;

    // LU decomposition
    for (idxType row = 0; row < row_num - 1; ++row) {
        idxType idx_cur = batch_stride * row + tid;
        idxType idx_next = batch_stride * (row + 1) + tid;
        idxType idx_tmp_cur = batch_count * row + tid;
        idxType idx_tmp_next = batch_count * (row + 1) + tid;

        valType d_cur = d[idx_cur];
        valType dl_next = dl[idx_next];
        valType new_dl_next;

        if (std::abs(d_cur) < std::abs(dl_next)) {
            // pivoting
            idxType k = p[idx_tmp_cur];
            p[idx_tmp_cur] = row + 1;
            p[idx_tmp_next] = k;

            valType d_next = d[idx_next];
            valType du_cur = du[idx_cur];
            valType du_next = du[idx_next];
            valType u2_cur = u2[idx_tmp_cur];
            valType x_cur = x[idx_cur];

            d[idx_cur] = dl_next;
            du[idx_cur] = d_next;
            u2[idx_tmp_cur] = du_next;
            d[idx_next] = du_cur;
            du[idx_next] = u2_cur;
            x[idx_cur] = x[idx_next];
            x[idx_next] = x_cur;
            new_dl_next = d_cur / THRESHOLD(dl_next, epslion_gtsv_batch_device);
        } else {
            p[idx_tmp_next] = row + 1;
            new_dl_next = dl_next / THRESHOLD(d_cur, epslion_gtsv_batch_device);
        }
        dl[idx_next] = new_dl_next;
        d[idx_next] = d[idx_next] - new_dl_next * du[idx_cur];
        du[idx_next] = du[idx_next] - new_dl_next * u2[idx_tmp_cur];
    }

    // Forward elimination
    valType tmp = static_cast<valType>(0);
    for (idxType row = 1; row < row_num; ++row) {
        idxType index = batch_stride * row + tid;
        tmp += x[index - batch_stride] * dl[index];
        if (p[batch_count * row + tid] <= row) {
            x[index] -= tmp;
            tmp = static_cast<valType>(0);
        }
    }

    // backward substitution
    idxType idm_0 = batch_stride * (row_num - 1) + tid;
    idxType idm_1 = batch_stride * (row_num - 2) + tid;
    x[idm_0] = x[idm_0] / THRESHOLD(d[idm_0], epslion_gtsv_batch_device);
    x[idm_1] = (x[idm_1] - du[idm_1] * x[idm_0]) / THRESHOLD(d[idm_1], epslion_gtsv_batch_device);

    for (int row = row_num - 3; row >= 0; --row) {
        idxType index = batch_stride * row + tid;
        x[index] = (x[index] - du[index] * x[index + batch_stride] -
                    u2[batch_count * row + tid] * x[index + batch_stride * 2]) /
                   THRESHOLD(d[index], epslion_gtsv_batch_device);
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspBatchedGtsvThomasKernel(idxType row_num, idxType batch_stride, idxType batch_count, valType* dl,
                                            valType* d, valType* du, valType* x, valType* du_tmp) {
    idxType tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= batch_count) {
        return;
    }

    // Forward elimination
    du_tmp[tid] = du[tid] / THRESHOLD(d[tid], epslion_gtsv_batch_device);
    x[tid] = x[tid] / THRESHOLD(d[tid], epslion_gtsv_batch_device);
    valType coef;

    for (idxType row = 1; row < row_num; ++row) {
        idxType index = batch_stride * row + tid;
        coef = d[index] - dl[index] * du_tmp[index - batch_stride];
        coef = 1.0f / THRESHOLD(coef, epslion_gtsv_batch_device);
        du_tmp[index] = du[index] * coef;
        x[index] = (x[index] - dl[index] * x[index - batch_stride]) * coef;
    }

    // backward substitution
    for (int row = row_num - 2; row >= 0; --row) {
        idxType index = batch_stride * row + tid;
        x[index] = x[index] - du_tmp[index] * x[index + batch_stride];
    }
}

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspGtsvBatchGivensQrKernel(idxType row_num, idxType batch_count, idxType batch_stride, valType* dl,
                                            valType* d, valType* du, valType* x, valType* r2) {
    idxType tid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if (tid >= batch_count) {
        return;
    }

    for (idxType row = 0; row < row_num - 1; ++row) {
        idxType idp_0 = batch_stride * (row + 0) + tid;
        idxType idp_1 = batch_stride * (row + 1) + tid;

        valType dlp_1 = dl[idp_1];
        valType dp_0 = d[idp_0];
        valType dp_1 = d[idp_1];
        valType dup_0 = du[idp_0];
        valType dup_1 = du[idp_1];
        valType Bp_0 = x[idp_0];
        valType Bp_1 = x[idp_1];

        valType r = sqrt(std::abs(dp_0 * mcsp_conj(dp_0) + dlp_1 * mcsp_conj(dlp_1)));
        valType c = mcsp_conj(dp_0) / THRESHOLD(r, epslion_gtsv_batch_device);
        valType s = mcsp_conj(dlp_1) / THRESHOLD(r, epslion_gtsv_batch_device);

        d[idp_0] = dp_0 * c + dlp_1 * s;
        d[idp_1] = dup_0 * -mcsp_conj(s) + dp_1 * mcsp_conj(c);

        du[idp_0] = dup_0 * c + dp_1 * s;
        du[idp_1] = dup_1 * mcsp_conj(c);
        r2[batch_count * row + tid] = dup_1 * s;

        x[idp_0] = Bp_0 * c + Bp_1 * s;
        x[idp_1] = Bp_0 * -mcsp_conj(s) + Bp_1 * mcsp_conj(c);
    }

    idxType idm_0 = batch_stride * (row_num - 1) + tid;
    idxType idm_1 = batch_stride * (row_num - 2) + tid;

    x[idm_0] = x[idm_0] / THRESHOLD(d[idm_0], epslion_gtsv_batch_device);
    x[idm_1] = (x[idm_1] - du[idm_1] * x[idm_0]) / THRESHOLD(d[idm_1], epslion_gtsv_batch_device);

    for (int row = row_num - 3; row >= 0; --row) {
        idxType index = batch_stride * row + tid;
        x[index] = (x[index] - du[index] * x[index + batch_stride] -
                    r2[batch_count * row + tid] * x[index + batch_stride * 2]) /
                   THRESHOLD(d[index], epslion_gtsv_batch_device);
    }
}

#endif