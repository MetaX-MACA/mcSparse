#ifndef GPSV_INTERLEAVED_BATCH_DEVICE_H_
#define GPSV_INTERLEAVED_BATCH_DEVICE_H_

#include "mcsp_config.h"
#include "mcsp_general_utility.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

constexpr double epslion_gpsv_device = 1e-7;

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspBatchedGpsvLUKernel(idxType row_num, idxType batch_stride, idxType batch_count, valType* ds,
                                        valType* dl, valType* dm, valType* du, valType* dw, valType* x) {
    idxType tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= batch_count) {
        return;
    }

    // LU decomposition
    for (idxType row = 0; row < row_num - 1; ++row) {
        idxType idx_cur = batch_stride * row + tid;
        idxType idx_next = batch_stride * (row + 1) + tid;

        valType dm_cur = dm[idx_cur];
        valType dl_next = dl[idx_next];
        valType new_dl_next = dl_next / THRESHOLD(dm_cur, epslion_gpsv_device);
        dl[idx_next] = new_dl_next;
        dm[idx_next] = dm[idx_next] - new_dl_next * du[idx_cur];
        du[idx_next] = du[idx_next] - new_dl_next * dw[idx_cur];

        if (row + 2 < row_num) {
            idxType idx_next_next = batch_stride * (row + 2) + tid;
            valType ds_next_next = ds[idx_next_next];
            valType new_ds_next_next = ds_next_next / THRESHOLD(dm_cur, epslion_gpsv_device);
            ds[idx_next_next] = new_ds_next_next;
            dl[idx_next_next] = dl[idx_next_next] - new_ds_next_next * du[idx_cur];
            dm[idx_next_next] = dm[idx_next_next] - new_ds_next_next * dw[idx_cur];
        }
    }

    // Forward elimination
    for (idxType row = 1; row < row_num; ++row) {
        idxType idx_cur = batch_stride * row + tid;
        idxType idx_back = batch_stride * (row - 1) + tid;
        idxType idx_back_back = batch_stride * (row - 2) + tid;
        if (row != 1) {
            x[idx_cur] = x[idx_cur] - x[idx_back] * dl[idx_cur] - x[idx_back_back] * ds[idx_cur];
        } else {
            x[idx_cur] = x[idx_cur] - x[idx_back] * dl[idx_cur];
        }
    }

    // backward substitution
    x[batch_stride * (row_num - 1) + tid] =
        x[batch_stride * (row_num - 1) + tid] / THRESHOLD(dm[batch_stride * (row_num - 1) + tid], epslion_gpsv_device);
    x[batch_stride * (row_num - 2) + tid] =
        (x[batch_stride * (row_num - 2) + tid] -
         du[batch_stride * (row_num - 2) + tid] * x[batch_stride * (row_num - 1) + tid]) /
        dm[batch_stride * (row_num - 2) + tid];

    for (int row = row_num - 3; row >= 0; --row) {
        int idx_cur = batch_stride * row + tid;
        int idx_back = batch_stride * (row + 1) + tid;
        int idx_back_back = batch_stride * (row + 2) + tid;
        x[idx_cur] = (x[idx_cur] - du[idx_cur] * x[idx_back] - dw[idx_cur] * x[idx_back_back]) /
                     THRESHOLD(dm[idx_cur], epslion_gpsv_device);
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspBatchedGpsvThomasKernel(idxType row_num, idxType batch_stride, idxType batch_count, valType* ds,
                                            valType* dl, valType* dm, valType* du, valType* dw, valType* x,
                                            valType* du1, valType* dw1) {
    idxType tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= batch_count) {
        return;
    }

    // Forward elimination
    du1[tid] = du[tid] / THRESHOLD(dm[tid], epslion_gpsv_device);
    dw1[tid] = dw[tid] / THRESHOLD(dm[tid], epslion_gpsv_device);
    x[tid] = x[tid] / THRESHOLD(dm[tid], epslion_gpsv_device);

    for (idxType row = 1; row < row_num; ++row) {
        idxType index_0 = batch_stride * row + tid;
        idxType index_1 = batch_stride * (row + 1) + tid;
        idxType minus_index = batch_stride * (row - 1) + tid;

        idxType count_index_0 = batch_count * row + tid;
        idxType count_minus_index = batch_count * (row - 1) + tid;

        valType dl_0 = dl[index_0];
        valType dm_0 = dm[index_0];
        valType du_0 = du[index_0];
        valType dw_0 = dw[index_0];
        valType b_0 = x[index_0];

        du1[count_index_0] = (du_0 - dw1[count_minus_index] * dl_0) /
                             THRESHOLD((dm_0 - du1[count_minus_index] * dl_0), epslion_gpsv_device);
        dw1[count_index_0] = dw_0 / THRESHOLD((dm_0 - du1[count_minus_index] * dl_0), epslion_gpsv_device);
        x[index_0] =
            (b_0 - x[minus_index] * dl_0) / THRESHOLD((dm_0 - du1[count_minus_index] * dl_0), epslion_gpsv_device);

        if (index_1 < row_num * batch_stride) {
            valType ds_1 = ds[index_1];
            valType dl_1 = dl[index_1];
            valType dm_1 = dm[index_1];
            valType b_1 = x[index_1];

            dl[index_1] = dl_1 - du1[count_minus_index] * ds_1;
            dm[index_1] = dm_1 - dw1[count_minus_index] * ds_1;
            x[index_1] = b_1 - x[minus_index] * ds_1;
        }
    }

    // backward substitution
    idxType index_0 = batch_stride * (row_num - 2) + tid;
    idxType index_1 = batch_stride * (row_num - 1) + tid;
    x[index_0] = x[index_0] - x[index_1] * du1[batch_count * (row_num - 2) + tid];

    for (int row = row_num - 3; row >= 0; --row) {
        index_0 = batch_stride * row + tid;
        index_1 = batch_stride * (row + 1) + tid;
        idxType index_2 = batch_stride * (row + 2) + tid;
        x[index_0] = x[index_0] - x[index_1] * du1[batch_count * row + tid] - x[index_2] * dw1[batch_count * row + tid];
    }
}

__device__ __forceinline__ float internalSqrt(float value) {
    return sqrt(value);
}

__device__ __forceinline__ double internalSqrt(double value) {
    return sqrt(value);
}

__device__ __forceinline__ float internalSqrt(mcspComplexFloat value) {
    return sqrt(value.x);
}

__device__ __forceinline__ double internalSqrt(mcspComplexDouble value) {
    return sqrt(value.x);
}

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspGpsvBatchHouseholderQrKernel(idxType row_num, idxType batch_count, idxType batch_stride,
                                                 valType* ds, valType* dl, valType* dm, valType* du, valType* dw,
                                                 valType* X, valType* t1, valType* t2) {
    idxType tid = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if (tid >= batch_count) {
        return;
    }

    for (idxType row = 0; row < row_num - 1; ++row) {
        idxType idp_0 = batch_stride * (row + 0) + tid;
        idxType idp_1 = batch_stride * (row + 1) + tid;
        idxType idp_2 = batch_stride * (row + 2) + tid;
        idxType idp_0c = batch_count * (row + 0) + tid;
        idxType idp_1c = batch_count * (row + 1) + tid;
        idxType idp_2c = batch_count * (row + 2) + tid;

        valType dlp_1 = dl[idp_1];
        valType dp_1 = dm[idp_1];
        valType dup_1 = du[idp_1];
        valType dwp_1 = dw[idp_1];
        valType Bp1 = X[idp_1];

        valType dsp_2 = static_cast<valType>(0);
        valType dlp_2 = static_cast<valType>(0);
        valType dp_2 = static_cast<valType>(0);
        valType dup_2 = static_cast<valType>(0);
        valType dwp_2 = static_cast<valType>(0);
        valType Bp_2 = static_cast<valType>(0);

        if (row != row_num - 2) {
            dsp_2 = ds[idp_2];
            dlp_2 = dl[idp_2];
            dp_2 = dm[idp_2];
            dup_2 = du[idp_2];
            dwp_2 = dw[idp_2];
            Bp_2 = X[idp_2];
        }

        valType new_dlp_1 = dlp_1;
        valType new_dsp_2 = dsp_2;
        valType dls_sq = new_dlp_1 * mcsp_conj(new_dlp_1) + new_dsp_2 * mcsp_conj(new_dsp_2);

        if (dls_sq == static_cast<valType>(0)) {
            continue;
        }

        valType diag = dm[idp_0];
        valType val = internalSqrt(diag * mcsp_conj(diag) + dls_sq);
        valType a_ii;
        if (diag != static_cast<valType>(0)) {
            a_ii = diag + val * diag / std::abs(diag);
        } else {
            a_ii = val;
        }
        valType sq = a_ii * mcsp_conj(a_ii);
        valType beta = static_cast<valType>(2) * sq / (dls_sq + sq);

        new_dlp_1 = new_dlp_1 / a_ii;
        new_dsp_2 = new_dsp_2 / a_ii;
        valType d1 = (dsp_2 * mcsp_conj(new_dsp_2) + dlp_1 * mcsp_conj(new_dlp_1) + diag) * beta;
        valType d2 = (dlp_2 * mcsp_conj(new_dsp_2) + dp_1 * mcsp_conj(new_dlp_1) + du[idp_0]) * beta;
        valType d3 = (dp_2 * mcsp_conj(new_dsp_2) + dup_1 * mcsp_conj(new_dlp_1) + dw[idp_0]) * beta;
        valType d4 = (dup_2 * mcsp_conj(new_dsp_2) + dwp_1 * mcsp_conj(new_dlp_1) + t1[idp_0c]) * beta;
        valType d5 = (dwp_2 * mcsp_conj(new_dsp_2) + t1[idp_1c] * mcsp_conj(new_dlp_1) + t2[idp_0c]) * beta;
        valType fs = (Bp_2 * mcsp_conj(new_dsp_2) + Bp1 * mcsp_conj(new_dlp_1) + X[idp_0]) * beta;

        dm[idp_0] -= d1;
        du[idp_0] -= d2;
        dw[idp_0] -= d3;
        t1[idp_0c] -= d4;
        t2[idp_0c] -= d5;
        X[idp_0] -= fs;

        dm[idp_1] = dp_1 - d2 * new_dlp_1;
        du[idp_1] = dup_1 - d3 * new_dlp_1;
        dw[idp_1] = dwp_1 - d4 * new_dlp_1;
        t1[idp_1c] = t1[idp_1c] - d5 * new_dlp_1;
        X[idp_1] = Bp1 - new_dlp_1 * fs;

        if (row != row_num - 2) {
            dl[idp_2] = dlp_2 - d2 * new_dsp_2;
            dm[idp_2] = dp_2 - d3 * new_dsp_2;
            du[idp_2] = dup_2 - d4 * new_dsp_2;
            dw[idp_2] = dwp_2 - d5 * new_dsp_2;
            X[idp_2] = Bp_2 - new_dsp_2 * fs;
        }
    }
    for (idxType idx = 1; idx <= row_num; ++idx) {
        idxType row = row_num - idx;
        idxType idp_0 = batch_stride * row + tid;
        idxType idp_0c = batch_count * row + tid;
        valType sum = static_cast<valType>(0);
        if (row + 1 < row_num) sum += du[idp_0] * X[batch_stride * (row + 1) + tid];
        if (row + 2 < row_num) sum += dw[idp_0] * X[batch_stride * (row + 2) + tid];
        if (row + 3 < row_num) sum += t1[idp_0c] * X[batch_stride * (row + 3) + tid];
        if (row + 4 < row_num) sum += t2[idp_0c] * X[batch_stride * (row + 4) + tid];
        if (dm[idp_0] != static_cast<valType>(0)) {
            X[idp_0] = (X[idp_0] - sum) / dm[idp_0];
        }
    }
}

#endif