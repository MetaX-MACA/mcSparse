#ifndef KERNELS_LEVEL2_CSR_SPSV_DEVICE_HPP__
#define KERNELS_LEVEL2_CSR_SPSV_DEVICE_HPP__

#include "device_reduce.hpp"
#include "internal_interface/mcsp_internal_conversion.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType, typename valType>
__global__ void mcspCsrTrmDiagRowNnzKernel(idxType m, const valType *csr_vals, const idxType *csr_rows,
                                           const idxType *csr_cols, idxType *diag_ind, idxType *zero_pivot,
                                           idxType *row_nnz, mcsparseIndexBase_t base) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType row = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    idxType col;

    int found_flag = 0;
    if (row < m) {
        const idxType row_start = csr_rows[row] - base;
        const idxType row_end = csr_rows[row + 1] - base;
        if (lane == 0) {
            row_nnz[row] = row_end - row_start;
        }
        for (idxType idx = row_start + lane; idx < row_end; idx += WARP_SIZE) {
            col = csr_cols[idx] - base;
            if ((found_flag == 0) && row == col) {
                diag_ind[row] = idx;
                found_flag = 1;
            }
            found_flag = __any(found_flag);
#ifndef __MACA__
            __syncwarp();
#endif
        }
        if ((lane == 0) && (found_flag == 0)) {
            zero_pivot[row] = row + base;
        }
    }
}

template <bool lower_flag, typename idxType, typename valType>
__global__ void mcspCsrTrmDepthKernel(idxType m, const valType *csr_vals, const idxType *csr_rows,
                                      const idxType *csr_cols, idxType *depth_buffer, mcsparseIndexBase_t base) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);

    if (gid < m) {
        idxType row;
        if constexpr (lower_flag) {
            row = gid;
        } else {
            row = m - 1 - gid;
        }
        idxType row_start = csr_rows[row] - base;
        idxType row_end = csr_rows[row + 1] - base;

        idxType depth = 0;
        idxType max_depth = 0;
        idxType col;
        bool zone_flag;
        for (idxType i = lane; i < (row_end - row_start); i += WARP_SIZE) {
            if constexpr (lower_flag) {
                col = csr_cols[row_start + i] - base;
                zone_flag = (col < row);
            } else {
                col = csr_cols[row_end - 1 - i] - base;
                zone_flag = (col > row);
            }
            if (zone_flag) {
                depth = atomicOr(&(depth_buffer[col]), 0);
                while (depth == 0) {
                    depth = atomicOr(&(depth_buffer[col]), 0);
                }
                max_depth = max(max_depth, depth);
            } else {
                break;
            }
        }

#pragma unroll
        // intra-warp reduce, op max
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
#ifndef __MACA__
            max_depth = max(max_depth, __shfl_xor_sync(UINT32_BIT_MASK, max_depth, i, WARP_SIZE));
#else
            max_depth = max(max_depth, __shfl_xor(max_depth, i, WARP_SIZE));
#endif
        }

        __threadfence();
        if (lane == 0) {
            atomicOr(&(depth_buffer[row]), max_depth + 1);
        }
    }
}

template <bool lower_flag, bool unity_flag, typename idxType, typename valType>
__global__ void mcspCsrSpsvSolveKernel(idxType m, valType alpha, const valType *csr_vals, const idxType *csr_rows,
                                       const idxType *csr_cols, const idxType *row_map, const idxType *diag_ind,
                                       const valType *x, valType *y, idxType *done_buffer, idxType *zero_pivot,
                                       mcsparseIndexBase_t base) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *buffer = reinterpret_cast<valType *>(smem);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;

    if (gid < m) {
        const idxType row = row_map[gid];
        idxType row_start = csr_rows[row] - base;
        idxType row_end = csr_rows[row + 1] - base;
        valType rhs = alpha * x[row];
        idxType row_diag_idx;
        idxType interval_start;
        idxType interval_end;
        valType diag_val;
        if constexpr (unity_flag) {
            interval_start = row_start;
            interval_end = row_end;
        } else {
            if constexpr (lower_flag) {
                interval_start = row_start;
                interval_end = diag_ind[row];
                diag_val = csr_vals[interval_end];
            } else {
                interval_start = diag_ind[row] + 1;
                interval_end = row_end;
                diag_val = csr_vals[interval_start - 1];
            }
        }

        if (diag_val == static_cast<valType>(0) && (!unity_flag)) {
            atomicMin(zero_pivot, row + base);
        }

        valType local_sum = (valType)0;
        for (idxType i = lane; i < (interval_end - interval_start); i += WARP_SIZE) {
            idxType col;
            idxType idx;
            if constexpr (lower_flag) {
                idx = interval_start + i;
                col = csr_cols[idx] - base;
                if constexpr (unity_flag) {
                    if (col >= row) {
                        continue;
                    }
                }
            } else {
                idx = interval_end - 1 - i;
                col = csr_cols[idx] - base;
                if constexpr (unity_flag) {
                    if (col <= row) {
                        continue;
                    }
                }
            }
            valType val_cur = csr_vals[idx];

            while (!atomicOr(&(done_buffer[col]), 0)) {
                ;
            }
            __threadfence();
            local_sum += y[col] * val_cur;
        }

        // intra-warp reduce, op sum
        buffer[threadIdx.x] = local_sum;
#pragma unroll
        for (unsigned int stride = 1; stride < WARP_SIZE; stride <<= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if ((lane & ((stride << 1) - 1)) == 0) {
                buffer[threadIdx.x] += buffer[threadIdx.x + stride];
            }
        }
        if (lane == 0) {
            if constexpr (unity_flag) {
                y[row] = rhs - buffer[threadIdx.x];
            } else {
                y[row] = (rhs - buffer[threadIdx.x]) / diag_val;
            }
            __threadfence();
            atomicOr(&(done_buffer[row]), 1);
        }
    }
}

#endif
