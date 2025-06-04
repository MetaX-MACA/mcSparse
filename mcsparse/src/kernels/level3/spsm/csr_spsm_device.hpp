#ifndef KERNELS_LEVEL3_CSR_SPSM_DEVICE_HPP__
#define KERNELS_LEVEL3_CSR_SPSM_DEVICE_HPP__

#include "device_reduce.hpp"
#include "mcsp_config.h"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE, bool lower_flag, bool unity_flag, typename idxType, typename valType>
__global__ void mcspCsrSpsmSolveKernel(idxType m, idxType nrhs, valType alpha, const valType *csr_vals,
                                       const idxType *csr_rows, const idxType *csr_cols, const idxType *row_map,
                                       idxType *zero_pivot, valType *B, idxType ldb, idxType *done_buffer,
                                       mcsparseIndexBase_t idx_base) {
    idxType tid = threadIdx.x;
    idxType bid = blockIdx.x;
    idxType tile_row = bid % m;
    idxType tile_col = bid / m;
    idxType row = row_map[tile_row];
    idxType Bj = tile_col * BLOCK_SIZE + tid;
    idxType Bij = row * ldb + Bj;
    idxType offset_col = tile_col * m;
    idxType start = csr_rows[row] - idx_base;
    idxType end = csr_rows[row + 1] - idx_base;
    idxType row_nnz = end - start;
    __shared__ idxType smem_col[BLOCK_SIZE];
    __shared__ valType smem_val[BLOCK_SIZE];
    smem_col[tid] = static_cast<idxType>(0);
    smem_val[tid] = static_cast<valType>(0);

    valType local_sum = (Bj < nrhs) ? alpha * B[Bij] : static_cast<valType>(0);
    valType Aii = static_cast<valType>(1);
    for (idxType j = start; j < end; ++j) {
        idxType k = (j - start) & (BLOCK_SIZE - 1);
        // pre-fetch matrix A to share memory to avoid duplicated global memory loading
        if (k == 0) {
            smem_col[tid] = (tid < row_nnz) ? (csr_cols[tid + start] - idx_base) : static_cast<idxType>(-1);
            smem_val[tid] = (tid < row_nnz) ? csr_vals[tid + start] : static_cast<valType>(-1);
        }
        __syncthreads();

        idxType A_local_col = smem_col[k];
        valType A_local_val = smem_val[k];

        if ((!unity_flag) && A_local_col == row && A_local_val == static_cast<valType>(0)) {
            if (tid == 0) {
                atomicMin(zero_pivot, row + idx_base);
            }
            A_local_val = static_cast<valType>(1);
        }

        if (lower_flag) {
            if (A_local_col > row) {
                continue;
            }
            if (A_local_col == row) {
                if (!unity_flag) {
                    Aii = static_cast<valType>(1) / A_local_val;
                }
                continue;
            }
        } else {
            if (A_local_col < row) {
                continue;
            }
            if (A_local_col == row) {
                if (!unity_flag) {
                    Aii = static_cast<valType>(1) / A_local_val;
                }
                continue;
            }
        }

        if (tid == 0) {
            while (!atomicOr(&(done_buffer[offset_col + A_local_col]), 0)) {
                ;
            }
        }
        __syncthreads();
        __threadfence();
        idxType Xij = A_local_col * ldb + Bj;
        if (Bj < nrhs) {
            local_sum = (-A_local_val * B[Xij]) + local_sum;
        } else {
            local_sum = static_cast<valType>(0);
        }
    }

    if (!unity_flag) {
        local_sum = Aii * local_sum;
    }
    if (Bj < nrhs) {
        B[Bij] = local_sum;
    }
    __threadfence();

    if (tid == 0) {
        atomicOr(&done_buffer[offset_col + row], 1);
    }
}

#endif
