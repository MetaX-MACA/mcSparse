#ifndef KERNELS_BSRMM_DEVICE_HPP__
#define KERNELS_BSRMM_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspBsrmmZeroAlphaKernel(idxType mb, idxType n, idxType block_dim, valType h_beta, valType* d_C,
                                         idxType ldc) {
    idxType tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
    idxType col_idx = tid / (block_dim * mb);
    if (col_idx < n) {
        idxType row_idx = tid % (block_dim * mb);
        d_C[ldc * col_idx + row_idx] = h_beta * d_C[ldc * col_idx + row_idx];
    }
}

template <unsigned int BLOCKSIZE, unsigned int SEGSIZE, typename idxType, typename valType>
__global__ void mcspBsrmmShareKernel(idxType mb, idxType n, idxType kb, mcsparseOperation_t trans_B,
                                     mcsparseDirection_t dir, mcsparseIndexBase_t bsr_base, const valType* bsr_vals,
                                     const idxType* bsr_rows_ind, const idxType* bsr_cols_ind, idxType block_dim,
                                     valType h_alpha, valType h_beta, const valType* d_B, idxType ldb, valType* d_C,
                                     idxType ldc) {
    idxType tid = threadIdx.x;
    idxType bsr_row_idx = blockIdx.x / block_dim;
    idxType block_row_idx = blockIdx.x % block_dim;
    idxType b_col = tid / WARP_SIZE;
    idxType segment_idx = tid % WARP_SIZE;

    idxType start_idx = bsr_rows_ind[bsr_row_idx] - bsr_base;
    idxType end_idx = bsr_rows_ind[bsr_row_idx + 1] - bsr_base;

    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType* shared_A = reinterpret_cast<valType*>(smem);

    for (idxType col_idx = b_col; col_idx < n; col_idx += SEGSIZE) {
        valType sum = 0;
        idxType warp_end_idx = start_idx + ((end_idx - start_idx + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        for (idxType idx = start_idx + segment_idx; idx < warp_end_idx; idx += WARP_SIZE) {
            if (b_col == 0 && idx < end_idx) {
                for (idxType i = 0; i < block_dim; ++i) {
                    idxType b_idx;
                    if (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE) {
                        b_idx = col_idx * ldb + (bsr_cols_ind[idx] - bsr_base) * block_dim + i;
                    } else {
                        b_idx = col_idx + ((bsr_cols_ind[idx] - bsr_base) * block_dim + i) * ldb;
                    }
                    if (dir == MCSPARSE_DIRECTION_ROW) {
                        shared_A[segment_idx * block_dim + i] =
                            bsr_vals[idx * block_dim * block_dim + block_row_idx * block_dim + i];
                    } else {
                        shared_A[segment_idx * block_dim + i] =
                            bsr_vals[idx * block_dim * block_dim + block_dim * i + block_row_idx];
                    }
                    sum += h_alpha * shared_A[segment_idx * block_dim + i] * d_B[b_idx];
                }
            }
            __syncthreads();
            if (b_col != 0 && idx < end_idx) {
                for (idxType i = 0; i < block_dim; ++i) {
                    idxType b_idx;
                    if (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE) {
                        b_idx = col_idx * ldb + (bsr_cols_ind[idx] - bsr_base) * block_dim + i;
                    } else {
                        b_idx = col_idx + ((bsr_cols_ind[idx] - bsr_base) * block_dim + i) * ldb;
                    }
                    sum += h_alpha * shared_A[segment_idx * block_dim + i] * d_B[b_idx];
                }
            }
            __syncthreads();
        }
        __syncthreads();
        sum = warpReduceSum<WARP_SIZE>(sum);
        if (segment_idx == (WARP_SIZE - 1)) {
            d_C[ldc * col_idx + bsr_row_idx * block_dim + block_row_idx] =
                h_beta * d_C[ldc * col_idx + bsr_row_idx * block_dim + block_row_idx] + sum;
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int SEGSIZE, typename idxType, typename valType>
__global__ void mcspBsrmmGlobalKernel(idxType mb, idxType n, idxType kb, mcsparseOperation_t trans_B,
                                      mcsparseDirection_t dir, mcsparseIndexBase_t bsr_base, const valType* bsr_vals,
                                      const idxType* bsr_rows_ind, const idxType* bsr_cols_ind, idxType block_dim,
                                      valType h_alpha, valType h_beta, const valType* d_B, idxType ldb, valType* d_C,
                                      idxType ldc) {
    idxType tid = threadIdx.x;
    idxType bsr_row_idx = blockIdx.x / block_dim;
    idxType block_row_idx = blockIdx.x % block_dim;
    idxType b_col = tid / WARP_SIZE;
    idxType segment_idx = tid % WARP_SIZE;

    idxType start_idx = bsr_rows_ind[bsr_row_idx] - bsr_base;
    idxType end_idx = bsr_rows_ind[bsr_row_idx + 1] - bsr_base;

    for (idxType col_idx = b_col; col_idx < n; col_idx += SEGSIZE) {
        valType sum = 0;
        for (idxType idx = start_idx + segment_idx; idx < end_idx; idx += WARP_SIZE) {
            for (idxType i = 0; i < block_dim; ++i) {
                idxType b_idx;
                if (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE) {
                    b_idx = col_idx * ldb + (bsr_cols_ind[idx] - bsr_base) * block_dim + i;
                } else {
                    b_idx = col_idx + ((bsr_cols_ind[idx] - bsr_base) * block_dim + i) * ldb;
                }
                if (dir == MCSPARSE_DIRECTION_ROW) {
                    sum += h_alpha * bsr_vals[idx * block_dim * block_dim + block_row_idx * block_dim + i] * d_B[b_idx];
                } else {
                    sum += h_alpha * bsr_vals[idx * block_dim * block_dim + block_dim * i + block_row_idx] * d_B[b_idx];
                }
            }
        }
        __syncthreads();
        sum = warpReduceSum<WARP_SIZE>(sum);
        if (segment_idx == (WARP_SIZE - 1)) {
            d_C[ldc * col_idx + bsr_row_idx * block_dim + block_row_idx] =
                h_beta * d_C[ldc * col_idx + bsr_row_idx * block_dim + block_row_idx] + sum;
        }
    }
}
#endif