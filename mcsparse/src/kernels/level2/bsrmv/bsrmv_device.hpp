#ifndef KERNELS_BSRMV_DEVICE_HPP__
#define KERNELS_BSRMV_DEVICE_HPP__

#include "block_reduce.hpp"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspBsrmvZeroAlphaKernel(idxType y_dim, valType h_beta, valType *y) {
    idxType tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
    if (tid < y_dim) {
        y[tid] = h_beta * y[tid];
    }
}

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspBsrmvKernel(idxType mb, idxType nb, mcsparseDirection_t dir, mcsparseIndexBase_t bsr_base,
                                const valType *bsr_vals, const idxType *bsr_rows_ind, const idxType *bsr_cols_ind,
                                idxType block_dim, valType h_alpha, const valType *x, valType h_beta, valType *y) {
    idxType tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
    idxType bsr_row_idx = tid / (BLOCKSIZE * block_dim);
    idxType block_row_idx = (tid % (BLOCKSIZE * block_dim)) / BLOCKSIZE;
    idxType segment_idx = (tid % (BLOCKSIZE * block_dim)) % BLOCKSIZE;

    idxType start_idx = bsr_rows_ind[bsr_row_idx] - bsr_base;
    idxType end_idx = bsr_rows_ind[bsr_row_idx + 1] - bsr_base;

    valType sum = 0;
    for (idxType idx = start_idx + segment_idx; idx < end_idx; idx += BLOCKSIZE) {
        for (idxType i = 0; i < block_dim; ++i) {
            if (dir == MCSPARSE_DIRECTION_ROW) {
                sum += h_alpha * bsr_vals[idx * block_dim * block_dim + block_row_idx * block_dim + i] *
                       x[(bsr_cols_ind[idx] - bsr_base) * block_dim + i];
            } else {
                sum += h_alpha * bsr_vals[idx * block_dim * block_dim + block_dim * i + block_row_idx] *
                       x[(bsr_cols_ind[idx] - bsr_base) * block_dim + i];
            }
        }
    }
    __syncthreads();
    sum = warpReduceSum<BLOCKSIZE>(sum);

    if (segment_idx == (BLOCKSIZE - 1)) {
        y[bsr_row_idx * block_dim + block_row_idx] = h_beta * y[bsr_row_idx * block_dim + block_row_idx] + sum;
    }
}
#endif