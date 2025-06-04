#ifndef KERNELS_BSRXMV_DEVICE_HPP__
#define KERNELS_BSRXMV_DEVICE_HPP__

#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspBsrxmvZeroAlphaKernel(idxType mask_size, const idxType *mask_ptr, idxType y_dim, idxType block_dim,
                                          valType h_beta, valType *y) {
    idxType tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
    if (tid < y_dim) {
        idxType bsr_row = tid / block_dim;
        idxType row_lan = tid % block_dim;
        bsr_row = mask_ptr[bsr_row] * block_dim + row_lan;
        y[bsr_row] = h_beta * y[bsr_row];
    }
}

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspBsrxmvKernel(idxType mask_size, idxType mb, idxType nb, mcsparseDirection_t dir,
                                 mcsparseIndexBase_t bsr_base, const valType *bsr_vals, const idxType *mask_ptr,
                                 const idxType *bsr_rows_ind, const idxType *bsr_ends_ind, const idxType *bsr_cols_ind,
                                 idxType block_dim, valType h_alpha, const valType *x, valType h_beta, valType *y) {
    idxType tid = BLOCKSIZE * blockIdx.x + threadIdx.x;
    idxType bsr_row_idx = tid / (BLOCKSIZE * block_dim);
    bsr_row_idx = mask_ptr[bsr_row_idx] - bsr_base;

    idxType block_row_idx = (tid % (BLOCKSIZE * block_dim)) / BLOCKSIZE;
    idxType segment_idx = (tid % (BLOCKSIZE * block_dim)) % BLOCKSIZE;

    idxType start_idx = bsr_rows_ind[bsr_row_idx] - bsr_base;
    idxType end_idx = bsr_ends_ind[bsr_row_idx] - bsr_base;

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