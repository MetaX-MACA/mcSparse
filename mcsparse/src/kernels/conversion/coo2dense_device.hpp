#ifndef KERNELS_COO2DENSE_DEVICE_HPP__
#define KERNELS_COO2DENSE_DEVICE_HPP__

#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE, typename idxType, typename valType>
__global__ void coo2denseKernel(idxType nnz, idxType lda, valType* dense_matrix, mcsparseIndexBase_t idx_base,
                                const valType* coo_vals, const idxType* coo_rows, const idxType* coo_cols) {
    const idxType idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    if (idx < nnz) {
        const uint64_t row_id = coo_rows[idx] - idx_base;
        const uint64_t col_id = coo_cols[idx] - idx_base;
        const uint64_t dense_id = col_id * lda + row_id;
        dense_matrix[dense_id] = coo_vals[idx];
    }
}

#endif