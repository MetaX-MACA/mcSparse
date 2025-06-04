#ifndef KERNELS_CONVERSION_PRUNE_DENSE2CSR_BY_PERCENTAGE_DEVICE_HPP__
#define KERNELS_CONVERSION_PRUNE_DENSE2CSR_BY_PERCENTAGE_DEVICE_HPP__

#include "mcsp_runtime_wrapper.h"
template <uint32_t BLOCK_SIZE, typename valType, typename idxType>
__global__ void absDenseMatrixKernel(idxType m, idxType n, idxType lda, const valType *input_matrix,
                                     valType *output_matrix) {
    const uint64_t idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (idx < (m * n)) {
        const uint64_t row_idx = idx % m;
        const uint64_t col_idx = idx / m;
        output_matrix[col_idx * m + row_idx] = std::abs(input_matrix[col_idx * lda + row_idx]);
    }
}

#endif