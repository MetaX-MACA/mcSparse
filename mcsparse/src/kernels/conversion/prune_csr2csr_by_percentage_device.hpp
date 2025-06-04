#ifndef KERNELS_CONVERSION_PRUNE_DENSE2CSR_BY_PERCENTAGE_DEVICE_HPP__
#define KERNELS_CONVERSION_PRUNE_DENSE2CSR_BY_PERCENTAGE_DEVICE_HPP__

#include "mcsp_runtime_wrapper.h"
template <uint32_t BLOCK_SIZE, typename valType, typename idxType>
__global__ void absCsrValueKernel(idxType nnz, const valType *input_matrix, valType *output_matrix) {
    const uint32_t idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    if (idx < nnz) {
        output_matrix[idx] = std::abs(input_matrix[idx]);
    }
}

#endif