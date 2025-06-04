#ifndef KERNELS_CONVERSION_NNZ_COMPRESS_DEVICE_HPP__
#define KERNELS_CONVERSION_NNZ_COMPRESS_DEVICE_HPP__

#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE, uint32_t SEGMENTS_PER_BLOCK, uint32_t SEGMENT_SIZE, uint32_t WF_SIZE, typename valType,
          typename idxType>
__global__ void nnzCompressKernel(mcspInt m, mcsparseIndexBase_t idx_base, const valType* csr_val_A,
                                  const idxType* csr_row_A, idxType* nnz_per_row, valType tol) {
    const uint32_t segment_id = threadIdx.x / SEGMENT_SIZE;
    const uint32_t segment_lan_id = threadIdx.x % SEGMENT_SIZE;
    const uint32_t row_idx = SEGMENTS_PER_BLOCK * blockIdx.x + segment_id;

    if (row_idx < m) {
        idxType start_idx = csr_row_A[row_idx] - idx_base;
        idxType end_idx = csr_row_A[row_idx + 1] - idx_base;

        uint32_t count = 0;

        for (idxType i = start_idx + segment_lan_id; i < end_idx; i += SEGMENT_SIZE) {
            const valType cur_value = csr_val_A[i];
            if (std::abs(cur_value) > std::abs(tol)) {
                count++;
            }
        }
        count = warpReduceSum<SEGMENT_SIZE>(count);
        if (segment_lan_id == (SEGMENT_SIZE - 1)) nnz_per_row[row_idx] = count;
    }
}

#endif