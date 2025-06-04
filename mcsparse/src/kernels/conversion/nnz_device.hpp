#ifndef KERNELS_CONVERSION_NNZ_DEVICE_HPP__
#define KERNELS_CONVERSION_NNZ_DEVICE_HPP__

#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE, uint32_t SEGMENTS_PER_BLOCK, uint32_t SEGMENT_SIZE, typename valType, typename idxType>
__global__ void nnzKernelRow(mcspInt m, mcspInt n, mcspInt lda, const valType* dense_matrix,
                             idxType* nnz_per_row_or_column, valType tol) {
    const uint32_t segment_id = threadIdx.x / SEGMENT_SIZE;
    const uint32_t segment_lan_id = threadIdx.x % SEGMENT_SIZE;
    const uint32_t row_idx = SEGMENTS_PER_BLOCK * blockIdx.x + segment_id;
    idxType end_idx = lda * n;
    idxType start_idx = row_idx;
    if (row_idx < m) {
        uint32_t count = 0;
        for (idxType i = start_idx + (segment_lan_id * lda); i < end_idx; i += (SEGMENT_SIZE * lda)) {
            const valType cur_value = dense_matrix[i];
#if defined(__MACA__)
            // white list: maca platform supports S,D,C,Z,H,half_complex,BF16,bf16_complex data types
            if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double> ||
                          std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble> ||
                          std::is_same_v<valType, __half> || std::is_same_v<valType, __half2> ||
                          std::is_same_v<valType, mcsp_bfloat16> || std::is_same_v<valType, mcsp_bfloat162>) {
                if (std::abs(cur_value) > std::abs(tol)) {
                    count++;
                }
            }
#else
            if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double> ||
                          std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble> ||
                          std::is_same_v<valType, __half>) {
                if (std::abs(cur_value) > std::abs(tol)) {
                    count++;
                }
            }
#endif
        }
        count = warpReduceSum<SEGMENT_SIZE>(count);
        if (segment_lan_id == (SEGMENT_SIZE - 1)) nnz_per_row_or_column[row_idx] = count;
    }
}

template <uint32_t BLOCK_SIZE, uint32_t SEGMENTS_PER_BLOCK, uint32_t SEGMENT_SIZE, typename valType, typename idxType>
__global__ void nnzKernelColumn(mcspInt m, mcspInt n, mcspInt lda, const valType* dense_matrix,
                                idxType* nnz_per_row_or_column, valType tol) {
    const uint32_t segment_id = threadIdx.x / SEGMENT_SIZE;
    const uint32_t segment_lan_id = threadIdx.x % SEGMENT_SIZE;
    const uint32_t col_idx = SEGMENTS_PER_BLOCK * blockIdx.x + segment_id;

    if (col_idx < n) {
        idxType start_idx = col_idx * lda;
        idxType end_idx = col_idx * lda + m;

        uint32_t count = 0;
        for (idxType i = start_idx + segment_lan_id; i < end_idx; i += SEGMENT_SIZE) {
            const valType cur_value = dense_matrix[i];
#if defined(__MACA__)
            // white list: maca platform supports S,D,C,Z,H,half_complex,BF16,bf16_complex data types
            if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double> ||
                          std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble> ||
                          std::is_same_v<valType, __half> || std::is_same_v<valType, __half2> ||
                          std::is_same_v<valType, mcsp_bfloat16> || std::is_same_v<valType, mcsp_bfloat162>) {
                if (std::abs(cur_value) > std::abs(tol)) {
                    count++;
                }
            }
#else
            if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double> ||
                          std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble> ||
                          std::is_same_v<valType, __half>) {
                if (std::abs(cur_value) > std::abs(tol)) {
                    count++;
                }
            }
#endif
        }
        count = warpReduceSum<SEGMENT_SIZE>(count);
        if (segment_lan_id == (SEGMENT_SIZE - 1)) nnz_per_row_or_column[col_idx] = count;
    }
}

#endif