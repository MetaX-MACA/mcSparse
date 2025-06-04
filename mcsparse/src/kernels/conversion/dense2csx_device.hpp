#ifndef KERNELS_CONVERSION_DENSE2CSX_DEVICE_HPP__
#define KERNELS_CONVERSION_DENSE2CSX_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE, uint32_t SEGMENTS_PER_BLOCK, uint32_t SEGMENT_SIZE, uint32_t WP_SIZE, typename valType,
          typename idxType>
__global__ void dense2csrKernel(idxType m, idxType n, idxType lda, const valType* dense_matrix,
                                mcsparseIndexBase_t idx_base, valType* csr_vals, const idxType* csr_rows,
                                idxType* csr_cols, valType tol) {
    const idxType segment_id = threadIdx.x / SEGMENT_SIZE;
    const idxType segment_lane_id = threadIdx.x % SEGMENT_SIZE;

    const idxType segment_id_in_warp = segment_id % (WP_SIZE / SEGMENT_SIZE);
    constexpr idxType bit_size = 63;

    const uint64_t filter_mask = (UINT64_BIT_MASK >> (bit_size - segment_lane_id));
    const uint64_t shifted_filter_mask = filter_mask << (SEGMENT_SIZE * segment_id_in_warp);

    const idxType row_idx = SEGMENTS_PER_BLOCK * blockIdx.x + segment_id;

    idxType end_idx = lda * n;
    idxType start_idx = row_idx;

    if (row_idx < m) {
        idxType row_start = csr_rows[row_idx] - idx_base;
        for (idxType i = start_idx + (segment_lane_id * lda); i < end_idx; i += (SEGMENT_SIZE * lda)) {
            const valType cur_value = dense_matrix[i];
            bool predicate = false;
#if defined(__MACA__)
            // white list: maca platform supports S,D,C,Z,H,half_complex,BF16,bf16_complex data types
            if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double> ||
                          std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble> ||
                          std::is_same_v<valType, __half> || std::is_same_v<valType, __half2> ||
                          std::is_same_v<valType, mcsp_bfloat16> || std::is_same_v<valType, mcsp_bfloat162>) {
                predicate = std::abs(cur_value) > std::abs(tol) ? true : false;
            }
#else
            if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double> ||
                          std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble> ||
                          std::is_same_v<valType, __half>) {
                predicate = std::abs(cur_value) > std::abs(tol) ? true : false;
            }
#endif

#ifndef __MACA__
            const uint64_t wavefront_mask = __ballot_sync(UINT32_BIT_MASK, predicate);
#else
            const uint64_t wavefront_mask = __ballot(predicate);
#endif
            const uint64_t count_previous_nnzs = __popcll(wavefront_mask & shifted_filter_mask);
            if (predicate) {
                csr_vals[row_start + count_previous_nnzs - 1] = cur_value;
                csr_cols[row_start + count_previous_nnzs - 1] = i / lda + idx_base;
            }
#ifndef __MACA__
            row_start += __shfl_sync(UINT32_BIT_MASK, count_previous_nnzs, SEGMENT_SIZE - 1, SEGMENT_SIZE);
#else
            row_start += __shfl(count_previous_nnzs, SEGMENT_SIZE - 1, SEGMENT_SIZE);
#endif
        }
    }
}

template <uint32_t BLOCK_SIZE, uint32_t SEGMENTS_PER_BLOCK, uint32_t SEGMENT_SIZE, uint32_t WP_SIZE, typename valType,
          typename idxType>
__global__ void dense2cscKernel(idxType m, idxType n, idxType lda, const valType* dense_matrix,
                                mcsparseIndexBase_t idx_base, valType* csc_vals, idxType* csc_rows,
                                const idxType* csc_cols, valType tol) {
    const idxType segment_id = threadIdx.x / SEGMENT_SIZE;
    const idxType segment_lane_id = threadIdx.x % SEGMENT_SIZE;

    const idxType segment_id_in_warp = segment_id % (WP_SIZE / SEGMENT_SIZE);
    constexpr idxType bit_size = 63;

    const uint64_t filter_mask = (UINT64_BIT_MASK >> (bit_size - segment_lane_id));
    const uint64_t shifted_filter_mask = filter_mask << (SEGMENT_SIZE * segment_id_in_warp);

    const idxType col_idx = SEGMENTS_PER_BLOCK * blockIdx.x + segment_id;

    if (col_idx < n) {
        idxType start_idx = col_idx * lda;
        idxType end_idx = col_idx * lda + m;
        idxType col_start = csc_cols[col_idx] - idx_base;
        for (idxType i = start_idx + segment_lane_id; i < end_idx; i += SEGMENT_SIZE) {
            const valType cur_value = dense_matrix[i];
            bool predicate = false;
#if defined(__MACA__)
            // white list: maca platform supports S,D,C,Z,H,half_complex,BF16,bf16_complex data types
            if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double> ||
                          std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble> ||
                          std::is_same_v<valType, __half> || std::is_same_v<valType, __half2> ||
                          std::is_same_v<valType, mcsp_bfloat16> || std::is_same_v<valType, mcsp_bfloat162>) {
                predicate = std::abs(cur_value) > std::abs(tol) ? true : false;
            }
#else
            if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double> ||
                          std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble> ||
                          std::is_same_v<valType, __half>) {
                predicate = std::abs(cur_value) > std::abs(tol) ? true : false;
            }
#endif

#ifndef __MACA__
            const uint64_t wavefront_mask = __ballot_sync(UINT32_BIT_MASK, predicate);
#else
            const uint64_t wavefront_mask = __ballot(predicate);
#endif
            const uint64_t count_previous_nnzs = __popcll(wavefront_mask & shifted_filter_mask);
            if (predicate) {
                csc_vals[col_start + count_previous_nnzs - 1] = cur_value;
                csc_rows[col_start + count_previous_nnzs - 1] = i % lda + idx_base;
            }
#ifndef __MACA__
            col_start += __shfl_sync(UINT32_BIT_MASK, count_previous_nnzs, SEGMENT_SIZE - 1, SEGMENT_SIZE);
#else
            col_start += __shfl(count_previous_nnzs, SEGMENT_SIZE - 1, SEGMENT_SIZE);
#endif
        }
    }
}

#endif