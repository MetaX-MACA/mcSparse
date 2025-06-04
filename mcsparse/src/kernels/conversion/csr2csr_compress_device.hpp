#ifndef KERNELS_CONVERSION_CSR2CSR_COMPRESS_DEVICE_HPP__
#define KERNELS_CONVERSION_CSR2CSR_COMPRESS_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE, typename idxType>
__global__ void FillRowDevice(idxType m, mcsparseIndexBase_t idx_base, const idxType* nnz_per_row, idxType* csr_row_C) {
    idxType idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= m) {
        return;
    }
    csr_row_C[idx + 1] = nnz_per_row[idx];
    if (idx == 0) {
        csr_row_C[0] = idx_base;
    }
}

template <uint32_t BLOCK_SIZE, uint32_t SEGMENTS_PER_BLOCK, uint32_t SEGMENT_SIZE, uint32_t WP_SIZE, typename valType,
          typename idxType>
__global__ void csr2csrCompressKernel(idxType m, idxType n, mcsparseIndexBase_t idx_base_A, const valType* csr_val_A,
                                      const idxType* csr_row_A, const idxType* csr_col_A, idxType nnz_A,
                                      mcsparseIndexBase_t idx_base_C, valType* csr_val_C, const idxType* csr_row_C,
                                      idxType* csr_col_C, valType tol) {
    const idxType segment_id = threadIdx.x / SEGMENT_SIZE;
    const idxType segment_lane_id = threadIdx.x % SEGMENT_SIZE;

    const idxType segment_id_in_warp = segment_id % (WP_SIZE / SEGMENT_SIZE);
    constexpr idxType bit_size = 63;

    const uint64_t filter_mask = (UINT64_BIT_MASK >> (bit_size - segment_lane_id));
    const uint64_t shifted_filter_mask = filter_mask << (SEGMENT_SIZE * segment_id_in_warp);

    const idxType row_index = SEGMENTS_PER_BLOCK * blockIdx.x + segment_id;

    if (row_index < m) {
        const idxType start_A = csr_row_A[row_index] - idx_base_A;
        const idxType end_A = csr_row_A[row_index + 1] - idx_base_A;

        idxType start_C = csr_row_C[row_index] - idx_base_C;

        for (idxType i = start_A + segment_lane_id; i < end_A; i += SEGMENT_SIZE) {
            const valType value = csr_val_A[i];

            const bool predicate = std::abs(value) > std::abs(tol) ? true : false;
#ifndef __MACA__
            const uint64_t wavefront_mask = __ballot_sync(UINT32_BIT_MASK, predicate);
#else
            const uint64_t wavefront_mask = __ballot(predicate);
#endif

            const uint64_t count_previous_nnzs = __popcll(wavefront_mask & shifted_filter_mask);

            if (predicate) {
                csr_val_C[start_C + count_previous_nnzs - 1] = value;
                csr_col_C[start_C + count_previous_nnzs - 1] = (csr_col_A[i] - idx_base_A) + idx_base_C;
            }
#ifndef __MACA__
            start_C +=
                __shfl_sync(UINT32_BIT_MASK, static_cast<int>(count_previous_nnzs), SEGMENT_SIZE - 1, SEGMENT_SIZE);
#else
            start_C += __shfl(static_cast<int>(count_previous_nnzs), SEGMENT_SIZE - 1, SEGMENT_SIZE);
#endif
        }
    }
}

#endif