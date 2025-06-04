#ifndef KERNELS_EXTRA_SPGEMM_SPGEMM_REUSE_DEVICE_HPP__
#define KERNELS_EXTRA_SPGEMM_SPGEMM_REUSE_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE, uint32_t SEGE_SIZE, typename idxType, typename valType>
__global__ void mcspSpgemmReuseKernel(const idxType *csr_rows_A, const idxType *csr_cols_A, const valType *csr_vals_A,
                                      mcsparseIndexBase_t base_A, const idxType *csr_rows_B, const idxType *csr_cols_B,
                                      const valType *csr_vals_B, mcsparseIndexBase_t base_B, idxType *csr_rows_C,
                                      idxType *csr_cols_C, valType *csr_vals_C, mcsparseIndexBase_t base_C) {
    idxType row_idx_A = blockIdx.x;
    idxType tid = threadIdx.x;
    idxType warp_idx = tid / WARP_SIZE;
    idxType warp_lane = tid % WARP_SIZE;

    idxType start_idx_A = csr_rows_A[row_idx_A] - base_A;
    idxType end_idx_A = csr_rows_A[row_idx_A + 1] - base_A;

    idxType start_idx_C = csr_rows_C[row_idx_A] - base_C;
    idxType end_idx_C = csr_rows_C[row_idx_A + 1] - base_C;
    idxType col_start_idx_C = csr_cols_C[start_idx_C] - base_C;

    for (int32_t col_idx_A = start_idx_A + warp_idx; col_idx_A < end_idx_A; col_idx_A += SEGE_SIZE) {
        idxType row_idx_B = csr_cols_A[col_idx_A] - base_A;
        valType cur_val_A = csr_vals_A[col_idx_A];
        idxType start_idx_B = csr_rows_B[row_idx_B] - base_B;
        idxType end_idx_B = csr_rows_B[row_idx_B + 1] - base_B;
        idxType cur_start_idx_B = start_idx_B;

        for (int32_t col_idx_C = start_idx_C + warp_lane; col_idx_C < end_idx_C; col_idx_C += WARP_SIZE) {
            idxType cur_col_C = csr_cols_C[col_idx_C] - base_C;
            for (int32_t col_idx_B = cur_start_idx_B; col_idx_B < end_idx_B; ++col_idx_B) {
                if ((csr_cols_B[col_idx_B] - base_B) == cur_col_C) {
                    cur_start_idx_B = col_idx_B;
                    if constexpr (std::is_same_v<valType, mcspComplexDouble> ||
                                  std::is_same_v<valType, mcspComplexFloat>) {
                        complexAtomicAddByPart_(&(csr_vals_C[col_idx_C]), cur_val_A * csr_vals_B[col_idx_B]);
#if defined(__MACA__)
                    } else if constexpr (std::is_same_v<valType, half2> || std::is_same_v<valType, mcsp_bfloat162>) {
                        atomicAdd(&(csr_vals_C[col_idx_C]), complex_mul(cur_val_A, csr_vals_B[col_idx_B]));
#endif
                    } else {
                        atomicAdd_(&(csr_vals_C[col_idx_C]), cur_val_A * csr_vals_B[col_idx_B]);
                    }
                    break;
                }
            }
        }
    }
}

#endif