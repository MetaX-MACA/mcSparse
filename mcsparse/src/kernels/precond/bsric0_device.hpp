#ifndef KERNELS_PRECOND_BSRIC0_DEVICE_HPP__
#define KERNELS_PRECOND_BSRIC0_DEVICE_HPP__

#include "block_reduce.hpp"
#include "mcsp_config.h"
#include "mcsp_general_utility.h"
#include "mcsp_hashtable_device.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE, unsigned int HASH_SIZE, typename idxType, typename valType>
__global__ void mcspBsric0HashTableKernel(mcsparseDirection_t dir, idxType m, idxType block_dim, valType *bsr_vals,
                                          const idxType *bsr_rows, const idxType *bsr_cols, const idxType *row_map,
                                          const idxType *diag_ind, idxType *zero_pivot, idxType *row_done_flag,
                                          mcsparseIndexBase_t idx_base) {
    extern __shared__ __align__(sizeof(idxType)) unsigned char smem[];
    idxType *col_table = reinterpret_cast<idxType *>(smem);
    idxType *idx_table = reinterpret_cast<idxType *>(col_table + HASH_SIZE);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;

    idxType *cur_col_table = col_table + wid * HASH_SIZE;
    idxType *cur_idx_table = idx_table + wid * HASH_SIZE;
    for (idxType i = lane; i < HASH_SIZE; i += WARP_SIZE) {
        cur_col_table[i] = (idxType)HASH_MAGIC_NULL_VAL;
        cur_idx_table[i] = (idxType)0;
    }
#ifndef __MACA__
    __syncwarp();
#endif
    if (gid < m) {
        const idxType bsr_row = row_map[gid];
        idxType block_row_begin = bsr_rows[bsr_row] - idx_base;
        idxType row_end = bsr_rows[bsr_row + 1] - idx_base;
        idxType block_row_diag = diag_ind[bsr_row];
        for (idxType i = block_row_begin + lane; i < row_end; i += WARP_SIZE) {
            idxType col = bsr_cols[i] - idx_base;
            mcspInsertHashTablePairs<HASH_SIZE, 79>(col, i, cur_col_table, cur_idx_table);
        }
#ifndef __MACA__
        __syncwarp();
#endif
        for (idxType row = lane; row < block_dim; row += WARP_SIZE) {
            valType row_sum = static_cast<valType>(0);

            // Process lower diagonal
            for (idxType row_idx = block_row_begin; row_idx < block_row_diag; row_idx++) {
                idxType block_col = bsr_cols[row_idx] - idx_base;
                idxType local_block_begin = bsr_rows[block_col] - idx_base;
                idxType local_block_diag = diag_ind[block_col];

                while (!atomicOr(&(row_done_flag[block_col]), 0)) {
                    ;
                }
                __threadfence();

                for (idxType col_idx = 0; col_idx < block_dim; ++col_idx) {
                    idxType cur_col = block_col * block_dim + col_idx;
                    valType diag_val =
                        bsr_vals[block_dim * block_dim * local_block_diag + block_dim * col_idx + col_idx];
                    valType cur_val = bsr_vals[BSR_IND(row_idx, row, col_idx, dir, block_dim)];

                    valType local_sum = static_cast<valType>(0);
                    for (idxType j = local_block_begin; j < local_block_diag + 1; ++j) {
                        idxType local_col = bsr_cols[j] - idx_base;
                        int hash_found;
                        idxType local_idx;
                        mcspReadHashTablePairs<HASH_SIZE, 79>(local_col, cur_col_table, cur_idx_table, &local_idx,
                                                              &hash_found);

                        if (hash_found == 1) {
                            for (idxType k = 0; k < block_dim; ++k) {
                                if (block_dim * local_col + k < cur_col) {
                                    valType val_j = bsr_vals[BSR_IND(j, col_idx, k, dir, block_dim)];
                                    valType val_cur_row = bsr_vals[BSR_IND(local_idx, row, k, dir, block_dim)];
                                    if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                                  std::is_same_v<valType, mcspComplexDouble>) {
                                        local_sum += std::conj(val_cur_row) * val_j;
                                    } else {
                                        local_sum += val_cur_row * val_j;
                                    }
                                }
                            }
                        }
                    }

                    cur_val = (cur_val - local_sum) / diag_val;
                    bsr_vals[BSR_IND(row_idx, row, col_idx, dir, block_dim)] = cur_val;

                    if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                  std::is_same_v<valType, mcspComplexDouble>) {
                        row_sum += std::conj(cur_val) * cur_val;
                    } else {
                        row_sum += cur_val * cur_val;
                    }
                }
            }

            // Process diagonal
            for (idxType block_col = 0; block_col < block_dim; ++block_col) {
                idxType row_diag = block_dim * block_dim * block_row_diag + block_dim * block_col + block_col;
                valType diag_val = bsr_vals[row_diag];
                if (row == block_col) {
                    diag_val = diag_val - row_sum;
                    if constexpr (std::is_same_v<valType, mcspComplexFloat>) {
                        if (diag_val.x <= static_cast<float>(0)) {
                            atomicMin(zero_pivot, bsr_row + idx_base);
                        }
                        bsr_vals[row_diag] = std::sqrt(diag_val.x);
                    } else if constexpr (std::is_same_v<valType, mcspComplexDouble>) {
                        if (diag_val.x <= static_cast<double>(0)) {
                            atomicMin(zero_pivot, bsr_row + idx_base);
                        }
                        bsr_vals[row_diag] = std::sqrt(diag_val.x);
                    } else if constexpr (std::is_same_v<valType, float>) {
                        if (diag_val <= static_cast<float>(0)) {
                            atomicMin(zero_pivot, bsr_row + idx_base);
                        }
                        bsr_vals[row_diag] = std::sqrt(diag_val);
                    } else if constexpr (std::is_same_v<valType, double>) {
                        if (diag_val <= static_cast<double>(0)) {
                            atomicMin(zero_pivot, bsr_row + idx_base);
                        }
                        bsr_vals[row_diag] = std::sqrt(diag_val);
                    }
                }
                __threadfence();
                diag_val = bsr_vals[row_diag];

                if (block_col < row) {
                    valType cur_val = bsr_vals[BSR_IND(block_row_diag, row, block_col, dir, block_dim)];
                    valType local_sum = static_cast<valType>(0);
                    valType val_cur_col;
                    valType val_cur_row;
                    for (idxType k = block_row_begin; k < block_row_diag; ++k) {
                        for (idxType idx = 0; idx < block_dim; ++idx) {
                            val_cur_col = bsr_vals[BSR_IND(k, block_col, idx, dir, block_dim)];
                            val_cur_row = bsr_vals[BSR_IND(k, row, idx, dir, block_dim)];
                            if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                          std::is_same_v<valType, mcspComplexDouble>) {
                                local_sum += std::conj(val_cur_row) * val_cur_col;
                            } else {
                                local_sum += val_cur_row * val_cur_col;
                            }
                        }
                    }

                    for (idxType idx = 0; idx < block_col; ++idx) {
                        val_cur_col = bsr_vals[BSR_IND(block_row_diag, block_col, idx, dir, block_dim)];
                        val_cur_row = bsr_vals[BSR_IND(block_row_diag, row, idx, dir, block_dim)];
                        if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                      std::is_same_v<valType, mcspComplexDouble>) {
                            local_sum += std::conj(val_cur_row) * val_cur_col;
                        } else {
                            local_sum += val_cur_row * val_cur_col;
                        }
                    }
                    cur_val = (cur_val - local_sum) / diag_val;
                    if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                  std::is_same_v<valType, mcspComplexDouble>) {
                        row_sum += std::conj(cur_val) * cur_val;
                    } else {
                        row_sum += cur_val * cur_val;
                    }
                    bsr_vals[BSR_IND(block_row_diag, row, block_col, dir, block_dim)] = cur_val;
                }
            }
        }

        __threadfence();
        if (lane == 0) {
            atomicOr(&(row_done_flag[bsr_row]), 1);
        }
    }
}

template <typename idxType, typename valType>
__global__ void mcspBsric0BsearchKernel(mcsparseDirection_t dir, idxType m, idxType block_dim, valType *bsr_vals,
                                        const idxType *bsr_rows, const idxType *bsr_cols, const idxType *row_map,
                                        const idxType *diag_ind, idxType *zero_pivot, idxType *row_done_flag,
                                        mcsparseIndexBase_t idx_base) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;
    if (gid < m) {
        const idxType bsr_row = row_map[gid];
        idxType block_row_begin = bsr_rows[bsr_row] - idx_base;
        idxType row_end = bsr_rows[bsr_row + 1] - idx_base;
        idxType block_row_diag = diag_ind[bsr_row];

        for (idxType row = lane; row < block_dim; row += WARP_SIZE) {
            valType row_sum = static_cast<valType>(0);

            // Process lower diagonal
            for (idxType row_idx = block_row_begin; row_idx < block_row_diag; row_idx++) {
                idxType block_col = bsr_cols[row_idx] - idx_base;
                idxType local_block_begin = bsr_rows[block_col] - idx_base;
                idxType local_block_diag = diag_ind[block_col];

                while (!atomicOr(&(row_done_flag[block_col]), 0)) {
                    ;
                }
                __threadfence();

                for (idxType col_idx = 0; col_idx < block_dim; ++col_idx) {
                    idxType cur_col = block_col * block_dim + col_idx;
                    valType diag_val =
                        bsr_vals[block_dim * block_dim * local_block_diag + block_dim * col_idx + col_idx];
                    valType cur_val = bsr_vals[BSR_IND(row_idx, row, col_idx, dir, block_dim)];

                    valType local_sum = static_cast<valType>(0);
                    for (idxType j = local_block_begin; j < local_block_diag + 1; ++j) {
                        idxType local_col = bsr_cols[j] - idx_base;
                        int left = block_row_begin;
                        int right = row_end - 1;
                        int mid;
                        idxType local_idx;
                        idxType col_mid;
                        idxType bsearch_found = 0;
                        while (left <= right) {
                            mid = (left + right) / 2;
                            col_mid = bsr_cols[mid] - idx_base;
                            if (col_mid == local_col) {
                                local_idx = mid;
                                bsearch_found = 1;
                                break;
                            } else if (col_mid < local_col) {
                                left = mid + 1;
                            } else {
                                right = mid - 1;
                            }
                        }
                        if (bsearch_found == 1) {
                            for (idxType k = 0; k < block_dim; ++k) {
                                if (block_dim * local_col + k < cur_col) {
                                    valType val_j = bsr_vals[BSR_IND(j, col_idx, k, dir, block_dim)];
                                    valType val_cur_row = bsr_vals[BSR_IND(local_idx, row, k, dir, block_dim)];
                                    if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                                  std::is_same_v<valType, mcspComplexDouble>) {
                                        local_sum += std::conj(val_cur_row) * val_j;
                                    } else {
                                        local_sum += val_cur_row * val_j;
                                    }
                                }
                            }
                        }
                    }
                    cur_val = (cur_val - local_sum) / diag_val;
                    bsr_vals[BSR_IND(row_idx, row, col_idx, dir, block_dim)] = cur_val;

                    if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                  std::is_same_v<valType, mcspComplexDouble>) {
                        row_sum += std::conj(cur_val) * cur_val;
                    } else {
                        row_sum += cur_val * cur_val;
                    }
                }
            }

            // Process diagonal
            for (idxType block_col = 0; block_col < block_dim; ++block_col) {
                idxType row_diag = block_dim * block_dim * block_row_diag + block_dim * block_col + block_col;
                valType diag_val = bsr_vals[row_diag];
                if (row == block_col) {
                    diag_val = diag_val - row_sum;
                    if constexpr (std::is_same_v<valType, mcspComplexFloat>) {
                        if (diag_val.x <= static_cast<float>(0)) {
                            atomicMin(zero_pivot, bsr_row + idx_base);
                        }
                        bsr_vals[row_diag] = std::sqrt(diag_val.x);
                    } else if constexpr (std::is_same_v<valType, mcspComplexDouble>) {
                        if (diag_val.x <= static_cast<double>(0)) {
                            atomicMin(zero_pivot, bsr_row + idx_base);
                        }
                        bsr_vals[row_diag] = std::sqrt(diag_val.x);
                    } else if constexpr (std::is_same_v<valType, float>) {
                        if (diag_val <= static_cast<float>(0)) {
                            atomicMin(zero_pivot, bsr_row + idx_base);
                        }
                        bsr_vals[row_diag] = std::sqrt(diag_val);
                    } else if constexpr (std::is_same_v<valType, double>) {
                        if (diag_val <= static_cast<double>(0)) {
                            atomicMin(zero_pivot, bsr_row + idx_base);
                        }
                        bsr_vals[row_diag] = std::sqrt(diag_val);
                    }
                }
                __threadfence();
                diag_val = bsr_vals[row_diag];

                if (block_col < row) {
                    valType cur_val = bsr_vals[BSR_IND(block_row_diag, row, block_col, dir, block_dim)];
                    valType local_sum = static_cast<valType>(0);
                    valType val_cur_col;
                    valType val_cur_row;
                    for (idxType k = block_row_begin; k < block_row_diag; ++k) {
                        for (idxType idx = 0; idx < block_dim; ++idx) {
                            val_cur_col = bsr_vals[BSR_IND(k, block_col, idx, dir, block_dim)];
                            val_cur_row = bsr_vals[BSR_IND(k, row, idx, dir, block_dim)];
                            if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                          std::is_same_v<valType, mcspComplexDouble>) {
                                local_sum += std::conj(val_cur_row) * val_cur_col;
                            } else {
                                local_sum += val_cur_row * val_cur_col;
                            }
                        }
                    }

                    for (idxType idx = 0; idx < block_col; ++idx) {
                        val_cur_col = bsr_vals[BSR_IND(block_row_diag, block_col, idx, dir, block_dim)];
                        val_cur_row = bsr_vals[BSR_IND(block_row_diag, row, idx, dir, block_dim)];
                        if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                      std::is_same_v<valType, mcspComplexDouble>) {
                            local_sum += std::conj(val_cur_row) * val_cur_col;
                        } else {
                            local_sum += val_cur_row * val_cur_col;
                        }
                    }
                    cur_val = (cur_val - local_sum) / diag_val;
                    if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                  std::is_same_v<valType, mcspComplexDouble>) {
                        row_sum += std::conj(cur_val) * cur_val;
                    } else {
                        row_sum += cur_val * cur_val;
                    }
                    bsr_vals[BSR_IND(block_row_diag, row, block_col, dir, block_dim)] = cur_val;
                }
            }
        }
        __threadfence();
        if (lane == 0) {
            atomicOr(&(row_done_flag[bsr_row]), 1);
        }
    }
}
#endif