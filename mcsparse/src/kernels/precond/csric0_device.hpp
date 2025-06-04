#ifndef KERNELS_PRECOND_CSRIC0_DEVICE_HPP__
#define KERNELS_PRECOND_CSRIC0_DEVICE_HPP__

#include "block_reduce.hpp"
#include "mcsp_config.h"
#include "mcsp_hashtable_device.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE, unsigned int HASH_SIZE, typename idxType, typename valType>
__global__ void mcspCsric0HashTableKernel(idxType m, valType *csr_vals, const idxType *csr_rows,
                                          const idxType *csr_cols, const idxType *row_map, const idxType *diag_ind,
                                          idxType *zero_pivot, idxType *row_done_flag, mcsparseIndexBase_t idx_base) {
    extern __shared__ __align__(sizeof(idxType)) unsigned char smem[];
    idxType *col_table = reinterpret_cast<idxType *>(smem);
    idxType *idx_table = reinterpret_cast<idxType *>(col_table + HASH_SIZE);
    valType *sum_data = reinterpret_cast<valType *>(col_table + 2 * HASH_SIZE);

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
        const idxType row = row_map[gid];
        idxType row_start = csr_rows[row] - idx_base;
        idxType row_end = csr_rows[row + 1] - idx_base;
        idxType row_diag_idx = diag_ind[row];
        for (idxType i = row_start + lane; i < row_end; i += WARP_SIZE) {
            idxType col = csr_cols[i] - idx_base;
            mcspInsertHashTablePairs<HASH_SIZE, 79>(col, i, cur_col_table, cur_idx_table);
        }
#ifndef __MACA__
        __syncwarp();
#endif
        for (idxType i = row_start; i < row_diag_idx; i++) {
            idxType col = csr_cols[i] - idx_base;
            idxType row_ref_start = csr_rows[col] - idx_base;
            idxType row_ref_end = csr_rows[col + 1] - idx_base;
            idxType row_ref_diag_idx = diag_ind[col];

            while (!atomicOr(&(row_done_flag[col]), 0)) {
                ;
            }
            __threadfence();
            valType val_ref_diag = csr_vals[row_ref_diag_idx];
            sum_data[lane] = 0;
            valType inner_sum = 0;
            for (idxType j = row_ref_start + lane; j < row_ref_diag_idx; j += WARP_SIZE) {
                idxType local_col = csr_cols[j] - idx_base;
                int hash_found;
                idxType local_idx;
                mcspReadHashTablePairs<HASH_SIZE, 79>(local_col, cur_col_table, cur_idx_table, &local_idx, &hash_found);
                if (hash_found == 1) {
                    if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                  std::is_same_v<valType, mcspComplexDouble>) {
                        inner_sum += std::conj(csr_vals[local_idx]) * csr_vals[j];
                    } else {
                        inner_sum += csr_vals[local_idx] * csr_vals[j];
                    }
                }
            }
            sum_data[lane] = inner_sum;
#ifndef __MACA__
            __syncwarp();
#endif
            mcprim::intra_block_reduce<WARP_SIZE>(lane, sum_data, mcprim::plus<valType>());
            if (lane == 0) {
                if (val_ref_diag == static_cast<valType>(0)) {
                    atomicMin(zero_pivot, col);
                    break;
                }
                val_ref_diag = static_cast<valType>(1) / val_ref_diag;
                csr_vals[i] = csr_vals[i] - sum_data[0];
                valType val_cur = csr_vals[i];
                val_cur *= val_ref_diag;
                csr_vals[i] = val_cur;
            }
        }

        __syncthreads();
        sum_data[lane] = 0;
        valType local_sum = 0;
        for (idxType i = row_start + lane; i < row_diag_idx; i += WARP_SIZE) {
            valType val_cur = csr_vals[i];
            if constexpr (std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble>) {
                local_sum += val_cur * std::conj(val_cur);
            } else {
                local_sum += val_cur * val_cur;
            }
        }
        sum_data[lane] = local_sum;
#ifndef __MACA__
        __syncwarp();
#endif
        mcprim::intra_block_reduce<WARP_SIZE>(lane, sum_data, mcprim::plus<valType>());
        if (lane == 0) {
            valType temp_val = csr_vals[row_diag_idx] - sum_data[0];
            csr_vals[row_diag_idx] = std::sqrt(std::abs(temp_val));
            if constexpr (std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble>) {
                if (temp_val.x <= 0) {
                    atomicMin(zero_pivot, row + idx_base);
                }
            } else {
                if (temp_val <= 0) {
                    atomicMin(zero_pivot, row + idx_base);
                }
            }
        }

        __threadfence();
        if (lane == 0) {
            atomicOr(&(row_done_flag[row]), 1);
        }
    }
}

template <typename idxType, typename valType>
__global__ void mcspCsric0BsearchKernel(idxType m, valType *csr_vals, const idxType *csr_rows, const idxType *csr_cols,
                                        const idxType *row_map, const idxType *diag_ind, idxType *zero_pivot,
                                        idxType *row_done_flag, mcsparseIndexBase_t idx_base) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *sum_data = reinterpret_cast<valType *>(smem);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;
    if (gid < m) {
        const idxType row = row_map[gid];
        idxType row_start = csr_rows[row] - idx_base;
        idxType row_end = csr_rows[row + 1] - idx_base;
        idxType row_diag_idx = diag_ind[row];
        for (idxType i = row_start; i < row_diag_idx; i++) {
            idxType col = csr_cols[i] - idx_base;
            idxType row_ref_start = csr_rows[col] - idx_base;
            idxType row_ref_end = csr_rows[col + 1] - idx_base;
            idxType row_ref_diag_idx = diag_ind[col];

            while (!atomicOr(&(row_done_flag[col]), 0)) {
                ;
            }
            __threadfence();
            valType val_ref_diag = csr_vals[row_ref_diag_idx];
            sum_data[lane] = 0;
            valType inner_sum = 0;
            for (idxType j = row_ref_start + lane; j < row_ref_diag_idx; j += WARP_SIZE) {
                idxType local_col = csr_cols[j] - idx_base;
                int left = row_start + 1;
                int right = row_end - 1;
                int mid;
                idxType local_idx;
                idxType col_mid;
                idxType bsearch_found = 0;
                while (left <= right) {
                    mid = (left + right) / 2;
                    col_mid = csr_cols[mid] - idx_base;
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
                    if constexpr (std::is_same_v<valType, mcspComplexFloat> ||
                                  std::is_same_v<valType, mcspComplexDouble>) {
                        inner_sum += std::conj(csr_vals[local_idx]) * csr_vals[j];
                    } else {
                        inner_sum += csr_vals[local_idx] * csr_vals[j];
                    }
                }
            }
            sum_data[lane] = inner_sum;
#ifndef __MACA__
            __syncwarp();
#endif
            mcprim::intra_block_reduce<WARP_SIZE>(lane, sum_data, mcprim::plus<valType>());
            if (lane == 0) {
                if (val_ref_diag == static_cast<valType>(0)) {
                    atomicMin(zero_pivot, col);
                    break;
                }
                csr_vals[i] = csr_vals[i] - sum_data[0];
                valType val_cur = csr_vals[i];
                val_ref_diag = static_cast<valType>(1) / val_ref_diag;
                val_cur *= val_ref_diag;
                csr_vals[i] = val_cur;
            }
        }

        __syncthreads();
        sum_data[lane] = 0;
        valType local_sum = 0;
        for (idxType i = row_start + lane; i < row_diag_idx; i += WARP_SIZE) {
            valType val_cur = csr_vals[i];
            if constexpr (std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble>) {
                local_sum += val_cur * std::conj(val_cur);
            } else {
                local_sum += val_cur * val_cur;
            }
        }
        sum_data[lane] = local_sum;
#ifndef __MACA__
        __syncwarp();
#endif
        mcprim::intra_block_reduce<WARP_SIZE>(lane, sum_data, mcprim::plus<valType>());
        if (lane == 0) {
            valType temp_val = csr_vals[row_diag_idx] - sum_data[0];
            csr_vals[row_diag_idx] = std::sqrt(std::abs(temp_val));
            if constexpr (std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble>) {
                if (temp_val.x < 0) {
                    atomicMin(zero_pivot, row + idx_base);
                }
            } else {
                if (temp_val < 0) {
                    atomicMin(zero_pivot, row + idx_base);
                }
            }
        }

        __threadfence();
        if (lane == 0) {
            atomicOr(&(row_done_flag[row]), 1);
        }
    }
}
#endif