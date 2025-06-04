#ifndef KERNELS_PRECOND_CSRILU0_DEVICE_HPP__
#define KERNELS_PRECOND_CSRILU0_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_hashtable_device.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE, unsigned int HASH_SIZE, typename idxType, typename valType>
__global__ void mcspCsrilu0HashTableKernel(idxType m, valType *csr_vals, const idxType *csr_rows,
                                           const idxType *csr_cols, const idxType *row_map, const idxType *diag_ind,
                                           idxType *row_done_flag, mcsparseIndexBase_t idx_base) {
    extern __shared__ __align__(sizeof(idxType)) unsigned char smem[];
    idxType *col_table = reinterpret_cast<idxType *>(smem);
    idxType *idx_table = reinterpret_cast<idxType *>(col_table + HASH_SIZE * BLOCK_SIZE / WARP_SIZE);

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

        idxType col;
        idxType row_ref_start;
        idxType row_ref_end;
        idxType row_ref_diag_idx;
        idxType local_col;
        idxType local_idx;
        valType val_cur;
        valType val_ref_diag;
        int hash_found;
        for (idxType i = row_start + lane; i < row_end; i += WARP_SIZE) {
            col = csr_cols[i] - idx_base;
            mcspInsertHashTablePairs<HASH_SIZE, 79>(col, i, cur_col_table, cur_idx_table);
        }
#ifndef __MACA__
        __syncwarp();
#endif
        for (idxType i = row_start; i < row_diag_idx; i++) {
            col = csr_cols[i] - idx_base;
            val_cur = csr_vals[i];
            row_ref_start = csr_rows[col] - idx_base;
            row_ref_end = csr_rows[col + 1] - idx_base;
            row_ref_diag_idx = diag_ind[col];

            while (!atomicOr(&(row_done_flag[col]), 0)) {
                ;
            }
            __threadfence();
            val_ref_diag = csr_vals[row_ref_diag_idx];
            val_cur /= val_ref_diag;
            csr_vals[i] = val_cur;
            for (idxType j = row_ref_diag_idx + 1 + lane; j < row_ref_end; j += WARP_SIZE) {
                local_col = csr_cols[j] - idx_base;
                mcspReadHashTablePairs<HASH_SIZE, 79>(local_col, cur_col_table, cur_idx_table, &local_idx, &hash_found);
                if (hash_found == 1) {
                    csr_vals[local_idx] = csr_vals[local_idx] - csr_vals[j] * val_cur;
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
__global__ void mcspCsrilu0BsearchKernel(idxType m, valType *csr_vals, const idxType *csr_rows, const idxType *csr_cols,
                                         const idxType *row_map, const idxType *diag_ind, idxType *row_done_flag,
                                         mcsparseIndexBase_t idx_base) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;

    if (gid < m) {
        const idxType row = row_map[gid];

        idxType row_start = csr_rows[row] - idx_base;
        idxType row_end = csr_rows[row + 1] - idx_base;
        idxType row_diag_idx = diag_ind[row];

        idxType col;
        idxType row_ref_start;
        idxType row_ref_end;
        idxType row_ref_diag_idx;
        idxType local_col;
        idxType local_idx;
        valType val_cur;
        valType val_ref_diag;

        int left;
        int right;
        int mid;
        idxType col_mid;

        for (idxType i = row_start; i < row_diag_idx; i++) {
            col = csr_cols[i] - idx_base;
            val_cur = csr_vals[i];
            row_ref_start = csr_rows[col] - idx_base;
            row_ref_end = csr_rows[col + 1] - idx_base;
            row_ref_diag_idx = diag_ind[col];

            while (!atomicOr(&(row_done_flag[col]), 0)) {
                ;
            }
            __threadfence();
            val_ref_diag = csr_vals[row_ref_diag_idx];
            val_cur /= val_ref_diag;
            csr_vals[i] = val_cur;
            for (idxType j = row_ref_diag_idx + 1 + lane; j < row_ref_end; j += WARP_SIZE) {
                local_col = csr_cols[j] - idx_base;
                left = row_start + 1;
                right = row_end - 1;
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
                    csr_vals[local_idx] = csr_vals[local_idx] - csr_vals[j] * val_cur;
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
__global__ void mcspCsrIlu0InfoCollectKernel(idxType m, const valType *csr_vals, const idxType *csr_rows,
                                             const idxType *csr_cols, idxType *diag_ind, idxType *zero_pivot,
                                             idxType *row_nnz) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType row = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    idxType col;

    int found_flag = 0;
    if (row < m) {
        const idxType row_start = csr_rows[row];
        const idxType row_end = csr_rows[row + 1];
        if (lane == 0) {
            row_nnz[row] = row_end - row_start;
        }
        for (idxType idx = row_start + lane; idx < row_end; idx += WARP_SIZE) {
            col = csr_cols[idx];
            if ((found_flag == 0) && row == csr_cols[idx]) {
                if (csr_vals[idx] == (valType)0) {
                    zero_pivot[row] = row;
                }
                diag_ind[row] = idx;
                found_flag = 1;
            }
            found_flag = __any(found_flag);
#ifndef __MACA__
            __syncwarp();
#endif
        }
        if ((lane == 0) && (found_flag == 0)) {
            zero_pivot[row] = row;
        }
    }
}

#endif
