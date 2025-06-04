#ifndef KERNELS_PRECOND_BSRILU0_DEVICE_HPP__
#define KERNELS_PRECOND_BSRILU0_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_general_utility.h"
#include "mcsp_hashtable_device.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE, unsigned int HASH_SIZE, typename idxType, typename valType>
__global__ void mcspBsrilu0HashTableKernel(mcsparseDirection_t dir, idxType mb, idxType block_dim, valType *bsr_vals,
                                           const idxType *bsr_rows, const idxType *bsr_cols, const idxType *row_map,
                                           const idxType *diag_ind, idxType *row_done_flag,
                                           mcsparseIndexBase_t idx_base, idxType *zero_pivot, int enable_boost,
                                           double tol, valType boost_val) {
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

    if (gid < mb) {
        bool pivot = false;
        const idxType row = row_map[gid];
        idxType row_start = bsr_rows[row] - idx_base;
        idxType row_end = bsr_rows[row + 1] - idx_base;
        idxType row_diag_idx = diag_ind[row];

        idxType col;
        idxType row_ref_start;
        idxType row_ref_end;
        idxType row_ref_diag_idx;
        idxType local_col;
        idxType local_idx;
        int hash_found;
        for (idxType i = row_start + lane; i < row_end; i += WARP_SIZE) {
            col = bsr_cols[i] - idx_base;
            mcspInsertHashTablePairs<HASH_SIZE, 79>(col, i, cur_col_table, cur_idx_table);
        }
#ifndef __MACA__
        __syncwarp();
#endif
        // Process lower diagonal
        for (idxType i = row_start; i < row_diag_idx; i++) {
            col = bsr_cols[i] - idx_base;
            row_ref_end = bsr_rows[col + 1] - idx_base;
            row_ref_diag_idx = diag_ind[col];

            while (!atomicOr(&(row_done_flag[col]), 0)) {
                ;
            }
            __threadfence();

            for (idxType b_idx = 0; b_idx < block_dim; ++b_idx) {
                valType diag = bsr_vals[BSR_IND(row_ref_diag_idx, b_idx, b_idx, dir, block_dim)];
                for (idxType b_k = lane; b_k < block_dim; b_k += WARP_SIZE) {
                    valType val = bsr_vals[BSR_IND(i, b_k, b_idx, dir, block_dim)];
                    val /= diag;
                    bsr_vals[BSR_IND(i, b_k, b_idx, dir, block_dim)] = val;
                    for (idxType b_j = b_idx + 1; b_j < block_dim; ++b_j) {
                        bsr_vals[BSR_IND(i, b_k, b_j, dir, block_dim)] =
                            bsr_vals[BSR_IND(i, b_k, b_j, dir, block_dim)] -
                            val * bsr_vals[BSR_IND(row_ref_diag_idx, b_idx, b_j, dir, block_dim)];
                    }
                }
            }
            for (idxType j = row_ref_diag_idx + 1; j < row_ref_end; ++j) {
                local_col = bsr_cols[j] - idx_base;
                mcspReadHashTablePairs<HASH_SIZE, 79>(local_col, cur_col_table, cur_idx_table, &local_idx, &hash_found);
                if (hash_found == 1) {
                    for (idxType bi = lane; bi < block_dim; bi += WARP_SIZE) {
                        for (idxType bj = 0; bj < block_dim; ++bj) {
                            valType sum = static_cast<valType>(0);

                            for (idxType bk = 0; bk < block_dim; ++bk) {
                                sum += bsr_vals[BSR_IND(i, bi, bk, dir, block_dim)] *
                                       bsr_vals[BSR_IND(j, bk, bj, dir, block_dim)];
                            }

                            bsr_vals[BSR_IND(local_idx, bi, bj, dir, block_dim)] -= sum;
                        }
                    }
                }
            }
        }
        // Process diagonal
        for (idxType b_idx = 0; b_idx < block_dim; ++b_idx) {
            valType diag = bsr_vals[BSR_IND(row_diag_idx, b_idx, b_idx, dir, block_dim)];
            if (enable_boost) {
                diag = (tol >= std::abs(diag)) ? boost_val : diag;
                if (lane == 0) {
                    bsr_vals[BSR_IND(row_diag_idx, b_idx, b_idx, dir, block_dim)] = diag;
                }
            } else {
                if (diag == static_cast<valType>(0)) {
                    pivot = true;
                    continue;
                }
            }

            for (idxType b_k = b_idx + 1 + lane; b_k < block_dim; b_k += WARP_SIZE) {
                valType val = bsr_vals[BSR_IND(row_diag_idx, b_k, b_idx, dir, block_dim)];
                val /= diag;
                bsr_vals[BSR_IND(row_diag_idx, b_k, b_idx, dir, block_dim)] = val;
                for (idxType b_j = b_idx + 1; b_j < block_dim; ++b_j) {
                    bsr_vals[BSR_IND(row_diag_idx, b_k, b_j, dir, block_dim)] =
                        bsr_vals[BSR_IND(row_diag_idx, b_k, b_j, dir, block_dim)] -
                        val * bsr_vals[BSR_IND(row_diag_idx, b_idx, b_j, dir, block_dim)];
                }
            }
        }
        // Process upper diagonal BSR blocks
        for (idxType j = row_diag_idx + 1; j < row_end; ++j) {
            for (idxType b_idx = 0; b_idx < block_dim; ++b_idx) {
                for (idxType b_k = lane; b_k < block_dim; b_k += WARP_SIZE) {
                    for (idxType b_j = b_idx + 1; b_j < block_dim; ++b_j) {
                        bsr_vals[BSR_IND(j, b_j, b_k, dir, block_dim)] =
                            bsr_vals[BSR_IND(j, b_j, b_k, dir, block_dim)] -
                            bsr_vals[BSR_IND(row_diag_idx, b_j, b_idx, dir, block_dim)] *
                                bsr_vals[BSR_IND(j, b_idx, b_k, dir, block_dim)];
                    }
                }
            }
        }

        __threadfence();

        if (lane == 0) {
            atomicOr(&(row_done_flag[row]), 1);
            if (pivot) {
                atomicMin(zero_pivot, row + idx_base);
            }
        }
    }
}

template <typename idxType, typename valType>
__global__ void mcspBsrilu0BsearchKernel(mcsparseDirection_t dir, idxType mb, idxType block_dim, valType *bsr_vals,
                                         const idxType *bsr_rows, const idxType *bsr_cols, const idxType *row_map,
                                         const idxType *diag_ind, idxType *row_done_flag, mcsparseIndexBase_t idx_base,
                                         idxType *zero_pivot, int enable_boost, double tol, valType boost_val) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;

    if (gid < mb) {
        bool pivot = false;
        const idxType row = row_map[gid];
        idxType row_start = bsr_rows[row] - idx_base;
        idxType row_end = bsr_rows[row + 1] - idx_base;
        idxType row_diag_idx = diag_ind[row];

        idxType col;
        idxType row_ref_start;
        idxType row_ref_end;
        idxType row_ref_diag_idx;
        idxType local_col;
        idxType local_idx;

        int left;
        int right;
        int mid;
        idxType col_mid;

        // Process lower diagonal
        for (idxType i = row_start; i < row_diag_idx; i++) {
            col = bsr_cols[i] - idx_base;
            row_ref_start = bsr_rows[col] - idx_base;
            row_ref_end = bsr_rows[col + 1] - idx_base;
            row_ref_diag_idx = diag_ind[col];

            while (!atomicOr(&(row_done_flag[col]), 0)) {
                ;
            }
            __threadfence();

            for (idxType b_idx = 0; b_idx < block_dim; ++b_idx) {
                valType diag = bsr_vals[BSR_IND(row_ref_diag_idx, b_idx, b_idx, dir, block_dim)];
                for (idxType b_k = lane; b_k < block_dim; b_k += WARP_SIZE) {
                    valType val = bsr_vals[BSR_IND(i, b_k, b_idx, dir, block_dim)];
                    val /= diag;
                    bsr_vals[BSR_IND(i, b_k, b_idx, dir, block_dim)] = val;
                    for (idxType b_j = b_idx + 1; b_j < block_dim; ++b_j) {
                        bsr_vals[BSR_IND(i, b_k, b_j, dir, block_dim)] =
                            bsr_vals[BSR_IND(i, b_k, b_j, dir, block_dim)] -
                            val * bsr_vals[BSR_IND(row_ref_diag_idx, b_idx, b_j, dir, block_dim)];
                    }
                }
            }

            for (idxType j = row_ref_diag_idx + 1 + lane; j < row_ref_end; j += WARP_SIZE) {
                local_col = bsr_cols[j] - idx_base;
                left = row_start + 1;
                right = row_end - 1;
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
                    // for (idxType bi = lane; bi < block_dim; bi += WARP_SIZE) {
                    for (idxType bi = 0; bi < block_dim; bi += 1) {
                        for (idxType bj = 0; bj < block_dim; ++bj) {
                            valType sum = static_cast<valType>(0);

                            for (idxType bk = 0; bk < block_dim; ++bk) {
                                sum += bsr_vals[BSR_IND(i, bi, bk, dir, block_dim)] *
                                       bsr_vals[BSR_IND(j, bk, bj, dir, block_dim)];
                            }

                            bsr_vals[BSR_IND(local_idx, bi, bj, dir, block_dim)] -= sum;
                        }
                    }
                }
            }
        }
        // Process diagonal
        for (idxType b_idx = 0; b_idx < block_dim; ++b_idx) {
            valType diag = bsr_vals[BSR_IND(row_diag_idx, b_idx, b_idx, dir, block_dim)];
            if (enable_boost) {
                diag = (tol >= std::abs(diag)) ? boost_val : diag;
                if (lane == 0) {
                    bsr_vals[BSR_IND(row_diag_idx, b_idx, b_idx, dir, block_dim)] = diag;
                }
            } else {
                if (diag == static_cast<valType>(0)) {
                    pivot = true;
                    continue;
                }
            }

            for (idxType b_k = b_idx + 1 + lane; b_k < block_dim; b_k += WARP_SIZE) {
                valType val = bsr_vals[BSR_IND(row_diag_idx, b_k, b_idx, dir, block_dim)];
                val /= diag;
                bsr_vals[BSR_IND(row_diag_idx, b_k, b_idx, dir, block_dim)] = val;
                for (idxType b_j = b_idx + 1; b_j < block_dim; ++b_j) {
                    bsr_vals[BSR_IND(row_diag_idx, b_k, b_j, dir, block_dim)] =
                        bsr_vals[BSR_IND(row_diag_idx, b_k, b_j, dir, block_dim)] -
                        val * bsr_vals[BSR_IND(row_diag_idx, b_idx, b_j, dir, block_dim)];
                }
            }
        }
        // Process upper diagonal BSR blocks
        for (idxType j = row_diag_idx + 1; j < row_end; ++j) {
            for (idxType b_idx = 0; b_idx < block_dim; ++b_idx) {
                for (idxType b_k = lane; b_k < block_dim; b_k += WARP_SIZE) {
                    for (idxType b_j = b_idx + 1; b_j < block_dim; ++b_j) {
                        bsr_vals[BSR_IND(j, b_j, b_k, dir, block_dim)] =
                            bsr_vals[BSR_IND(j, b_j, b_k, dir, block_dim)] -
                            bsr_vals[BSR_IND(row_diag_idx, b_j, b_idx, dir, block_dim)] *
                                bsr_vals[BSR_IND(j, b_idx, b_k, dir, block_dim)];
                    }
                }
            }
        }
        __threadfence();

        if (lane == 0) {
            atomicOr(&(row_done_flag[row]), 1);
            if (pivot) {
                atomicMin(zero_pivot, row + idx_base);
            }
        }
    }
}

#endif
