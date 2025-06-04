#ifndef KERNELS_LEVEL3_BSRSM_DEVICE_HPP__
#define KERNELS_LEVEL3_BSRSM_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"

template <bool lower_flag, bool unity_flag, uint32_t N_COL, uint32_t BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspBsrsmSolveKernel(idxType m, idxType n, idxType block_dim, mcsparseIndexBase_t bsr_base,
                                     mcsparseDirection_t dir, valType alpha, const valType *bsr_vals,
                                     const idxType *bsr_rows, const idxType *bsr_cols, const idxType *row_map,
                                     const idxType *diag_ind, valType *x, idxType ldx, const valType *b, idxType ldb,
                                     idxType *bsr_done_buffer, idxType *done_buffer, idxType *zero_pivot,
                                     mcsparseOperation_t transXY) {
    idxType gidx = blockIdx.x;
    idxType gidy = blockIdx.y;
    idxType tid = threadIdx.x;
    idxType cur_block_row = gidx % block_dim;
    gidx = gidx / block_dim;

    idxType lane = tid % WARP_SIZE;
    idxType cur_bx_col = tid / WARP_SIZE + gidy * N_COL;

    if (cur_bx_col >= n || gidx >= m) {
        return;
    }

    idxType *cur_done = done_buffer + cur_bx_col * m * block_dim;
    idxType *cur_bsr_done = bsr_done_buffer + cur_bx_col * m;
    idxType bsr_row = row_map[gidx];
    idxType bsr_row_start = bsr_rows[bsr_row] - bsr_base;
    idxType bsr_row_end = bsr_rows[bsr_row + 1] - bsr_base;

    idxType bx_idx;
    if (transXY == MCSPARSE_OPERATION_NON_TRANSPOSE) {
        bx_idx = ldb * cur_bx_col + bsr_row * block_dim + cur_block_row;
    } else {
        bx_idx = cur_bx_col + (bsr_row * block_dim + cur_block_row) * ldb;
    }
    valType rhs = alpha * b[bx_idx];

    idxType row_diag_idx;
    idxType interval_start;
    idxType interval_end;

    row_diag_idx = diag_ind[bsr_row];

    if constexpr (lower_flag) {
        interval_start = bsr_row_start;
        interval_end = row_diag_idx;
    } else {
        interval_start = row_diag_idx + 1;
        interval_end = bsr_row_end;
    }

    valType local_sum = static_cast<valType>(0);
    idxType warp_end_idx = ((interval_end - interval_start + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    for (idxType i = lane; i < warp_end_idx; i += WARP_SIZE) {
        idxType bsr_col;
        idxType bsr_idx;
        bool loop_flag = false;
        if constexpr (lower_flag) {
            bsr_idx = interval_start + i;
            loop_flag = (bsr_idx < interval_end);
            if (loop_flag) {
                bsr_col = bsr_cols[bsr_idx] - bsr_base;
                if (bsr_col >= bsr_row) {
                    continue;
                }
            }
        } else {
            loop_flag = (interval_end >= interval_start + 1 + i);
            if (loop_flag) {
                bsr_idx = interval_end - 1 - i;
                bsr_col = bsr_cols[bsr_idx] - bsr_base;
                if (bsr_col <= bsr_row) {
                    continue;
                }
            }
        }
        if (loop_flag) {
            while (!atomicOr(&(cur_bsr_done[bsr_col]), 0)) {
                ;
            }
            __threadfence();
            if (dir == MCSPARSE_DIRECTION_ROW) {
                for (idxType block_col = 0; block_col < block_dim; ++block_col) {
                    if (transXY == MCSPARSE_OPERATION_NON_TRANSPOSE) {
                        bx_idx = ldx * cur_bx_col + bsr_col * block_dim + block_col;
                    } else {
                        bx_idx = cur_bx_col + ldx * (bsr_col * block_dim + block_col);
                    }
                    local_sum +=
                        x[bx_idx] * bsr_vals[bsr_idx * block_dim * block_dim + cur_block_row * block_dim + block_col];
                }
            } else {
                for (idxType block_col = 0; block_col < block_dim; ++block_col) {
                    if (transXY == MCSPARSE_OPERATION_NON_TRANSPOSE) {
                        bx_idx = ldx * cur_bx_col + bsr_col * block_dim + block_col;
                    } else {
                        bx_idx = cur_bx_col + ldx * (bsr_col * block_dim + block_col);
                    }
                    local_sum +=
                        x[bx_idx] * bsr_vals[bsr_idx * block_dim * block_dim + block_dim * block_col + cur_block_row];
                }
            }
        }
        __syncthreads();
    }
    local_sum = warpReduceSum<WARP_SIZE>(local_sum);

    if (lane == (WARP_SIZE - 1)) {
        valType diag_val;
        if (dir == MCSPARSE_DIRECTION_ROW) {
            diag_val = bsr_vals[row_diag_idx * block_dim * block_dim + cur_block_row * block_dim + cur_block_row];
        } else {
            diag_val = bsr_vals[row_diag_idx * block_dim * block_dim + block_dim * cur_block_row + cur_block_row];
        }
        if (diag_val == static_cast<valType>(0) && (!unity_flag)) {
            atomicMin(zero_pivot, bsr_row + bsr_base);
        }

        if constexpr (lower_flag) {
            interval_start = 0;
            interval_end = cur_block_row;
        } else {
            interval_start = cur_block_row + 1;
            interval_end = block_dim;
        }

        if (dir == MCSPARSE_DIRECTION_ROW) {
            for (idxType block_col = interval_start; block_col < interval_end; ++block_col) {
                idxType block_idx = cur_block_row * block_dim + block_col;
                while (!atomicOr(&(cur_done[bsr_row * block_dim + block_col]), 0)) {
                    ;
                }
                __threadfence();
                if (transXY == MCSPARSE_OPERATION_NON_TRANSPOSE) {
                    bx_idx = ldx * cur_bx_col + bsr_row * block_dim + block_col;
                } else {
                    bx_idx = cur_bx_col + ldx * (bsr_row * block_dim + block_col);
                }
                local_sum += x[bx_idx] * bsr_vals[row_diag_idx * block_dim * block_dim + block_idx];
            }
        } else {
            for (idxType block_col = interval_start; block_col < interval_end; ++block_col) {
                idxType block_idx = block_dim * block_col + cur_block_row;
                while (!atomicOr(&(cur_done[bsr_row * block_dim + block_col]), 0)) {
                    ;
                }
                __threadfence();
                if (transXY == MCSPARSE_OPERATION_NON_TRANSPOSE) {
                    bx_idx = ldx * cur_bx_col + bsr_row * block_dim + block_col;
                } else {
                    bx_idx = cur_bx_col + ldx * (bsr_row * block_dim + block_col);
                }
                local_sum += x[bx_idx] * bsr_vals[row_diag_idx * block_dim * block_dim + block_idx];
            }
        }

        if (transXY == MCSPARSE_OPERATION_NON_TRANSPOSE) {
            bx_idx = ldx * cur_bx_col + bsr_row * block_dim + cur_block_row;
        } else {
            bx_idx = cur_bx_col + ldx * (bsr_row * block_dim + cur_block_row);
        }
        if constexpr (unity_flag) {
            x[bx_idx] = rhs - local_sum;
        } else {
            x[bx_idx] = (rhs - local_sum) / diag_val;
        }

        __threadfence();
        atomicOr(&(cur_done[bsr_row * block_dim + cur_block_row]), 1);
        if (lower_flag) {
            if (cur_block_row == (block_dim - 1)) {
                atomicOr(&(cur_bsr_done[bsr_row]), 1);
            }
        } else {
            if (cur_block_row == 0) {
                atomicOr(&(cur_bsr_done[bsr_row]), 1);
            }
        }
    }
}

#endif