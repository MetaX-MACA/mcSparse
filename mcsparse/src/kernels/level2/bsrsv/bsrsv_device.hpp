#ifndef KERNELS_LEVEL2_BSRSV_DEVICE_HPP__
#define KERNELS_LEVEL2_BSRSV_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"

template <bool lower_flag, bool unity_flag, typename idxType, typename valType>
__global__ void mcspBsrsvSolveSharedKernel(idxType m, idxType block_dim, mcsparseIndexBase_t bsr_base,
                                           mcsparseDirection_t dir, valType alpha, const valType *bsr_vals,
                                           const idxType *bsr_rows, const idxType *bsr_cols, const idxType *row_map,
                                           const idxType *diag_ind, const valType *x, valType *y, idxType *done_buffer,
                                           idxType *zero_pivot) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *shared_y = reinterpret_cast<valType *>(smem);
    idxType *row_done = reinterpret_cast<idxType *>(smem + block_dim * WARP_SIZE * sizeof(valType));

    idxType tid = threadIdx.x;
    idxType gid = blockIdx.x;
    idxType cur_row = tid / WARP_SIZE;
    idxType lane = tid & (WARP_SIZE - 1);
    idxType bsr_row = row_map[gid];
    idxType bsr_row_start = bsr_rows[bsr_row] - bsr_base;
    idxType bsr_row_end = bsr_rows[bsr_row + 1] - bsr_base;
    valType rhs = alpha * x[bsr_row * block_dim + cur_row];

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
        if (cur_row == 0 && loop_flag) {
            while (!atomicOr(&(done_buffer[bsr_col]), 0)) {
                ;
            }
            __threadfence();
            for (idxType block_col = 0; block_col < block_dim; ++block_col) {
                shared_y[lane * block_dim + block_col] = y[bsr_col * block_dim + block_col];
                if (dir == MCSPARSE_DIRECTION_ROW) {
                    local_sum += shared_y[lane * block_dim + block_col] *
                                 bsr_vals[bsr_idx * block_dim * block_dim + cur_row * block_dim + block_col];
                } else {
                    local_sum += shared_y[lane * block_dim + block_col] *
                                 bsr_vals[bsr_idx * block_dim * block_dim + block_dim * block_col + cur_row];
                }
            }
        }
        __syncthreads();

        if (cur_row != 0 && loop_flag) {
            for (idxType block_col = 0; block_col < block_dim; ++block_col) {
                if (dir == MCSPARSE_DIRECTION_ROW) {
                    local_sum += shared_y[lane * block_dim + block_col] *
                                 bsr_vals[bsr_idx * block_dim * block_dim + cur_row * block_dim + block_col];
                } else {
                    local_sum += shared_y[lane * block_dim + block_col] *
                                 bsr_vals[bsr_idx * block_dim * block_dim + block_dim * block_col + cur_row];
                }
            }
        }
        __syncthreads();
    }
    local_sum = warpReduceSum<WARP_SIZE>(local_sum);
    row_done[cur_row] = 0;
    __syncthreads();

    if (lane == (WARP_SIZE - 1)) {
        valType diag_val;
        if (dir == MCSPARSE_DIRECTION_ROW) {
            diag_val = bsr_vals[row_diag_idx * block_dim * block_dim + cur_row * block_dim + cur_row];
        } else {
            diag_val = bsr_vals[row_diag_idx * block_dim * block_dim + block_dim * cur_row + cur_row];
        }
        if (diag_val == static_cast<valType>(0) && (!unity_flag)) {
            atomicMin(zero_pivot, bsr_row + bsr_base);
        }

        if constexpr (lower_flag) {
            interval_start = 0;
            interval_end = cur_row;
        } else {
            interval_start = cur_row + 1;
            interval_end = block_dim;
        }

        for (idxType block_col = interval_start; block_col < interval_end; ++block_col) {
            idxType block_idx;
            if (dir == MCSPARSE_DIRECTION_ROW) {
                block_idx = cur_row * block_dim + block_col;
            } else {
                block_idx = block_dim * block_col + cur_row;
            }
            while (!atomicOr(&(row_done[block_col]), 0)) {
                ;
            }
            __threadfence();
            local_sum +=
                y[bsr_row * block_dim + block_col] * bsr_vals[row_diag_idx * block_dim * block_dim + block_idx];
        }

        if constexpr (unity_flag) {
            y[bsr_row * block_dim + cur_row] = rhs - local_sum;
        } else {
            y[bsr_row * block_dim + cur_row] = (rhs - local_sum) / diag_val;
        }
        __threadfence();
        atomicOr(&(row_done[cur_row]), 1);
        if (lower_flag) {
            if (cur_row == (block_dim - 1)) {
                atomicOr(&(done_buffer[bsr_row]), 1);
            }
        } else {
            if (cur_row == 0) {
                atomicOr(&(done_buffer[bsr_row]), 1);
            }
        }
    }
}

template <bool lower_flag, bool unity_flag, typename idxType, typename valType>
__global__ void mcspBsrsvSolveGlobalKernel(idxType m, idxType block_dim, mcsparseIndexBase_t bsr_base,
                                           mcsparseDirection_t dir, valType alpha, const valType *bsr_vals,
                                           const idxType *bsr_rows, const idxType *bsr_cols, const idxType *row_map,
                                           const idxType *diag_ind, const valType *x, valType *y, idxType *done_buffer,
                                           idxType *done_row, idxType *zero_pivot) {
    idxType gid = blockIdx.x;
    idxType cur_row = gid % block_dim;
    gid = gid / block_dim;
    idxType lane = threadIdx.x;

    idxType bsr_row = row_map[gid];
    idxType bsr_row_start = bsr_rows[bsr_row] - bsr_base;
    idxType bsr_row_end = bsr_rows[bsr_row + 1] - bsr_base;
    valType rhs = alpha * x[bsr_row * block_dim + cur_row];

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
            while (!atomicOr(&(done_buffer[bsr_col]), 0)) {
                ;
            }
            __threadfence();
            for (idxType block_col = 0; block_col < block_dim; ++block_col) {
                if (dir == MCSPARSE_DIRECTION_ROW) {
                    local_sum += y[bsr_col * block_dim + block_col] *
                                 bsr_vals[bsr_idx * block_dim * block_dim + cur_row * block_dim + block_col];
                } else {
                    local_sum += y[bsr_col * block_dim + block_col] *
                                 bsr_vals[bsr_idx * block_dim * block_dim + block_dim * block_col + cur_row];
                }
            }
        }
        __syncthreads();
    }
    local_sum = warpReduceSum<WARP_SIZE>(local_sum);

    if (lane == (WARP_SIZE - 1)) {
        valType diag_val;
        if (dir == MCSPARSE_DIRECTION_ROW) {
            diag_val = bsr_vals[row_diag_idx * block_dim * block_dim + cur_row * block_dim + cur_row];
        } else {
            diag_val = bsr_vals[row_diag_idx * block_dim * block_dim + block_dim * cur_row + cur_row];
        }
        if (diag_val == static_cast<valType>(0) && (!unity_flag)) {
            atomicMin(zero_pivot, bsr_row + bsr_base);
        }

        if constexpr (lower_flag) {
            interval_start = 0;
            interval_end = cur_row;
        } else {
            interval_start = cur_row + 1;
            interval_end = block_dim;
        }

        for (idxType block_col = interval_start; block_col < interval_end; ++block_col) {
            idxType block_idx;
            if (dir == MCSPARSE_DIRECTION_ROW) {
                block_idx = cur_row * block_dim + block_col;
            } else {
                block_idx = block_dim * block_col + cur_row;
            }
            while (!atomicOr(&(done_row[bsr_row * block_dim + block_col]), 0)) {
                ;
            }
            __threadfence();
            local_sum +=
                y[bsr_row * block_dim + block_col] * bsr_vals[row_diag_idx * block_dim * block_dim + block_idx];
        }

        if constexpr (unity_flag) {
            y[bsr_row * block_dim + cur_row] = rhs - local_sum;
        } else {
            y[bsr_row * block_dim + cur_row] = (rhs - local_sum) / diag_val;
        }

        __threadfence();
        atomicOr(&(done_row[bsr_row * block_dim + cur_row]), 1);
        if (lower_flag) {
            if (cur_row == (block_dim - 1)) {
                atomicOr(&(done_buffer[bsr_row]), 1);
            }
        } else {
            if (cur_row == 0) {
                atomicOr(&(done_buffer[bsr_row]), 1);
            }
        }
    }
}
#endif