#ifndef KERNELS_GENERIC_SDDMM_DEVICE_HPP__
#define KERNELS_GENERIC_SDDMM_DEVICE_HPP__

#include "block_reduce.hpp"
#include "common/mcsp_types.h"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE = 512, typename idxType, typename valType>
__global__ void mcspSddmmCooKernel(mcsparseOperation_t opA, mcsparseOperation_t opB, mcsparseOrder_t orderA,
                                   mcsparseOrder_t orderB, mcsparseIndexBase_t idx_base, valType alpha, idxType row_num,
                                   idxType col_num, idxType A_row_num, idxType A_col_num, idxType lda,
                                   const valType *A_vals, idxType ldb, const valType *B_vals, valType beta,
                                   idxType C_nnz, valType *C_coo_vals, idxType *C_coo_rows, idxType *C_coo_cols,
                                   int64_t A_stride = 0, int64_t B_stride = 0, int64_t C_nnz_stride = 0) {
    idxType batch_idx = blockIdx.y;
    const valType *cur_vals_A = A_vals + batch_idx * A_stride;
    const valType *cur_vals_B = B_vals + batch_idx * B_stride;
    valType *cur_coo_vals_C = C_coo_vals + batch_idx * C_nnz_stride;
    idxType *cur_coo_rows_C = C_coo_rows + batch_idx * C_nnz_stride;
    idxType *cur_coo_cols_C = C_coo_cols + batch_idx * C_nnz_stride;

    __shared__ valType sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    sdata[tid] = static_cast<valType>(0);

    idxType coo_position = blockIdx.x;
    idxType C_row_id = cur_coo_rows_C[coo_position] - idx_base;
    idxType C_col_id = cur_coo_cols_C[coo_position] - idx_base;

    const valType *a_offset =
        (orderA == MCSPARSE_ORDER_COL)
            ? ((opA == MCSPARSE_OPERATION_NON_TRANSPOSE) ? (cur_vals_A + C_row_id) : (cur_vals_A + lda * C_row_id))
            : ((opA == MCSPARSE_OPERATION_NON_TRANSPOSE) ? (cur_vals_A + lda * C_row_id) : (cur_vals_A + C_row_id));
    idxType a_incr = (orderA == MCSPARSE_ORDER_COL) ? ((opA == MCSPARSE_OPERATION_NON_TRANSPOSE) ? lda : 1)
                                                    : ((opA == MCSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : lda);
    const valType *b_offset =
        (orderB == MCSPARSE_ORDER_COL)
            ? ((opB == MCSPARSE_OPERATION_NON_TRANSPOSE) ? (cur_vals_B + ldb * C_col_id) : (cur_vals_B + C_col_id))
            : ((opB == MCSPARSE_OPERATION_NON_TRANSPOSE) ? (cur_vals_B + C_col_id) : (cur_vals_B + ldb * C_col_id));
    idxType b_incr = (orderB == MCSPARSE_ORDER_COL) ? ((opB == MCSPARSE_OPERATION_NON_TRANSPOSE) ? 1 : ldb)
                                                    : ((opB == MCSPARSE_OPERATION_NON_TRANSPOSE) ? ldb : 1);

    idxType elimnate_num = (opA == MCSPARSE_OPERATION_NON_TRANSPOSE) ? A_col_num : A_row_num;
    for (idxType k = tid; k < elimnate_num; k += blockDim.x) {
        sdata[tid] += a_offset[k * a_incr] * b_offset[k * b_incr];
    }
    __syncthreads();

    mcprim::intra_block_reduce<BLOCK_SIZE>(tid, sdata, mcprim::plus<valType>());

    if (tid == 0) {
        cur_coo_vals_C[coo_position] = alpha * sdata[0] + beta * cur_coo_vals_C[coo_position];
    }
}

#endif  // KERNELS_GENERIC_SDDMM_DEVICE_HPP__
