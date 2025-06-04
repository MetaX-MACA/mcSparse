#ifndef KERNELS_LEVEL3_SPMM_BLOCKELL_SPMM_DEVICE_HPP__
#define KERNELS_LEVEL3_SPMM_BLOCKELL_SPMM_DEVICE_HPP__

#include <stdio.h>

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE, typename idxType, typename ComputeType, typename inputType, typename outputType>
__global__ void mcspSpmmBlockEllNnKernel(mcsparseOrder_t orderB, mcsparseOrder_t orderC, idxType m, idxType n,
                                         const ComputeType alpha, const idxType ellBlockSize, const idxType ellCols,
                                         const inputType *ellValue, const idxType *ellColInd, const inputType *mtx_B,
                                         idxType ldb, const ComputeType beta, outputType *mtx_C, idxType ldc,
                                         mcsparseIndexBase_t idx_base) {
    idxType row_C = blockIdx.y * blockDim.y + threadIdx.y;
    idxType col_C = blockIdx.x * blockDim.x + threadIdx.x;

    idxType loc1 = 0;
    idxType loc2 = 0;
    if ((row_C < m) && (col_C < n)) {
        ComputeType tmp_val_oneC = GetTypedValue<ComputeType>(0);
        idxType row_block = row_C / ellBlockSize;
        int32_t val_blk_col = 0;
        ComputeType val_oneA = GetTypedValue<ComputeType>(0);
        ComputeType val_oneB = GetTypedValue<ComputeType>(0);
        ComputeType tmpval2 = GetTypedValue<ComputeType>(0);

        for (idxType id_blk_col = 0; id_blk_col < ellCols / ellBlockSize; id_blk_col++) {
            loc1 = row_block * (ellCols / ellBlockSize) + id_blk_col;
            val_blk_col = (int32_t)(ellColInd[loc1]);
            if (val_blk_col == -1) {  // if this block is empty (all zero)
                break;
            }

            tmpval2 = 0.0;
            for (idxType id_in_ell_blk = 0; id_in_ell_blk < ellBlockSize; id_in_ell_blk++) {
                loc1 = row_C * ellCols + (id_blk_col * ellBlockSize + id_in_ell_blk);
                loc2 = (orderB == MCSPARSE_ORDER_COL) ? (col_C * ldb + (val_blk_col * ellBlockSize + id_in_ell_blk))
                                                      : ((val_blk_col * ellBlockSize + id_in_ell_blk) * ldb + col_C);
                if constexpr (std::is_same_v<ComputeType, inputType> &&
                              std::is_same_v<ComputeType, outputType>) {  // uniform-precision
                    val_oneA = ellValue[loc1];
                    val_oneB = mtx_B[loc2];
                } else {  // mixed-precision
#if defined(__MACA__)
                    val_oneA = GetFloatFromLowReal(ellValue[loc1]);
                    val_oneB = GetFloatFromLowReal(mtx_B[loc2]);
#endif
                }
                tmpval2 = tmpval2 + val_oneA * val_oneB;
            }
            tmp_val_oneC = tmp_val_oneC + tmpval2;
        }

        loc1 = (orderC == MCSPARSE_ORDER_COL) ? (col_C * ldc + row_C) : (row_C * ldc + col_C);
        if constexpr (std::is_same_v<ComputeType, inputType> && std::is_same_v<ComputeType, outputType>) {
            mtx_C[loc1] = alpha * tmp_val_oneC + beta * mtx_C[loc1];
        } else {
#if defined(__MACA__)
            mtx_C[loc1] = GetTypedValue<outputType>(alpha * tmp_val_oneC + beta * GetFloatFromLowReal(mtx_C[loc1]));
#endif
        }
    }
}

#endif
