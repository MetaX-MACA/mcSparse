#include <assert.h>
#include <stdio.h>

#include "blockell_spmm_device.hpp"
#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "coo_spmm_device.hpp"
#include "csr_spmm_device.hpp"
#include "mcsp_config.h"
#include "mcsp_dense_transpose_device.hpp"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_transpose_sparse.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "utils/mcsp_logger.h"

template <unsigned int BLOCK_SIZE = 512, typename valType, typename idxType>
mcspStatus_t mcspCsrSpmmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                 idxType m, idxType n, idxType k, idxType nnz, const valType *alpha,
                                 const mcspMatDescr_t descr, const valType *csr_vals, const idxType *csr_rows,
                                 const idxType *csr_cols, const valType *mtx_B, idxType ldb, const valType *beta,
                                 valType *mtx_C, idxType ldc, mcsparseOrder_t orderB = MCSPARSE_ORDER_COL,
                                 mcsparseOrder_t orderC = MCSPARSE_ORDER_COL) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || k < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || beta == nullptr || csr_vals == nullptr || csr_rows == nullptr || csr_cols == nullptr ||
        mtx_B == nullptr || mtx_C == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE || descr->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr->fill_mode != MCSPARSE_FILL_MODE_FULL || descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    valType h_beta = getScalarToHost(beta, handle->ptr_mode);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    if (nnz == 0 && h_beta == static_cast<valType>(1)) {
        return MCSP_STATUS_SUCCESS;
    }

    idxType ldbt = ldb;
    valType *Bt = const_cast<valType *>(mtx_B);
    mcStream_t stream = mcspGetStreamInternal(handle);
    if (trans_B == MCSPARSE_OPERATION_TRANSPOSE) {
        const int trans_dimX = 32;
        const int trans_dimY = 8;
        dim3 trans_blocks((n - 1) / trans_dimX + 1, 1);
        dim3 trans_threads(trans_dimX * trans_dimY);
        ldbt = k;
        MACA_ASSERT(mcMalloc((void **)&Bt, ldb * k * sizeof(valType)));
        MACA_ASSERT(mcMemsetAsync(Bt, 0, ldb * k * sizeof(valType), stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        mcLaunchKernelGGL((mcspDenseTransposeKernel<trans_dimX, trans_dimY>), trans_blocks, trans_threads, 0, stream,
                           n, k, mtx_B, ldb, Bt, ldbt);
    }

    constexpr int nElem = BLOCK_SIZE;
    int gElem = nElem / WARP_SIZE;
    int b_round = 32;
    int nblock_x = (m + b_round - 1) / b_round * b_round;
    int nblock_y = (n + gElem - 1) / gElem;
    nblock_y = (nblock_y + b_round - 1) / b_round * b_round;
    dim3 grid(nblock_x, nblock_y);
    mcLaunchKernelGGL((mcspSpmmCsrNnKernel<BLOCK_SIZE>), grid, dim3(nElem), 0, stream, m, n, h_alpha, csr_vals,
                       csr_rows, csr_cols, Bt, ldbt, h_beta, mtx_C, ldc, descr->base, orderC);
    MACA_ASSERT(mcStreamSynchronize(stream));
    if (trans_B == MCSPARSE_OPERATION_TRANSPOSE) {
        MACA_ASSERT(mcFree(Bt));
    }
    return MCSP_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE = 512, typename idxType, typename computeType, typename inputType,
          typename outputType>
mcspStatus_t mcspBatchedCsrSpmmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                        idxType m, idxType n, idxType k, idxType nnz, const computeType *alpha,
                                        const mcspMatDescr_t descr, const inputType *csr_vals, const idxType *csr_rows,
                                        const idxType *csr_cols, int64_t batch_A, int64_t row_stride,
                                        int64_t col_stride, const inputType *mtx_B, idxType ldb, int64_t batch_B,
                                        int64_t stride_B, const computeType *beta, outputType *mtx_C, idxType ldc,
                                        int64_t batch_C, int64_t stride_C, mcsparseOrder_t orderB = MCSPARSE_ORDER_COL,
                                        mcsparseOrder_t orderC = MCSPARSE_ORDER_COL) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || k < 0 || nnz < 0 || ldb < 0 || ldc < 0 || !(batch_C == batch_A && batch_C == batch_B)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || beta == nullptr || csr_vals == nullptr || csr_rows == nullptr || csr_cols == nullptr ||
        mtx_B == nullptr || mtx_C == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE || descr->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr->fill_mode != MCSPARSE_FILL_MODE_FULL || descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    computeType h_beta = getScalarToHost(beta, handle->ptr_mode);
    computeType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    if (nnz == 0 && h_beta == GetTypedValue<computeType>(1)) {
        return MCSP_STATUS_SUCCESS;
    }

    idxType ldbt = ldb;
    inputType *Bt = const_cast<inputType *>(mtx_B);
    mcStream_t stream = mcspGetStreamInternal(handle);
    if ((trans_B == MCSPARSE_OPERATION_TRANSPOSE && orderB == MCSPARSE_ORDER_COL) ||
        (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE && orderB == MCSPARSE_ORDER_ROW)) {
        ldbt = k;
        MACA_ASSERT(mcMalloc((void **)&Bt, batch_B * ldb * k * sizeof(inputType)));
        MACA_ASSERT(mcMemsetAsync(Bt, 0, batch_B * ldb * k * sizeof(inputType), stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        const int trans_dimX = 32;
        const int trans_dimY = 8;
        dim3 trans_blocks((n - 1) / trans_dimX + 1, batch_B);
        dim3 trans_threads(trans_dimX * trans_dimY);

        mcLaunchKernelGGL((mcspDenseTransposeKernel<trans_dimX, trans_dimY>), trans_blocks, trans_threads, 0, stream,
                           n, k, mtx_B, ldb, Bt, ldbt, stride_B);
        MACA_ASSERT(mcStreamSynchronize(stream));
    }

    constexpr int nElem = BLOCK_SIZE;
    int gElem = nElem / WARP_SIZE;
    int b_round = 32;
    int nblock_x = (m + b_round - 1) / b_round * b_round;
    int nblock_y = (n + gElem - 1) / gElem;
    nblock_y = (nblock_y + b_round - 1) / b_round * b_round;
    dim3 grid(nblock_x, nblock_y, batch_C);

    if constexpr (std::is_same_v<computeType, __half2> || std::is_same_v<computeType, mcsp_bfloat162>) {
        mcLaunchKernelGGL((mcspBatchedSpmmCsrNnKernelLowPrecisionComplex<BLOCK_SIZE, idxType, computeType>), grid,
                           dim3(nElem), nElem * sizeof(computeType), stream, m, n, h_alpha, csr_vals, csr_rows,
                           csr_cols, row_stride, col_stride, Bt, ldbt, stride_B, h_beta, mtx_C, ldc, stride_C,
                           descr->base, orderC);
    } else {
        size_t compute_type_size = GetMacaDataTypeSize(GetMacaDataTypeFromTypename<computeType>());
        mcLaunchKernelGGL((mcspBatchedSpmmCsrNnKernel<BLOCK_SIZE, idxType, computeType, inputType, outputType>), grid,
                           dim3(nElem), nElem * compute_type_size, stream, m, n, h_alpha, csr_vals, csr_rows, csr_cols,
                           row_stride, col_stride, Bt, ldbt, stride_B, h_beta, mtx_C, ldc, stride_C, descr->base,
                           orderC);
    }
    MACA_ASSERT(mcStreamSynchronize(stream));

    if ((trans_B == MCSPARSE_OPERATION_TRANSPOSE && orderB == MCSPARSE_ORDER_COL) ||
        (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE && orderB == MCSPARSE_ORDER_ROW)) {
        MACA_ASSERT(mcFree(Bt));
    }
    return MCSP_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE = 512, typename computeType, typename idxType, typename inputType,
          typename outputType>
mcspStatus_t mcspBatchedCooSpmmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                        idxType m, idxType n, idxType k, idxType nnz, const computeType *alpha,
                                        const mcspMatDescr_t descr, const inputType *coo_vals, const idxType *coo_rows,
                                        const idxType *coo_cols, int64_t batch_A, int64_t row_stride,
                                        int64_t col_stride, const inputType *mtx_B, idxType ldb, int64_t batch_B,
                                        int64_t stride_B, const computeType *beta, outputType *mtx_C, idxType ldc,
                                        int64_t batch_C, int64_t stride_C, mcsparseOrder_t orderB = MCSPARSE_ORDER_COL,
                                        mcsparseOrder_t orderC = MCSPARSE_ORDER_COL) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || k < 0 || nnz < 0 || ldb < 0 || ldc < 0 || !(batch_C == batch_A && batch_C == batch_B)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || beta == nullptr || coo_vals == nullptr || coo_rows == nullptr || coo_cols == nullptr ||
        mtx_B == nullptr || mtx_C == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE || descr->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr->fill_mode != MCSPARSE_FILL_MODE_FULL || descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    computeType h_beta = getScalarToHost(beta, handle->ptr_mode);
    computeType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    if (nnz == 0 && h_beta == GetTypedValue<computeType>(1)) {
        return MCSP_STATUS_SUCCESS;
    }

    idxType ldbt = ldb;
    inputType *Bt = const_cast<inputType *>(mtx_B);
    mcStream_t stream = mcspGetStreamInternal(handle);
    constexpr idxType block_size = 512;

    if ((trans_B == MCSPARSE_OPERATION_TRANSPOSE && orderB == MCSPARSE_ORDER_COL) ||
        (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE && orderB == MCSPARSE_ORDER_ROW)) {
        ldbt = k;
        MACA_ASSERT(mcMalloc((void **)&Bt, batch_B * ldb * k * sizeof(inputType)));
        MACA_ASSERT(mcMemsetAsync(Bt, 0, batch_B * ldb * k * sizeof(inputType), stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        const int trans_dimX = 32;
        const int trans_dimY = 8;
        dim3 trans_blocks((n - 1) / trans_dimX + 1, batch_B);
        dim3 trans_threads(trans_dimX * trans_dimY);

        mcLaunchKernelGGL((mcspDenseTransposeKernel<trans_dimX, trans_dimY>), trans_blocks, trans_threads, 0, stream,
                           n, k, mtx_B, ldb, Bt, ldbt, stride_B);
        MACA_ASSERT(mcStreamSynchronize(stream));
    }

    if constexpr (std::is_same_v<computeType, __half2> || std::is_same_v<computeType, mcsp_bfloat162>) {
        idxType blocks = (m * n - 1) / block_size + 1;
        mcLaunchKernelGGL((mcspSpmmCooLowPrecisionComplexScalingKernel<block_size>), dim3(blocks, batch_C),
                           dim3(block_size), 0, stream, m, n, ldc, stride_C, h_beta, mtx_C, orderC);
        MACA_ASSERT(mcStreamSynchronize(stream));

        blocks = (nnz - 1) / block_size + 1;
        mcLaunchKernelGGL((mcspSpmmCooLowPrecisionComplexBodyKernel<block_size>), dim3(blocks, batch_C),
                           dim3(block_size), 0, stream, n, nnz, h_alpha, coo_rows, coo_cols, coo_vals, row_stride,
                           col_stride, Bt, ldbt, stride_B, mtx_C, ldc, stride_C, descr->base, orderC);
    } else {
        idxType blocks = (m * n - 1) / block_size + 1;
        mcLaunchKernelGGL((mcspSpmmCooScalingKernel<block_size>), dim3(blocks, batch_C), dim3(block_size), 0, stream,
                           m, n, ldc, stride_C, h_beta, mtx_C, orderC);
        MACA_ASSERT(mcStreamSynchronize(stream));

        blocks = (nnz - 1) / block_size + 1;
        mcLaunchKernelGGL((mcspSpmmCooBodyKernel<block_size>), dim3(blocks, batch_C), dim3(block_size), 0, stream, n,
                           nnz, h_alpha, coo_rows, coo_cols, coo_vals, row_stride, col_stride, Bt, ldbt, stride_B,
                           mtx_C, ldc, stride_C, descr->base, orderC);
    }

    if ((trans_B == MCSPARSE_OPERATION_TRANSPOSE && orderB == MCSPARSE_ORDER_COL) ||
        (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE && orderB == MCSPARSE_ORDER_ROW)) {
        MACA_ASSERT(mcFree(Bt));
    }
    return MCSP_STATUS_SUCCESS;
}

const int BLOCKELL_SIZE = 16;
template <unsigned int BLOCK_SIZE = 512, typename computeType, typename idxType, typename inputType,
          typename outputType>
mcspStatus_t mcspBlockEllSpmmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                      mcsparseOrder_t orderB, mcsparseOrder_t orderC, idxType m, idxType n, idxType k,
                                      const computeType *alpha, const mcspMatDescr_t descr, const idxType ellBlockSize,
                                      const idxType ellCols, const inputType *ellValue, const idxType *ellColInd,
                                      const inputType *mtx_B, idxType ldb, const computeType *beta, outputType *mtx_C,
                                      idxType ldc) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || k < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (ellBlockSize < 0 || ellCols < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || beta == nullptr || ellValue == nullptr || ellColInd == nullptr || mtx_B == nullptr ||
        mtx_C == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    computeType h_beta = getScalarToHost(beta, handle->ptr_mode);
    computeType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    mcStream_t stream = mcspGetStreamInternal(handle);

    idxType ldbt = ldb;
    inputType *Bt = const_cast<inputType *>(mtx_B);
    if (trans_B == MCSPARSE_OPERATION_TRANSPOSE) {
        if (orderB == MCSPARSE_ORDER_COL) {
            trans_B = MCSPARSE_OPERATION_NON_TRANSPOSE;
            orderB = MCSPARSE_ORDER_ROW;
            ldbt = n;
        } else {
            trans_B = MCSPARSE_OPERATION_NON_TRANSPOSE;
            orderB = MCSPARSE_ORDER_COL;
            ldbt = k;
        }
    }

    constexpr idxType block_sz = BLOCKELL_SIZE;
    dim3 blk(block_sz, block_sz);
    int nblock_x = CEIL(n, block_sz);
    int nblock_y = CEIL(m, block_sz);  // y direction -> row
    dim3 grid(nblock_x, nblock_y);     // 2D grid、2D block

    mcLaunchKernelGGL((mcspSpmmBlockEllNnKernel<BLOCK_SIZE>), grid, blk, 0, stream, orderB, orderC, m, n, h_alpha,
                       ellBlockSize, ellCols, ellValue, ellColInd, Bt, ldbt, h_beta, mtx_C, ldc, descr->base);
    MACA_ASSERT(mcStreamSynchronize(stream));

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspSpMM_bufferSizeImpl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                     const void *alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, const void *beta,
                                     mcspDnMatDescr_t matC, macaDataType computeType, mcsparseSpMMAlg_t alg,
                                     size_t *bufferSize) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (matA == nullptr || matB == nullptr || matC == nullptr || alpha == nullptr || beta == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (matA->row_num >> 32 != 0 || matA->col_num >> 32 != 0 || matA->nnz >> 32 != 0 || matB->row_num >> 32 != 0 ||
        matB->col_num >> 32 != 0 || matB->ld >> 32 != 0 || matC->row_num >> 32 != 0 || matC->col_num >> 32 != 0 ||
        matC->ld >> 32 != 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (matA->format != MCSPARSE_FORMAT_CSR && matA->format != MCSPARSE_FORMAT_COO &&
        matA->format != MCSPARSE_FORMAT_BLOCKED_ELL && matA->format != MCSPARSE_FORMAT_CSC) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    *bufferSize = MIN_BUFFER_SIZE;

    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (matA->format == MCSPARSE_FORMAT_CSC) {
        macaDataType inputType = matA->valueType;
        stat = CalculateAssistBufferSizeForTranspose<idxType>(handle, matA, inputType, bufferSize, MCSPARSE_FORMAT_CSC);
    }

    return stat;
}

template <typename idxType>
mcspStatus_t mcspUnifiedSpMMImpl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void *alpha, mcspSpMatDescr_t matA, void *working_vals, idxType *working_rows,
                                 idxType *working_cols, mcsparseFormat_t working_format, mcspDnMatDescr_t matB,
                                 const void *beta, mcspDnMatDescr_t matC, macaDataType computeType,
                                 mcsparseSpMMAlg_t alg, void *externalBuffer) {
    mcStream_t stream = mcspGetStreamInternal(handle);
    switch (working_format) {
        case MCSPARSE_FORMAT_CSR: {
            switch (computeType) {
                case MACA_R_32F:
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (float *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (float *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (float *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case MACA_R_64F:
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (double *)alpha, matA->mat_descr, (double *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (double *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (double *)beta, (double *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case MACA_C_32F:
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr,
                        (mcspComplexFloat *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                        matA->batchCount, matA->offsetsBatchStride, matA->batchStride, (mcspComplexFloat *)matB->values,
                        (idxType)matB->ld, matB->batchCount, matB->batchStride, (mcspComplexFloat *)beta,
                        (mcspComplexFloat *)matC->values, (idxType)matC->ld, matC->batchCount, matC->batchStride,
                        matB->order, matC->order);

                case MACA_C_64F:
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcspComplexDouble *)alpha, matA->mat_descr,
                        (mcspComplexDouble *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                        matA->batchCount, matA->offsetsBatchStride, matA->batchStride,
                        (mcspComplexDouble *)matB->values, (idxType)matB->ld, matB->batchCount, matB->batchStride,
                        (mcspComplexDouble *)beta, (mcspComplexDouble *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

#if defined(__MACA__)
                case MACA_R_16F:
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (__half *)alpha, matA->mat_descr, (__half *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (__half *)beta, (__half *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);
#endif
#ifdef __MACA__
                case MACA_R_16BF:
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcsp_bfloat16 *)alpha, matA->mat_descr, (mcsp_bfloat16 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (mcsp_bfloat16 *)beta, (mcsp_bfloat16 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

                case MACA_C_16F:
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (__half2 *)alpha, matA->mat_descr, (__half2 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half2 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (__half2 *)beta, (__half2 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

                case MACA_C_16BF:
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcsp_bfloat162 *)alpha, matA->mat_descr, (mcsp_bfloat162 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat162 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (mcsp_bfloat162 *)beta, (mcsp_bfloat162 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_COO: {
            switch (computeType) {
                case MACA_R_32F:
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (float *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (float *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (float *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case MACA_R_64F:
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (double *)alpha, matA->mat_descr, (double *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (double *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (double *)beta, (double *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case MACA_C_32F:
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr,
                        (mcspComplexFloat *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                        matA->batchCount, matA->offsetsBatchStride, matA->batchStride, (mcspComplexFloat *)matB->values,
                        (idxType)matB->ld, matB->batchCount, matB->batchStride, (mcspComplexFloat *)beta,
                        (mcspComplexFloat *)matC->values, (idxType)matC->ld, matC->batchCount, matC->batchStride,
                        matB->order, matC->order);

                case MACA_C_64F:
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcspComplexDouble *)alpha, matA->mat_descr,
                        (mcspComplexDouble *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                        matA->batchCount, matA->offsetsBatchStride, matA->batchStride,
                        (mcspComplexDouble *)matB->values, (idxType)matB->ld, matB->batchCount, matB->batchStride,
                        (mcspComplexDouble *)beta, (mcspComplexDouble *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

#if defined(__MACA__)
                case MACA_R_16F:
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (__half *)alpha, matA->mat_descr, (__half *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (__half *)beta, (__half *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);
                case MACA_R_16BF:
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcsp_bfloat16 *)alpha, matA->mat_descr, (mcsp_bfloat16 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (mcsp_bfloat16 *)beta, (mcsp_bfloat16 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);
                case MACA_C_16F:
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (__half2 *)alpha, matA->mat_descr, (__half2 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half2 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (__half2 *)beta, (__half2 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);
                case MACA_C_16BF:
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcsp_bfloat162 *)alpha, matA->mat_descr, (mcsp_bfloat162 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat162 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (mcsp_bfloat162 *)beta, (mcsp_bfloat162 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_BLOCKED_ELL: {
            switch (computeType) {
                case MACA_R_32F:
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (float *)alpha, matA->mat_descr, (idxType)matA->ellBlockSize,
                        (idxType)matA->ellCols, (float *)matA->ellValue, (idxType *)matA->ellColInd,
                        (float *)matB->values, (idxType)matB->ld, (float *)beta, (float *)matC->values,
                        (idxType)matC->ld);
                case MACA_R_64F:
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (double *)alpha, matA->mat_descr, (idxType)matA->ellBlockSize,
                        (idxType)matA->ellCols, (double *)matA->ellValue, (idxType *)matA->ellColInd,
                        (double *)matB->values, (idxType)matB->ld, (double *)beta, (double *)matC->values,
                        (idxType)matC->ld);
                case MACA_C_32F:
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (mcspComplexFloat *)alpha, matA->mat_descr, (idxType)matA->ellBlockSize,
                        (idxType)matA->ellCols, (mcspComplexFloat *)matA->ellValue, (idxType *)matA->ellColInd,
                        (mcspComplexFloat *)matB->values, (idxType)matB->ld, (mcspComplexFloat *)beta,
                        (mcspComplexFloat *)matC->values, (idxType)matC->ld);
                case MACA_C_64F:
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (mcspComplexDouble *)alpha, matA->mat_descr,
                        (idxType)matA->ellBlockSize, (idxType)matA->ellCols, (mcspComplexDouble *)matA->ellValue,
                        (idxType *)matA->ellColInd, (mcspComplexDouble *)matB->values, (idxType)matB->ld,
                        (mcspComplexDouble *)beta, (mcspComplexDouble *)matC->values, (idxType)matC->ld);
#if defined(__MACA__)
                case MACA_R_16F:
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (__half *)alpha, matA->mat_descr, (idxType)matA->ellBlockSize,
                        (idxType)matA->ellCols, (__half *)matA->ellValue, (idxType *)matA->ellColInd,
                        (__half *)matB->values, (idxType)matB->ld, (__half *)beta, (__half *)matC->values,
                        (idxType)matC->ld);
                case MACA_R_16BF:
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (mcsp_bfloat16 *)alpha, matA->mat_descr, (idxType)matA->ellBlockSize,
                        (idxType)matA->ellCols, (mcsp_bfloat16 *)matA->ellValue, (idxType *)matA->ellColInd,
                        (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, (mcsp_bfloat16 *)beta,
                        (mcsp_bfloat16 *)matC->values, (idxType)matC->ld);
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspMixedSpMMImpl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void *alpha,
                               mcspSpMatDescr_t matA, void *working_vals, idxType *working_rows, idxType *working_cols,
                               mcsparseFormat_t working_format, mcspDnMatDescr_t matB, const void *beta,
                               mcspDnMatDescr_t matC, macaDataType computeType, mcsparseSpMMAlg_t alg,
                               void *externalBuffer) {
    macaDataType inputType = matA->valueType;
    macaDataType outputType = matC->valueType;
    uint64_t mixedType = GetMixedDataType(inputType, outputType);
    mcStream_t stream = mcspGetStreamInternal(handle);

    switch (working_format) {
        case MCSPARSE_FORMAT_CSR: {
            switch (mixedType) {
#if defined(__MACA__)
                case GetMixedDataType(MACA_R_16F, MACA_R_32F):
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (__half *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (float *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_16BF, MACA_R_32F):
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (mcsp_bfloat16 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (float *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_8I, MACA_R_32F):
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (int8_t *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (int8_t *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (float *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_16F, MACA_R_16F):
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (__half *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (__half *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_16BF, MACA_R_16BF):
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (mcsp_bfloat16 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (mcsp_bfloat16 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_C_16F, MACA_C_16F):
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr, (__half2 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half2 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (mcspComplexFloat *)beta, (__half2 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_C_16BF, MACA_C_16BF):
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr, (mcsp_bfloat162 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat162 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (mcspComplexFloat *)beta, (mcsp_bfloat162 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_8I, MACA_R_32I):
                    return mcspBatchedCsrSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (int32_t *)alpha, matA->mat_descr, (int8_t *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (int8_t *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (int32_t *)beta, (int32_t *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_COO: {
            switch (mixedType) {
#if defined(__MACA__)
                case GetMixedDataType(MACA_R_16F, MACA_R_32F):
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (__half *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (float *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_16BF, MACA_R_32F):
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (mcsp_bfloat16 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (float *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_8I, MACA_R_32F):
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (int8_t *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (int8_t *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (float *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_16F, MACA_R_16F):
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (__half *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (__half *)matC->values, (idxType)matC->ld, matC->batchCount,
                        matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_16BF, MACA_R_16BF):
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (mcsp_bfloat16 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (float *)beta, (mcsp_bfloat16 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_C_16F, MACA_C_16F):
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr, (__half2 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (__half2 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (mcspComplexFloat *)beta, (__half2 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_C_16BF, MACA_C_16BF):
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr, (mcsp_bfloat162 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (mcsp_bfloat162 *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (mcspComplexFloat *)beta, (mcsp_bfloat162 *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);

                case GetMixedDataType(MACA_R_8I, MACA_R_32I):
                    return mcspBatchedCooSpmmTemplate(
                        handle, opA, opB, (idxType)matA->row_num, (idxType)matC->col_num, (idxType)matA->col_num,
                        (idxType)matA->nnz, (int32_t *)alpha, matA->mat_descr, (int8_t *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, matA->batchCount, matA->offsetsBatchStride,
                        matA->batchStride, (int8_t *)matB->values, (idxType)matB->ld, matB->batchCount,
                        matB->batchStride, (int32_t *)beta, (int32_t *)matC->values, (idxType)matC->ld,
                        matC->batchCount, matC->batchStride, matB->order, matC->order);
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_BLOCKED_ELL: {
            switch (mixedType) {
#if defined(__MACA__)
                case GetMixedDataType(MACA_R_16F, MACA_R_16F):
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (float *)alpha, matA->mat_descr, (idxType)matA->ellBlockSize,
                        (idxType)matA->ellCols, (__half *)matA->ellValue, (idxType *)matA->ellColInd,
                        (__half *)matB->values, (idxType)matB->ld, (float *)beta, (__half *)matC->values,
                        (idxType)matC->ld);

                case GetMixedDataType(MACA_R_16F, MACA_R_32F):
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (float *)alpha, matA->mat_descr, (idxType)matA->ellBlockSize,
                        (idxType)matA->ellCols, (__half *)matA->ellValue, (idxType *)matA->ellColInd,
                        (__half *)matB->values, (idxType)matB->ld, (float *)beta, (float *)matC->values,
                        (idxType)matC->ld);

                case GetMixedDataType(MACA_R_16BF, MACA_R_16BF):
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (float *)alpha, matA->mat_descr, (idxType)matA->ellBlockSize,
                        (idxType)matA->ellCols, (mcsp_bfloat16 *)matA->ellValue, (idxType *)matA->ellColInd,
                        (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, (float *)beta, (mcsp_bfloat16 *)matC->values,
                        (idxType)matC->ld);

                case GetMixedDataType(MACA_R_16BF, MACA_R_32F):
                    return mcspBlockEllSpmmTemplate(
                        handle, opA, opB, matB->order, matC->order, (idxType)matA->row_num, (idxType)matC->col_num,
                        (idxType)matA->col_num, (float *)alpha, matA->mat_descr, (idxType)matA->ellBlockSize,
                        (idxType)matA->ellCols, (mcsp_bfloat16 *)matA->ellValue, (idxType *)matA->ellColInd,
                        (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, (float *)beta, (float *)matC->values,
                        (idxType)matC->ld);
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspSpMMImpl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void *alpha,
                          mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, const void *beta, mcspDnMatDescr_t matC,
                          macaDataType computeType, mcsparseSpMMAlg_t alg, void *externalBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (matA == nullptr || matB == nullptr || matC == nullptr || alpha == nullptr || beta == nullptr ||
        externalBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (matA->row_num >> 32 != 0 || matA->col_num >> 32 != 0 || matA->nnz >> 32 != 0 || matB->row_num >> 32 != 0 ||
        matB->col_num >> 32 != 0 || matB->ld >> 32 != 0 || matC->row_num >> 32 != 0 || matC->col_num >> 32 != 0 ||
        matC->ld >> 32 != 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (matA->format != MCSPARSE_FORMAT_CSR && matA->format != MCSPARSE_FORMAT_COO &&
        matA->format != MCSPARSE_FORMAT_BLOCKED_ELL && matA->format != MCSPARSE_FORMAT_CSC) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    macaDataType inputType = matA->valueType;
    macaDataType outputType = matC->valueType;
    void *working_rows = nullptr;
    void *working_cols = nullptr;
    void *working_vals = nullptr;
    mcsparseFormat_t working_format = matA->format;
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    mcStream_t stream = mcspGetStreamInternal(handle);

    if (working_format == MCSPARSE_FORMAT_CSC) {
        if (externalBuffer != nullptr && matA->is_buffersize_called == 0) {
            size_t total_buffer_size = 0;
            mcspStatus_t stat = mcspSpMM_bufferSizeImpl<idxType>(handle, opA, opB, alpha, matA, matB, beta, matC,
                                                                 computeType, alg, &total_buffer_size);
            if (stat != MCSP_STATUS_SUCCESS) {
                return MCSP_STATUS_INTERNAL_ERROR;
            }
            LOG_FS_WARN("Running SpMV without calling SpMV_buffersize for matrix A ahead.\n");
            LOG_FS_WARN("Enough buffer with buffersize %d bytes should be guaranteed or the program may crash.\n",
                        total_buffer_size);
            MACA_ASSERT(mcMemsetAsync(externalBuffer, 0, total_buffer_size, stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
        }
        idxType *buffer_head = reinterpret_cast<idxType *>(externalBuffer);
        matA->to_csr_rows = (void *)buffer_head;
        buffer_head += (matA->row_num + 1);
        matA->to_csr_cols = (void *)buffer_head;
        matA->to_csr_vals = (void *)((reinterpret_cast<char *>(externalBuffer) + matA->assist_index_buffer_size));
        void *csc2csr_buffer = (void *)(reinterpret_cast<char *>(externalBuffer) + matA->fixed_length_buffer_size);
        stat = mcspTransposeSparseByCsr2Csc(handle, opA, (idxType)matA->col_num, (idxType)matA->row_num,
                                            (idxType)matA->nnz, matA->vals, (idxType *)matA->cols,
                                            (idxType *)matA->rows, matA->to_csr_vals, (idxType *)matA->to_csr_cols,
                                            (idxType *)matA->to_csr_rows, matA->idxBase, inputType, csc2csr_buffer);
        MACA_ASSERT(mcStreamSynchronize(stream));
        if (stat != MCSP_STATUS_SUCCESS) {
            return stat;
        }
        working_rows = matA->to_csr_rows;
        working_cols = matA->to_csr_cols;
        working_vals = matA->to_csr_vals;
        working_format = MCSPARSE_FORMAT_CSR;
    } else {
        working_rows = matA->rows;
        working_cols = matA->cols;
        working_vals = matA->vals;
    }

    if (computeType == inputType && computeType == outputType) {
        stat = mcspUnifiedSpMMImpl<idxType>(handle, opA, opB, alpha, matA, working_vals, (idxType *)working_rows,
                                            (idxType *)working_cols, working_format, matB, beta, matC, computeType, alg,
                                            externalBuffer);
    } else {
        stat = mcspMixedSpMMImpl<idxType>(handle, opA, opB, alpha, matA, working_vals, (idxType *)working_rows,
                                          (idxType *)working_cols, working_format, matB, beta, matC, computeType, alg,
                                          externalBuffer);
    }

    return stat;
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute CSR-based SpMM in single and double precision in GPU, csr_scalar algorithm
 * ref: Design Principles for Sparse Matrix Multiplication on the GPU, 2018
 * C = alpha * A * B + beta * C
 */
mcspStatus_t mcspScsrSpmm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, mcspInt nnz, const float *alpha, const mcspMatDescr_t descr,
                          const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols, const float *mtx_B,
                          mcspInt ldb, const float *beta, float *mtx_C, mcspInt ldc) {
    return mcspCsrSpmmTemplate(handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, csr_vals, csr_rows, csr_cols,
                               mtx_B, ldb, beta, mtx_C, ldc);
}

mcspStatus_t mcspDcsrSpmm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, mcspInt nnz, const double *alpha, const mcspMatDescr_t descr,
                          const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols, const double *mtx_B,
                          mcspInt ldb, const double *beta, double *mtx_C, mcspInt ldc) {
    return mcspCsrSpmmTemplate(handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, csr_vals, csr_rows, csr_cols,
                               mtx_B, ldb, beta, mtx_C, ldc);
}

mcspStatus_t mcspCcsrSpmm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, mcspInt nnz, const mcspComplexFloat *alpha, const mcspMatDescr_t descr,
                          const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                          const mcspComplexFloat *mtx_B, mcspInt ldb, const mcspComplexFloat *beta,
                          mcspComplexFloat *mtx_C, mcspInt ldc) {
    return mcspCsrSpmmTemplate(handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, csr_vals, csr_rows, csr_cols,
                               mtx_B, ldb, beta, mtx_C, ldc);
}

mcspStatus_t mcspZcsrSpmm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, mcspInt nnz, const mcspComplexDouble *alpha, const mcspMatDescr_t descr,
                          const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                          const mcspComplexDouble *mtx_B, mcspInt ldb, const mcspComplexDouble *beta,
                          mcspComplexDouble *mtx_C, mcspInt ldc) {
    return mcspCsrSpmmTemplate(handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, csr_vals, csr_rows, csr_cols,
                               mtx_B, ldb, beta, mtx_C, ldc);
}

mcspStatus_t mcspCuinScsrmm2(mcspHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m, int n,
                           int k, int nnz, const float *alpha, const mcspMatDescr_t descrA, const float *csrValA,
                           const int *csrRowPtrA, const int *csrColIndA, const float *B, int ldb, const float *beta,
                           float *C, int ldc) {
    return mcspCsrSpmmTemplate(handle, transA, transB, (mcspInt)m, (mcspInt)n, (mcspInt)k, (mcspInt)nnz, alpha, descrA,
                               csrValA, (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, B, (mcspInt)ldb, beta, C,
                               (mcspInt)ldc);
}

mcspStatus_t mcspCuinDcsrmm2(mcspHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m, int n,
                           int k, int nnz, const double *alpha, const mcspMatDescr_t descrA, const double *csrValA,
                           const int *csrRowPtrA, const int *csrColIndA, const double *B, int ldb, const double *beta,
                           double *C, int ldc) {
    return mcspCsrSpmmTemplate(handle, transA, transB, (mcspInt)m, (mcspInt)n, (mcspInt)k, (mcspInt)nnz, alpha, descrA,
                               csrValA, (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, B, (mcspInt)ldb, beta, C,
                               (mcspInt)ldc);
}

mcspStatus_t mcspCuinCcsrmm2(mcspHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m, int n,
                           int k, int nnz, const mcspComplexFloat *alpha, const mcspMatDescr_t descrA,
                           const mcspComplexFloat *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                           const mcspComplexFloat *B, int ldb, const mcspComplexFloat *beta, mcspComplexFloat *C,
                           int ldc) {
    return mcspCsrSpmmTemplate(handle, transA, transB, (mcspInt)m, (mcspInt)n, (mcspInt)k, (mcspInt)nnz, alpha, descrA,
                               csrValA, (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, B, (mcspInt)ldb, beta, C,
                               (mcspInt)ldc);
}

mcspStatus_t mcspCuinZcsrmm2(mcspHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m, int n,
                           int k, int nnz, const mcspComplexDouble *alpha, const mcspMatDescr_t descrA,
                           const mcspComplexDouble *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                           const mcspComplexDouble *B, int ldb, const mcspComplexDouble *beta, mcspComplexDouble *C,
                           int ldc) {
    return mcspCsrSpmmTemplate(handle, transA, transB, (mcspInt)m, (mcspInt)n, (mcspInt)k, (mcspInt)nnz, alpha, descrA,
                               csrValA, (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, B, (mcspInt)ldb, beta, C,
                               (mcspInt)ldc);
}

mcspStatus_t mcspSpMM_bufferSize(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void *alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, const void *beta,
                                 mcspDnMatDescr_t matC, macaDataType computeType, mcsparseSpMMAlg_t alg,
                                 size_t *bufferSize) {
    if (matA->rowIdxType == MCSPARSE_INDEX_32I && matA->colIdxType == MCSPARSE_INDEX_32I) {
        return mcspSpMM_bufferSizeImpl<mcspInt>(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
                                                bufferSize);
    } else if (matA->rowIdxType == MCSPARSE_INDEX_64I && matA->colIdxType == MCSPARSE_INDEX_64I) {
        return mcspSpMM_bufferSizeImpl<int64_t>(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
                                                bufferSize);
    } else {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

mcspStatus_t mcspSpMM_preprocess(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void *alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, const void *beta,
                                 mcspDnMatDescr_t matC, macaDataType computeType, mcsparseSpMMAlg_t alg,
                                 void *externalBuffer) {
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpMM(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void *alpha,
                      mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, const void *beta, mcspDnMatDescr_t matC,
                      macaDataType computeType, mcsparseSpMMAlg_t alg, void *externalBuffer) {
    if (matA->rowIdxType == MCSPARSE_INDEX_32I && matA->colIdxType == MCSPARSE_INDEX_32I) {
        return mcspSpMMImpl<mcspInt>(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
    } else if (matA->rowIdxType == MCSPARSE_INDEX_64I && matA->colIdxType == MCSPARSE_INDEX_64I) {
        return mcspSpMMImpl<int64_t>(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
    } else {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

#ifdef __cplusplus
}
#endif
