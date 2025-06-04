#include "common/mcsp_types.h"
#include "internal_interface/mcsp_internal_conversion.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "sddmm_device.hpp"

mcspStatus_t mcspSddmmBufferSizeImpl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                     const void *alpha, const mcspDnMatDescr_t A, const mcspDnMatDescr_t B,
                                     const void *beta, mcspSpMatDescr_t C, macaDataType compute_type,
                                     mcsparseSDDMMAlg_t alg, size_t *buffer_size) {
    *buffer_size = MIN_BUFFER_SIZE;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSddmmPreprocessImpl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                     const void *alpha, const mcspDnMatDescr_t A, const mcspDnMatDescr_t B,
                                     const void *beta, mcspSpMatDescr_t C, macaDataType compute_type,
                                     mcsparseSDDMMAlg_t alg, void *temp_buffer) {
    return MCSP_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE = 512, typename valType, typename idxType>
mcspStatus_t mcspSddmmCsrTemplate(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                  mcsparseOrder_t orderA, mcsparseOrder_t orderB, mcsparseIndexBase_t idx_base,
                                  const valType *alpha, idxType row_num, idxType col_num, idxType A_row_num,
                                  idxType A_col_num, idxType B_row_num, idxType B_col_num, idxType lda,
                                  const valType *A_vals, idxType ldb, const valType *B_vals, const valType *beta,
                                  idxType C_nnz, valType *C_csr_vals, idxType *C_csr_rows, idxType *C_csr_cols,
                                  int64_t batch_count = 1, int64_t A_stride = 0, int64_t B_stride = 0,
                                  int64_t C_row_stride = 0, int64_t C_col_stride = 0) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    // beta != 1 can be optimized
    valType h_beta = getScalarToHost(beta, handle->ptr_mode);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    if ((A_row_num == 0 || A_col_num == 0 || lda == 0 || B_row_num == 0 || B_col_num == 0 || ldb == 0) &&
        h_beta == static_cast<valType>(1)) {
        return MCSP_STATUS_SUCCESS;
    }

    idxType *C_coo_rows = nullptr;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMalloc((void **)&(C_coo_rows), batch_count * C_nnz * sizeof(idxType)));
    MACA_ASSERT(mcMemsetAsync(C_coo_rows, 0, batch_count * C_nnz * sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    for (int64_t batch = 0; batch < batch_count; ++batch) {
        MACA_ASSERT(mcspCsr2Coo(handle, C_csr_rows + batch * C_row_stride, C_nnz, row_num,
                                C_coo_rows + batch * C_col_stride, idx_base));
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid(C_nnz, batch_count);
    mcLaunchKernelGGL(mcspSddmmCooKernel, grid, block, 0, stream, opA, opB, orderA, orderB, idx_base, h_alpha, row_num,
                       col_num, A_row_num, A_col_num, lda, A_vals, ldb, B_vals, h_beta, C_nnz, C_csr_vals, C_coo_rows,
                       C_csr_cols, A_stride, B_stride, C_col_stride);
    MACA_ASSERT(mcStreamSynchronize(stream));
    MACA_ASSERT(mcFree(C_coo_rows));
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspSddmmCsrImpl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void *alpha,
                              const mcspDnMatDescr_t A, const mcspDnMatDescr_t B, const void *beta, mcspSpMatDescr_t C,
                              macaDataType compute_type, mcsparseSDDMMAlg_t alg, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr ||
        C->mat_descr == nullptr || A->mat_descr == nullptr || B->mat_descr == nullptr || A->values == nullptr ||
        B->values == nullptr || C->rows == nullptr || C->cols == nullptr || C->vals == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (opA == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE || opB == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE ||
        C->format != MCSPARSE_FORMAT_CSR || C->mat_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        C->mat_descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (A->row_num < 0 || A->col_num < 0 || A->ld < 0 || B->row_num < 0 || B->col_num < 0 || B->ld < 0 ||
        C->row_num < 0 || C->col_num < 0 || C->nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if ((opA == MCSPARSE_OPERATION_NON_TRANSPOSE && opB == MCSPARSE_OPERATION_NON_TRANSPOSE &&
         A->col_num != B->row_num) ||
        (opA == MCSPARSE_OPERATION_TRANSPOSE && opB == MCSPARSE_OPERATION_NON_TRANSPOSE && A->row_num != B->row_num) ||
        (opA == MCSPARSE_OPERATION_NON_TRANSPOSE && opB == MCSPARSE_OPERATION_TRANSPOSE && A->col_num != B->col_num) ||
        (opA == MCSPARSE_OPERATION_TRANSPOSE && opB == MCSPARSE_OPERATION_TRANSPOSE && A->row_num != B->col_num)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if ((A->order == MCSPARSE_ORDER_COL && A->ld < A->row_num) ||
        (A->order == MCSPARSE_ORDER_ROW && A->ld < A->col_num) ||
        (B->order == MCSPARSE_ORDER_COL && B->ld < B->row_num) ||
        (B->order == MCSPARSE_ORDER_ROW && B->ld < B->col_num) ||
        !(C->batchCount == B->batchCount && C->batchCount == A->batchCount)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (C->nnz == 0 || C->row_num == 0 || C->col_num == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    switch (compute_type) {
        case MACA_R_32F:
            return mcspSddmmCsrTemplate<512, float, idxType>(
                handle, opA, opB, A->order, B->order, C->idxBase, (float *)alpha, (idxType)C->row_num,
                (idxType)C->col_num, (idxType)A->row_num, (idxType)A->col_num, (idxType)B->row_num, (idxType)B->col_num,
                (idxType)A->ld, (float *)A->values, (idxType)B->ld, (float *)B->values, (float *)beta, (idxType)C->nnz,
                (float *)C->vals, (idxType *)C->rows, (idxType *)C->cols, C->batchCount, A->batchStride, B->batchStride,
                C->offsetsBatchStride, C->batchStride);
        case MACA_R_64F:
            return mcspSddmmCsrTemplate<512, double, idxType>(
                handle, opA, opB, A->order, B->order, C->idxBase, (double *)alpha, (idxType)C->row_num,
                (idxType)C->col_num, (idxType)A->row_num, (idxType)A->col_num, (idxType)B->row_num, (idxType)B->col_num,
                (idxType)A->ld, (double *)A->values, (idxType)B->ld, (double *)B->values, (double *)beta,
                (idxType)C->nnz, (double *)C->vals, (idxType *)C->rows, (idxType *)C->cols, C->batchCount,
                A->batchStride, B->batchStride, C->offsetsBatchStride, C->batchStride);
        case MACA_C_32F:
            return mcspSddmmCsrTemplate<512, mcspComplexFloat, idxType>(
                handle, opA, opB, A->order, B->order, C->idxBase, (mcspComplexFloat *)alpha, (idxType)C->row_num,
                (idxType)C->col_num, (idxType)A->row_num, (idxType)A->col_num, (idxType)B->row_num, (idxType)B->col_num,
                (idxType)A->ld, (mcspComplexFloat *)A->values, (idxType)B->ld, (mcspComplexFloat *)B->values,
                (mcspComplexFloat *)beta, (idxType)C->nnz, (mcspComplexFloat *)C->vals, (idxType *)C->rows,
                (idxType *)C->cols, C->batchCount, A->batchStride, B->batchStride, C->offsetsBatchStride,
                C->batchStride);
        case MACA_C_64F:
            return mcspSddmmCsrTemplate<512, mcspComplexDouble, idxType>(
                handle, opA, opB, A->order, B->order, C->idxBase, (mcspComplexDouble *)alpha, (idxType)C->row_num,
                (idxType)C->col_num, (idxType)A->row_num, (idxType)A->col_num, (idxType)B->row_num, (idxType)B->col_num,
                (idxType)A->ld, (mcspComplexDouble *)A->values, (idxType)B->ld, (mcspComplexDouble *)B->values,
                (mcspComplexDouble *)beta, (idxType)C->nnz, (mcspComplexDouble *)C->vals, (idxType *)C->rows,
                (idxType *)C->cols, C->batchCount, A->batchStride, B->batchStride, C->offsetsBatchStride,
                C->batchStride);
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSddmmBufferSize(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void *alpha, const mcspDnMatDescr_t A, const mcspDnMatDescr_t B,
                                 const void *beta, mcspSpMatDescr_t C, macaDataType compute_type,
                                 mcsparseSDDMMAlg_t alg, size_t *buffer_size) {
    return mcspSddmmBufferSizeImpl(handle, opA, opB, alpha, A, B, beta, C, compute_type, alg, buffer_size);
}

mcspStatus_t mcspSddmmPreprocess(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void *alpha, const mcspDnMatDescr_t A, const mcspDnMatDescr_t B,
                                 const void *beta, mcspSpMatDescr_t C, macaDataType compute_type,
                                 mcsparseSDDMMAlg_t alg, void *temp_buffer) {
    return mcspSddmmPreprocessImpl(handle, opA, opB, alpha, A, B, beta, C, compute_type, alg, temp_buffer);
}

mcspStatus_t mcspSddmm(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void *alpha,
                       const mcspDnMatDescr_t A, const mcspDnMatDescr_t B, const void *beta, mcspSpMatDescr_t C,
                       macaDataType compute_type, mcsparseSDDMMAlg_t alg, void *temp_buffer) {
    return mcspSddmmCsrImpl<mcspInt>(handle, opA, opB, alpha, A, B, beta, C, compute_type, alg, temp_buffer);
}

#ifdef __cplusplus
}
#endif
