#include <assert.h>
#include <stdio.h>

#include "common/mcsp_types.h"
#include "gemmi_device.hpp"
#include "internal_interface/mcsp_internal_conversion.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE = 512, typename valType, typename idxType>
mcspStatus_t mcspGemmiTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, idxType m,
                               idxType n, idxType k, idxType nnz, const valType *alpha, const valType *A, idxType lda,
                               const mcspMatDescr_t descr, const valType *csr_vals, const idxType *csr_rows,
                               const idxType *csr_cols, const valType *beta, valType *C, idxType ldc) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || k < 0 || lda < 0 || ldc < 0 || nnz < 0 || ldc < m || lda < m) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || beta == nullptr || csr_vals == nullptr || csr_rows == nullptr || csr_cols == nullptr ||
        A == nullptr || C == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (!(trans_A == MCSPARSE_OPERATION_NON_TRANSPOSE && trans_B == MCSPARSE_OPERATION_TRANSPOSE)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    valType h_beta = getScalarToHost(beta, handle->ptr_mode);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    if (h_alpha == static_cast<valType>(0) && h_beta == static_cast<valType>(0)) {
        return MCSP_STATUS_SUCCESS;
    }

    // @TODO(zhiming): scale C
    if (m == 0 || n == 0 || lda == 0 || nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    // Each threadblock processes (BLOCK_SIZE rows x 1 column)
    dim3 block(BLOCK_SIZE);
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, n);
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL((mcspGemmiNTKernel<BLOCK_SIZE>), grid, block, 0, stream, m, n, k, h_alpha, A, lda, csr_vals,
                       csr_rows, csr_cols, h_beta, C, ldc, descr->base);

    return MCSP_STATUS_SUCCESS;
}

template <typename valType, typename idxType>
mcspStatus_t mcspCuinGemmiTemplate(mcspHandle_t handle, idxType m, idxType n, idxType k, idxType nnz, const valType *alpha,
                             const valType *A, idxType lda, const valType *csc_vals, const idxType *csc_cols,
                             const idxType *csc_rows, const valType *beta, valType *C, idxType ldc) {
    mcspMatDescr_t descr = nullptr;
    mcspStatus_t stat = mcspCreateMatDescr(&(descr));
    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }
    stat = mcspGemmiTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, MCSPARSE_OPERATION_TRANSPOSE, m, n, k, nnz,
                             alpha, A, lda, descr, csc_vals, csc_cols, csc_rows, beta, C, ldc);
    return stat;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSgemmi(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                        mcspInt n, mcspInt k, mcspInt nnz, const float *alpha, const float *A, mcspInt lda,
                        const mcspMatDescr_t descr, const float *csr_vals, const mcspInt *csr_rows,
                        const mcspInt *csr_cols, const float *beta, float *C, mcspInt ldc) {
    return mcspGemmiTemplate(handle, trans_A, trans_B, m, n, k, nnz, alpha, A, lda, descr, csr_vals, csr_rows, csr_cols,
                             beta, C, ldc);
}

mcspStatus_t mcspDgemmi(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                        mcspInt n, mcspInt k, mcspInt nnz, const double *alpha, const double *A, mcspInt lda,
                        const mcspMatDescr_t descr, const double *csr_vals, const mcspInt *csr_rows,
                        const mcspInt *csr_cols, const double *beta, double *C, mcspInt ldc) {
    return mcspGemmiTemplate(handle, trans_A, trans_B, m, n, k, nnz, alpha, A, lda, descr, csr_vals, csr_rows, csr_cols,
                             beta, C, ldc);
}

mcspStatus_t mcspCgemmi(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                        mcspInt n, mcspInt k, mcspInt nnz, const mcspComplexFloat *alpha, const mcspComplexFloat *A,
                        mcspInt lda, const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals,
                        const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspComplexFloat *beta,
                        mcspComplexFloat *C, mcspInt ldc) {
    return mcspGemmiTemplate(handle, trans_A, trans_B, m, n, k, nnz, alpha, A, lda, descr, csr_vals, csr_rows, csr_cols,
                             beta, C, ldc);
}

mcspStatus_t mcspZgemmi(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                        mcspInt n, mcspInt k, mcspInt nnz, const mcspComplexDouble *alpha, const mcspComplexDouble *A,
                        mcspInt lda, const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                        const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspComplexDouble *beta,
                        mcspComplexDouble *C, mcspInt ldc) {
    return mcspGemmiTemplate(handle, trans_A, trans_B, m, n, k, nnz, alpha, A, lda, descr, csr_vals, csr_rows, csr_cols,
                             beta, C, ldc);
}

mcspStatus_t mcspCuinSgemmi(mcspHandle_t handle, int m, int n, int k, int nnz, const float *alpha, const float *A,
                          int lda, const float *csc_vals, const int *csc_cols, const int *csc_rows, const float *beta,
                          float *C, int ldc) {
    return mcspCuinGemmiTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, (mcspInt)nnz, alpha, A, (mcspInt)lda, csc_vals,
                           (mcspInt *)csc_cols, (mcspInt *)csc_rows, beta, C, (mcspInt)ldc);
}

mcspStatus_t mcspCuinDgemmi(mcspHandle_t handle, int m, int n, int k, int nnz, const double *alpha, const double *A,
                          int lda, const double *csc_vals, const int *csc_cols, const int *csc_rows, const double *beta,
                          double *C, int ldc) {
    return mcspCuinGemmiTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, (mcspInt)nnz, alpha, A, (mcspInt)lda, csc_vals,
                           (mcspInt *)csc_cols, (mcspInt *)csc_rows, beta, C, (mcspInt)ldc);
}

mcspStatus_t mcspCuinCgemmi(mcspHandle_t handle, int m, int n, int k, int nnz, const mcspComplexFloat *alpha,
                          const mcspComplexFloat *A, int lda, const mcspComplexFloat *csc_vals, const int *csc_cols,
                          const int *csc_rows, const mcspComplexFloat *beta, mcspComplexFloat *C, int ldc) {
    return mcspCuinGemmiTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, (mcspInt)nnz, alpha, A, (mcspInt)lda, csc_vals,
                           (mcspInt *)csc_cols, (mcspInt *)csc_rows, beta, C, (mcspInt)ldc);
}

mcspStatus_t mcspCuinZgemmi(mcspHandle_t handle, int m, int n, int k, int nnz, const mcspComplexDouble *alpha,
                          const mcspComplexDouble *A, int lda, const mcspComplexDouble *csc_vals, const int *csc_cols,
                          const int *csc_rows, const mcspComplexDouble *beta, mcspComplexDouble *C, int ldc) {
    return mcspCuinGemmiTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, (mcspInt)nnz, alpha, A, (mcspInt)lda, csc_vals,
                           (mcspInt *)csc_cols, (mcspInt *)csc_rows, beta, C, (mcspInt)ldc);
}

#ifdef __cplusplus
}
#endif
