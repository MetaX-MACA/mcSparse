#include "bsrmm_device.hpp"
#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspBsrmmTemplate(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans_A,
                               mcsparseOperation_t trans_B, idxType mb, idxType n, idxType kb, idxType nnzb,
                               const valType* alpha, const mcspMatDescr_t bsr_descr, const valType* bsr_vals,
                               const idxType* bsr_rows_ind, const idxType* bsr_cols_ind, idxType block_dim,
                               const valType* d_B, idxType ldb, const valType* beta, valType* d_C, idxType ldc) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE ||
        (trans_B != MCSPARSE_OPERATION_TRANSPOSE && trans_B != MCSPARSE_OPERATION_NON_TRANSPOSE)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mb < 0 || n < 0 || kb < 0 || nnzb < 0 || block_dim <= 1) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (mb == 0 || kb == 0 || n == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if ((trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE && ldb < kb * block_dim) ||
        (trans_B == MCSPARSE_OPERATION_TRANSPOSE && ldb < n) || ldc < mb * block_dim) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || bsr_descr == nullptr || beta == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (bsr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    valType h_beta = getScalarToHost(beta, handle->ptr_mode);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    mcStream_t stream = mcspGetStreamInternal(handle);

    if (nnzb == 0 || h_alpha == static_cast<valType>(0)) {
        constexpr idxType block_size = 512;
        idxType blocks = (block_dim * mb * n + block_size - 1) / block_size;
        mcLaunchKernelGGL(mcspBsrmmZeroAlphaKernel<block_size>, dim3(blocks), dim3(block_size), 0, stream, mb, n,
                           block_dim, h_beta, d_C, ldc);
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows_ind == nullptr || bsr_cols_ind == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    constexpr idxType block_size = 512;
    constexpr idxType segement_size = block_size / WARP_SIZE;
    idxType blocks = mb * block_dim;
    if (block_dim < WARP_SIZE) {
        mcLaunchKernelGGL((mcspBsrmmShareKernel<block_size, segement_size>), dim3(blocks), dim3(block_size),
                           block_dim * WARP_SIZE * sizeof(*bsr_vals), stream, mb, n, kb, trans_B, dir, bsr_descr->base,
                           bsr_vals, bsr_rows_ind, bsr_cols_ind, block_dim, h_alpha, h_beta, d_B, ldb, d_C, ldc);
    } else {
        mcLaunchKernelGGL((mcspBsrmmGlobalKernel<block_size, segement_size>), dim3(blocks), dim3(block_size), 0,
                           stream, mb, n, kb, trans_B, dir, bsr_descr->base, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                           block_dim, h_alpha, h_beta, d_B, ldb, d_C, ldc);
    }

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSbsrmm(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans_A,
                        mcsparseOperation_t trans_B, mcspInt mb, mcspInt n, mcspInt kb, mcspInt nnzb,
                        const float* alpha, const mcspMatDescr_t bsr_descr, const float* bsr_vals,
                        const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim, const float* d_B,
                        mcspInt ldb, const float* beta, float* d_C, mcspInt ldc) {
    return mcspBsrmmTemplate(handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind,
                             bsr_cols_ind, block_dim, d_B, ldb, beta, d_C, ldc);
}

mcspStatus_t mcspDbsrmm(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans_A,
                        mcsparseOperation_t trans_B, mcspInt mb, mcspInt n, mcspInt kb, mcspInt nnzb,
                        const double* alpha, const mcspMatDescr_t bsr_descr, const double* bsr_vals,
                        const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim, const double* d_B,
                        mcspInt ldb, const double* beta, double* d_C, mcspInt ldc) {
    return mcspBsrmmTemplate(handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind,
                             bsr_cols_ind, block_dim, d_B, ldb, beta, d_C, ldc);
}

mcspStatus_t mcspCbsrmm(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans_A,
                        mcsparseOperation_t trans_B, mcspInt mb, mcspInt n, mcspInt kb, mcspInt nnzb,
                        const mcspComplexFloat* alpha, const mcspMatDescr_t bsr_descr, const mcspComplexFloat* bsr_vals,
                        const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                        const mcspComplexFloat* d_B, mcspInt ldb, const mcspComplexFloat* beta, mcspComplexFloat* d_C,
                        mcspInt ldc) {
    return mcspBsrmmTemplate(handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind,
                             bsr_cols_ind, block_dim, d_B, ldb, beta, d_C, ldc);
}

mcspStatus_t mcspZbsrmm(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans_A,
                        mcsparseOperation_t trans_B, mcspInt mb, mcspInt n, mcspInt kb, mcspInt nnzb,
                        const mcspComplexDouble* alpha, const mcspMatDescr_t bsr_descr,
                        const mcspComplexDouble* bsr_vals, const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind,
                        mcspInt block_dim, const mcspComplexDouble* d_B, mcspInt ldb, const mcspComplexDouble* beta,
                        mcspComplexDouble* d_C, mcspInt ldc) {
    return mcspBsrmmTemplate(handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind,
                             bsr_cols_ind, block_dim, d_B, ldb, beta, d_C, ldc);
}

mcspStatus_t mcspCuinSbsrmm(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                          mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const float* alpha,
                          const mcspMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA,
                          const int* bsrSortedColIndA, const int blockSize, const float* B, const int ldb,
                          const float* beta, float* C, int ldc) {
    return mcspBsrmmTemplate(handle, dirA, transA, transB, (mcspInt)mb, (mcspInt)n, (mcspInt)kb, (mcspInt)nnzb, alpha,
                             descrA, bsrSortedValA, (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA,
                             (mcspInt)blockSize, B, (mcspInt)ldb, beta, C, (mcspInt)ldc);
}

mcspStatus_t mcspCuinDbsrmm(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                          mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const double* alpha,
                          const mcspMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA,
                          const int* bsrSortedColIndA, const int blockSize, const double* B, const int ldb,
                          const double* beta, double* C, int ldc) {
    return mcspBsrmmTemplate(handle, dirA, transA, transB, (mcspInt)mb, (mcspInt)n, (mcspInt)kb, (mcspInt)nnzb, alpha,
                             descrA, bsrSortedValA, (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA,
                             (mcspInt)blockSize, B, (mcspInt)ldb, beta, C, (mcspInt)ldc);
}

mcspStatus_t mcspCuinCbsrmm(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                          mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const mcFloatComplex* alpha,
                          const mcspMatDescr_t descrA, const mcFloatComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                          const int* bsrSortedColIndA, const int blockSize, const mcFloatComplex* B, const int ldb,
                          const mcFloatComplex* beta, mcFloatComplex* C, int ldc) {
    return mcspBsrmmTemplate(handle, dirA, transA, transB, (mcspInt)mb, (mcspInt)n, (mcspInt)kb, (mcspInt)nnzb, alpha,
                             descrA, bsrSortedValA, (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA,
                             (mcspInt)blockSize, B, (mcspInt)ldb, beta, C, (mcspInt)ldc);
}

mcspStatus_t mcspCuinZbsrmm(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                          mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const mcDoubleComplex* alpha,
                          const mcspMatDescr_t descrA, const mcDoubleComplex* bsrSortedValA,
                          const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize,
                          const mcDoubleComplex* B, const int ldb, const mcDoubleComplex* beta, mcDoubleComplex* C,
                          int ldc) {
    return mcspBsrmmTemplate(handle, dirA, transA, transB, (mcspInt)mb, (mcspInt)n, (mcspInt)kb, (mcspInt)nnzb, alpha,
                             descrA, bsrSortedValA, (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA,
                             (mcspInt)blockSize, B, (mcspInt)ldb, beta, C, (mcspInt)ldc);
}

#ifdef __cplusplus
}
#endif