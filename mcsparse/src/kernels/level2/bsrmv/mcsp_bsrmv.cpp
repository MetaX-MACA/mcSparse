#include "bsrmv_device.hpp"
#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspBsrmvTemplate(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, idxType mb,
                               idxType nb, idxType nnzb, const valType* alpha, const mcspMatDescr_t bsr_descr,
                               const valType* bsr_vals, const idxType* bsr_rows_ind, const idxType* bsr_cols_ind,
                               idxType block_dim, const valType* x, const valType* beta, valType* y) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mb < 0 || nb < 0 || nnzb < 0 || block_dim <= 1) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || bsr_descr == nullptr || beta == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (mb == 0 || nb == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (x == nullptr || y == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (bsr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    valType h_beta = getScalarToHost(beta, handle->ptr_mode);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    mcStream_t stream = mcspGetStreamInternal(handle);

    if (nnzb == 0 || h_alpha == static_cast<valType>(0)) {
        idxType y_dim = mb * block_dim;
        constexpr idxType block_size = 512;
        idxType blocks = (y_dim + block_size - 1) / block_size;
        mcLaunchKernelGGL(mcspBsrmvZeroAlphaKernel<block_size>, dim3(blocks), dim3(block_size), 0, stream, y_dim,
                           h_beta, y);
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows_ind == nullptr || bsr_cols_ind == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    constexpr idxType block_size = WARP_SIZE;
    idxType blocks = mb * block_dim;
    mcLaunchKernelGGL((mcspBsrmvKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, mb, nb, dir,
                       bsr_descr->base, bsr_vals, bsr_rows_ind, bsr_cols_ind, block_dim, h_alpha, x, h_beta, y);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSbsrmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb, mcspInt nb,
                        mcspInt nnzb, const float* alpha, const mcspMatDescr_t bsr_descr, const float* bsr_vals,
                        const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim, const float* x,
                        const float* beta, float* y) {
    return mcspBsrmvTemplate(handle, dir, trans, mb, nb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                             block_dim, x, beta, y);
}

mcspStatus_t mcspDbsrmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb, mcspInt nb,
                        mcspInt nnzb, const double* alpha, const mcspMatDescr_t bsr_descr, const double* bsr_vals,
                        const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim, const double* x,
                        const double* beta, double* y) {
    return mcspBsrmvTemplate(handle, dir, trans, mb, nb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                             block_dim, x, beta, y);
}

mcspStatus_t mcspCbsrmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb, mcspInt nb,
                        mcspInt nnzb, const mcspComplexFloat* alpha, const mcspMatDescr_t bsr_descr,
                        const mcspComplexFloat* bsr_vals, const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind,
                        mcspInt block_dim, const mcspComplexFloat* x, const mcspComplexFloat* beta,
                        mcspComplexFloat* y) {
    return mcspBsrmvTemplate(handle, dir, trans, mb, nb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                             block_dim, x, beta, y);
}

mcspStatus_t mcspZbsrmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb, mcspInt nb,
                        mcspInt nnzb, const mcspComplexDouble* alpha, const mcspMatDescr_t bsr_descr,
                        const mcspComplexDouble* bsr_vals, const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind,
                        mcspInt block_dim, const mcspComplexDouble* x, const mcspComplexDouble* beta,
                        mcspComplexDouble* y) {
    return mcspBsrmvTemplate(handle, dir, trans, mb, nb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                             block_dim, x, beta, y);
}

mcspStatus_t mcspCuinSbsrmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb, int nb,
                          int nnzb, const float* alpha, const mcspMatDescr_t descrA, const float* bsrSortedValA,
                          const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const float* x,
                          const float* beta, float* y) {
    return mcspBsrmvTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, alpha, descrA,
                             bsrSortedValA, (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim,
                             x, beta, y);
}

mcspStatus_t mcspCuinDbsrmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb, int nb,
                          int nnzb, const double* alpha, const mcspMatDescr_t descrA, const double* bsrSortedValA,
                          const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const double* x,
                          const double* beta, double* y) {
    return mcspBsrmvTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, alpha, descrA,
                             bsrSortedValA, (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim,
                             x, beta, y);
}

mcspStatus_t mcspCuinCbsrmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb, int nb,
                          int nnzb, const mcFloatComplex* alpha, const mcspMatDescr_t descrA,
                          const mcFloatComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA,
                          int blockDim, const mcFloatComplex* x, const mcFloatComplex* beta, mcFloatComplex* y) {
    return mcspBsrmvTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, alpha, descrA,
                             bsrSortedValA, (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim,
                             x, beta, y);
}

mcspStatus_t mcspCuinZbsrmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb, int nb,
                          int nnzb, const mcDoubleComplex* alpha, const mcspMatDescr_t descrA,
                          const mcDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                          const int* bsrSortedColIndA, int blockDim, const mcDoubleComplex* x,
                          const mcDoubleComplex* beta, mcDoubleComplex* y) {
    return mcspBsrmvTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, alpha, descrA,
                             bsrSortedValA, (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim,
                             x, beta, y);
}

#ifdef __cplusplus
}
#endif