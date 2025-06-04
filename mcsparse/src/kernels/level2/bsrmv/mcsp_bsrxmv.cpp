#include "bsrxmv_device.hpp"
#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspBsrxmvTemplate(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans,
                                idxType mask_size, idxType mb, idxType nb, idxType nnzb, const valType* alpha,
                                const mcspMatDescr_t bsr_descr, const valType* bsr_vals, const idxType* mask_ptr,
                                const idxType* bsr_rows_ind, const idxType* bsr_ends_ind, const idxType* bsr_cols_ind,
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

    if (mb == 0 || nb == 0 || mask_size == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (x == nullptr || y == nullptr || mask_ptr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (bsr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    valType h_beta = getScalarToHost(beta, handle->ptr_mode);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    mcStream_t stream = mcspGetStreamInternal(handle);

    if (nnzb == 0 || h_alpha == static_cast<valType>(0)) {
        idxType y_dim = mask_size * block_dim;
        constexpr idxType block_size = 512;
        idxType blocks = (y_dim + block_size - 1) / block_size;
        mcLaunchKernelGGL(mcspBsrxmvZeroAlphaKernel<block_size>, dim3(blocks), dim3(block_size), 0, stream, mask_size,
                           mask_ptr, y_dim, block_dim, h_beta, y);
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows_ind == nullptr || bsr_cols_ind == nullptr || bsr_ends_ind == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    constexpr idxType block_size = WARP_SIZE;
    idxType blocks = mask_size * block_dim;
    mcLaunchKernelGGL((mcspBsrxmvKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, mask_size, mb, nb,
                       dir, bsr_descr->base, bsr_vals, mask_ptr, bsr_rows_ind, bsr_ends_ind, bsr_cols_ind, block_dim,
                       h_alpha, x, h_beta, y);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSbsrxmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mask_size,
                         mcspInt mb, mcspInt nb, mcspInt nnzb, const float* alpha, const mcspMatDescr_t bsr_descr,
                         const float* bsr_vals, const mcspInt* mask_ptr, const mcspInt* bsr_rows_ind,
                         const mcspInt* bsr_ends_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim, const float* x,
                         const float* beta, float* y) {
    return mcspBsrxmvTemplate(handle, dir, trans, mask_size, mb, nb, nnzb, alpha, bsr_descr, bsr_vals, mask_ptr,
                              bsr_rows_ind, bsr_ends_ind, bsr_cols_ind, block_dim, x, beta, y);
}

mcspStatus_t mcspDbsrxmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mask_size,
                         mcspInt mb, mcspInt nb, mcspInt nnzb, const double* alpha, const mcspMatDescr_t bsr_descr,
                         const double* bsr_vals, const mcspInt* mask_ptr, const mcspInt* bsr_rows_ind,
                         const mcspInt* bsr_ends_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim, const double* x,
                         const double* beta, double* y) {
    return mcspBsrxmvTemplate(handle, dir, trans, mask_size, mb, nb, nnzb, alpha, bsr_descr, bsr_vals, mask_ptr,
                              bsr_rows_ind, bsr_ends_ind, bsr_cols_ind, block_dim, x, beta, y);
}

mcspStatus_t mcspCbsrxmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mask_size,
                         mcspInt mb, mcspInt nb, mcspInt nnzb, const mcspComplexFloat* alpha,
                         const mcspMatDescr_t bsr_descr, const mcspComplexFloat* bsr_vals, const mcspInt* mask_ptr,
                         const mcspInt* bsr_rows_ind, const mcspInt* bsr_ends_ind, const mcspInt* bsr_cols_ind,
                         mcspInt block_dim, const mcspComplexFloat* x, const mcspComplexFloat* beta,
                         mcspComplexFloat* y) {
    return mcspBsrxmvTemplate(handle, dir, trans, mask_size, mb, nb, nnzb, alpha, bsr_descr, bsr_vals, mask_ptr,
                              bsr_rows_ind, bsr_ends_ind, bsr_cols_ind, block_dim, x, beta, y);
}

mcspStatus_t mcspZbsrxmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mask_size,
                         mcspInt mb, mcspInt nb, mcspInt nnzb, const mcspComplexDouble* alpha,
                         const mcspMatDescr_t bsr_descr, const mcspComplexDouble* bsr_vals, const mcspInt* mask_ptr,
                         const mcspInt* bsr_rows_ind, const mcspInt* bsr_ends_ind, const mcspInt* bsr_cols_ind,
                         mcspInt block_dim, const mcspComplexDouble* x, const mcspComplexDouble* beta,
                         mcspComplexDouble* y) {
    return mcspBsrxmvTemplate(handle, dir, trans, mask_size, mb, nb, nnzb, alpha, bsr_descr, bsr_vals, mask_ptr,
                              bsr_rows_ind, bsr_ends_ind, bsr_cols_ind, block_dim, x, beta, y);
}


mcspStatus_t mcspCuinSbsrxmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int sizeOfMask,
                           int mb, int nb, int nnzb, const float* alpha, const mcspMatDescr_t descrA,
                           const float* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA,
                           const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const float* x,
                           const float* beta, float* y) {
    return mcspBsrxmvTemplate(handle, dirA, transA, (mcspInt)sizeOfMask, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, alpha,
                              descrA, bsrSortedValA, (mcspInt*)bsrSortedMaskPtrA, (mcspInt*)bsrSortedRowPtrA,
                              (mcspInt*)bsrSortedEndPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, x, beta, y);
}

mcspStatus_t mcspCuinDbsrxmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int sizeOfMask,
                           int mb, int nb, int nnzb, const double* alpha, const mcspMatDescr_t descrA,
                           const double* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA,
                           const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const double* x,
                           const double* beta, double* y) {
    return mcspBsrxmvTemplate(handle, dirA, transA, (mcspInt)sizeOfMask, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, alpha,
                              descrA, bsrSortedValA, (mcspInt*)bsrSortedMaskPtrA, (mcspInt*)bsrSortedRowPtrA,
                              (mcspInt*)bsrSortedEndPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, x, beta, y);
}

mcspStatus_t mcspCuinCbsrxmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int sizeOfMask,
                           int mb, int nb, int nnzb, const mcFloatComplex* alpha, const mcspMatDescr_t descrA,
                           const mcFloatComplex* bsrSortedValA, const int* bsrSortedMaskPtrA,
                           const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA,
                           int blockDim, const mcFloatComplex* x, const mcFloatComplex* beta, mcFloatComplex* y) {
    return mcspBsrxmvTemplate(handle, dirA, transA, (mcspInt)sizeOfMask, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, alpha,
                              descrA, bsrSortedValA, (mcspInt*)bsrSortedMaskPtrA, (mcspInt*)bsrSortedRowPtrA,
                              (mcspInt*)bsrSortedEndPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, x, beta, y);
}

mcspStatus_t mcspCuinZbsrxmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int sizeOfMask,
                           int mb, int nb, int nnzb, const mcDoubleComplex* alpha, const mcspMatDescr_t descrA,
                           const mcDoubleComplex* bsrSortedValA, const int* bsrSortedMaskPtrA,
                           const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA,
                           int blockDim, const mcDoubleComplex* x, const mcDoubleComplex* beta, mcDoubleComplex* y) {
    return mcspBsrxmvTemplate(handle, dirA, transA, (mcspInt)sizeOfMask, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, alpha,
                              descrA, bsrSortedValA, (mcspInt*)bsrSortedMaskPtrA, (mcspInt*)bsrSortedRowPtrA,
                              (mcspInt*)bsrSortedEndPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, x, beta, y);
}

#ifdef __cplusplus
}
#endif