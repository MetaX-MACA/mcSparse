#include "common/mcsp_types.h"
#include "coo2dense_device.hpp"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspCoo2DenseTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz,
                                   const mcspMatDescr_t mcsp_descr_A, const valType *coo_vals, const idxType *coo_rows,
                                   const idxType *coo_cols, valType *dense_matrix, idxType lda, mcsparseOrder_t order) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (mcsp_descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (m < 0 || n < 0 || ((order == MCSPARSE_ORDER_ROW) && lda < n) || ((order == MCSPARSE_ORDER_COL) && lda < m)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0 || nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || coo_vals == nullptr || coo_rows == nullptr ||
        coo_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    mcStream_t stream = mcspGetStreamInternal(handle);
    if (order == MCSPARSE_ORDER_COL) {
        MACA_ASSERT(mcMemsetAsync(dense_matrix, 0, (lda * n) * sizeof(*dense_matrix), stream));
    } else {
        MACA_ASSERT(mcMemsetAsync(dense_matrix, 0, (m * lda) * sizeof(*dense_matrix), stream));
    }
    MACA_ASSERT(mcStreamSynchronize(stream));
    constexpr uint32_t nElem = 512;
    uint32_t nBlock = (nnz + nElem - 1) / nElem;
    if (order == MCSPARSE_ORDER_COL) {
        mcLaunchKernelGGL(coo2denseKernel<nElem>, dim3(nBlock), dim3(nElem), 0, stream, nnz, lda, dense_matrix,
                           mcsp_descr_A->base, coo_vals, coo_rows, coo_cols);
    } else {
        mcLaunchKernelGGL(coo2denseKernel<nElem>, dim3(nBlock), dim3(nElem), 0, stream, nnz, lda, dense_matrix,
                           mcsp_descr_A->base, coo_vals, coo_cols, coo_rows);
    }

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                            const float *coo_vals, const mcspInt *coo_rows, const mcspInt *coo_cols,
                            float *dense_matrix, mcspInt lda) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 MCSPARSE_ORDER_COL);
}

mcspStatus_t mcspDcoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                            const double *coo_vals, const mcspInt *coo_rows, const mcspInt *coo_cols,
                            double *dense_matrix, mcspInt lda) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 MCSPARSE_ORDER_COL);
}

mcspStatus_t mcspCcoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *coo_vals, const mcspInt *coo_rows, const mcspInt *coo_cols,
                            mcspComplexFloat *dense_matrix, mcspInt lda) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 MCSPARSE_ORDER_COL);
}

mcspStatus_t mcspZcoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *coo_vals, const mcspInt *coo_rows, const mcspInt *coo_cols,
                            mcspComplexDouble *dense_matrix, mcspInt lda) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 MCSPARSE_ORDER_COL);
}

// for generic
mcspStatus_t mcspSgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                   const mcspMatDescr_t mcsp_descr_A, const float *coo_vals, const mcspInt *coo_rows,
                                   const mcspInt *coo_cols, float *dense_matrix, mcspInt lda, mcsparseOrder_t order_B) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 order_B);
}

mcspStatus_t mcspDgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                   const mcspMatDescr_t mcsp_descr_A, const double *coo_vals, const mcspInt *coo_rows,
                                   const mcspInt *coo_cols, double *dense_matrix, mcspInt lda,
                                   mcsparseOrder_t order_B) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 order_B);
}

mcspStatus_t mcspCgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *coo_vals,
                                   const mcspInt *coo_rows, const mcspInt *coo_cols, mcspComplexFloat *dense_matrix,
                                   mcspInt lda, mcsparseOrder_t order_B) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 order_B);
}

mcspStatus_t mcspZgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *coo_vals,
                                   const mcspInt *coo_rows, const mcspInt *coo_cols, mcspComplexDouble *dense_matrix,
                                   mcspInt lda, mcsparseOrder_t order_B) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 order_B);
}

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                      const mcspMatDescr_t mcsp_descr_A, const __half *coo_vals,
                                      const mcspInt *coo_rows, const mcspInt *coo_cols, __half *dense_matrix,
                                      mcspInt lda, mcsparseOrder_t order_B) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 order_B);
}
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                      const mcspMatDescr_t mcsp_descr_A, const __half2 *coo_vals,
                                      const mcspInt *coo_rows, const mcspInt *coo_cols, __half2 *dense_matrix,
                                      mcspInt lda, mcsparseOrder_t order_B) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 order_B);
}

mcspStatus_t mcspR16BgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                      const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16 *coo_vals,
                                      const mcspInt *coo_rows, const mcspInt *coo_cols, mcsp_bfloat16 *dense_matrix,
                                      mcspInt lda, mcsparseOrder_t order_B) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 order_B);
}

mcspStatus_t mcspC16BgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                      const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162 *coo_vals,
                                      const mcspInt *coo_rows, const mcspInt *coo_cols, mcsp_bfloat162 *dense_matrix,
                                      mcspInt lda, mcsparseOrder_t order_B) {
    return mcspCoo2DenseTemplate(handle, m, n, nnz, mcsp_descr_A, coo_vals, coo_rows, coo_cols, dense_matrix, lda,
                                 order_B);
}
#endif

#ifdef __cplusplus
}
#endif