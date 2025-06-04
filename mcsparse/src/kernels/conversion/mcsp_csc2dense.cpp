#include "common/mcsp_types.h"
#include "coo2csr_device.hpp"
#include "coo2dense_device.hpp"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspCsc2DenseTemplate(mcspHandle_t handle, idxType m, idxType n, const mcspMatDescr_t mcsp_descr_A,
                                   const valType *csc_vals, const idxType *csc_rows, const idxType *csc_cols,
                                   valType *dense_matrix, idxType lda, mcsparseOrder_t B_order = MCSPARSE_ORDER_COL) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (mcsp_descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (m < 0 || n < 0 || ((B_order == MCSPARSE_ORDER_ROW) && lda < n) ||
        ((B_order == MCSPARSE_ORDER_COL) && lda < m)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || csc_vals == nullptr || csc_rows == nullptr ||
        csc_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType nnz, nnz_;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemcpyAsync(&nnz, &csc_cols[n], sizeof(*csc_cols), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcMemcpyAsync(&nnz_, &csc_cols[0], sizeof(*csc_cols), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    nnz -= nnz_;
    idxType *coo_cols = nullptr;
    bool use_buffer_pool = handle->mcspUsePoolBuffer((void **)&coo_cols, nnz * sizeof(*coo_cols));
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc(&coo_cols, nnz * sizeof(*coo_cols)));
    }
    constexpr uint32_t nElem = 512;
    uint32_t nBlock = (n * WARP_SIZE + nElem - 1) / nElem;
    mcLaunchKernelGGL(mcspCsr2CooKernel, dim3(nBlock), dim3(nElem), 0, stream, n, csc_cols, coo_cols,
                       mcsp_descr_A->base);

    if (B_order == MCSPARSE_ORDER_COL) {
        MACA_ASSERT(mcMemsetAsync(dense_matrix, 0, (lda * n) * sizeof(*dense_matrix), stream));
    } else {
        MACA_ASSERT(mcMemsetAsync(dense_matrix, 0, (m * lda) * sizeof(*dense_matrix), stream));
    }
    MACA_ASSERT(mcStreamSynchronize(stream));
    nBlock = (nnz + nElem - 1) / nElem;
    if (B_order == MCSPARSE_ORDER_COL) {
        mcLaunchKernelGGL(coo2denseKernel<nElem>, dim3(nBlock), dim3(nElem), 0, stream, nnz, lda, dense_matrix,
                           mcsp_descr_A->base, csc_vals, csc_rows, coo_cols);
    } else {
        mcLaunchKernelGGL(coo2denseKernel<nElem>, dim3(nBlock), dim3(nElem), 0, stream, nnz, lda, dense_matrix,
                           mcsp_descr_A->base, csc_vals, coo_cols, csc_rows);
    }

    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(coo_cols));
    }
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                            float *dense_matrix, mcspInt lda) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda);
}

mcspStatus_t mcspDcsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                            double *dense_matrix, mcspInt lda) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda);
}

mcspStatus_t mcspCcsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                            mcspComplexFloat *dense_matrix, mcspInt lda) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda);
}

mcspStatus_t mcspZcsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                            mcspComplexDouble *dense_matrix, mcspInt lda) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda);
}

mcspStatus_t mcspCuinScsc2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const float *csc_vals, const int *csc_rows, const int *csc_cols, float *dense_matrix,
                              int lda) {
    return mcspCsc2DenseTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csc_vals, (mcspInt *)csc_rows,
                                 (mcspInt *)csc_cols, dense_matrix, (mcspInt)lda);
}

mcspStatus_t mcspCuinDcsc2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const double *csc_vals, const int *csc_rows, const int *csc_cols, double *dense_matrix,
                              int lda) {
    return mcspCsc2DenseTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csc_vals, (mcspInt *)csc_rows,
                                 (mcspInt *)csc_cols, dense_matrix, (mcspInt)lda);
}

mcspStatus_t mcspCuinCcsc2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat *csc_vals, const int *csc_rows, const int *csc_cols,
                              mcspComplexFloat *dense_matrix, int lda) {
    return mcspCsc2DenseTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csc_vals, (mcspInt *)csc_rows,
                                 (mcspInt *)csc_cols, dense_matrix, (mcspInt)lda);
}

mcspStatus_t mcspCuinZcsc2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble *csc_vals, const int *csc_rows, const int *csc_cols,
                              mcspComplexDouble *dense_matrix, int lda) {
    return mcspCsc2DenseTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csc_vals, (mcspInt *)csc_rows,
                                 (mcspInt *)csc_cols, dense_matrix, (mcspInt)lda);
}

// for generic
mcspStatus_t mcspSgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const float *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                   float *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspDgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const double *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                   double *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspCgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const mcspComplexFloat *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                   mcspComplexFloat *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspZgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const mcspComplexDouble *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                   mcspComplexDouble *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda, B_order);
}

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const __half *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                      __half *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda, B_order);
}
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const __half2 *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                      __half2 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspR16BgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const mcsp_bfloat16 *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                      mcsp_bfloat16 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspC16BgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const mcsp_bfloat162 *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                      mcsp_bfloat162 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsc2DenseTemplate(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda, B_order);
}
#endif

#ifdef __cplusplus
}
#endif