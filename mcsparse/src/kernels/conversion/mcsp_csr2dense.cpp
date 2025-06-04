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
mcspStatus_t mcspCsr2DenseTemplate(mcspHandle_t handle, idxType m, idxType n, const mcspMatDescr_t mcsp_descr_A,
                                   const valType *csr_vals, const idxType *csr_rows, const idxType *csr_cols,
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

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || csr_vals == nullptr || csr_rows == nullptr ||
        csr_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType nnz, nnz_;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemcpyAsync(&nnz, &csr_rows[m], sizeof(*csr_rows), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcMemcpyAsync(&nnz_, &csr_rows[0], sizeof(*csr_rows), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    nnz -= nnz_;

    idxType *coo_rows = nullptr;
    bool use_buffer_pool = handle->mcspUsePoolBuffer((void **)&coo_rows, nnz * sizeof(*coo_rows));
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc(&coo_rows, nnz * sizeof(*coo_rows)));
    }
    constexpr uint32_t nElem = 512;
    uint32_t nBlock = (m * WARP_SIZE + nElem - 1) / nElem;

    mcLaunchKernelGGL(mcspCsr2CooKernel, dim3(nBlock), dim3(nElem), 0, stream, m, csr_rows, coo_rows,
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
                           mcsp_descr_A->base, csr_vals, coo_rows, csr_cols);
    } else {
        mcLaunchKernelGGL(coo2denseKernel<nElem>, dim3(nBlock), dim3(nElem), 0, stream, nnz, lda, dense_matrix,
                           mcsp_descr_A->base, csr_vals, csr_cols, coo_rows);
    }

    if (use_buffer_pool) {
        handle->mcspReturnPoolBuffer();
    } else {
        MACA_ASSERT(mcFree(coo_rows));
    }
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                            float *dense_matrix, mcspInt lda) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda);
}

mcspStatus_t mcspDcsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                            double *dense_matrix, mcspInt lda) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda);
}

mcspStatus_t mcspCcsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                            mcspComplexFloat *dense_matrix, mcspInt lda) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda);
}

mcspStatus_t mcspZcsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                            mcspComplexDouble *dense_matrix, mcspInt lda) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda);
}

mcspStatus_t mcspCuinScsr2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const float *csr_vals, const int *csr_rows, const int *csr_cols, float *dense_matrix,
                              int lda) {
    return mcspCsr2DenseTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csr_vals, (mcspInt *)csr_rows,
                                 (mcspInt *)csr_cols, dense_matrix, (mcspInt)lda);
}

mcspStatus_t mcspCuinDcsr2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const double *csr_vals, const int *csr_rows, const int *csr_cols, double *dense_matrix,
                              int lda) {
    return mcspCsr2DenseTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csr_vals, (mcspInt *)csr_rows,
                                 (mcspInt *)csr_cols, dense_matrix, (mcspInt)lda);
}

mcspStatus_t mcspCuinCcsr2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat *csr_vals, const int *csr_rows, const int *csr_cols,
                              mcspComplexFloat *dense_matrix, int lda) {
    return mcspCsr2DenseTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csr_vals, (mcspInt *)csr_rows,
                                 (mcspInt *)csr_cols, dense_matrix, (mcspInt)lda);
}

mcspStatus_t mcspCuinZcsr2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble *csr_vals, const int *csr_rows, const int *csr_cols,
                              mcspComplexDouble *dense_matrix, int lda) {
    return mcspCsr2DenseTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csr_vals, (mcspInt *)csr_rows,
                                 (mcspInt *)csr_cols, dense_matrix, (mcspInt)lda);
}

// for generic
mcspStatus_t mcspSgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                   float *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspDgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                   double *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspCgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                   mcspComplexFloat *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspZgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                   mcspComplexDouble *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda, B_order);
}

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const __half *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                      __half *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda, B_order);
}
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const __half2 *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                      __half2 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspR16BgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const mcsp_bfloat16 *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                      mcsp_bfloat16 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda, B_order);
}

mcspStatus_t mcspC16BgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const mcsp_bfloat162 *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                      mcsp_bfloat162 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order) {
    return mcspCsr2DenseTemplate(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda, B_order);
}
#endif

#ifdef __cplusplus
}
#endif