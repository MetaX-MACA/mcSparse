#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_interface.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspDense2CooTemplate(mcspHandle_t handle, idxType m, idxType n, const mcspMatDescr_t mcsp_descr_A,
                                   const valType *dense_matrix, idxType lda, idxType *nnz_per_row, valType *coo_vals,
                                   idxType *coo_rows, idxType *coo_cols, mcsparseOrder_t A_order = MCSPARSE_ORDER_COL) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (mcsp_descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (m < 0 || n < 0 || ((A_order == MCSPARSE_ORDER_ROW) && lda < n) ||
        ((A_order == MCSPARSE_ORDER_COL) && lda < m)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || nnz_per_row == nullptr || coo_vals == nullptr ||
        coo_rows == nullptr || coo_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    idxType *csr_rows = nullptr;
    bool use_buffer_pool;
    use_buffer_pool = handle->mcspUsePoolBuffer((void **)&csr_rows, (m + 1) * sizeof(*csr_rows));
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc((void **)&csr_rows, (m + 1) * sizeof(*csr_rows)));
    }
    if constexpr (std::is_same_v<valType, float>) {
        mcspSgenericDense2Csr(handle, A_order, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, csr_rows,
                              coo_cols);
    } else if constexpr (std::is_same_v<valType, double>) {
        mcspDgenericDense2Csr(handle, A_order, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, csr_rows,
                              coo_cols);
    } else if constexpr (std::is_same_v<valType, mcspComplexFloat>) {
        mcspCgenericDense2Csr(handle, A_order, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, csr_rows,
                              coo_cols);
    } else if constexpr (std::is_same_v<valType, mcspComplexDouble>) {
        mcspZgenericDense2Csr(handle, A_order, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, csr_rows,
                              coo_cols);
    } else if constexpr (std::is_same_v<valType, __half>) {
#if defined(__MACA__)
        mcspR16FgenericDense2Csr(handle, A_order, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals,
                                 csr_rows, coo_cols);
#endif
    } else if constexpr (std::is_same_v<valType, __half2>) {
#ifdef __MACA__
        mcspC16FgenericDense2Csr(handle, A_order, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals,
                                 csr_rows, coo_cols);
#endif
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
#ifdef __MACA__
        mcspR16BFgenericDense2Csr(handle, A_order, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals,
                                  csr_rows, coo_cols);
#endif
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat162>) {
#ifdef __MACA__
        mcspC16BFgenericDense2Csr(handle, A_order, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals,
                                  csr_rows, coo_cols);
#endif
    }

    idxType nnz, nnz_;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemcpyAsync(&nnz, &csr_rows[m], sizeof(*csr_rows), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcMemcpyAsync(&nnz_, &csr_rows[0], sizeof(*csr_rows), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    nnz -= nnz_;
    mcspCsr2Coo(handle, csr_rows, nnz, m, coo_rows, mcsp_descr_A->base);

    if (use_buffer_pool) {
        handle->mcspReturnPoolBuffer();
    } else {
        MACA_ASSERT(mcFree(csr_rows));
    }
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSdense2Coo(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *dense_matrix, mcspInt lda, mcspInt *nnz_per_row, float *coo_vals,
                            mcspInt *coo_rows, mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols);
}

mcspStatus_t mcspDdense2Coo(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *dense_matrix, mcspInt lda, mcspInt *nnz_per_row, double *coo_vals,
                            mcspInt *coo_rows, mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols);
}

mcspStatus_t mcspCdense2Coo(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *dense_matrix, mcspInt lda, mcspInt *nnz_per_row,
                            mcspComplexFloat *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols);
}

mcspStatus_t mcspZdense2Coo(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *dense_matrix, mcspInt lda, mcspInt *nnz_per_row,
                            mcspComplexDouble *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols);
}

// for generic
mcspStatus_t mcspSgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const float *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row, float *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols, A_order);
}

mcspStatus_t mcspDgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const double *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row, double *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols, A_order);
}

mcspStatus_t mcspCgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row, mcspComplexFloat *coo_vals, mcspInt *coo_rows,
                                   mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols, A_order);
}

mcspStatus_t mcspZgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *dense_matrix,
                                   mcspInt lda, mcspInt *nnz_per_row, mcspComplexDouble *coo_vals, mcspInt *coo_rows,
                                   mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols, A_order);
}

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row, __half *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols, A_order);
}
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half2 *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row, __half2 *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols, A_order);
}

mcspStatus_t mcspR16BFgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row, mcsp_bfloat16 *coo_vals, mcspInt *coo_rows,
                                       mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols, A_order);
}

mcspStatus_t mcspC16BFgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row, mcsp_bfloat162 *coo_vals, mcspInt *coo_rows,
                                       mcspInt *coo_cols) {
    return mcspDense2CooTemplate(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row, coo_vals, coo_rows,
                                 coo_cols, A_order);
}
#endif
#ifdef __cplusplus
}
#endif