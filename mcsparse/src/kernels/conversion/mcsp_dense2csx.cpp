#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "csr2csr_compress_device.hpp"
#include "dense2csx_device.hpp"
#include "device_scan.hpp"
#include "mcsp_config.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspDense2CsxTemplate(mcspHandle_t handle, mcsparseDirection_t dir, idxType m, idxType n,
                                   const mcspMatDescr_t mcsp_descr_A, const valType *dense_matrix, idxType lda,
                                   idxType *nnz_per_row_or_column, valType *csx_vals, idxType *csx_rows,
                                   idxType *csx_cols, mcsparseOrder_t A_order = MCSPARSE_ORDER_COL) {
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

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || nnz_per_row_or_column == nullptr || csx_vals == nullptr ||
        csx_rows == nullptr || csx_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType per_num = (dir == MCSPARSE_DIRECTION_ROW) ? m : n;
    constexpr uint32_t block_size = 512;
    uint32_t grid_size = (per_num + block_size - 1) / block_size;
    mcStream_t stream = mcspGetStreamInternal(handle);

    idxType *tmp_csx_row_or_column;
    MACA_ASSERT(mcMalloc(&tmp_csx_row_or_column, (per_num + 1) * sizeof(*tmp_csx_row_or_column)));
    mcLaunchKernelGGL(FillRowDevice<block_size>, dim3(grid_size), dim3(block_size), 0, stream, per_num,
                       mcsp_descr_A->base, nnz_per_row_or_column, tmp_csx_row_or_column);

    idxType buffer_size;
    void *buffer_device;
    bool use_buffer_pool;

    idxType *csx_row_or_column = (dir == MCSPARSE_DIRECTION_ROW) ? csx_rows : csx_cols;
    mcprim::inclusive_scan(nullptr, buffer_size, tmp_csx_row_or_column, csx_row_or_column, per_num + 1, stream);
    use_buffer_pool = handle->mcspUsePoolBuffer(&buffer_device, buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc(&buffer_device, buffer_size));
    }
    mcprim::inclusive_scan(buffer_device, buffer_size, tmp_csx_row_or_column, csx_row_or_column, per_num + 1, stream);

    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    grid_size = (per_num + segments_per_block - 1) / segments_per_block;

    if (dir == MCSPARSE_DIRECTION_ROW) {
        if (A_order == MCSPARSE_ORDER_COL) {
            mcLaunchKernelGGL((dense2csrKernel<block_size, segments_per_block, segment_size, WARP_SIZE>),
                               dim3(grid_size), dim3(block_size), 0, stream, m, n, lda, dense_matrix, mcsp_descr_A->base,
                               csx_vals, csx_rows, csx_cols, GetTypedValue<valType>(0));
        } else {
            mcLaunchKernelGGL((dense2cscKernel<block_size, segments_per_block, segment_size, WARP_SIZE>),
                               dim3(grid_size), dim3(block_size), 0, stream, n, m, lda, dense_matrix, mcsp_descr_A->base,
                               csx_vals, csx_cols, csx_rows, GetTypedValue<valType>(0));
        }
    } else {
        if (A_order == MCSPARSE_ORDER_COL) {
            mcLaunchKernelGGL((dense2cscKernel<block_size, segments_per_block, segment_size, WARP_SIZE>),
                               dim3(grid_size), dim3(block_size), 0, stream, m, n, lda, dense_matrix, mcsp_descr_A->base,
                               csx_vals, csx_rows, csx_cols, GetTypedValue<valType>(0));
        } else {
            mcLaunchKernelGGL((dense2csrKernel<block_size, segments_per_block, segment_size, WARP_SIZE>),
                               dim3(grid_size), dim3(block_size), 0, stream, n, m, lda, dense_matrix, mcsp_descr_A->base,
                               csx_vals, csx_cols, csx_rows, GetTypedValue<valType>(0));
        }
    }

    MACA_ASSERT(mcFree(tmp_csx_row_or_column));
    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(buffer_device));
    }
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSdense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column, float *csr_vals,
                            mcspInt *csr_rows, mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols);
}

mcspStatus_t mcspDdense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column, double *csr_vals,
                            mcspInt *csr_rows, mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols);
}

mcspStatus_t mcspCdense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column,
                            mcspComplexFloat *csr_vals, mcspInt *csr_rows, mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols);
}

mcspStatus_t mcspZdense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column,
                            mcspComplexDouble *csr_vals, mcspInt *csr_rows, mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols);
}

mcspStatus_t mcspSdense2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column, float *csc_vals,
                            mcspInt *csc_rows, mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols);
}

mcspStatus_t mcspDdense2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column, double *csc_vals,
                            mcspInt *csc_rows, mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols);
}

mcspStatus_t mcspCdense2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column,
                            mcspComplexFloat *csc_vals, mcspInt *csc_rows, mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols);
}

mcspStatus_t mcspZdense2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column,
                            mcspComplexDouble *csc_vals, mcspInt *csc_rows, mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols);
}

mcspStatus_t mcspCuinSdense2csr(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const float *dense_matrix, int lda, int *nnz_per_row_or_column, float *csr_vals,
                              int *csr_rows, int *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix,
                                 (mcspInt)lda, (mcspInt *)nnz_per_row_or_column, csr_vals, (mcspInt *)csr_rows,
                                 (mcspInt *)csr_cols);
}

mcspStatus_t mcspCuinDdense2csr(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const double *dense_matrix, int lda, int *nnz_per_row_or_column, double *csr_vals,
                              int *csr_rows, int *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix,
                                 (mcspInt)lda, (mcspInt *)nnz_per_row_or_column, csr_vals, (mcspInt *)csr_rows,
                                 (mcspInt *)csr_cols);
}

mcspStatus_t mcspCuinCdense2csr(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat *dense_matrix, int lda, int *nnz_per_row_or_column,
                              mcspComplexFloat *csr_vals, int *csr_rows, int *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix,
                                 (mcspInt)lda, (mcspInt *)nnz_per_row_or_column, csr_vals, (mcspInt *)csr_rows,
                                 (mcspInt *)csr_cols);
}

mcspStatus_t mcspCuinZdense2csr(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble *dense_matrix, int lda, int *nnz_per_row_or_column,
                              mcspComplexDouble *csr_vals, int *csr_rows, int *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix,
                                 (mcspInt)lda, (mcspInt *)nnz_per_row_or_column, csr_vals, (mcspInt *)csr_rows,
                                 (mcspInt *)csr_cols);
}

mcspStatus_t mcspCuinSdense2csc(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const float *dense_matrix, int lda, int *nnz_per_row_or_column, float *csc_vals,
                              int *csc_rows, int *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix,
                                 (mcspInt)lda, (mcspInt *)nnz_per_row_or_column, csc_vals, (mcspInt *)csc_rows,
                                 (mcspInt *)csc_cols);
}

mcspStatus_t mcspCuinDdense2csc(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const double *dense_matrix, int lda, int *nnz_per_row_or_column, double *csc_vals,
                              int *csc_rows, int *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix,
                                 (mcspInt)lda, (mcspInt *)nnz_per_row_or_column, csc_vals, (mcspInt *)csc_rows,
                                 (mcspInt *)csc_cols);
}

mcspStatus_t mcspCuinCdense2csc(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat *dense_matrix, int lda, int *nnz_per_row_or_column,
                              mcspComplexFloat *csc_vals, int *csc_rows, int *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix,
                                 (mcspInt)lda, (mcspInt *)nnz_per_row_or_column, csc_vals, (mcspInt *)csc_rows,
                                 (mcspInt *)csc_cols);
}

mcspStatus_t mcspCuinZdense2csc(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble *dense_matrix, int lda, int *nnz_per_row_or_column,
                              mcspComplexDouble *csc_vals, int *csc_rows, int *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix,
                                 (mcspInt)lda, (mcspInt *)nnz_per_row_or_column, csc_vals, (mcspInt *)csc_rows,
                                 (mcspInt *)csc_cols);
}

// for generic
mcspStatus_t mcspSgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const float *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, float *csr_vals, mcspInt *csr_rows,
                                   mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols, A_order);
}

mcspStatus_t mcspDgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const double *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, double *csr_vals, mcspInt *csr_rows,
                                   mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols, A_order);
}

mcspStatus_t mcspCgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, mcspComplexFloat *csr_vals, mcspInt *csr_rows,
                                   mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols, A_order);
}

mcspStatus_t mcspZgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *dense_matrix,
                                   mcspInt lda, mcspInt *nnz_per_row_or_column, mcspComplexDouble *csr_vals,
                                   mcspInt *csr_rows, mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols, A_order);
}

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row_or_column, __half *csr_vals, mcspInt *csr_rows,
                                      mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols, A_order);
}
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half2 *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row_or_column, __half2 *csr_vals, mcspInt *csr_rows,
                                      mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols, A_order);
}

mcspStatus_t mcspR16BFgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row_or_column, mcsp_bfloat16 *csr_vals,
                                       mcspInt *csr_rows, mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols, A_order);
}

mcspStatus_t mcspC16BFgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row_or_column, mcsp_bfloat162 *csr_vals,
                                       mcspInt *csr_rows, mcspInt *csr_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_ROW, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csr_vals, csr_rows, csr_cols, A_order);
}
#endif

mcspStatus_t mcspSgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const float *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, float *csc_vals, mcspInt *csc_rows,
                                   mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols, A_order);
}

mcspStatus_t mcspDgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const double *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, double *csc_vals, mcspInt *csc_rows,
                                   mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols, A_order);
}

mcspStatus_t mcspCgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, mcspComplexFloat *csc_vals, mcspInt *csc_rows,
                                   mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols, A_order);
}

mcspStatus_t mcspZgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *dense_matrix,
                                   mcspInt lda, mcspInt *nnz_per_row_or_column, mcspComplexDouble *csc_vals,
                                   mcspInt *csc_rows, mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols, A_order);
}

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row_or_column, __half *csc_vals, mcspInt *csc_rows,
                                      mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols, A_order);
}
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half2 *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row_or_column, __half2 *csc_vals, mcspInt *csc_rows,
                                      mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols, A_order);
}

mcspStatus_t mcspR16BFgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row_or_column, mcsp_bfloat16 *csc_vals,
                                       mcspInt *csc_rows, mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols, A_order);
}

mcspStatus_t mcspC16BFgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row_or_column, mcsp_bfloat162 *csc_vals,
                                       mcspInt *csc_rows, mcspInt *csc_cols) {
    return mcspDense2CsxTemplate(handle, MCSPARSE_DIRECTION_COLUMN, m, n, mcsp_descr_A, dense_matrix, lda,
                                 nnz_per_row_or_column, csc_vals, csc_rows, csc_cols, A_order);
}
#endif

#ifdef __cplusplus
}
#endif