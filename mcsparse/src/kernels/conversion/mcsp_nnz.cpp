#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "device_reduce.hpp"
#include "mcsp_config.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"
#include "nnz_device.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspNnzTemplate(mcspHandle_t handle, mcsparseDirection_t dir, idxType m, idxType n,
                             const mcspMatDescr_t mcsp_descr_A, const valType* dense_matrix, idxType lda,
                             idxType* nnz_per_row_or_column, idxType* nnz,
                             mcsparseOrder_t A_order = MCSPARSE_ORDER_COL) {
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

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || nnz_per_row_or_column == nullptr || nnz == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    idxType per_num = (dir == MCSPARSE_DIRECTION_ROW) ? m : n;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemsetAsync(nnz_per_row_or_column, 0, per_num * sizeof(*nnz_per_row_or_column), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    constexpr uint32_t block_size = 512;
    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    uint32_t grid_size = (per_num + segments_per_block - 1) / segments_per_block;
    if (dir == MCSPARSE_DIRECTION_ROW) {
        if (A_order == MCSPARSE_ORDER_COL) {
            mcLaunchKernelGGL((nnzKernelRow<block_size, segments_per_block, segment_size>), dim3(grid_size),
                               dim3(block_size), 0, stream, m, n, lda, dense_matrix, nnz_per_row_or_column,
                               GetTypedValue<valType>(0));
        } else {
            mcLaunchKernelGGL((nnzKernelColumn<block_size, segments_per_block, segment_size>), dim3(grid_size),
                               dim3(block_size), 0, stream, n, m, lda, dense_matrix, nnz_per_row_or_column,
                               GetTypedValue<valType>(0));
        }
    } else {
        if (A_order == MCSPARSE_ORDER_COL) {
            mcLaunchKernelGGL((nnzKernelColumn<block_size, segments_per_block, segment_size>), dim3(grid_size),
                               dim3(block_size), 0, stream, m, n, lda, dense_matrix, nnz_per_row_or_column,
                               GetTypedValue<valType>(0));
        } else {
            mcLaunchKernelGGL((nnzKernelRow<block_size, segments_per_block, segment_size>), dim3(grid_size),
                               dim3(block_size), 0, stream, n, m, lda, dense_matrix, nnz_per_row_or_column,
                               GetTypedValue<valType>(0));
        }
    }

    idxType buffer_size;
    void* buffer_device;
    idxType* dnnz;
    bool use_buffer_pool;
    MACA_ASSERT(mcMalloc(&dnnz, sizeof(*dnnz)));
    mcprim::reduce(nullptr, buffer_size, nnz_per_row_or_column, dnnz, per_num, mcprim::plus<idxType>(), stream);
    use_buffer_pool = handle->mcspUsePoolBuffer(&buffer_device, buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc(&buffer_device, buffer_size));
    }
    mcprim::reduce(buffer_device, buffer_size, nnz_per_row_or_column, dnnz, per_num, mcprim::plus<idxType>()), stream;
    if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
        MACA_ASSERT(mcMemcpyAsync(nnz, dnnz, sizeof(*dnnz), mcMemcpyDeviceToHost, stream));
    } else {
        MACA_ASSERT(mcMemcpyAsync(nnz, dnnz, sizeof(*dnnz), mcMemcpyDeviceToDevice, stream));
    }
    MACA_ASSERT(mcStreamSynchronize(stream));
    MACA_ASSERT(mcFree(dnnz));
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

mcspStatus_t mcspSnnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                      const mcspMatDescr_t mcsp_descr_A, const float* dense_matrix, mcspInt lda,
                      mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz);
}

mcspStatus_t mcspDnnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                      const mcspMatDescr_t mcsp_descr_A, const double* dense_matrix, mcspInt lda,
                      mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz);
}

mcspStatus_t mcspCnnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                      const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat* dense_matrix, mcspInt lda,
                      mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz);
}

mcspStatus_t mcspZnnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                      const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble* dense_matrix, mcspInt lda,
                      mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz);
}
mcspStatus_t mcspCuinSnnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                        const float* dense_matrix, int lda, int* nnz_per_row_or_column, int* nnz) {
    return mcspNnzTemplate(handle, dir, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix, (mcspInt)lda,
                           (mcspInt*)nnz_per_row_or_column, (mcspInt*)nnz);
}

mcspStatus_t mcspCuinDnnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                        const double* dense_matrix, int lda, int* nnz_per_row_or_column, int* nnz) {
    return mcspNnzTemplate(handle, dir, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix, (mcspInt)lda,
                           (mcspInt*)nnz_per_row_or_column, (mcspInt*)nnz);
}

mcspStatus_t mcspCuinCnnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                        const mcspComplexFloat* dense_matrix, int lda, int* nnz_per_row_or_column, int* nnz) {
    return mcspNnzTemplate(handle, dir, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix, (mcspInt)lda,
                           (mcspInt*)nnz_per_row_or_column, (mcspInt*)nnz);
}

mcspStatus_t mcspCuinZnnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                        const mcspComplexDouble* dense_matrix, int lda, int* nnz_per_row_or_column, int* nnz) {
    return mcspNnzTemplate(handle, dir, (mcspInt)m, (mcspInt)n, mcsp_descr_A, dense_matrix, (mcspInt)lda,
                           (mcspInt*)nnz_per_row_or_column, (mcspInt*)nnz);
}

// for generic
mcspStatus_t mcspSgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                             mcspInt n, const mcspMatDescr_t mcsp_descr_A, const float* dense_matrix, mcspInt lda,
                             mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, B_dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz, A_order);
}

mcspStatus_t mcspDgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                             mcspInt n, const mcspMatDescr_t mcsp_descr_A, const double* dense_matrix, mcspInt lda,
                             mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, B_dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz, A_order);
}

mcspStatus_t mcspCgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                             mcspInt n, const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat* dense_matrix,
                             mcspInt lda, mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, B_dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz, A_order);
}

mcspStatus_t mcspZgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                             mcspInt n, const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble* dense_matrix,
                             mcspInt lda, mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, B_dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz, A_order);
}

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                                mcspInt n, const mcspMatDescr_t mcsp_descr_A, const __half* dense_matrix, mcspInt lda,
                                mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, B_dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz, A_order);
}
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                                mcspInt n, const mcspMatDescr_t mcsp_descr_A, const __half2* dense_matrix, mcspInt lda,
                                mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, B_dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz, A_order);
}

mcspStatus_t mcspR16BFgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                                 mcspInt n, const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16* dense_matrix,
                                 mcspInt lda, mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, B_dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz, A_order);
}

mcspStatus_t mcspC16BFgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                                 mcspInt n, const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162* dense_matrix,
                                 mcspInt lda, mcspInt* nnz_per_row_or_column, mcspInt* nnz) {
    return mcspNnzTemplate(handle, B_dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz, A_order);
}
#endif

#ifdef __cplusplus
}
#endif