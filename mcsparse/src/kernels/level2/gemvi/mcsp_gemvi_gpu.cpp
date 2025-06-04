#include <assert.h>
#include <stdio.h>

#include "common/mcsp_types.h"
#include "gemvi_device.hpp"
#include "mcsp_config.h"
#include "mcsp_dense_transpose_device.hpp"
#include "mcsp_handle.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType, typename valType>
mcspStatus_t mcspGemviBuffersizeTemplate(mcspHandle_t handle, mcsparseOperation_t trans, idxType m, idxType n,
                                         idxType nnz, size_t *buffer_size) {
    *buffer_size = MIN_BUFFER_SIZE;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGemviBufferSizeImpl(mcspHandle_t handle, mcsparseOperation_t trans, idxType m, idxType n, idxType nnz,
                                     int *buffer_size) {
    *buffer_size = MIN_BUFFER_SIZE;
    return MCSP_STATUS_SUCCESS;
}

template <uint32_t BLOCK_SIZE = 512, typename valType, typename idxType>
mcspStatus_t mcspGemviTemplate(mcspHandle_t handle, mcsparseOperation_t trans, idxType m, idxType n,
                               const valType *alpha, const valType *A, idxType lda, idxType nnz, const valType *x_val,
                               const idxType *x_ind, const valType *beta, valType *y, mcsparseIndexBase_t idx_base,
                               void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || lda < 0 || lda < m || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || beta == nullptr || A == nullptr || x_val == nullptr || x_ind == nullptr || y == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    valType h_beta = getScalarToHost(beta, handle->ptr_mode);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    if (h_alpha == static_cast<valType>(0) && h_beta == static_cast<valType>(0)) {
        return MCSP_STATUS_SUCCESS;
    }

    // @TODO(zhiming): scale y
    if (m == 0 || n == 0 || lda == 0 || nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double>) {
        if (trans == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    idxType ldat = lda;
    valType *At = const_cast<valType *>(A);
    mcStream_t stream = mcspGetStreamInternal(handle);
    if (trans == MCSPARSE_OPERATION_TRANSPOSE || trans == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        const int trans_dimX = 32;
        const int trans_dimY = 8;
        dim3 trans_blocks((n - 1) / trans_dimX + 1);
        dim3 trans_threads(trans_dimX * trans_dimY);
        ldat = n;
        MACA_ASSERT(mcMalloc((void **)&At, lda * n * sizeof(valType)));
        MACA_ASSERT(mcMemsetAsync(At, 0, lda * n * sizeof(valType), stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        mcLaunchKernelGGL((mcspDenseTransposeKernel<trans_dimX, trans_dimY>), trans_blocks, trans_threads, 0, stream, n, m,
                           A, lda, At, ldat);
    }

    if (trans == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        if constexpr (std::is_same_v<valType, mcspComplexFloat> || std::is_same_v<valType, mcspComplexDouble>) {
            const int block_size = 512;
            const int n_blocks = (lda * n + block_size - 1) / block_size;
            mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream, lda * n,
                               At);
        }
    }

    int block_size = BLOCK_SIZE;
    int block_num = (m + WARP_SIZE - 1) / WARP_SIZE;
    mcLaunchKernelGGL((mcspGemviKernel<BLOCK_SIZE>), dim3(block_num), dim3(block_size), BLOCK_SIZE * sizeof(*A),
                       stream, m, n, h_alpha, At, ldat, nnz, x_val, x_ind, h_beta, y, idx_base, temp_buffer);
    MACA_ASSERT(mcStreamSynchronize(stream));

    if (trans == MCSPARSE_OPERATION_TRANSPOSE || trans == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        MACA_ASSERT(mcFree(At));
    }
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSgemviBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, mcspInt nnz,
                                  size_t *buffer_size) {
    return mcspGemviBuffersizeTemplate<mcspInt, float>(handle, trans, m, n, nnz, buffer_size);
}

mcspStatus_t mcspDgemviBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, mcspInt nnz,
                                  size_t *buffer_size) {
    return mcspGemviBuffersizeTemplate<mcspInt, double>(handle, trans, m, n, nnz, buffer_size);
}

mcspStatus_t mcspCgemviBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, mcspInt nnz,
                                  size_t *buffer_size) {
    return mcspGemviBuffersizeTemplate<mcspInt, mcspComplexFloat>(handle, trans, m, n, nnz, buffer_size);
}

mcspStatus_t mcspZgemviBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, mcspInt nnz,
                                  size_t *buffer_size) {
    return mcspGemviBuffersizeTemplate<mcspInt, mcspComplexDouble>(handle, trans, m, n, nnz, buffer_size);
}

mcspStatus_t mcspSgemvi(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, const float *alpha,
                        const float *A, mcspInt lda, mcspInt nnz, const float *x_val, const mcspInt *x_ind,
                        const float *beta, float *y, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspGemviTemplate(handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
}

mcspStatus_t mcspDgemvi(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, const double *alpha,
                        const double *A, mcspInt lda, mcspInt nnz, const double *x_val, const mcspInt *x_ind,
                        const double *beta, double *y, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspGemviTemplate(handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
}

mcspStatus_t mcspCgemvi(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n,
                        const mcspComplexFloat *alpha, const mcspComplexFloat *A, mcspInt lda, mcspInt nnz,
                        const mcspComplexFloat *x_val, const mcspInt *x_ind, const mcspComplexFloat *beta,
                        mcspComplexFloat *y, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspGemviTemplate(handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
}

mcspStatus_t mcspZgemvi(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n,
                        const mcspComplexDouble *alpha, const mcspComplexDouble *A, mcspInt lda, mcspInt nnz,
                        const mcspComplexDouble *x_val, const mcspInt *x_ind, const mcspComplexDouble *beta,
                        mcspComplexDouble *y, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspGemviTemplate(handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
}

mcspStatus_t mcspCuinSgemvi_bufferSize(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                     int *buffer_size) {
    return mcspGemviBufferSizeImpl<mcspInt, float>(handle, trans, m, n, nnz, buffer_size);
}

mcspStatus_t mcspCuinDgemvi_bufferSize(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                     int *buffer_size) {
    return mcspGemviBufferSizeImpl<mcspInt, double>(handle, trans, m, n, nnz, buffer_size);
}

mcspStatus_t mcspCuinCgemvi_bufferSize(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                     int *buffer_size) {
    return mcspGemviBufferSizeImpl<mcspInt, mcspComplexFloat>(handle, trans, m, n, nnz, buffer_size);
}

mcspStatus_t mcspCuinZgemvi_bufferSize(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                     int *buffer_size) {
    return mcspGemviBufferSizeImpl<mcspInt, mcspComplexDouble>(handle, trans, m, n, nnz, buffer_size);
}

mcspStatus_t mcspCuinSgemvi(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, const float *alpha,
                          const float *A, int lda, int nnz, const float *x_val, const int *x_ind, const float *beta,
                          float *y, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspGemviTemplate(handle, trans, (mcspInt)m, (mcspInt)n, alpha, A, (mcspInt)lda, (mcspInt)nnz, x_val,
                             (mcspInt *)x_ind, beta, y, idx_base, temp_buffer);
}

mcspStatus_t mcspCuinDgemvi(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, const double *alpha,
                          const double *A, int lda, int nnz, const double *x_val, const int *x_ind, const double *beta,
                          double *y, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspGemviTemplate(handle, trans, (mcspInt)m, (mcspInt)n, alpha, A, (mcspInt)lda, (mcspInt)nnz, x_val,
                             (mcspInt *)x_ind, beta, y, idx_base, temp_buffer);
}

mcspStatus_t mcspCuinCgemvi(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, const mcspComplexFloat *alpha,
                          const mcspComplexFloat *A, int lda, int nnz, const mcspComplexFloat *x_val, const int *x_ind,
                          const mcspComplexFloat *beta, mcspComplexFloat *y, mcsparseIndexBase_t idx_base,
                          void *temp_buffer) {
    return mcspGemviTemplate(handle, trans, (mcspInt)m, (mcspInt)n, alpha, A, (mcspInt)lda, (mcspInt)nnz, x_val,
                             (mcspInt *)x_ind, beta, y, idx_base, temp_buffer);
}

mcspStatus_t mcspCuinZgemvi(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, const mcspComplexDouble *alpha,
                          const mcspComplexDouble *A, int lda, int nnz, const mcspComplexDouble *x_val,
                          const int *x_ind, const mcspComplexDouble *beta, mcspComplexDouble *y,
                          mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspGemviTemplate(handle, trans, (mcspInt)m, (mcspInt)n, alpha, A, (mcspInt)lda, (mcspInt)nnz, x_val,
                             (mcspInt *)x_ind, beta, y, idx_base, temp_buffer);
}

#ifdef __cplusplus
}
#endif
