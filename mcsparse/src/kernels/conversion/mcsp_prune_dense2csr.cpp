#include "common/mcsp_types.h"
#include "csr2csr_compress_device.hpp"
#include "dense2csx_device.hpp"
#include "device_reduce.hpp"
#include "device_scan.hpp"
#include "mcsp_config.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"
#include "nnz_device.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspPruneDense2CsrBufferSizeTemplate(mcspHandle_t handle, idxType m, idxType n,
                                                  const valType *dense_matrix, idxType lda, const valType *threshold,
                                                  const mcspMatDescr_t mcsp_descr_A, const valType *csr_vals,
                                                  const idxType *csr_rows, const idxType *csr_cols,
                                                  size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || lda < m) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (mcsp_descr_A == nullptr || threshold == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m == 0 || n == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    if (dense_matrix == nullptr || csr_rows == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *buffer_size = 2 * (m + 1) * sizeof(idxType);

    idxType temp_buffer_size = 0;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::reduce(nullptr, temp_buffer_size, (idxType *)nullptr, (idxType *)nullptr, m, mcprim::plus<idxType>(),
                   stream);
    *buffer_size += temp_buffer_size;
    mcprim::inclusive_scan(nullptr, temp_buffer_size, (idxType *)nullptr, (idxType *)nullptr, m + 1, stream);
    *buffer_size += temp_buffer_size;

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspPruneDense2CsrNnzTemplate(mcspHandle_t handle, idxType m, idxType n, const valType *dense_matrix,
                                           idxType lda, const valType *threshold, const mcspMatDescr_t mcsp_descr_A,
                                           idxType *csr_rows, idxType *nnz, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || lda < m) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || csr_rows == nullptr || threshold == nullptr ||
        temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    valType h_threshold = getScalarToHost(threshold, handle->ptr_mode);
    if constexpr (std::is_same_v<valType, __half>) {
        float float_threshold = __half2float(h_threshold);
        if (float_threshold < 0.f) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    } else {
        if (h_threshold < static_cast<valType>(0)) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    char *buffer_ptr = reinterpret_cast<char *>(temp_buffer);

    idxType *nnz_per_row = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += m * sizeof(idxType);
    idxType *dnnz = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += 1 * sizeof(idxType);
    idxType *tmp_csr_row = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += (m + 1) * sizeof(idxType);

    constexpr uint32_t block_size = 512;
    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    uint32_t grid_size = (m + segments_per_block - 1) / segments_per_block;
    mcStream_t stream = mcspGetStreamInternal(handle);

    mcLaunchKernelGGL((nnzKernelRow<block_size, segments_per_block, segment_size>), dim3(grid_size), dim3(block_size),
                       0, stream, m, n, lda, dense_matrix, nnz_per_row, h_threshold);

    idxType buffer_size;
    mcprim::reduce(nullptr, buffer_size, nnz_per_row, dnnz, m, mcprim::plus<idxType>(), stream);
    mcprim::reduce((void *)buffer_ptr, buffer_size, nnz_per_row, dnnz, m, mcprim::plus<idxType>(), stream);
    if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
        MACA_ASSERT(mcMemcpyAsync(nnz, dnnz, sizeof(*dnnz), mcMemcpyDeviceToHost, stream));
    } else {
        MACA_ASSERT(mcMemcpyAsync(nnz, dnnz, sizeof(*dnnz), mcMemcpyDeviceToDevice, stream));
    }
    MACA_ASSERT(mcStreamSynchronize(stream));
    buffer_ptr += buffer_size;

    grid_size = (m + block_size - 1) / block_size;
    mcLaunchKernelGGL(FillRowDevice<block_size>, dim3(grid_size), dim3(block_size), 0, stream, m, mcsp_descr_A->base,
                       nnz_per_row, tmp_csr_row);

    mcprim::inclusive_scan(nullptr, buffer_size, tmp_csr_row, csr_rows, m + 1, stream);
    mcprim::inclusive_scan((void *)buffer_ptr, buffer_size, tmp_csr_row, csr_rows, m + 1, stream);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspPruneDense2CsrTemplate(mcspHandle_t handle, idxType m, idxType n, const valType *dense_matrix,
                                        idxType lda, const valType *threshold, const mcspMatDescr_t mcsp_descr_A,
                                        valType *csr_vals, const idxType *csr_rows, idxType *csr_cols,
                                        void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || lda < m) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || csr_vals == nullptr || csr_rows == nullptr ||
        csr_cols == nullptr || threshold == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    valType h_threshold = getScalarToHost(threshold, handle->ptr_mode);
    if constexpr (std::is_same_v<valType, __half>) {
        float float_threshold = __half2float(h_threshold);
        if (float_threshold < 0.f) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    } else {
        if (h_threshold < static_cast<valType>(0)) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    constexpr uint32_t block_size = 512;
    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    uint32_t grid_size = (m + segments_per_block - 1) / segments_per_block;
    mcStream_t stream = mcspGetStreamInternal(handle);

    mcLaunchKernelGGL((dense2csrKernel<block_size, segments_per_block, segment_size, WARP_SIZE>), dim3(grid_size),
                       dim3(block_size), 0, stream, m, n, lda, dense_matrix, mcsp_descr_A->base, csr_vals, csr_rows,
                       csr_cols, h_threshold);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSpruneDense2CsrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix,
                                           mcspInt lda, const float *threshold, const mcspMatDescr_t mcsp_descr_A,
                                           const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                           size_t *buffer_size) {
    return mcspPruneDense2CsrBufferSizeTemplate(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_vals,
                                                csr_rows, csr_cols, buffer_size);
}

mcspStatus_t mcspDpruneDense2CsrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix,
                                           mcspInt lda, const double *threshold, const mcspMatDescr_t mcsp_descr_A,
                                           const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                           size_t *buffer_size) {
    return mcspPruneDense2CsrBufferSizeTemplate(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_vals,
                                                csr_rows, csr_cols, buffer_size);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneDense2CsrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix,
                                           mcspInt lda, const __half *threshold, const mcspMatDescr_t mcsp_descr_A,
                                           const __half *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                           size_t *buffer_size) {
    return mcspPruneDense2CsrBufferSizeTemplate(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_vals,
                                                csr_rows, csr_cols, buffer_size);
}
#endif

mcspStatus_t mcspSpruneDense2CsrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix, mcspInt lda,
                                    const float *threshold, const mcspMatDescr_t mcsp_descr_A, mcspInt *csr_rows,
                                    mcspInt *nnz, void *temp_buffer) {
    return mcspPruneDense2CsrNnzTemplate(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_rows, nnz,
                                         temp_buffer);
}

mcspStatus_t mcspDpruneDense2CsrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix, mcspInt lda,
                                    const double *threshold, const mcspMatDescr_t mcsp_descr_A, mcspInt *csr_rows,
                                    mcspInt *nnz, void *temp_buffer) {
    return mcspPruneDense2CsrNnzTemplate(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_rows, nnz,
                                         temp_buffer);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneDense2CsrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix, mcspInt lda,
                                    const __half *threshold, const mcspMatDescr_t mcsp_descr_A, mcspInt *csr_rows,
                                    mcspInt *nnz, void *temp_buffer) {
    return mcspPruneDense2CsrNnzTemplate(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_rows, nnz,
                                         temp_buffer);
}
#endif

mcspStatus_t mcspSpruneDense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix, mcspInt lda,
                                 const float *threshold, const mcspMatDescr_t mcsp_descr_A, float *csr_vals,
                                 const mcspInt *csr_rows, mcspInt *csr_cols, void *temp_buffer) {
    return mcspPruneDense2CsrTemplate(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_vals, csr_rows,
                                      csr_cols, temp_buffer);
}

mcspStatus_t mcspDpruneDense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix, mcspInt lda,
                                 const double *threshold, const mcspMatDescr_t mcsp_descr_A, double *csr_vals,
                                 const mcspInt *csr_rows, mcspInt *csr_cols, void *temp_buffer) {
    return mcspPruneDense2CsrTemplate(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_vals, csr_rows,
                                      csr_cols, temp_buffer);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneDense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix, mcspInt lda,
                                 const __half *threshold, const mcspMatDescr_t mcsp_descr_A, __half *csr_vals,
                                 const mcspInt *csr_rows, mcspInt *csr_cols, void *temp_buffer) {
    return mcspPruneDense2CsrTemplate(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_vals, csr_rows,
                                      csr_cols, temp_buffer);
}
#endif

mcspStatus_t mcspCuinSpruneDense2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                                 const float *threshold, const mcspMatDescr_t mcsp_descr_A,
                                                 const float *csr_vals, const int *csr_rows, const int *csr_cols,
                                                 size_t *buffer_size) {
    return mcspPruneDense2CsrBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda, threshold,
                                                mcsp_descr_A, csr_vals, (mcspInt *)csr_rows, (mcspInt *)csr_cols,
                                                buffer_size);
}

mcspStatus_t mcspCuinDpruneDense2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                                 const double *threshold, const mcspMatDescr_t mcsp_descr_A,
                                                 const double *csr_vals, const int *csr_rows, const int *csr_cols,
                                                 size_t *buffer_size) {
    return mcspPruneDense2CsrBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda, threshold,
                                                mcsp_descr_A, csr_vals, (mcspInt *)csr_rows, (mcspInt *)csr_cols,
                                                buffer_size);
}

mcspStatus_t mcspCuinSpruneDense2csrNnz(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                      const float *threshold, const mcspMatDescr_t mcsp_descr_A, int *csr_rows,
                                      int *nnz, void *temp_buffer) {
    return mcspPruneDense2CsrNnzTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda, threshold,
                                         mcsp_descr_A, (mcspInt *)csr_rows, (mcspInt *)nnz, temp_buffer);
}

mcspStatus_t mcspCuinDpruneDense2csrNnz(mcspHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                      const double *threshold, const mcspMatDescr_t mcsp_descr_A, int *csr_rows,
                                      int *nnz, void *temp_buffer) {
    return mcspPruneDense2CsrNnzTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda, threshold,
                                         mcsp_descr_A, (mcspInt *)csr_rows, (mcspInt *)nnz, temp_buffer);
}

mcspStatus_t mcspCuinSpruneDense2csr(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                   const float *threshold, const mcspMatDescr_t mcsp_descr_A, float *csr_vals,
                                   const int *csr_rows, int *csr_cols, void *temp_buffer) {
    return mcspPruneDense2CsrTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda, threshold,
                                      mcsp_descr_A, csr_vals, (mcspInt *)csr_rows, (mcspInt *)csr_cols, temp_buffer);
}

mcspStatus_t mcspCuinDpruneDense2csr(mcspHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                   const double *threshold, const mcspMatDescr_t mcsp_descr_A, double *csr_vals,
                                   const int *csr_rows, int *csr_cols, void *temp_buffer) {
    return mcspPruneDense2CsrTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda, threshold,
                                      mcsp_descr_A, csr_vals, (mcspInt *)csr_rows, (mcspInt *)csr_cols, temp_buffer);
}
#ifdef __cplusplus
}
#endif