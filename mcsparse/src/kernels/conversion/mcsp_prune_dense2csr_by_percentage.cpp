#include "common/mcsp_types.h"
#include "csr2csr_compress_device.hpp"
#include "dense2csx_device.hpp"
#include "device_radix_sort.hpp"
#include "device_reduce.hpp"
#include "mcsp_config.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"
#include "nnz_device.hpp"
#include "prune_dense2csr_by_percentage_device.hpp"

template <typename idxType, typename valType, typename perType>
mcspStatus_t mcspPruneDense2CsrByPercentageBufferSizeTemplate(mcspHandle_t handle, idxType m, idxType n,
                                                              const valType *dense_matrix, idxType lda,
                                                              perType percentage, const mcspMatDescr_t mcsp_descr_A,
                                                              const valType *csr_vals, const idxType *csr_rows,
                                                              const idxType *csr_cols, const mcspPruneInfo_t info,
                                                              size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || lda < m || percentage < static_cast<perType>(0.0) ||
        percentage > static_cast<perType>(100.0)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (mcsp_descr_A == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m == 0 || n == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    if (dense_matrix == nullptr || csr_rows == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType temp_buffer_size = 0;
    valType *dense_output = nullptr;
    mcStream_t stream = mcspGetStreamInternal(handle);
    if constexpr (std::is_same_v<valType, float>) {
        mcprim::radix_sort_keys(nullptr, temp_buffer_size, (uint32_t *)dense_matrix, (uint32_t *)dense_output, m * n,
                                stream);
    } else if constexpr (std::is_same_v<valType, double>) {
        mcprim::radix_sort_keys(nullptr, temp_buffer_size, (uint64_t *)dense_matrix, (uint64_t *)dense_output, m * n,
                                stream);
    } else if constexpr (std::is_same_v<valType, __half>) {
        mcprim::radix_sort_keys(nullptr, temp_buffer_size, (uint16_t *)dense_matrix, (uint16_t *)dense_output, m * n,
                                stream);
    }
    *buffer_size = temp_buffer_size + 2 * m * n * sizeof(valType);
    mcprim::reduce(nullptr, temp_buffer_size, (idxType *)nullptr, (idxType *)nullptr, m, mcprim::plus<idxType>(),
                   stream);
    *buffer_size += temp_buffer_size;
    mcprim::inclusive_scan(nullptr, temp_buffer_size, (idxType *)nullptr, (idxType *)nullptr, m + 1, stream);
    *buffer_size += temp_buffer_size;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType, typename perType>
mcspStatus_t mcspPruneDense2CsrNnzByPercentageTemplate(mcspHandle_t handle, idxType m, idxType n,
                                                       const valType *dense_matrix, idxType lda, perType percentage,
                                                       const mcspMatDescr_t mcsp_descr_A, idxType *csr_rows,
                                                       idxType *nnz, const mcspPruneInfo_t info, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || lda < m || percentage < static_cast<perType>(0.0) ||
        percentage > static_cast<perType>(100.0)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || csr_rows == nullptr || nnz == nullptr ||
        temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    char *buffer_ptr = reinterpret_cast<char *>(temp_buffer);
    valType *output_matrix = reinterpret_cast<valType *>(buffer_ptr);
    buffer_ptr += (m * n) * sizeof(valType);
    valType *sorted_matrix = reinterpret_cast<valType *>(buffer_ptr);
    buffer_ptr += (m * n) * sizeof(valType);

    int nnz_A = m * n;
    int pos = std::ceil(nnz_A * (percentage / 100)) - 1;
    pos = std::min(pos, nnz_A - 1);
    pos = std::max(pos, 0);

    constexpr uint32_t block_size = 512;
    uint32_t grid_size = (m * n + block_size - 1) / block_size;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL(absDenseMatrixKernel<block_size>, dim3(grid_size), dim3(block_size), 0, stream, m, n, lda,
                       dense_matrix, output_matrix);

    idxType buffer_size = 0;
    if constexpr (std::is_same_v<valType, float>) {
        mcprim::radix_sort_keys(nullptr, buffer_size, (uint32_t *)output_matrix, (uint32_t *)sorted_matrix, m * n,
                                stream);
        mcprim::radix_sort_keys((void *)buffer_ptr, buffer_size, (uint32_t *)output_matrix, (uint32_t *)sorted_matrix,
                                m * n, stream);
    } else if constexpr (std::is_same_v<valType, double>) {
        mcprim::radix_sort_keys(nullptr, buffer_size, (uint64_t *)output_matrix, (uint64_t *)sorted_matrix, m * n,
                                stream);
        mcprim::radix_sort_keys((void *)buffer_ptr, buffer_size, (uint64_t *)output_matrix, (uint64_t *)sorted_matrix,
                                m * n, stream);
    } else if constexpr (std::is_same_v<valType, __half>) {
        mcprim::radix_sort_keys(nullptr, buffer_size, (uint16_t *)output_matrix, (uint16_t *)sorted_matrix, m * n,
                                stream);
        mcprim::radix_sort_keys((void *)buffer_ptr, buffer_size, (uint16_t *)output_matrix, (uint16_t *)sorted_matrix,
                                m * n, stream);
    }
    valType tol = 0;
    MACA_ASSERT(mcMemcpyAsync(&tol, &sorted_matrix[pos], sizeof(valType), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    buffer_ptr = reinterpret_cast<char *>(temp_buffer);
    idxType *nnz_per_row = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += m * sizeof(idxType);
    idxType *dnnz = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += 1 * sizeof(idxType);
    idxType *tmp_csr_row = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += (m + 1) * sizeof(idxType);

    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    grid_size = (m + segments_per_block - 1) / segments_per_block;

    mcLaunchKernelGGL((nnzKernelRow<block_size, segments_per_block, segment_size>), dim3(grid_size), dim3(block_size),
                       0, stream, m, n, lda, dense_matrix, nnz_per_row, tol);

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
    MACA_ASSERT(mcMemcpyAsync(temp_buffer, &tol, sizeof(valType), mcMemcpyHostToDevice, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType, typename perType>
mcspStatus_t mcspPruneDense2CsrByPercentageTemplate(mcspHandle_t handle, idxType m, idxType n,
                                                    const valType *dense_matrix, idxType lda, perType percentage,
                                                    const mcspMatDescr_t mcsp_descr_A, valType *csr_vals,
                                                    const idxType *csr_rows, idxType *csr_cols,
                                                    const mcspPruneInfo_t info, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || lda < m || percentage < static_cast<perType>(0.0) ||
        percentage > static_cast<perType>(100.0)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || dense_matrix == nullptr || csr_vals == nullptr || csr_rows == nullptr ||
        csr_cols == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    valType tol = 0;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemcpyAsync(&tol, temp_buffer, sizeof(valType), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    constexpr uint32_t block_size = 512;
    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    uint32_t grid_size = (m + segments_per_block - 1) / segments_per_block;

    mcLaunchKernelGGL((dense2csrKernel<block_size, segments_per_block, segment_size, WARP_SIZE>), dim3(grid_size),
                       dim3(block_size), 0, stream, m, n, lda, dense_matrix, mcsp_descr_A->base, csr_vals, csr_rows,
                       csr_cols, tol);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSpruneDense2CsrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n,
                                                       const float *dense_matrix, mcspInt lda, float percentage,
                                                       const mcspMatDescr_t mcsp_descr_A, const float *csr_vals,
                                                       const mcspInt *csr_rows, const mcspInt *csr_cols,
                                                       const mcspPruneInfo_t info, size_t *buffer_size) {
    return mcspPruneDense2CsrByPercentageBufferSizeTemplate(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                            csr_vals, csr_rows, csr_cols, info, buffer_size);
}

mcspStatus_t mcspDpruneDense2CsrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n,
                                                       const double *dense_matrix, mcspInt lda, double percentage,
                                                       const mcspMatDescr_t mcsp_descr_A, const double *csr_vals,
                                                       const mcspInt *csr_rows, const mcspInt *csr_cols,
                                                       const mcspPruneInfo_t info, size_t *buffer_size) {
    return mcspPruneDense2CsrByPercentageBufferSizeTemplate(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                            csr_vals, csr_rows, csr_cols, info, buffer_size);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneDense2CsrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n,
                                                       const __half *dense_matrix, mcspInt lda, float percentage,
                                                       const mcspMatDescr_t mcsp_descr_A, const __half *csr_vals,
                                                       const mcspInt *csr_rows, const mcspInt *csr_cols,
                                                       const mcspPruneInfo_t info, size_t *buffer_size) {
    return mcspPruneDense2CsrByPercentageBufferSizeTemplate(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                            csr_vals, csr_rows, csr_cols, info, buffer_size);
}
#endif

mcspStatus_t mcspSpruneDense2CsrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix,
                                                mcspInt lda, float percentage, const mcspMatDescr_t mcsp_descr_A,
                                                mcspInt *csr_rows, mcspInt *nnz, const mcspPruneInfo_t info,
                                                void *temp_buffer) {
    return mcspPruneDense2CsrNnzByPercentageTemplate(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                     csr_rows, nnz, info, temp_buffer);
}

mcspStatus_t mcspDpruneDense2CsrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix,
                                                mcspInt lda, double percentage, const mcspMatDescr_t mcsp_descr_A,
                                                mcspInt *csr_rows, mcspInt *nnz, const mcspPruneInfo_t info,
                                                void *temp_buffer) {
    return mcspPruneDense2CsrNnzByPercentageTemplate(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                     csr_rows, nnz, info, temp_buffer);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneDense2CsrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix,
                                                mcspInt lda, float percentage, const mcspMatDescr_t mcsp_descr_A,
                                                mcspInt *csr_rows, mcspInt *nnz, const mcspPruneInfo_t info,
                                                void *temp_buffer) {
    return mcspPruneDense2CsrNnzByPercentageTemplate(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                     csr_rows, nnz, info, temp_buffer);
}
#endif

mcspStatus_t mcspSpruneDense2CsrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix,
                                             mcspInt lda, float percentage, const mcspMatDescr_t mcsp_descr_A,
                                             float *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                                             const mcspPruneInfo_t info, void *temp_buffer) {
    return mcspPruneDense2CsrByPercentageTemplate(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A, csr_vals,
                                                  csr_rows, csr_cols, info, temp_buffer);
}

mcspStatus_t mcspDpruneDense2CsrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix,
                                             mcspInt lda, double percentage, const mcspMatDescr_t mcsp_descr_A,
                                             double *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                                             const mcspPruneInfo_t info, void *temp_buffer) {
    return mcspPruneDense2CsrByPercentageTemplate(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A, csr_vals,
                                                  csr_rows, csr_cols, info, temp_buffer);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneDense2CsrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix,
                                             mcspInt lda, float percentage, const mcspMatDescr_t mcsp_descr_A,
                                             __half *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                                             const mcspPruneInfo_t info, void *temp_buffer) {
    return mcspPruneDense2CsrByPercentageTemplate(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A, csr_vals,
                                                  csr_rows, csr_cols, info, temp_buffer);
}
#endif

mcspStatus_t mcspCuinSpruneDense2csrByPercentage_bufferSizeExt(mcspHandle_t handle, int m, int n,
                                                             const float *dense_matrix, int lda, float percentage,
                                                             const mcspMatDescr_t mcsp_descr_A, const float *csr_vals,
                                                             const int *csr_rows, const int *csr_cols,
                                                             const mcspPruneInfo_t info, size_t *buffer_size) {
    return mcspPruneDense2CsrByPercentageBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda,
                                                            percentage, mcsp_descr_A, csr_vals, (mcspInt *)csr_rows,
                                                            (mcspInt *)csr_cols, info, buffer_size);
}

mcspStatus_t mcspCuinDpruneDense2csrByPercentage_bufferSizeExt(mcspHandle_t handle, int m, int n,
                                                             const double *dense_matrix, int lda, double percentage,
                                                             const mcspMatDescr_t mcsp_descr_A, const double *csr_vals,
                                                             const int *csr_rows, const int *csr_cols,
                                                             const mcspPruneInfo_t info, size_t *buffer_size) {
    return mcspPruneDense2CsrByPercentageBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda,
                                                            percentage, mcsp_descr_A, csr_vals, (mcspInt *)csr_rows,
                                                            (mcspInt *)csr_cols, info, buffer_size);
}

mcspStatus_t mcspCuinSpruneDense2csrNnzByPercentage(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                                  float percentage, const mcspMatDescr_t mcsp_descr_A, int *csr_rows,
                                                  int *nnz, const mcspPruneInfo_t info, void *temp_buffer) {
    return mcspPruneDense2CsrNnzByPercentageTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda,
                                                     percentage, mcsp_descr_A, (mcspInt *)csr_rows, (mcspInt *)nnz,
                                                     info, temp_buffer);
}

mcspStatus_t mcspCuinDpruneDense2csrNnzByPercentage(mcspHandle_t handle, int m, int n, const double *dense_matrix,
                                                  int lda, double percentage, const mcspMatDescr_t mcsp_descr_A,
                                                  int *csr_rows, int *nnz, const mcspPruneInfo_t info,
                                                  void *temp_buffer) {
    return mcspPruneDense2CsrNnzByPercentageTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda,
                                                     percentage, mcsp_descr_A, (mcspInt *)csr_rows, (mcspInt *)nnz,
                                                     info, temp_buffer);
}

mcspStatus_t mcspCuinSpruneDense2csrByPercentage(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                               float percentage, const mcspMatDescr_t mcsp_descr_A, float *csr_vals,
                                               const int *csr_rows, int *csr_cols, const mcspPruneInfo_t info,
                                               void *temp_buffer) {
    return mcspPruneDense2CsrByPercentageTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda,
                                                  percentage, mcsp_descr_A, csr_vals, (mcspInt *)csr_rows,
                                                  (mcspInt *)csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCuinDpruneDense2csrByPercentage(mcspHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                               double percentage, const mcspMatDescr_t mcsp_descr_A, double *csr_vals,
                                               const int *csr_rows, int *csr_cols, const mcspPruneInfo_t info,
                                               void *temp_buffer) {
    return mcspPruneDense2CsrByPercentageTemplate(handle, (mcspInt)m, (mcspInt)n, dense_matrix, (mcspInt)lda,
                                                  percentage, mcsp_descr_A, csr_vals, (mcspInt *)csr_rows,
                                                  (mcspInt *)csr_cols, info, temp_buffer);
}

#ifdef __cplusplus
}
#endif