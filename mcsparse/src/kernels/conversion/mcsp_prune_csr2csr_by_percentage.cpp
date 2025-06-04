#include "common/mcsp_types.h"
#include "csr2csr_compress_device.hpp"
#include "device_radix_sort.hpp"
#include "device_reduce.hpp"
#include "device_scan.hpp"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"
#include "nnz_compress_device.hpp"
#include "prune_csr2csr_by_percentage_device.hpp"

template <typename idxType, typename valType, typename perType>
mcspStatus_t mcspPruneCsr2csrByPercentageBufferSizeTemplate(
    mcspHandle_t handle, idxType m, idxType n, idxType nnz_A, const mcspMatDescr_t mcsp_descr_A,
    const valType *csr_vals_A, const idxType *csr_rows_A, const idxType *csr_cols_A, perType percentage,
    const mcspMatDescr_t mcsp_descr_C, const valType *csr_vals_C, const idxType *csr_rows_C, const idxType *csr_cols_C,
    const mcspPruneInfo_t info, size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz_A < 0 || percentage < static_cast<perType>(0.0) ||
        percentage > static_cast<perType>(100.0)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (mcsp_descr_A == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m == 0 || n == 0 || nnz_A == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    if (csr_vals_A == nullptr || csr_rows_A == nullptr || csr_cols_A == nullptr || csr_rows_C == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType temp_buffer_size = 0;
    valType *csr_val_output = nullptr;
    mcStream_t stream = mcspGetStreamInternal(handle);
    if constexpr (std::is_same_v<valType, float>) {
        mcprim::radix_sort_keys(nullptr, temp_buffer_size, (uint32_t *)csr_vals_A, (uint32_t *)csr_val_output, nnz_A,
                                stream);
    } else if constexpr (std::is_same_v<valType, double>) {
        mcprim::radix_sort_keys(nullptr, temp_buffer_size, (uint64_t *)csr_vals_A, (uint64_t *)csr_val_output, nnz_A,
                                stream);
    } else if constexpr (std::is_same_v<valType, __half>) {
        mcprim::radix_sort_keys(nullptr, temp_buffer_size, (uint16_t *)csr_vals_A, (uint16_t *)csr_val_output, nnz_A,
                                stream);
    }
    *buffer_size = temp_buffer_size + 2 * nnz_A * sizeof(valType) + 2 * (m + 1) * sizeof(idxType);
    mcprim::reduce(nullptr, temp_buffer_size, (idxType *)nullptr, (idxType *)nullptr, m, mcprim::plus<idxType>(),
                   stream);
    *buffer_size += temp_buffer_size;
    mcprim::inclusive_scan(nullptr, temp_buffer_size, (idxType *)nullptr, (idxType *)nullptr, m + 1, stream);
    *buffer_size += temp_buffer_size;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType, typename perType>
mcspStatus_t mcspPruneCsr2csrNnzByPercentageTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz_A,
                                                     const mcspMatDescr_t mcsp_descr_A, const valType *csr_vals_A,
                                                     const idxType *csr_rows_A, const idxType *csr_cols_A,
                                                     perType percentage, const mcspMatDescr_t mcsp_descr_C,
                                                     idxType *csr_rows_C, idxType *nnz_C, const mcspPruneInfo_t info,
                                                     void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz_A < 0 || percentage < static_cast<perType>(0.0) ||
        percentage > static_cast<perType>(100.0)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0 || nnz_A == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || csr_vals_A == nullptr || csr_rows_A == nullptr || csr_cols_A == nullptr ||
        mcsp_descr_C == nullptr || csr_rows_C == nullptr || nnz_C == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    char *buffer_ptr = reinterpret_cast<char *>(temp_buffer);
    valType *abs_matrix = reinterpret_cast<valType *>(buffer_ptr);
    buffer_ptr += nnz_A * sizeof(valType);
    valType *sorted_matrix = reinterpret_cast<valType *>(buffer_ptr);
    buffer_ptr += nnz_A * sizeof(valType);

    int total_num = nnz_A;
    int pos = std::ceil(total_num * (percentage / 100)) - 1;
    pos = std::min(pos, total_num - 1);
    pos = std::max(pos, 0);

    constexpr uint32_t block_size = 512;
    uint32_t grid_size = (nnz_A + block_size - 1) / block_size;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL(absCsrValueKernel<block_size>, dim3(grid_size), dim3(block_size), 0, stream, nnz_A, csr_vals_A,
                       abs_matrix);

    idxType buffer_size = 0;
    if constexpr (std::is_same_v<valType, float>) {
        mcprim::radix_sort_keys(nullptr, buffer_size, (uint32_t *)abs_matrix, (uint32_t *)sorted_matrix, nnz_A, stream);
        mcprim::radix_sort_keys((void *)buffer_ptr, buffer_size, (uint32_t *)abs_matrix, (uint32_t *)sorted_matrix,
                                nnz_A, stream);
    } else if constexpr (std::is_same_v<valType, double>) {
        mcprim::radix_sort_keys(nullptr, buffer_size, (uint64_t *)abs_matrix, (uint64_t *)sorted_matrix, nnz_A, stream);
        mcprim::radix_sort_keys((void *)buffer_ptr, buffer_size, (uint64_t *)abs_matrix, (uint64_t *)sorted_matrix,
                                nnz_A, stream);
    } else if constexpr (std::is_same_v<valType, __half>) {
        mcprim::radix_sort_keys(nullptr, buffer_size, (uint16_t *)abs_matrix, (uint16_t *)sorted_matrix, nnz_A, stream);
        mcprim::radix_sort_keys((void *)buffer_ptr, buffer_size, (uint16_t *)abs_matrix, (uint16_t *)sorted_matrix,
                                nnz_A, stream);
    }
    valType tol = 0;
    MACA_ASSERT(mcMemcpyAsync(&tol, &sorted_matrix[pos], sizeof(valType), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    buffer_ptr = reinterpret_cast<char *>(temp_buffer);
    idxType *nnz_per_row = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += m * sizeof(idxType);
    idxType *dnnz_C = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += 1 * sizeof(idxType);
    idxType *tmp_csr_row = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += (m + 1) * sizeof(idxType);

    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    grid_size = (m + segments_per_block - 1) / segments_per_block;

    mcLaunchKernelGGL((nnzCompressKernel<block_size, segments_per_block, segment_size, WARP_SIZE>), dim3(grid_size),
                       dim3(block_size), 0, stream, m, mcsp_descr_A->base, csr_vals_A, csr_rows_A, nnz_per_row, tol);

    mcprim::reduce(nullptr, buffer_size, nnz_per_row, dnnz_C, m, mcprim::plus<idxType>(), stream);
    mcprim::reduce((void *)buffer_ptr, buffer_size, nnz_per_row, dnnz_C, m, mcprim::plus<idxType>(), stream);
    if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
        MACA_ASSERT(mcMemcpyAsync(nnz_C, dnnz_C, sizeof(*dnnz_C), mcMemcpyDeviceToHost, stream));
    } else {
        MACA_ASSERT(mcMemcpyAsync(nnz_C, dnnz_C, sizeof(*dnnz_C), mcMemcpyDeviceToDevice, stream));
    }
    MACA_ASSERT(mcStreamSynchronize(stream));
    buffer_ptr += buffer_size;

    grid_size = (m + block_size - 1) / block_size;
    mcLaunchKernelGGL(FillRowDevice<block_size>, dim3(grid_size), dim3(block_size), 0, stream, m, mcsp_descr_A->base,
                       nnz_per_row, tmp_csr_row);

    mcprim::inclusive_scan(nullptr, buffer_size, tmp_csr_row, csr_rows_C, m + 1, stream);
    mcprim::inclusive_scan((void *)buffer_ptr, buffer_size, tmp_csr_row, csr_rows_C, m + 1, stream);
    MACA_ASSERT(mcMemcpyAsync(temp_buffer, &tol, sizeof(valType), mcMemcpyHostToDevice, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType, typename perType>
mcspStatus_t mcspPruneCsr2csrByPercentageTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz_A,
                                                  const mcspMatDescr_t mcsp_descr_A, const valType *csr_vals_A,
                                                  const idxType *csr_rows_A, const idxType *csr_cols_A,
                                                  perType percentage, const mcspMatDescr_t mcsp_descr_C,
                                                  valType *csr_vals_C, const idxType *csr_rows_C, idxType *csr_cols_C,
                                                  const mcspPruneInfo_t info, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz_A < 0 || percentage < static_cast<perType>(0.0) ||
        percentage > static_cast<perType>(100.0)) {
        return MCSP_STATUS_INVALID_SIZE;
    }
    if (m == 0 || n == 0 || nnz_A == 0) {
        return MCSP_STATUS_SUCCESS;
    }
    if (mcsp_descr_A == nullptr || csr_vals_A == nullptr || csr_rows_A == nullptr || csr_cols_A == nullptr ||
        mcsp_descr_C == nullptr || csr_vals_C == nullptr || csr_rows_C == nullptr || csr_cols_C == nullptr ||
        temp_buffer == nullptr) {
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

    mcLaunchKernelGGL((csr2csrCompressKernel<block_size, segments_per_block, segment_size, WARP_SIZE>),
                       dim3(grid_size), dim3(block_size), 0, stream, m, n, mcsp_descr_A->base, csr_vals_A, csr_rows_A,
                       csr_cols_A, nnz_A, mcsp_descr_C->base, csr_vals_C, csr_rows_C, csr_cols_C, tol);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSpruneCsr2csrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                                     const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                                     const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                                                     float percentage, const mcspMatDescr_t mcsp_descr_C,
                                                     const float *csr_vals_C, const mcspInt *csr_rows_C,
                                                     const mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                                     size_t *buffer_size) {
    return mcspPruneCsr2csrByPercentageBufferSizeTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A,
                                                          csr_cols_A, percentage, mcsp_descr_C, csr_vals_C, csr_rows_C,
                                                          csr_cols_C, info, buffer_size);
}

mcspStatus_t mcspDpruneCsr2csrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                                     const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                                     const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                                                     double percentage, const mcspMatDescr_t mcsp_descr_C,
                                                     const double *csr_vals_C, const mcspInt *csr_rows_C,
                                                     const mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                                     size_t *buffer_size) {
    return mcspPruneCsr2csrByPercentageBufferSizeTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A,
                                                          csr_cols_A, percentage, mcsp_descr_C, csr_vals_C, csr_rows_C,
                                                          csr_cols_C, info, buffer_size);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneCsr2csrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                                     const mcspMatDescr_t mcsp_descr_A, const __half *csr_vals_A,
                                                     const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                                                     float percentage, const mcspMatDescr_t mcsp_descr_C,
                                                     const __half *csr_vals_C, const mcspInt *csr_rows_C,
                                                     const mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                                     size_t *buffer_size) {
    return mcspPruneCsr2csrByPercentageBufferSizeTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A,
                                                          csr_cols_A, percentage, mcsp_descr_C, csr_vals_C, csr_rows_C,
                                                          csr_cols_C, info, buffer_size);
}
#endif

mcspStatus_t mcspSpruneCsr2csrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                              const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                              const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, float percentage,
                                              const mcspMatDescr_t mcsp_descr_C, mcspInt *csr_rows_C, mcspInt *nnz_C,
                                              const mcspPruneInfo_t info, void *temp_buffer) {
    return mcspPruneCsr2csrNnzByPercentageTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A,
                                                   csr_cols_A, percentage, mcsp_descr_C, csr_rows_C, nnz_C, info,
                                                   temp_buffer);
}

mcspStatus_t mcspDpruneCsr2csrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                              const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                              const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, double percentage,
                                              const mcspMatDescr_t mcsp_descr_C, mcspInt *csr_rows_C, mcspInt *nnz_C,
                                              const mcspPruneInfo_t info, void *temp_buffer) {
    return mcspPruneCsr2csrNnzByPercentageTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A,
                                                   csr_cols_A, percentage, mcsp_descr_C, csr_rows_C, nnz_C, info,
                                                   temp_buffer);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneCsr2csrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                              const mcspMatDescr_t mcsp_descr_A, const __half *csr_vals_A,
                                              const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, float percentage,
                                              const mcspMatDescr_t mcsp_descr_C, mcspInt *csr_rows_C, mcspInt *nnz_C,
                                              const mcspPruneInfo_t info, void *temp_buffer) {
    return mcspPruneCsr2csrNnzByPercentageTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A,
                                                   csr_cols_A, percentage, mcsp_descr_C, csr_rows_C, nnz_C, info,
                                                   temp_buffer);
}
#endif

mcspStatus_t mcspSpruneCsr2csrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                           const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                           const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, float percentage,
                                           const mcspMatDescr_t mcsp_descr_C, float *csr_vals_C,
                                           const mcspInt *csr_rows_C, mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                           void *temp_buffer) {
    return mcspPruneCsr2csrByPercentageTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A, csr_cols_A,
                                                percentage, mcsp_descr_C, csr_vals_C, csr_rows_C, csr_cols_C, info,
                                                temp_buffer);
}

mcspStatus_t mcspDpruneCsr2csrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                           const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                           const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, double percentage,
                                           const mcspMatDescr_t mcsp_descr_C, double *csr_vals_C,
                                           const mcspInt *csr_rows_C, mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                           void *temp_buffer) {
    return mcspPruneCsr2csrByPercentageTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A, csr_cols_A,
                                                percentage, mcsp_descr_C, csr_vals_C, csr_rows_C, csr_cols_C, info,
                                                temp_buffer);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneCsr2csrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                           const mcspMatDescr_t mcsp_descr_A, const __half *csr_vals_A,
                                           const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, float percentage,
                                           const mcspMatDescr_t mcsp_descr_C, __half *csr_vals_C,
                                           const mcspInt *csr_rows_C, mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                           void *temp_buffer) {
    return mcspPruneCsr2csrByPercentageTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A, csr_cols_A,
                                                percentage, mcsp_descr_C, csr_vals_C, csr_rows_C, csr_cols_C, info,
                                                temp_buffer);
}
#endif

mcspStatus_t mcspCuinSpruneCsr2csrByPercentage_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz_A,
                                                           const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                                           const int *csr_rows_A, const int *csr_cols_A,
                                                           float percentage, const mcspMatDescr_t mcsp_descr_C,
                                                           const float *csr_vals_C, const int *csr_rows_C,
                                                           const int *csr_cols_C, const mcspPruneInfo_t info,
                                                           size_t *buffer_size) {
    return mcspPruneCsr2csrByPercentageBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A,
                                                          csr_vals_A, (mcspInt *)csr_rows_A, (mcspInt *)csr_cols_A,
                                                          percentage, mcsp_descr_C, csr_vals_C, (mcspInt *)csr_rows_C,
                                                          (mcspInt *)csr_cols_C, info, buffer_size);
}

mcspStatus_t mcspCuinDpruneCsr2csrByPercentage_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz_A,
                                                           const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                                           const int *csr_rows_A, const int *csr_cols_A,
                                                           double percentage, const mcspMatDescr_t mcsp_descr_C,
                                                           const double *csr_vals_C, const int *csr_rows_C,
                                                           const int *csr_cols_C, const mcspPruneInfo_t info,
                                                           size_t *buffer_size) {
    return mcspPruneCsr2csrByPercentageBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A,
                                                          csr_vals_A, (mcspInt *)csr_rows_A, (mcspInt *)csr_cols_A,
                                                          percentage, mcsp_descr_C, csr_vals_C, (mcspInt *)csr_rows_C,
                                                          (mcspInt *)csr_cols_C, info, buffer_size);
}

mcspStatus_t mcspCuinSpruneCsr2csrNnzByPercentage(mcspHandle_t handle, int m, int n, int nnz_A,
                                                const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                                const int *csr_rows_A, const int *csr_cols_A, float percentage,
                                                const mcspMatDescr_t mcsp_descr_C, int *csr_rows_C, int *nnz_C,
                                                const mcspPruneInfo_t info, void *temp_buffer) {
    return mcspPruneCsr2csrNnzByPercentageTemplate(
        handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A, csr_vals_A, (mcspInt *)csr_rows_A,
        (mcspInt *)csr_cols_A, percentage, mcsp_descr_C, (mcspInt *)csr_rows_C, (mcspInt *)nnz_C, info, temp_buffer);
}

mcspStatus_t mcspCuinDpruneCsr2csrNnzByPercentage(mcspHandle_t handle, int m, int n, int nnz_A,
                                                const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                                const int *csr_rows_A, const int *csr_cols_A, double percentage,
                                                const mcspMatDescr_t mcsp_descr_C, int *csr_rows_C, int *nnz_C,
                                                const mcspPruneInfo_t info, void *temp_buffer) {
    return mcspPruneCsr2csrNnzByPercentageTemplate(
        handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A, csr_vals_A, (mcspInt *)csr_rows_A,
        (mcspInt *)csr_cols_A, percentage, mcsp_descr_C, (mcspInt *)csr_rows_C, (mcspInt *)nnz_C, info, temp_buffer);
}

mcspStatus_t mcspCuinSpruneCsr2csrByPercentage(mcspHandle_t handle, int m, int n, int nnz_A,
                                             const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                             const int *csr_rows_A, const int *csr_cols_A, float percentage,
                                             const mcspMatDescr_t mcsp_descr_C, float *csr_vals_C,
                                             const int *csr_rows_C, int *csr_cols_C, const mcspPruneInfo_t info,
                                             void *temp_buffer) {
    return mcspPruneCsr2csrByPercentageTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A,
                                                csr_vals_A, (mcspInt *)csr_rows_A, (mcspInt *)csr_cols_A, percentage,
                                                mcsp_descr_C, csr_vals_C, (mcspInt *)csr_rows_C, (mcspInt *)csr_cols_C,
                                                info, temp_buffer);
}

mcspStatus_t mcspCuinDpruneCsr2csrByPercentage(mcspHandle_t handle, int m, int n, int nnz_A,
                                             const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                             const int *csr_rows_A, const int *csr_cols_A, double percentage,
                                             const mcspMatDescr_t mcsp_descr_C, double *csr_vals_C,
                                             const int *csr_rows_C, int *csr_cols_C, const mcspPruneInfo_t info,
                                             void *temp_buffer) {
    return mcspPruneCsr2csrByPercentageTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A,
                                                csr_vals_A, (mcspInt *)csr_rows_A, (mcspInt *)csr_cols_A, percentage,
                                                mcsp_descr_C, csr_vals_C, (mcspInt *)csr_rows_C, (mcspInt *)csr_cols_C,
                                                info, temp_buffer);
}
#ifdef __cplusplus
}
#endif
