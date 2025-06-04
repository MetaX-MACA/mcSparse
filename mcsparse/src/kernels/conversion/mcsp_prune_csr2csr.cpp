#include "common/mcsp_types.h"
#include "csr2csr_compress_device.hpp"
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

template <typename idxType, typename valType>
mcspStatus_t mcspPruneCsr2csrBufferSizeTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz_A,
                                                const mcspMatDescr_t mcsp_descr_A, const valType *csr_val_A,
                                                const idxType *csr_row_A, const idxType *csr_col_A, const valType *tol,
                                                const mcspMatDescr_t mcsp_descr_C, const valType *csr_val_C,
                                                const idxType *csr_row_C, const idxType *csr_col_C,
                                                size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz_A < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }
    if (mcsp_descr_A == nullptr || tol == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (m == 0 || n == 0 || nnz_A == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    if (csr_val_A == nullptr || csr_row_A == nullptr || csr_col_A == nullptr || csr_row_C == nullptr) {
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
mcspStatus_t mcspPruneCsr2csrNnzTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz_A,
                                         const mcspMatDescr_t mcsp_descr_A, const valType *csr_val_A,
                                         const idxType *csr_row_A, const idxType *csr_col_A, const valType *tol,
                                         const mcspMatDescr_t mcsp_descr_C, idxType *csr_row_C, idxType *nnz_C,
                                         void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz_A < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0 || nnz_A == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || csr_val_A == nullptr || csr_row_A == nullptr || csr_col_A == nullptr ||
        tol == nullptr || mcsp_descr_C == nullptr || csr_row_C == nullptr || nnz_C == nullptr ||
        temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    valType h_tol = getScalarToHost(tol, handle->ptr_mode);
    if constexpr (std::is_same_v<valType, __half>) {
        float float_threshold = __half2float(h_tol);
        if (float_threshold < 0.f) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    } else {
        if (h_tol < static_cast<valType>(0)) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    char *buffer_ptr = reinterpret_cast<char *>(temp_buffer);
    idxType *nnz_per_row = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += m * sizeof(idxType);
    idxType *dnnz_C = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += 1 * sizeof(idxType);
    idxType *tmp_csr_row = reinterpret_cast<idxType *>(buffer_ptr);
    buffer_ptr += (m + 1) * sizeof(idxType);

    constexpr uint32_t block_size = 512;
    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    uint32_t grid_size = (m + segments_per_block - 1) / segments_per_block;
    mcStream_t stream = mcspGetStreamInternal(handle);

    mcLaunchKernelGGL((nnzCompressKernel<block_size, segments_per_block, segment_size, WARP_SIZE>), dim3(grid_size),
                       dim3(block_size), 0, stream, m, mcsp_descr_A->base, csr_val_A, csr_row_A, nnz_per_row, h_tol);

    idxType buffer_size;
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

    mcprim::inclusive_scan(nullptr, buffer_size, tmp_csr_row, csr_row_C, m + 1, stream);
    mcprim::inclusive_scan((void *)buffer_ptr, buffer_size, tmp_csr_row, csr_row_C, m + 1, stream);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspPruneCsr2csrTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz_A,
                                      const mcspMatDescr_t mcsp_descr_A, const valType *csr_val_A,
                                      const idxType *csr_row_A, const idxType *csr_col_A, const valType *tol,
                                      const mcspMatDescr_t mcsp_descr_C, valType *csr_val_C, const idxType *csr_row_C,
                                      idxType *csr_col_C, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz_A < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0 || nnz_A == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (mcsp_descr_A == nullptr || csr_val_A == nullptr || csr_row_A == nullptr || csr_col_A == nullptr ||
        tol == nullptr || mcsp_descr_C == nullptr || csr_val_C == nullptr || csr_row_C == nullptr ||
        csr_col_C == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    valType h_tol = getScalarToHost(tol, handle->ptr_mode);
    if constexpr (std::is_same_v<valType, __half>) {
        float float_threshold = __half2float(h_tol);
        if (float_threshold < 0.f) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    } else {
        if (h_tol < static_cast<valType>(0)) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    constexpr uint32_t block_size = 512;
    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    uint32_t grid_size = (m + segments_per_block - 1) / segments_per_block;
    mcStream_t stream = mcspGetStreamInternal(handle);

    mcLaunchKernelGGL((csr2csrCompressKernel<block_size, segments_per_block, segment_size, WARP_SIZE>),
                       dim3(grid_size), dim3(block_size), 0, stream, m, n, mcsp_descr_A->base, csr_val_A, csr_row_A,
                       csr_col_A, nnz_A, mcsp_descr_C->base, csr_val_C, csr_row_C, csr_col_C, h_tol);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSpruneCsr2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                         const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A,
                                         const mcspInt *csr_row_A, const mcspInt *csr_col_A, const float *tol,
                                         const mcspMatDescr_t mcsp_descr_C, const float *csr_val_C,
                                         const mcspInt *csr_row_C, const mcspInt *csr_col_C, size_t *buffer_size) {
    return mcspPruneCsr2csrBufferSizeTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                              mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, buffer_size);
}

mcspStatus_t mcspDpruneCsr2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                         const mcspMatDescr_t mcsp_descr_A, const double *csr_val_A,
                                         const mcspInt *csr_row_A, const mcspInt *csr_col_A, const double *tol,
                                         const mcspMatDescr_t mcsp_descr_C, const double *csr_val_C,
                                         const mcspInt *csr_row_C, const mcspInt *csr_col_C, size_t *buffer_size) {
    return mcspPruneCsr2csrBufferSizeTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                              mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, buffer_size);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneCsr2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                         const mcspMatDescr_t mcsp_descr_A, const __half *csr_val_A,
                                         const mcspInt *csr_row_A, const mcspInt *csr_col_A, const __half *tol,
                                         const mcspMatDescr_t mcsp_descr_C, const __half *csr_val_C,
                                         const mcspInt *csr_row_C, const mcspInt *csr_col_C, size_t *buffer_size) {
    return mcspPruneCsr2csrBufferSizeTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                              mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, buffer_size);
}
#endif

mcspStatus_t mcspSpruneCsr2csrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                  const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A, const mcspInt *csr_row_A,
                                  const mcspInt *csr_col_A, const float *tol, const mcspMatDescr_t mcsp_descr_C,
                                  mcspInt *csr_row_C, mcspInt *nnz_C, void *temp_buffer) {
    return mcspPruneCsr2csrNnzTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                       mcsp_descr_C, csr_row_C, nnz_C, temp_buffer);
}

mcspStatus_t mcspDpruneCsr2csrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                  const mcspMatDescr_t mcsp_descr_A, const double *csr_val_A, const mcspInt *csr_row_A,
                                  const mcspInt *csr_col_A, const double *tol, const mcspMatDescr_t mcsp_descr_C,
                                  mcspInt *csr_row_C, mcspInt *nnz_C, void *temp_buffer) {
    return mcspPruneCsr2csrNnzTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                       mcsp_descr_C, csr_row_C, nnz_C, temp_buffer);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneCsr2csrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                  const mcspMatDescr_t mcsp_descr_A, const __half *csr_val_A, const mcspInt *csr_row_A,
                                  const mcspInt *csr_col_A, const __half *tol, const mcspMatDescr_t mcsp_descr_C,
                                  mcspInt *csr_row_C, mcspInt *nnz_C, void *temp_buffer) {
    return mcspPruneCsr2csrNnzTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                       mcsp_descr_C, csr_row_C, nnz_C, temp_buffer);
}
#endif

mcspStatus_t mcspSpruneCsr2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                               const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A, const mcspInt *csr_row_A,
                               const mcspInt *csr_col_A, const float *tol, const mcspMatDescr_t mcsp_descr_C,
                               float *csr_val_C, const mcspInt *csr_row_C, mcspInt *csr_col_C, void *temp_buffer) {
    return mcspPruneCsr2csrTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                    mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, temp_buffer);
}

mcspStatus_t mcspDpruneCsr2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                               const mcspMatDescr_t mcsp_descr_A, const double *csr_val_A, const mcspInt *csr_row_A,
                               const mcspInt *csr_col_A, const double *tol, const mcspMatDescr_t mcsp_descr_C,
                               double *csr_val_C, const mcspInt *csr_row_C, mcspInt *csr_col_C, void *temp_buffer) {
    return mcspPruneCsr2csrTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                    mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, temp_buffer);
}

#if defined(__MACA__)
mcspStatus_t mcspHpruneCsr2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                               const mcspMatDescr_t mcsp_descr_A, const __half *csr_val_A, const mcspInt *csr_row_A,
                               const mcspInt *csr_col_A, const __half *tol, const mcspMatDescr_t mcsp_descr_C,
                               __half *csr_val_C, const mcspInt *csr_row_C, mcspInt *csr_col_C, void *temp_buffer) {
    return mcspPruneCsr2csrTemplate(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                    mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, temp_buffer);
}
#endif

mcspStatus_t mcspCuinSpruneCsr2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz_A,
                                               const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A,
                                               const int *csr_row_A, const int *csr_col_A, const float *tol,
                                               const mcspMatDescr_t mcsp_descr_C, const float *csr_val_C,
                                               const int *csr_row_C, const int *csr_col_C, size_t *buffer_size) {
    return mcspPruneCsr2csrBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A, csr_val_A,
                                              (mcspInt *)csr_row_A, (mcspInt *)csr_col_A, tol, mcsp_descr_C, csr_val_C,
                                              (mcspInt *)csr_row_C, (mcspInt *)csr_col_C, buffer_size);
}

mcspStatus_t mcspCuinDpruneCsr2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz_A,
                                               const mcspMatDescr_t mcsp_descr_A, const double *csr_val_A,
                                               const int *csr_row_A, const int *csr_col_A, const double *tol,
                                               const mcspMatDescr_t mcsp_descr_C, const double *csr_val_C,
                                               const int *csr_row_C, const int *csr_col_C, size_t *buffer_size) {
    return mcspPruneCsr2csrBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A, csr_val_A,
                                              (mcspInt *)csr_row_A, (mcspInt *)csr_col_A, tol, mcsp_descr_C, csr_val_C,
                                              (mcspInt *)csr_row_C, (mcspInt *)csr_col_C, buffer_size);
}

mcspStatus_t mcspCuinSpruneCsr2csrNnz(mcspHandle_t handle, int m, int n, int nnz_A, const mcspMatDescr_t mcsp_descr_A,
                                    const float *csr_val_A, const int *csr_row_A, const int *csr_col_A,
                                    const float *tol, const mcspMatDescr_t mcsp_descr_C, int *csr_row_C, int *nnz_C,
                                    void *temp_buffer) {
    return mcspPruneCsr2csrNnzTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A, csr_val_A,
                                       (mcspInt *)csr_row_A, (mcspInt *)csr_col_A, tol, mcsp_descr_C,
                                       (mcspInt *)csr_row_C, (mcspInt *)nnz_C, temp_buffer);
}

mcspStatus_t mcspCuinDpruneCsr2csrNnz(mcspHandle_t handle, int m, int n, int nnz_A, const mcspMatDescr_t mcsp_descr_A,
                                    const double *csr_val_A, const int *csr_row_A, const int *csr_col_A,
                                    const double *tol, const mcspMatDescr_t mcsp_descr_C, int *csr_row_C, int *nnz_C,
                                    void *temp_buffer) {
    return mcspPruneCsr2csrNnzTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A, csr_val_A,
                                       (mcspInt *)csr_row_A, (mcspInt *)csr_col_A, tol, mcsp_descr_C,
                                       (mcspInt *)csr_row_C, (mcspInt *)nnz_C, temp_buffer);
}

mcspStatus_t mcspCuinSpruneCsr2csr(mcspHandle_t handle, int m, int n, int nnz_A, const mcspMatDescr_t mcsp_descr_A,
                                 const float *csr_val_A, const int *csr_row_A, const int *csr_col_A, const float *tol,
                                 const mcspMatDescr_t mcsp_descr_C, float *csr_val_C, const int *csr_row_C,
                                 int *csr_col_C, void *temp_buffer) {
    return mcspPruneCsr2csrTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A, csr_val_A,
                                    (mcspInt *)csr_row_A, (mcspInt *)csr_col_A, tol, mcsp_descr_C, csr_val_C,
                                    (mcspInt *)csr_row_C, (mcspInt *)csr_col_C, temp_buffer);
}

mcspStatus_t mcspCuinDpruneCsr2csr(mcspHandle_t handle, int m, int n, int nnz_A, const mcspMatDescr_t mcsp_descr_A,
                                 const double *csr_val_A, const int *csr_row_A, const int *csr_col_A, const double *tol,
                                 const mcspMatDescr_t mcsp_descr_C, double *csr_val_C, const int *csr_row_C,
                                 int *csr_col_C, void *temp_buffer) {
    return mcspPruneCsr2csrTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz_A, mcsp_descr_A, csr_val_A,
                                    (mcspInt *)csr_row_A, (mcspInt *)csr_col_A, tol, mcsp_descr_C, csr_val_C,
                                    (mcspInt *)csr_row_C, (mcspInt *)csr_col_C, temp_buffer);
}

#ifdef __cplusplus
}
#endif