#include "common/mcsp_types.h"
#include "gthr_device.hpp"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspCsru2csrBufferSizeTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz, valType* csr_vals,
                                            const idxType* csr_rows, idxType* csr_cols, mcspCsru2csrInfo_t info,
                                            size_t* buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (csr_rows == nullptr || csr_cols == nullptr || csr_vals == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (n == 0 || m == 0 || nnz == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    MACA_ASSERT(mcMalloc((void**)&info->perm, nnz * sizeof(idxType)));

    mcspCsrSortBufferSize(handle, m, n, nnz, csr_rows, csr_cols, buffer_size);
    *buffer_size = *buffer_size + nnz * (sizeof(valType) + sizeof(idxType));

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsru2csrTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz, mcspMatDescr_t mcsp_descr_A,
                                  valType* csr_vals, const idxType* csr_rows, idxType* csr_cols,
                                  mcspCsru2csrInfo_t info, void* temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (n == 0 || m == 0 || nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (csr_rows == nullptr || csr_cols == nullptr || csr_vals == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    valType* csr_vals_buffer = (valType*)temp_buffer;
    temp_buffer = (void*)(csr_vals_buffer + nnz);

    mcspCreateIdentityPermutation(handle, nnz, (mcspInt*)info->perm);
    mcspCsrSort(handle, m, n, nnz, mcsp_descr_A, csr_rows, csr_cols, (mcspInt*)info->perm, temp_buffer);
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemcpyAsync(csr_vals_buffer, csr_vals, nnz * sizeof(valType), mcMemcpyDeviceToDevice, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    if constexpr (std::is_same_v<valType, float>) {
        mcspSgthr(handle, nnz, csr_vals_buffer, csr_vals, (mcspInt*)info->perm, mcsp_descr_A->base);
    } else if constexpr (std::is_same_v<valType, double>) {
        mcspDgthr(handle, nnz, csr_vals_buffer, csr_vals, (mcspInt*)info->perm, mcsp_descr_A->base);
    } else if constexpr (std::is_same_v<valType, mcspComplexFloat>) {
        mcspCgthr(handle, nnz, csr_vals_buffer, csr_vals, (mcspInt*)info->perm, mcsp_descr_A->base);
    } else if constexpr (std::is_same_v<valType, mcspComplexDouble>) {
        mcspZgthr(handle, nnz, csr_vals_buffer, csr_vals, (mcspInt*)info->perm, mcsp_descr_A->base);
    } else {
        return MCSP_STATUS_TYPE_MISMATCH;
    }

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsru2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, float* csr_vals,
                                     const mcspInt* csr_rows, mcspInt* csr_cols, mcspCsru2csrInfo_t info,
                                     size_t* buffer_size) {
    return mcspCsru2csrBufferSizeTemplate(handle, m, n, nnz, csr_vals, csr_rows, csr_cols, info, buffer_size);
}

mcspStatus_t mcspDcsru2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, double* csr_vals,
                                     const mcspInt* csr_rows, mcspInt* csr_cols, mcspCsru2csrInfo_t info,
                                     size_t* buffer_size) {
    return mcspCsru2csrBufferSizeTemplate(handle, m, n, nnz, csr_vals, csr_rows, csr_cols, info, buffer_size);
}

mcspStatus_t mcspCcsru2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspComplexFloat* csr_vals,
                                     const mcspInt* csr_rows, mcspInt* csr_cols, mcspCsru2csrInfo_t info,
                                     size_t* buffer_size) {
    return mcspCsru2csrBufferSizeTemplate(handle, m, n, nnz, csr_vals, csr_rows, csr_cols, info, buffer_size);
}

mcspStatus_t mcspZcsru2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                     mcspComplexDouble* csr_vals, const mcspInt* csr_rows, mcspInt* csr_cols,
                                     mcspCsru2csrInfo_t info, size_t* buffer_size) {
    return mcspCsru2csrBufferSizeTemplate(handle, m, n, nnz, csr_vals, csr_rows, csr_cols, info, buffer_size);
}

mcspStatus_t mcspScsru2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           float* csr_vals, const mcspInt* csr_rows, mcspInt* csr_cols, mcspCsru2csrInfo_t info,
                           void* temp_buffer) {
    return mcspCsru2csrTemplate(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
}

mcspStatus_t mcspDcsru2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           double* csr_vals, const mcspInt* csr_rows, mcspInt* csr_cols, mcspCsru2csrInfo_t info,
                           void* temp_buffer) {
    return mcspCsru2csrTemplate(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCcsru2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           mcspComplexFloat* csr_vals, const mcspInt* csr_rows, mcspInt* csr_cols,
                           mcspCsru2csrInfo_t info, void* temp_buffer) {
    return mcspCsru2csrTemplate(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
}

mcspStatus_t mcspZcsru2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           mcspComplexDouble* csr_vals, const mcspInt* csr_rows, mcspInt* csr_cols,
                           mcspCsru2csrInfo_t info, void* temp_buffer) {
    return mcspCsru2csrTemplate(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCuinScsru2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, float* csr_vals,
                                           const int* csr_rows, int* csr_cols, mcspCsru2csrInfo_t info,
                                           size_t* buffer_size) {
    return mcspCsru2csrBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, csr_vals, (mcspInt*)csr_rows,
                                          (mcspInt*)csr_cols, info, buffer_size);
}

mcspStatus_t mcspCuinDcsru2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, double* csr_vals,
                                           const int* csr_rows, int* csr_cols, mcspCsru2csrInfo_t info,
                                           size_t* buffer_size) {
    return mcspCsru2csrBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, csr_vals, (mcspInt*)csr_rows,
                                          (mcspInt*)csr_cols, info, buffer_size);
}

mcspStatus_t mcspCuinCcsru2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, mcspComplexFloat* csr_vals,
                                           const int* csr_rows, int* csr_cols, mcspCsru2csrInfo_t info,
                                           size_t* buffer_size) {
    return mcspCsru2csrBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, csr_vals, (mcspInt*)csr_rows,
                                          (mcspInt*)csr_cols, info, buffer_size);
}

mcspStatus_t mcspCuinZcsru2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, mcspComplexDouble* csr_vals,
                                           const int* csr_rows, int* csr_cols, mcspCsru2csrInfo_t info,
                                           size_t* buffer_size) {
    return mcspCsru2csrBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, csr_vals, (mcspInt*)csr_rows,
                                          (mcspInt*)csr_cols, info, buffer_size);
}

mcspStatus_t mcspCuinScsru2csr(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             float* csr_vals, const int* csr_rows, int* csr_cols, mcspCsru2csrInfo_t info,
                             void* temp_buffer) {
    return mcspCsru2csrTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, mcsp_descr_A, csr_vals,
                                (mcspInt*)csr_rows, (mcspInt*)csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCuinDcsru2csr(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             double* csr_vals, const int* csr_rows, int* csr_cols, mcspCsru2csrInfo_t info,
                             void* temp_buffer) {
    return mcspCsru2csrTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, mcsp_descr_A, csr_vals,
                                (mcspInt*)csr_rows, (mcspInt*)csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCuinCcsru2csr(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             mcspComplexFloat* csr_vals, const int* csr_rows, int* csr_cols, mcspCsru2csrInfo_t info,
                             void* temp_buffer) {
    return mcspCsru2csrTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, mcsp_descr_A, csr_vals,
                                (mcspInt*)csr_rows, (mcspInt*)csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCuinZcsru2csr(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             mcspComplexDouble* csr_vals, const int* csr_rows, int* csr_cols, mcspCsru2csrInfo_t info,
                             void* temp_buffer) {
    return mcspCsru2csrTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, mcsp_descr_A, csr_vals,
                                (mcspInt*)csr_rows, (mcspInt*)csr_cols, info, temp_buffer);
}

#ifdef __cplusplus
}
#endif