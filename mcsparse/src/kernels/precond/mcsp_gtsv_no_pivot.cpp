#include "common/mcsp_types.h"
#include "gtsv_no_pivot_device.hpp"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvNopivotBufferSizeImpl(mcspHandle_t handle, idxType m, idxType n, const valType* dl,
                                           const valType* d, const valType* du, const valType* B, idxType ldb,
                                           size_t* buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 3 || n < 0 || ldb < m) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m == 0 || n == 0) {
        *buffer_size = 4;
        return MCSP_STATUS_SUCCESS;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || B == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (m <= 512) {
        *buffer_size = 5 * ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);
    } else {
        *buffer_size = 8 * ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvNopivotPcr(mcspHandle_t handle, idxType m, idxType n, const valType* dl, const valType* d,
                                const valType* du, valType* B, idxType ldb, void* temp_buffer) {
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    valType* temp_dl = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);

    valType* temp_d = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);

    valType* temp_du = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);

    valType* temp_B = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);

    valType* X = reinterpret_cast<valType*>(ptr);

    constexpr idxType block_size = 512;
    idxType blocks = n;
    idxType iter;
    idxType pow2m;
    mcStream_t stream = mcspGetStreamInternal(handle);
    if ((m & (m - 1)) == 0) {
        iter = static_cast<idxType>(log2(m)) - 1;
        pow2m = m;
    } else {
        iter = static_cast<idxType>(log2(m));
        pow2m = pow(2, iter + 1);
    }
    mcLaunchKernelGGL((mcspGtsvPcrKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, m, n, ldb, iter,
                       pow2m, dl, d, du, B, temp_dl, temp_d, temp_du, temp_B, X);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvNopivotPcrLarge(mcspHandle_t handle, idxType m, idxType n, const valType* dl, const valType* d,
                                     const valType* du, valType* B, idxType ldb, void* temp_buffer) {
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    valType* dl_from = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);
    valType* d_from = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);
    valType* du_from = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);
    valType* B_from = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);

    valType* dl_to = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);
    valType* d_to = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);
    valType* du_to = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(n * m * sizeof(valType), ALIGNED_SIZE);
    valType* B_to = reinterpret_cast<valType*>(ptr);
    valType* X = B_to;

    constexpr idxType block_size = 512;
    idxType blocks = (n * m) / block_size + 1;
    idxType iter;
    idxType pow2m;
    if ((m & (m - 1)) == 0) {
        iter = static_cast<idxType>(log2(m)) - 1;
        pow2m = m;
    } else {
        iter = static_cast<idxType>(log2(m));
        pow2m = pow(2, iter + 1);
    }
    mcStream_t stream = mcspGetStreamInternal(handle);

    mcLaunchKernelGGL((mcspGtsvPcrInitValKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, m, n, ldb, dl,
                       d, du, B, dl_from, d_from, du_from, B_from);
    MACA_ASSERT(mcStreamSynchronize(stream));
    idxType stride = 1;
    for (idxType i = 0; i < iter; i++) {
        mcLaunchKernelGGL((mcspGtsvPcrIterKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, m, n, ldb,
                           stride, dl_from, d_from, du_from, B_from, dl_to, d_to, du_to, B_to);
        stride *= 2;
        MACA_ASSERT(mcMemcpyAsync(dl_from, dl_to, 4 * ALIGN(n * m * sizeof(valType), ALIGNED_SIZE),
                                  mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }
    mcLaunchKernelGGL((mcspGtsvPcrGetXKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, m, n, ldb,
                       stride, pow2m, dl_from, d_from, du_from, B_from, X);
    MACA_ASSERT(mcStreamSynchronize(stream));
    mcLaunchKernelGGL((mcspGtsvPcrSetXKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, m, n, ldb, B, X);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvNopivotImpl(mcspHandle_t handle, idxType m, idxType n, const valType* dl, const valType* d,
                                 const valType* du, valType* B, idxType ldb, void* temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 3 || n < 0 || ldb < m) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || B == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m <= 512) {
        return mcspGtsvNopivotPcr(handle, m, n, dl, d, du, B, ldb, temp_buffer);
    } else {
        return mcspGtsvNopivotPcrLarge(handle, m, n, dl, d, du, B, ldb, temp_buffer);
    }
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsv_nopivot10xImpl(mcspHandle_t handle, idxType m, idxType n, const valType* dl, const valType* d,
                                     const valType* du, valType* B, idxType ldb) {
    size_t buffer_size = 0;
    mcspStatus_t ret = mcspGtsvNopivotBufferSizeImpl(handle, m, n, dl, d, du, B, ldb, &buffer_size);
    if (ret != MCSP_STATUS_SUCCESS) {
        return ret;
    }
    void* buffer;
    bool use_buffer_pool = handle->mcspUsePoolBuffer((void**)&buffer, buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc((void**)&buffer, buffer_size));
    }

    ret = mcspGtsvNopivotImpl(handle, m, n, dl, d, du, B, ldb, buffer);

    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(buffer));
    }

    return ret;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSgtsvNopivotBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const float* dl, const float* d,
                                        const float* du, const float* B, mcspInt ldb, size_t* buffer_size) {
    return mcspGtsvNopivotBufferSizeImpl(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

mcspStatus_t mcspDgtsvNopivotBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const double* dl, const double* d,
                                        const double* du, const double* B, mcspInt ldb, size_t* buffer_size) {
    return mcspGtsvNopivotBufferSizeImpl(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

mcspStatus_t mcspCgtsvNopivotBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* dl,
                                        const mcspComplexFloat* d, const mcspComplexFloat* du,
                                        const mcspComplexFloat* B, mcspInt ldb, size_t* buffer_size) {
    return mcspGtsvNopivotBufferSizeImpl(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

mcspStatus_t mcspZgtsvNopivotBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* dl,
                                        const mcspComplexDouble* d, const mcspComplexDouble* du,
                                        const mcspComplexDouble* B, mcspInt ldb, size_t* buffer_size) {
    return mcspGtsvNopivotBufferSizeImpl(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

mcspStatus_t mcspSgtsvNopivot(mcspHandle_t handle, mcspInt m, mcspInt n, const float* dl, const float* d,
                              const float* du, float* B, mcspInt ldb, void* temp_buffer) {
    return mcspGtsvNopivotImpl(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

mcspStatus_t mcspDgtsvNopivot(mcspHandle_t handle, mcspInt m, mcspInt n, const double* dl, const double* d,
                              const double* du, double* B, mcspInt ldb, void* temp_buffer) {
    return mcspGtsvNopivotImpl(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

mcspStatus_t mcspCgtsvNopivot(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* dl,
                              const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* B, mcspInt ldb,
                              void* temp_buffer) {
    return mcspGtsvNopivotImpl(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

mcspStatus_t mcspZgtsvNopivot(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* dl,
                              const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B,
                              mcspInt ldb, void* temp_buffer) {
    return mcspGtsvNopivotImpl(handle, m, n, dl, d, du, B, ldb, temp_buffer);
}

mcspStatus_t mcspCuinSgtsv2_nopivot_bufferSizeExt(mcspHandle_t handle, int m, int n, const float* dl, const float* d,
                                                const float* du, const float* B, int ldb, size_t* buffer_size) {
    return mcspGtsvNopivotBufferSizeImpl(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, buffer_size);
}

mcspStatus_t mcspCuinDgtsv2_nopivot_bufferSizeExt(mcspHandle_t handle, int m, int n, const double* dl, const double* d,
                                                const double* du, const double* B, int ldb, size_t* buffer_size) {
    return mcspGtsvNopivotBufferSizeImpl(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, buffer_size);
}

mcspStatus_t mcspCuinCgtsv2_nopivot_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcspComplexFloat* dl,
                                                const mcspComplexFloat* d, const mcspComplexFloat* du,
                                                const mcspComplexFloat* B, int ldb, size_t* buffer_size) {
    return mcspGtsvNopivotBufferSizeImpl(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, buffer_size);
}

mcspStatus_t mcspCuinZgtsv2_nopivot_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcspComplexDouble* dl,
                                                const mcspComplexDouble* d, const mcspComplexDouble* du,
                                                const mcspComplexDouble* B, int ldb, size_t* buffer_size) {
    return mcspGtsvNopivotBufferSizeImpl(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, buffer_size);
}

mcspStatus_t mcspCuinSgtsv2_nopivot(mcspHandle_t handle, int m, int n, const float* dl, const float* d, const float* du,
                                  float* B, int ldb, void* temp_buffer) {
    return mcspGtsvNopivotImpl(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, temp_buffer);
}

mcspStatus_t mcspCuinDgtsv2_nopivot(mcspHandle_t handle, int m, int n, const double* dl, const double* d,
                                  const double* du, double* B, int ldb, void* temp_buffer) {
    return mcspGtsvNopivotImpl(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, temp_buffer);
}

mcspStatus_t mcspCuinCgtsv2_nopivot(mcspHandle_t handle, int m, int n, const mcspComplexFloat* dl,
                                  const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* B, int ldb,
                                  void* temp_buffer) {
    return mcspGtsvNopivotImpl(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, temp_buffer);
}

mcspStatus_t mcspCuinZgtsv2_nopivot(mcspHandle_t handle, int m, int n, const mcspComplexDouble* dl,
                                  const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B,
                                  int ldb, void* temp_buffer) {
    return mcspGtsvNopivotImpl(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, temp_buffer);
}

mcspStatus_t mcspSgtsv_nopivot10x(mcspHandle_t handle, mcspInt m, mcspInt n, const float* dl, const float* d,
                                  const float* du, float* B, mcspInt ldb) {
    return mcspGtsv_nopivot10xImpl(handle, m, n, dl, d, du, B, ldb);
}

mcspStatus_t mcspDgtsv_nopivot10x(mcspHandle_t handle, mcspInt m, mcspInt n, const double* dl, const double* d,
                                  const double* du, double* B, mcspInt ldb) {
    return mcspGtsv_nopivot10xImpl(handle, m, n, dl, d, du, B, ldb);
}

mcspStatus_t mcspCgtsv_nopivot10x(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* dl,
                                  const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* B,
                                  mcspInt ldb) {
    return mcspGtsv_nopivot10xImpl(handle, m, n, dl, d, du, B, ldb);
}

mcspStatus_t mcspZgtsv_nopivot10x(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* dl,
                                  const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B,
                                  mcspInt ldb) {
    return mcspGtsv_nopivot10xImpl(handle, m, n, dl, d, du, B, ldb);
}
#ifdef __cplusplus
}
#endif