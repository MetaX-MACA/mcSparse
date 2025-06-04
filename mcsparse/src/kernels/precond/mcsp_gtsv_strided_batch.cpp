// refer to C. Klein, R. Strzodka Tridiagonal GPU Solver with Scaled
// Partial Pivoting at maximum Bandwith. ICPP2021

#include <cstddef>
#include <vector>

#include "common/mcsp_types.h"
#include "gtsv_device.hpp"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_runtime_wrapper.h"

// size of the partition of matrix
#define PARTITION_SIZE 32

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvStridedBatchBuffersizeTemplate(mcspHandle_t handle, idxType m, const valType *dl, const valType *d,
                                                    const valType *du, const valType *x, idxType batch_count,
                                                    idxType batch_stride, size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || x == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 3 || batch_count < 0 || batch_stride < std::max((idxType)1, m)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    // reduce the size of the tridiagonal system recursively, and store the reduced system in the buffer.
    *buffer_size = 0;
    for (; m > 2; m = m / PARTITION_SIZE * 2 + 2) {
        *buffer_size = *buffer_size + (m * (5 * batch_count) + 2 * batch_count) * sizeof(valType);
    }
    *buffer_size = ALIGN(*buffer_size, ALIGNED_SIZE);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
void mcspGtsvRecursive(mcspHandle_t handle, idxType m, idxType n, valType *dl, valType *d, valType *du, valType *B,
                       valType *x);

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvStridedBatchTemplate(mcspHandle_t handle, idxType m, const valType *dl, const valType *d,
                                          const valType *du, valType *x, idxType batch_count, idxType batch_stride,
                                          void *buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || x == nullptr || buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 3 || batch_count < 0 || batch_stride < std::max((idxType)1, m)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    valType *dl_buffer = reinterpret_cast<valType *>(buffer);
    valType *d_buffer = dl_buffer + m * batch_count;
    valType *du_buffer = d_buffer + m * batch_count;
    valType *B_buffer = du_buffer + m * batch_count;
    valType *x_buffer = B_buffer + m * batch_count + 1;
    mcStream_t stream = mcspGetStreamInternal(handle);
    for (int col = 0; col < batch_count; ++col) {
        MACA_ASSERT(mcMemcpyAsync(dl_buffer + col * m, dl + col * batch_stride, m * sizeof(valType),
                                  mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcMemcpyAsync(d_buffer + col * m, d + col * batch_stride, m * sizeof(valType),
                                  mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcMemcpyAsync(du_buffer + col * m, du + col * batch_stride, m * sizeof(valType),
                                  mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcMemcpyAsync(B_buffer + col * m, x + col * batch_stride, m * sizeof(valType),
                                  mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }
    mcspGtsvRecursive(handle, m, batch_count, dl_buffer, d_buffer, du_buffer, B_buffer, x_buffer);
    for (int col = 0; col < batch_count; ++col) {
        MACA_ASSERT(mcMemcpyAsync(x + col * batch_stride, x_buffer + col * (m + 2), m * sizeof(valType),
                                  mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvStridedBatch10xTemplate(mcspHandle_t handle, idxType m, const valType *dl, const valType *d,
                                             const valType *du, valType *x, idxType batch_count, idxType batch_stride) {
    size_t buffer_size = 0;
    mcspStatus_t ret =
        mcspGtsvStridedBatchBuffersizeTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, &buffer_size);
    if (ret != MCSP_STATUS_SUCCESS) {
        return ret;
    }
    void *buffer;
    bool use_buffer_pool = handle->mcspUsePoolBuffer((void **)&buffer, buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc((void **)&buffer, buffer_size));
    }

    ret = mcspGtsvStridedBatchTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, buffer);

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

mcspStatus_t mcspSgtsvStridedBatchBuffersize(mcspHandle_t handle, mcspInt m, const float *dl, const float *d,
                                             const float *du, const float *x, mcspInt batch_count, mcspInt batch_stride,
                                             size_t *buffer_size) {
    return mcspGtsvStridedBatchBuffersizeTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size);
}

mcspStatus_t mcspDgtsvStridedBatchBuffersize(mcspHandle_t handle, mcspInt m, const double *dl, const double *d,
                                             const double *du, const double *x, mcspInt batch_count,
                                             mcspInt batch_stride, size_t *buffer_size) {
    return mcspGtsvStridedBatchBuffersizeTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size);
}

mcspStatus_t mcspCgtsvStridedBatchBuffersize(mcspHandle_t handle, mcspInt m, const mcspComplexFloat *dl,
                                             const mcspComplexFloat *d, const mcspComplexFloat *du,
                                             const mcspComplexFloat *x, mcspInt batch_count, mcspInt batch_stride,
                                             size_t *buffer_size) {
    return mcspGtsvStridedBatchBuffersizeTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size);
}

mcspStatus_t mcspZgtsvStridedBatchBuffersize(mcspHandle_t handle, mcspInt m, const mcspComplexDouble *dl,
                                             const mcspComplexDouble *d, const mcspComplexDouble *du,
                                             const mcspComplexDouble *x, mcspInt batch_count, mcspInt batch_stride,
                                             size_t *buffer_size) {
    return mcspGtsvStridedBatchBuffersizeTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size);
}

mcspStatus_t mcspSgtsvStridedBatch(mcspHandle_t handle, mcspInt m, const float *dl, const float *d, const float *du,
                                   float *x, mcspInt batch_count, mcspInt batch_stride, void *buffer) {
    return mcspGtsvStridedBatchTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, buffer);
}

mcspStatus_t mcspDgtsvStridedBatch(mcspHandle_t handle, mcspInt m, const double *dl, const double *d, const double *du,
                                   double *x, mcspInt batch_count, mcspInt batch_stride, void *buffer) {
    return mcspGtsvStridedBatchTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, buffer);
}

mcspStatus_t mcspCgtsvStridedBatch(mcspHandle_t handle, mcspInt m, const mcspComplexFloat *dl,
                                   const mcspComplexFloat *d, const mcspComplexFloat *du, mcspComplexFloat *x,
                                   mcspInt batch_count, mcspInt batch_stride, void *buffer) {
    return mcspGtsvStridedBatchTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, buffer);
}

mcspStatus_t mcspZgtsvStridedBatch(mcspHandle_t handle, mcspInt m, const mcspComplexDouble *dl,
                                   const mcspComplexDouble *d, const mcspComplexDouble *du, mcspComplexDouble *x,
                                   mcspInt batch_count, mcspInt batch_stride, void *buffer) {
    return mcspGtsvStridedBatchTemplate(handle, m, dl, d, du, x, batch_count, batch_stride, buffer);
}

mcspStatus_t mcspCuinSgtsv2StridedBatch_bufferSizeExt(mcspHandle_t handle, int m, const float *dl, const float *d,
                                                    const float *du, const float *x, int batchCount, int batchStride,
                                                    size_t *bufferSizeInBytes) {
    return mcspGtsvStridedBatchBuffersizeTemplate(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batchCount,
                                                  (mcspInt)batchStride, bufferSizeInBytes);
}

mcspStatus_t mcspCuinDgtsv2StridedBatch_bufferSizeExt(mcspHandle_t handle, int m, const double *dl, const double *d,
                                                    const double *du, const double *x, int batchCount, int batchStride,
                                                    size_t *bufferSizeInBytes) {
    return mcspGtsvStridedBatchBuffersizeTemplate(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batchCount,
                                                  (mcspInt)batchStride, bufferSizeInBytes);
}

mcspStatus_t mcspCuinCgtsv2StridedBatch_bufferSizeExt(mcspHandle_t handle, int m, const mcFloatComplex *dl,
                                                    const mcFloatComplex *d, const mcFloatComplex *du,
                                                    const mcFloatComplex *x, int batchCount, int batchStride,
                                                    size_t *bufferSizeInBytes) {
    return mcspGtsvStridedBatchBuffersizeTemplate(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batchCount,
                                                  (mcspInt)batchStride, bufferSizeInBytes);
}

mcspStatus_t mcspCuinZgtsv2StridedBatch_bufferSizeExt(mcspHandle_t handle, int m, const mcDoubleComplex *dl,
                                                    const mcDoubleComplex *d, const mcDoubleComplex *du,
                                                    const mcDoubleComplex *x, int batchCount, int batchStride,
                                                    size_t *bufferSizeInBytes) {
    return mcspGtsvStridedBatchBuffersizeTemplate(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batchCount,
                                                  (mcspInt)batchStride, bufferSizeInBytes);
}

mcspStatus_t mcspCuinSgtsv2StridedBatch(mcspHandle_t handle, int m, const float *dl, const float *d, const float *du,
                                      float *x, int batchCount, int batchStride, void *pBuffer) {
    return mcspGtsvStridedBatchTemplate(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batchCount, (mcspInt)batchStride,
                                        pBuffer);
}

mcspStatus_t mcspCuinDgtsv2StridedBatch(mcspHandle_t handle, int m, const double *dl, const double *d, const double *du,
                                      double *x, int batchCount, int batchStride, void *pBuffer) {
    return mcspGtsvStridedBatchTemplate(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batchCount, (mcspInt)batchStride,
                                        pBuffer);
}

mcspStatus_t mcspCuinCgtsv2StridedBatch(mcspHandle_t handle, int m, const mcFloatComplex *dl, const mcFloatComplex *d,
                                      const mcFloatComplex *du, mcFloatComplex *x, int batchCount, int batchStride,
                                      void *pBuffer) {
    return mcspGtsvStridedBatchTemplate(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batchCount, (mcspInt)batchStride,
                                        pBuffer);
}

mcspStatus_t mcspCuinZgtsv2StridedBatch(mcspHandle_t handle, int m, const mcDoubleComplex *dl, const mcDoubleComplex *d,
                                      const mcDoubleComplex *du, mcDoubleComplex *x, int batchCount, int batchStride,
                                      void *pBuffer) {
    return mcspGtsvStridedBatchTemplate(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batchCount, (mcspInt)batchStride,
                                        pBuffer);
}

mcspStatus_t mcspSgtsvStridedBatch10x(mcspHandle_t handle, mcspInt m, const float *dl, const float *d, const float *du,
                                      float *x, mcspInt batch_count, mcspInt batch_stride) {
    return mcspGtsvStridedBatch10xTemplate(handle, m, dl, d, du, x, batch_count, batch_stride);
}

mcspStatus_t mcspDgtsvStridedBatch10x(mcspHandle_t handle, mcspInt m, const double *dl, const double *d,
                                      const double *du, double *x, mcspInt batch_count, mcspInt batch_stride) {
    return mcspGtsvStridedBatch10xTemplate(handle, m, dl, d, du, x, batch_count, batch_stride);
}

mcspStatus_t mcspCgtsvStridedBatch10x(mcspHandle_t handle, mcspInt m, const mcspComplexFloat *dl,
                                      const mcspComplexFloat *d, const mcspComplexFloat *du, mcspComplexFloat *x,
                                      mcspInt batch_count, mcspInt batch_stride) {
    return mcspGtsvStridedBatch10xTemplate(handle, m, dl, d, du, x, batch_count, batch_stride);
}

mcspStatus_t mcspZgtsvStridedBatch10x(mcspHandle_t handle, mcspInt m, const mcspComplexDouble *dl,
                                      const mcspComplexDouble *d, const mcspComplexDouble *du, mcspComplexDouble *x,
                                      mcspInt batch_count, mcspInt batch_stride) {
    return mcspGtsvStridedBatch10xTemplate(handle, m, dl, d, du, x, batch_count, batch_stride);
}

#ifdef __cplusplus
}
#endif