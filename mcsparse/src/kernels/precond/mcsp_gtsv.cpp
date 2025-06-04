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
mcspStatus_t mcspGtsvBuffersizeTemplate(mcspHandle_t handle, idxType m, idxType n, const valType *dl, const valType *d,
                                        const valType *du, const valType *B, idxType ldb, size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || B == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 3 || n < 0 || ldb < std::max((idxType)1, m)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    // reduce the size of the tridiagonal system recursively, and store the reduced system in the buffer.
    *buffer_size = 0;
    for (; m > 2; m = m / PARTITION_SIZE * 2 + 2) {
        *buffer_size = *buffer_size + (m * (5 * n) + 2 * n) * sizeof(valType);
    }
    *buffer_size = ALIGN(*buffer_size, ALIGNED_SIZE);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
void mcspGtsvRecursive(mcspHandle_t handle, idxType m, idxType n, valType *dl, valType *d, valType *du, valType *B,
                       valType *x) {
    const idxType m_direct_solve = PARTITION_SIZE;
    mcStream_t stream = mcspGetStreamInternal(handle);

    if (m <= m_direct_solve) {
        mcLaunchKernelGGL((mcspGtsvBatchGivensQrKernel<PARTITION_SIZE>), dim3(1, n), dim3(1, 1), 0, stream, m, dl, d,
                           du, B);
        MACA_ASSERT(mcStreamSynchronize(stream));
        for (int col = 0; col < n; ++col) {
            MACA_ASSERT(
                mcMemcpyAsync(x + col * (m + 2), B + col * m, m * sizeof(valType), mcMemcpyDeviceToDevice, stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
        }
        return;
    }

    // an array with size of m + 2 stores x[-1] to x[m], where x[-1] and x[m] equal 0
    MACA_ASSERT(mcMemsetAsync(x - 1, 0, (m + 2) * n * sizeof(valType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    // number of full size partitions of matrix
    idxType PARTITION_NUM = (m - 1) / PARTITION_SIZE;

    idxType m_res = m - PARTITION_SIZE * PARTITION_NUM;
    idxType m_next = m_res == 1 ? PARTITION_NUM * 2 + 1 : (PARTITION_NUM + 1) * 2;
    valType *dl_next = dl + m * (5 * n) + 2 * n;
    valType *d_next = dl_next + m_next * n;
    valType *du_next = d_next + m_next * n;
    valType *B_next = du_next + m_next * n;
    // x[-1] to x[m]
    valType *x_next = B_next + m_next * n + 1;

    idxType THREAD_NUM;
    if (m_res > 2) {
        THREAD_NUM = PARTITION_NUM + 1;
    } else {
        THREAD_NUM = PARTITION_NUM;
    }
    if (m_res <= 2 && m_res > 0) {
        for (int col = 0; col < n; ++col) {
            MACA_ASSERT(mcMemcpyAsync(dl_next + PARTITION_NUM * 2 + col * m_next, dl + m - m_res + col * m,
                                      m_res * sizeof(valType), mcMemcpyDeviceToDevice, stream));
            MACA_ASSERT(mcMemcpyAsync(d_next + PARTITION_NUM * 2 + col * m_next, d + m - m_res + col * m,
                                      m_res * sizeof(valType), mcMemcpyDeviceToDevice, stream));
            MACA_ASSERT(mcMemcpyAsync(du_next + PARTITION_NUM * 2 + col * m_next, du + m - m_res + col * m,
                                      m_res * sizeof(valType), mcMemcpyDeviceToDevice, stream));
            MACA_ASSERT(mcMemcpyAsync(B_next + PARTITION_NUM * 2 + col * m_next, B + (m - m_res) + col * m,
                                      m_res * sizeof(valType), mcMemcpyDeviceToDevice, stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
        }
    }

    int BLOCK_SIZE = 128;
    int n_block = (THREAD_NUM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem_size = sizeof(valType) * 4 * BLOCK_SIZE;
    size_t smem_size_sub = (sizeof(valType) * 4 + sizeof(short) * PARTITION_SIZE) * BLOCK_SIZE;

    mcLaunchKernelGGL((eliminateBandKernel<PARTITION_SIZE>), dim3(n_block, n), dim3(BLOCK_SIZE, 1), smem_size, stream,
                       THREAD_NUM, m, m_next, dl, d, du, B, dl_next, d_next, du_next, B_next);
    mcLaunchKernelGGL((eliminateBandReverseKernel<PARTITION_SIZE>), dim3(n_block, n), dim3(BLOCK_SIZE, 1), smem_size,
                       stream, THREAD_NUM, m, m_next, dl, d, du, B, dl_next, d_next, du_next, B_next);
    MACA_ASSERT(mcStreamSynchronize(stream));

    mcspGtsvRecursive(handle, m_next, n, dl_next, d_next, du_next, B_next, x_next);

    if (m_res <= 2 && m_res > 0) {
        for (int col = 0; col < n; ++col) {
            MACA_ASSERT(mcMemcpyAsync(x + (m - m_res) + col * (m + 2), x_next + PARTITION_NUM * 2 + col * (m_next + 2),
                                      m_res * sizeof(valType), mcMemcpyDeviceToDevice, stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
        }
    }

    mcLaunchKernelGGL((copyBoundaryKernel<PARTITION_SIZE>), dim3(n_block, n), dim3(BLOCK_SIZE, 1), 0, stream,
                       THREAD_NUM, m, m_next, x_next, x);
    MACA_ASSERT(mcStreamSynchronize(stream));
    mcLaunchKernelGGL((substitutionKernel<PARTITION_SIZE>), dim3(n_block, n), dim3(BLOCK_SIZE, 1), smem_size_sub,
                       stream, THREAD_NUM, m, m_next, x_next, dl, d, du, B, x);
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvTemplate(mcspHandle_t handle, idxType m, idxType n, const valType *dl, const valType *d,
                              const valType *du, valType *B, idxType ldb, void *buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || B == nullptr || buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 3 || n < 0 || ldb < std::max((idxType)1, m)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    valType *dl_buffer = reinterpret_cast<valType *>(buffer);
    valType *d_buffer = dl_buffer + m * n;
    valType *du_buffer = d_buffer + m * n;
    valType *B_buffer = du_buffer + m * n;
    valType *x_buffer = B_buffer + m * n + 1;
    mcStream_t stream = mcspGetStreamInternal(handle);
    for (int col = 0; col < n; ++col) {
        MACA_ASSERT(mcMemcpyAsync(dl_buffer + col * m, dl, m * sizeof(valType), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcMemcpyAsync(d_buffer + col * m, d, m * sizeof(valType), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcMemcpyAsync(du_buffer + col * m, du, m * sizeof(valType), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(
            mcMemcpyAsync(B_buffer + col * m, B + col * ldb, m * sizeof(valType), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }
    mcspGtsvRecursive(handle, m, n, dl_buffer, d_buffer, du_buffer, B_buffer, x_buffer);
    for (int col = 0; col < n; ++col) {
        MACA_ASSERT(mcMemcpyAsync(B + col * ldb, x_buffer + col * (m + 2), m * sizeof(valType), mcMemcpyDeviceToDevice,
                                  stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsv10xTemplate(mcspHandle_t handle, idxType m, idxType n, const valType *dl, const valType *d,
                                 const valType *du, valType *B, idxType ldb) {
    size_t buffer_size = 0;
    mcspStatus_t ret = mcspGtsvBuffersizeTemplate(handle, m, n, dl, d, du, B, ldb, &buffer_size);
    if (ret != MCSP_STATUS_SUCCESS) {
        return ret;
    }
    void *buffer;
    bool use_buffer_pool = handle->mcspUsePoolBuffer((void **)&buffer, buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc((void **)&buffer, buffer_size));
    }

    ret = mcspGtsvTemplate(handle, m, n, dl, d, du, B, ldb, buffer);

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

mcspStatus_t mcspSgtsvBuffersize(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dl, const float *d,
                                 const float *du, const float *B, mcspInt ldb, size_t *buffer_size) {
    return mcspGtsvBuffersizeTemplate(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

mcspStatus_t mcspDgtsvBuffersize(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dl, const double *d,
                                 const double *du, const double *B, mcspInt ldb, size_t *buffer_size) {
    return mcspGtsvBuffersizeTemplate(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

mcspStatus_t mcspCgtsvBuffersize(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat *dl,
                                 const mcspComplexFloat *d, const mcspComplexFloat *du, const mcspComplexFloat *B,
                                 mcspInt ldb, size_t *buffer_size) {
    return mcspGtsvBuffersizeTemplate(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

mcspStatus_t mcspZgtsvBuffersize(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble *dl,
                                 const mcspComplexDouble *d, const mcspComplexDouble *du, const mcspComplexDouble *B,
                                 mcspInt ldb, size_t *buffer_size) {
    return mcspGtsvBuffersizeTemplate(handle, m, n, dl, d, du, B, ldb, buffer_size);
}

mcspStatus_t mcspSgtsv(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dl, const float *d, const float *du,
                       float *B, mcspInt ldb, void *buffer) {
    return mcspGtsvTemplate(handle, m, n, dl, d, du, B, ldb, buffer);
}

mcspStatus_t mcspDgtsv(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dl, const double *d, const double *du,
                       double *B, mcspInt ldb, void *buffer) {
    return mcspGtsvTemplate(handle, m, n, dl, d, du, B, ldb, buffer);
}

mcspStatus_t mcspCgtsv(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat *dl, const mcspComplexFloat *d,
                       const mcspComplexFloat *du, mcspComplexFloat *B, mcspInt ldb, void *buffer) {
    return mcspGtsvTemplate(handle, m, n, dl, d, du, B, ldb, buffer);
}

mcspStatus_t mcspZgtsv(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble *dl,
                       const mcspComplexDouble *d, const mcspComplexDouble *du, mcspComplexDouble *B, mcspInt ldb,
                       void *buffer) {
    return mcspGtsvTemplate(handle, m, n, dl, d, du, B, ldb, buffer);
}

mcspStatus_t mcspCuinSgtsv2_bufferSizeExt(mcspHandle_t handle, int m, int n, const float *dl, const float *d,
                                        const float *du, const float *B, int ldb, size_t *bufferSizeInBytes) {
    return mcspGtsvBuffersizeTemplate(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, bufferSizeInBytes);
}

mcspStatus_t mcspCuinDgtsv2_bufferSizeExt(mcspHandle_t handle, int m, int n, const double *dl, const double *d,
                                        const double *du, const double *B, int ldb, size_t *bufferSizeInBytes) {
    return mcspGtsvBuffersizeTemplate(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, bufferSizeInBytes);
}

mcspStatus_t mcspCuinCgtsv2_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcFloatComplex *dl,
                                        const mcFloatComplex *d, const mcFloatComplex *du, const mcFloatComplex *B,
                                        int ldb, size_t *bufferSizeInBytes) {
    return mcspGtsvBuffersizeTemplate(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, bufferSizeInBytes);
}

mcspStatus_t mcspCuinZgtsv2_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcDoubleComplex *dl,
                                        const mcDoubleComplex *d, const mcDoubleComplex *du, const mcDoubleComplex *B,
                                        int ldb, size_t *bufferSizeInBytes) {
    return mcspGtsvBuffersizeTemplate(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, bufferSizeInBytes);
}

mcspStatus_t mcspCuinSgtsv2(mcspHandle_t handle, int m, int n, const float *dl, const float *d, const float *du, float *B,
                          int ldb, void *pBuffer) {
    return mcspGtsvTemplate(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, pBuffer);
}

mcspStatus_t mcspCuinDgtsv2(mcspHandle_t handle, int m, int n, const double *dl, const double *d, const double *du,
                          double *B, int ldb, void *pBuffer) {
    return mcspGtsvTemplate(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, pBuffer);
}

mcspStatus_t mcspCuinCgtsv2(mcspHandle_t handle, int m, int n, const mcFloatComplex *dl, const mcFloatComplex *d,
                          const mcFloatComplex *du, mcFloatComplex *B, int ldb, void *pBuffer) {
    return mcspGtsvTemplate(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, pBuffer);
}

mcspStatus_t mcspCuinZgtsv2(mcspHandle_t handle, int m, int n, const mcDoubleComplex *dl, const mcDoubleComplex *d,
                          const mcDoubleComplex *du, mcDoubleComplex *B, int ldb, void *pBuffer) {
    return mcspGtsvTemplate(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb, pBuffer);
}

mcspStatus_t mcspSgtsv10x(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dl, const float *d, const float *du,
                          float *B, mcspInt ldb) {
    return mcspGtsv10xTemplate(handle, m, n, dl, d, du, B, ldb);
}

mcspStatus_t mcspDgtsv10x(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dl, const double *d,
                          const double *du, double *B, mcspInt ldb) {
    return mcspGtsv10xTemplate(handle, m, n, dl, d, du, B, ldb);
}

mcspStatus_t mcspCgtsv10x(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat *dl,
                          const mcspComplexFloat *d, const mcspComplexFloat *du, mcspComplexFloat *B, mcspInt ldb) {
    return mcspGtsv10xTemplate(handle, m, n, dl, d, du, B, ldb);
}

mcspStatus_t mcspZgtsv10x(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble *dl,
                          const mcspComplexDouble *d, const mcspComplexDouble *du, mcspComplexDouble *B, mcspInt ldb) {
    return mcspGtsv10xTemplate(handle, m, n, dl, d, du, B, ldb);
}

#ifdef __cplusplus
}
#endif