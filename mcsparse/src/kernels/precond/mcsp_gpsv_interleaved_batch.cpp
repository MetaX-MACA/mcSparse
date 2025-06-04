#include "common/mcsp_types.h"
#include "gpsv_interleaved_batch_device.hpp"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"

template <typename idxType, typename valType>
mcspStatus_t mcspGpsvInterleavedBatchBuffersizeImpl(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg,
                                                    idxType row_num, valType* ds, valType* dl, valType* d, valType* du,
                                                    valType* dw, valType* x, idxType batch_count, idxType batch_stride,
                                                    size_t* buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (row_num < 4 || batch_count < 0 || batch_stride < batch_count) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (batch_count == 0) {
        *buffer_size = ALIGN(MIN_BUFFER_SIZE, ALIGNED_SIZE);
        return MCSP_STATUS_SUCCESS;
    }

    if (ds == nullptr || dl == nullptr || d == nullptr || du == nullptr || dw == nullptr || x == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    switch (alg) {
        case MCSPARSE_GPSV_INTERLEAVED_ALG_DEFAULT:
        case MCSPARSE_GPSV_INTERLEAVED_ALG_QR:
        case MCSPARSE_GPSV_INTERLEAVED_ALG_THOMAS: {
            *buffer_size = 0;
            *buffer_size += ALIGN(sizeof(valType) * row_num * batch_count, ALIGNED_SIZE);
            *buffer_size += ALIGN(sizeof(valType) * row_num * batch_count, ALIGNED_SIZE);
            break;
        }
        case MCSPARSE_GPSV_INTERLEAVED_ALG_LU: {
            *buffer_size = ALIGN(MIN_BUFFER_SIZE, ALIGNED_SIZE);
            break;
        }
        default: {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGpsvInterleavedBatchLUImpl(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, idxType row_num,
                                            valType* ds, valType* dl, valType* d, valType* du, valType* dw, valType* x,
                                            idxType batch_count, idxType batch_stride, void* temp_buffer) {
    constexpr uint32_t block_size = 256;
    u_int32_t blocks = (batch_count - 1) / block_size + 1;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL((mcspBatchedGpsvLUKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, row_num,
                       batch_stride, batch_count, ds, dl, d, du, dw, x);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGpsvInterleavedBatchThomasImpl(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, idxType row_num,
                                                valType* ds, valType* dl, valType* d, valType* du, valType* dw,
                                                valType* x, idxType batch_count, idxType batch_stride,
                                                void* temp_buffer) {
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    valType* du1 = reinterpret_cast<valType*>(temp_buffer);
    ptr += ALIGN(sizeof(valType) * row_num * batch_count, ALIGNED_SIZE);
    valType* dw1 = reinterpret_cast<valType*>(reinterpret_cast<void*>(ptr));

    constexpr uint32_t block_size = 256;
    u_int32_t blocks = (batch_count - 1) / block_size + 1;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL((mcspBatchedGpsvThomasKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, row_num,
                       batch_stride, batch_count, ds, dl, d, du, dw, x, du1, dw1);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGpsvInterleavedBatchQrImpl(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, idxType row_num,
                                            valType* ds, valType* dl, valType* d, valType* du, valType* dw, valType* x,
                                            idxType batch_count, idxType batch_stride, void* temp_buffer) {
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    valType* dt1 = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(sizeof(valType) * row_num * batch_count, ALIGNED_SIZE);
    valType* dt2 = reinterpret_cast<valType*>(ptr);
    mcStream_t stream = mcspGetStreamInternal(handle);

    MACA_ASSERT(mcMemsetAsync(dt1, 0, sizeof(valType) * row_num * batch_count, stream));
    MACA_ASSERT(mcMemsetAsync(dt2, 0, sizeof(valType) * row_num * batch_count, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    constexpr uint32_t block_size = 256;
    u_int32_t blocks = (batch_count - 1) / block_size + 1;
    mcLaunchKernelGGL((mcspGpsvBatchHouseholderQrKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream,
                       row_num, batch_count, batch_stride, ds, dl, d, du, dw, x, dt1, dt2);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGpsvInterleavedBatchImpl(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, idxType row_num,
                                          valType* ds, valType* dl, valType* d, valType* du, valType* dw, valType* x,
                                          idxType batch_count, idxType batch_stride, void* temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (row_num < 4 || batch_count < 0 || batch_stride < batch_count) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (batch_count == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (temp_buffer == nullptr || ds == nullptr || dl == nullptr || d == nullptr || du == nullptr || dw == nullptr ||
        x == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    switch (alg) {
        case MCSPARSE_GPSV_INTERLEAVED_ALG_DEFAULT:
        case MCSPARSE_GPSV_INTERLEAVED_ALG_QR: {
            return mcspGpsvInterleavedBatchQrImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                                  temp_buffer);
        }
        case MCSPARSE_GPSV_INTERLEAVED_ALG_THOMAS: {
            return mcspGpsvInterleavedBatchThomasImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count,
                                                      batch_stride, temp_buffer);
        }
        case MCSPARSE_GPSV_INTERLEAVED_ALG_LU: {
            return mcspGpsvInterleavedBatchLUImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                                  temp_buffer);
        }
        default: {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }
}

template <typename valType>
mcspStatus_t mcspCuinGpsvInterleavedBatch_bufferSizeExtImpl(mcspHandle_t handle, int alg, int row_num, valType* ds,
                                                          valType* dl, valType* d, valType* du, valType* dw, valType* x,
                                                          int batch_count, size_t* buffer_size) {
    mcsparseGpsvInterleavedAlg_t alg_;
    switch (alg) {
        case 0: {
            alg_ = MCSPARSE_GPSV_INTERLEAVED_ALG_DEFAULT;
            break;
        }
        case 1: {
            alg_ = MCSPARSE_GPSV_INTERLEAVED_ALG_QR;
            break;
        }
        case 2: {
            alg_ = MCSPARSE_GPSV_INTERLEAVED_ALG_THOMAS;
            break;
        }
        case 3: {
            alg_ = MCSPARSE_GPSV_INTERLEAVED_ALG_LU;
            break;
        }
        default: {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }
    return mcspGpsvInterleavedBatchBuffersizeImpl(handle, alg_, (mcspInt)row_num, ds, dl, d, du, dw, x,
                                                  (mcspInt)batch_count, (mcspInt)batch_count, buffer_size);
}

template <typename valType>
mcspStatus_t mcspCuinGpsvInterleavedBatchImpl(mcspHandle_t handle, int alg, int row_num, valType* ds, valType* dl,
                                            valType* d, valType* du, valType* dw, valType* x, int batch_count,
                                            void* temp_buffer) {
    mcsparseGpsvInterleavedAlg_t alg_;
    switch (alg) {
        case 0: {
            alg_ = MCSPARSE_GPSV_INTERLEAVED_ALG_DEFAULT;
            break;
        }
        case 1: {
            alg_ = MCSPARSE_GPSV_INTERLEAVED_ALG_QR;
            break;
        }
        case 2: {
            alg_ = MCSPARSE_GPSV_INTERLEAVED_ALG_THOMAS;
            break;
        }
        case 3: {
            alg_ = MCSPARSE_GPSV_INTERLEAVED_ALG_LU;
            break;
        }
        default: {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }
    return mcspGpsvInterleavedBatchImpl(handle, alg_, (mcspInt)row_num, ds, dl, d, du, dw, x, (mcspInt)batch_count,
                                        (mcspInt)batch_count, temp_buffer);
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSgpsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                                 float* ds, float* dl, float* d, float* du, float* dw, float* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size) {
    return mcspGpsvInterleavedBatchBuffersizeImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                                  buffer_size);
}

mcspStatus_t mcspDgpsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                                 double* ds, double* dl, double* d, double* du, double* dw, double* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size) {
    return mcspGpsvInterleavedBatchBuffersizeImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                                  buffer_size);
}

mcspStatus_t mcspCgpsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                                 mcspComplexFloat* ds, mcspComplexFloat* dl, mcspComplexFloat* d,
                                                 mcspComplexFloat* du, mcspComplexFloat* dw, mcspComplexFloat* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size) {
    return mcspGpsvInterleavedBatchBuffersizeImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                                  buffer_size);
}

mcspStatus_t mcspZgpsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                                 mcspComplexDouble* ds, mcspComplexDouble* dl, mcspComplexDouble* d,
                                                 mcspComplexDouble* du, mcspComplexDouble* dw, mcspComplexDouble* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size) {
    return mcspGpsvInterleavedBatchBuffersizeImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                                  buffer_size);
}

mcspStatus_t mcspSgpsvInterleavedBatch(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                       float* ds, float* dl, float* d, float* du, float* dw, float* x,
                                       mcspInt batch_count, mcspInt batch_stride, void* temp_buffer) {
    return mcspGpsvInterleavedBatchImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                        temp_buffer);
}

mcspStatus_t mcspDgpsvInterleavedBatch(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                       double* ds, double* dl, double* d, double* du, double* dw, double* x,
                                       mcspInt batch_count, mcspInt batch_stride, void* temp_buffer) {
    return mcspGpsvInterleavedBatchImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                        temp_buffer);
}

mcspStatus_t mcspCgpsvInterleavedBatch(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                       mcspComplexFloat* ds, mcspComplexFloat* dl, mcspComplexFloat* d,
                                       mcspComplexFloat* du, mcspComplexFloat* dw, mcspComplexFloat* x,
                                       mcspInt batch_count, mcspInt batch_stride, void* temp_buffer) {
    return mcspGpsvInterleavedBatchImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                        temp_buffer);
}

mcspStatus_t mcspZgpsvInterleavedBatch(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                       mcspComplexDouble* ds, mcspComplexDouble* dl, mcspComplexDouble* d,
                                       mcspComplexDouble* du, mcspComplexDouble* dw, mcspComplexDouble* x,
                                       mcspInt batch_count, mcspInt batch_stride, void* temp_buffer) {
    return mcspGpsvInterleavedBatchImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, batch_stride,
                                        temp_buffer);
}

mcspStatus_t mcspCuinSgpsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, float* ds, float* dl,
                                                       float* d, float* du, float* dw, float* x, int batch_count,
                                                       size_t* buffer_size) {
    return mcspCuinGpsvInterleavedBatch_bufferSizeExtImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count,
                                                        buffer_size);
}

mcspStatus_t mcspCuinDgpsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, double* ds,
                                                       double* dl, double* d, double* du, double* dw, double* x,
                                                       int batch_count, size_t* buffer_size) {
    return mcspCuinGpsvInterleavedBatch_bufferSizeExtImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count,
                                                        buffer_size);
}

mcspStatus_t mcspCuinCgpsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, mcspComplexFloat* ds,
                                                       mcspComplexFloat* dl, mcspComplexFloat* d, mcspComplexFloat* du,
                                                       mcspComplexFloat* dw, mcspComplexFloat* x, int batch_count,
                                                       size_t* buffer_size) {
    return mcspCuinGpsvInterleavedBatch_bufferSizeExtImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count,
                                                        buffer_size);
}

mcspStatus_t mcspCuinZgpsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, mcspComplexDouble* ds,
                                                       mcspComplexDouble* dl, mcspComplexDouble* d,
                                                       mcspComplexDouble* du, mcspComplexDouble* dw,
                                                       mcspComplexDouble* x, int batch_count, size_t* buffer_size) {
    return mcspCuinGpsvInterleavedBatch_bufferSizeExtImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count,
                                                        buffer_size);
}

mcspStatus_t mcspCuinSgpsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, float* ds, float* dl, float* d,
                                         float* du, float* dw, float* x, int batch_count, void* temp_buffer) {
    return mcspCuinGpsvInterleavedBatchImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, temp_buffer);
}

mcspStatus_t mcspCuinDgpsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, double* ds, double* dl, double* d,
                                         double* du, double* dw, double* x, int batch_count, void* temp_buffer) {
    return mcspCuinGpsvInterleavedBatchImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, temp_buffer);
}

mcspStatus_t mcspCuinCgpsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, mcspComplexFloat* ds,
                                         mcspComplexFloat* dl, mcspComplexFloat* d, mcspComplexFloat* du,
                                         mcspComplexFloat* dw, mcspComplexFloat* x, int batch_count,
                                         void* temp_buffer) {
    return mcspCuinGpsvInterleavedBatchImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, temp_buffer);
}

mcspStatus_t mcspCuinZgpsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, mcspComplexDouble* ds,
                                         mcspComplexDouble* dl, mcspComplexDouble* d, mcspComplexDouble* du,
                                         mcspComplexDouble* dw, mcspComplexDouble* x, int batch_count,
                                         void* temp_buffer) {
    return mcspCuinGpsvInterleavedBatchImpl(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, temp_buffer);
}

#ifdef __cplusplus
}
#endif