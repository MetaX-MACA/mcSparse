#include "common/mcsp_types.h"
#include "gtsv_interleaved_batch_device.hpp"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvInterleavedBatchBuffersizeImpl(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg,
                                                    idxType row_num, const valType* dl, const valType* d,
                                                    const valType* du, const valType* x, idxType batch_count,
                                                    idxType batch_stride, size_t* buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (row_num < 3 || batch_count < 0 || batch_stride < batch_count) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (batch_count == 0) {
        *buffer_size = ALIGN(MIN_BUFFER_SIZE, ALIGNED_SIZE);
        return MCSP_STATUS_SUCCESS;
    }

    if (dl == nullptr || d == nullptr || du == nullptr || x == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    switch (alg) {
        case MCSPARSE_GTSV_INTERLEAVED_ALG_DEFAULT:
        case MCSPARSE_GTSV_INTERLEAVED_ALG_QR:
        case MCSPARSE_GTSV_INTERLEAVED_ALG_THOMAS: {
            *buffer_size = 0;
            *buffer_size += ALIGN(sizeof(valType) * row_num * batch_count, ALIGNED_SIZE);
            break;
        }
        case MCSPARSE_GTSV_INTERLEAVED_ALG_LU: {
            *buffer_size = 0;
            *buffer_size += ALIGN(sizeof(valType) * row_num * batch_count, ALIGNED_SIZE);
            *buffer_size += ALIGN(sizeof(idxType) * row_num * batch_count, ALIGNED_SIZE);
            break;
        }
        default: {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvInterleavedBatchLUImpl(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, idxType row_num,
                                            valType* dl, valType* d, valType* du, valType* x, idxType batch_count,
                                            idxType batch_stride, void* temp_buffer) {
    char* ptr = reinterpret_cast<char*>(temp_buffer);
    valType* u2 = reinterpret_cast<valType*>(ptr);
    ptr += ALIGN(sizeof(valType) * row_num * batch_count, ALIGNED_SIZE);
    idxType* p = reinterpret_cast<idxType*>(ptr);
    mcStream_t stream = mcspGetStreamInternal(handle);

    MACA_ASSERT(mcMemsetAsync(u2, 0, sizeof(valType) * row_num * batch_count, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    constexpr uint32_t block_size = 256;
    u_int32_t blocks = (batch_count - 1) / block_size + 1;
    mcLaunchKernelGGL((mcspBatchedGtsvLUKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, row_num,
                       batch_stride, batch_count, dl, d, du, x, u2, p);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvInterleavedBatchThomasImpl(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, idxType row_num,
                                                valType* dl, valType* d, valType* du, valType* x, idxType batch_count,
                                                idxType batch_stride, void* temp_buffer) {
    valType* du_tmp = reinterpret_cast<valType*>(temp_buffer);

    constexpr uint32_t block_size = 256;
    u_int32_t blocks = (batch_count - 1) / block_size + 1;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL((mcspBatchedGtsvThomasKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, row_num,
                       batch_stride, batch_count, dl, d, du, x, du_tmp);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvInterleavedBatchQrImpl(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, idxType row_num,
                                            valType* dl, valType* d, valType* du, valType* x, idxType batch_count,
                                            idxType batch_stride, void* temp_buffer) {
    valType* r2 = reinterpret_cast<valType*>(temp_buffer);
    mcStream_t stream = mcspGetStreamInternal(handle);

    MACA_ASSERT(mcMemsetAsync(r2, 0, sizeof(valType) * row_num * batch_count, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    constexpr uint32_t block_size = 256;
    u_int32_t blocks = (batch_count - 1) / block_size + 1;
    mcLaunchKernelGGL((mcspGtsvBatchGivensQrKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, row_num,
                       batch_count, batch_stride, dl, d, du, x, r2);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGtsvInterleavedBatchImpl(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, idxType row_num,
                                          valType* dl, valType* d, valType* du, valType* x, idxType batch_count,
                                          idxType batch_stride, void* temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (row_num < 3 || batch_count < 0 || batch_stride < batch_count) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (batch_count == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (temp_buffer == nullptr || dl == nullptr || d == nullptr || du == nullptr || x == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    switch (alg) {
        case MCSPARSE_GTSV_INTERLEAVED_ALG_DEFAULT:
        case MCSPARSE_GTSV_INTERLEAVED_ALG_QR: {
            return mcspGtsvInterleavedBatchQrImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride,
                                                  temp_buffer);
        }
        case MCSPARSE_GTSV_INTERLEAVED_ALG_THOMAS: {
            return mcspGtsvInterleavedBatchThomasImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride,
                                                      temp_buffer);
        }
        case MCSPARSE_GTSV_INTERLEAVED_ALG_LU: {
            return mcspGtsvInterleavedBatchLUImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride,
                                                  temp_buffer);
        }
        default: {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }
}

template <typename valType>
mcspStatus_t mcspCuinGtsvInterleavedBatch_bufferSizeExtImpl(mcspHandle_t handle, int alg, int row_num, const valType* dl,
                                                          const valType* d, const valType* du, const valType* x,
                                                          int batch_count, size_t* buffer_size) {
    mcsparseGtsvInterleavedAlg_t alg_;
    switch (alg) {
        case 0: {
            alg_ = MCSPARSE_GTSV_INTERLEAVED_ALG_QR;
            break;
        }
        case 1: {
            alg_ = MCSPARSE_GTSV_INTERLEAVED_ALG_THOMAS;
            break;
        }
        case 2: {
            alg_ = MCSPARSE_GTSV_INTERLEAVED_ALG_LU;
            break;
        }
        default: {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }
    return mcspGtsvInterleavedBatchBuffersizeImpl(handle, alg_, (mcspInt)row_num, dl, d, du, x, (mcspInt)batch_count,
                                                  (mcspInt)batch_count, buffer_size);
}

template <typename valType>
mcspStatus_t mcspCuinGtsvInterleavedBatchImpl(mcspHandle_t handle, int alg, int row_num, valType* dl, valType* d,
                                            valType* du, valType* x, int batch_count, void* temp_buffer) {
    mcsparseGtsvInterleavedAlg_t alg_;
    switch (alg) {
        case 0: {
            alg_ = MCSPARSE_GTSV_INTERLEAVED_ALG_QR;
            break;
        }
        case 1: {
            alg_ = MCSPARSE_GTSV_INTERLEAVED_ALG_THOMAS;
            break;
        }
        case 2: {
            alg_ = MCSPARSE_GTSV_INTERLEAVED_ALG_LU;
            break;
        }
        default: {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }
    return mcspGtsvInterleavedBatchImpl(handle, alg_, (mcspInt)row_num, dl, d, du, x, (mcspInt)batch_count,
                                        (mcspInt)batch_count, temp_buffer);
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSgtsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                                 const float* dl, const float* d, const float* du, const float* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size) {
    return mcspGtsvInterleavedBatchBuffersizeImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride,
                                                  buffer_size);
}

mcspStatus_t mcspDgtsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                                 const double* dl, const double* d, const double* du, const double* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size) {
    return mcspGtsvInterleavedBatchBuffersizeImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride,
                                                  buffer_size);
}

mcspStatus_t mcspCgtsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                                 const mcspComplexFloat* dl, const mcspComplexFloat* d,
                                                 const mcspComplexFloat* du, const mcspComplexFloat* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size) {
    return mcspGtsvInterleavedBatchBuffersizeImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride,
                                                  buffer_size);
}

mcspStatus_t mcspZgtsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                                 const mcspComplexDouble* dl, const mcspComplexDouble* d,
                                                 const mcspComplexDouble* du, const mcspComplexDouble* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size) {
    return mcspGtsvInterleavedBatchBuffersizeImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride,
                                                  buffer_size);
}

mcspStatus_t mcspSgtsvInterleavedBatch(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                       float* dl, float* d, float* du, float* x, mcspInt batch_count,
                                       mcspInt batch_stride, void* temp_buffer) {
    return mcspGtsvInterleavedBatchImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride, temp_buffer);
}

mcspStatus_t mcspDgtsvInterleavedBatch(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                       double* dl, double* d, double* du, double* x, mcspInt batch_count,
                                       mcspInt batch_stride, void* temp_buffer) {
    return mcspGtsvInterleavedBatchImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride, temp_buffer);
}

mcspStatus_t mcspCgtsvInterleavedBatch(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                       mcspComplexFloat* dl, mcspComplexFloat* d, mcspComplexFloat* du,
                                       mcspComplexFloat* x, mcspInt batch_count, mcspInt batch_stride,
                                       void* temp_buffer) {
    return mcspGtsvInterleavedBatchImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride, temp_buffer);
}

mcspStatus_t mcspZgtsvInterleavedBatch(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                       mcspComplexDouble* dl, mcspComplexDouble* d, mcspComplexDouble* du,
                                       mcspComplexDouble* x, mcspInt batch_count, mcspInt batch_stride,
                                       void* temp_buffer) {
    return mcspGtsvInterleavedBatchImpl(handle, alg, row_num, dl, d, du, x, batch_count, batch_stride, temp_buffer);
}

mcspStatus_t mcspCuinSgtsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, const float* dl,
                                                       const float* d, const float* du, const float* x, int batch_count,
                                                       size_t* buffer_size) {
    return mcspCuinGtsvInterleavedBatch_bufferSizeExtImpl(handle, alg, row_num, dl, d, du, x, batch_count, buffer_size);
}

mcspStatus_t mcspCuinDgtsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, const double* dl,
                                                       const double* d, const double* du, const double* x,
                                                       int batch_count, size_t* buffer_size) {
    return mcspCuinGtsvInterleavedBatch_bufferSizeExtImpl(handle, alg, row_num, dl, d, du, x, batch_count, buffer_size);
}

mcspStatus_t mcspCuinCgtsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num,
                                                       const mcspComplexFloat* dl, const mcspComplexFloat* d,
                                                       const mcspComplexFloat* du, const mcspComplexFloat* x,
                                                       int batch_count, size_t* buffer_size) {
    return mcspCuinGtsvInterleavedBatch_bufferSizeExtImpl(handle, alg, row_num, dl, d, du, x, batch_count, buffer_size);
}

mcspStatus_t mcspCuinZgtsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num,
                                                       const mcspComplexDouble* dl, const mcspComplexDouble* d,
                                                       const mcspComplexDouble* du, const mcspComplexDouble* x,
                                                       int batch_count, size_t* buffer_size) {
    return mcspCuinGtsvInterleavedBatch_bufferSizeExtImpl(handle, alg, row_num, dl, d, du, x, batch_count, buffer_size);
}

mcspStatus_t mcspCuinSgtsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, float* dl, float* d, float* du,
                                         float* x, int batch_count, void* temp_buffer) {
    return mcspCuinGtsvInterleavedBatchImpl(handle, alg, row_num, dl, d, du, x, batch_count, temp_buffer);
}

mcspStatus_t mcspCuinDgtsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, double* dl, double* d, double* du,
                                         double* x, int batch_count, void* temp_buffer) {
    return mcspCuinGtsvInterleavedBatchImpl(handle, alg, row_num, dl, d, du, x, batch_count, temp_buffer);
}

mcspStatus_t mcspCuinCgtsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, mcspComplexFloat* dl,
                                         mcspComplexFloat* d, mcspComplexFloat* du, mcspComplexFloat* x,
                                         int batch_count, void* temp_buffer) {
    return mcspCuinGtsvInterleavedBatchImpl(handle, alg, row_num, dl, d, du, x, batch_count, temp_buffer);
}

mcspStatus_t mcspCuinZgtsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, mcspComplexDouble* dl,
                                         mcspComplexDouble* d, mcspComplexDouble* du, mcspComplexDouble* x,
                                         int batch_count, void* temp_buffer) {
    return mcspCuinGtsvInterleavedBatchImpl(handle, alg, row_num, dl, d, du, x, batch_count, temp_buffer);
}

#ifdef __cplusplus
}
#endif