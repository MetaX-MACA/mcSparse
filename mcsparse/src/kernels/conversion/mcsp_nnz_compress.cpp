#include "common/mcsp_types.h"
#include "device_reduce.hpp"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"
#include "nnz_compress_device.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspNnzCompressTemplate(mcspHandle_t handle, idxType m, const mcspMatDescr_t mcsp_descr_A,
                                     const valType* csr_val_A, const idxType* csr_row_A, idxType* nnz_per_row,
                                     idxType* nnz_C, valType tol) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (mcsp_descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (m < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (csr_row_A == nullptr || nnz_per_row == nullptr || nnz_C == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType nnz_A, nnz_A_;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemcpyAsync(&nnz_A, &csr_row_A[m], sizeof(*csr_row_A), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcMemcpyAsync(&nnz_A_, &csr_row_A[0], sizeof(*csr_row_A), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    nnz_A -= nnz_A_;
    if (nnz_A < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }
    if (nnz_A != 0 && csr_val_A == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    constexpr uint32_t block_size = 512;
    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    uint32_t grid_size = (m + segments_per_block - 1) / segments_per_block;

    mcLaunchKernelGGL((nnzCompressKernel<block_size, segments_per_block, segment_size, WARP_SIZE>), dim3(grid_size),
                       dim3(block_size), 0, stream, m, mcsp_descr_A->base, csr_val_A, csr_row_A, nnz_per_row, tol);

    idxType buffer_size;
    void* buffer_device;
    idxType* dnnz_C;
    bool use_buffer_pool;

    MACA_ASSERT(mcMalloc(&dnnz_C, sizeof(*dnnz_C)));
    mcprim::reduce(nullptr, buffer_size, nnz_per_row, dnnz_C, m, mcprim::plus<idxType>(), stream);
    use_buffer_pool = handle->mcspUsePoolBuffer(&buffer_device, buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc(&buffer_device, buffer_size));
    }
    mcprim::reduce(buffer_device, buffer_size, nnz_per_row, dnnz_C, m, mcprim::plus<idxType>(), stream);
    if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
        MACA_ASSERT(mcMemcpyAsync(nnz_C, dnnz_C, sizeof(*dnnz_C), mcMemcpyDeviceToHost, stream));
    } else {
        MACA_ASSERT(mcMemcpyAsync(nnz_C, dnnz_C, sizeof(*dnnz_C), mcMemcpyDeviceToDevice, stream));
    }
    MACA_ASSERT(mcStreamSynchronize(stream));
    MACA_ASSERT(mcFree(dnnz_C));
    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(buffer_device));
    }
    return MCSP_STATUS_SUCCESS;
}
#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSnnzCompress(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t mcsp_descr_A, const float* csr_val_A,
                              const mcspInt* csr_row_A, mcspInt* nnz_per_row, mcspInt* nnz_C, float tol) {
    return mcspNnzCompressTemplate(handle, m, mcsp_descr_A, csr_val_A, csr_row_A, nnz_per_row, nnz_C, tol);
}

mcspStatus_t mcspDnnzCompress(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t mcsp_descr_A,
                              const double* csr_val_A, const mcspInt* csr_row_A, mcspInt* nnz_per_row, mcspInt* nnz_C,
                              double tol) {
    return mcspNnzCompressTemplate(handle, m, mcsp_descr_A, csr_val_A, csr_row_A, nnz_per_row, nnz_C, tol);
}

mcspStatus_t mcspCnnzCompress(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat* csr_val_A, const mcspInt* csr_row_A, mcspInt* nnz_per_row,
                              mcspInt* nnz_C, mcspComplexFloat tol) {
    return mcspNnzCompressTemplate(handle, m, mcsp_descr_A, csr_val_A, csr_row_A, nnz_per_row, nnz_C, tol);
}

mcspStatus_t mcspZnnzCompress(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble* csr_val_A, const mcspInt* csr_row_A, mcspInt* nnz_per_row,
                              mcspInt* nnz_C, mcspComplexDouble tol) {
    return mcspNnzCompressTemplate(handle, m, mcsp_descr_A, csr_val_A, csr_row_A, nnz_per_row, nnz_C, tol);
}

mcspStatus_t mcspCuinSnnz_compress(mcspHandle_t handle, int m, const mcspMatDescr_t mcsp_descr_A, const float* csr_val_A,
                                 const int* csr_row_A, int* nnz_per_row, int* nnz_C, float tol) {
    return mcspNnzCompressTemplate(handle, (mcspInt)m, mcsp_descr_A, csr_val_A, (mcspInt*)csr_row_A,
                                   (mcspInt*)nnz_per_row, (mcspInt*)nnz_C, tol);
}

mcspStatus_t mcspCuinDnnz_compress(mcspHandle_t handle, int m, const mcspMatDescr_t mcsp_descr_A, const double* csr_val_A,
                                 const int* csr_row_A, int* nnz_per_row, int* nnz_C, double tol) {
    return mcspNnzCompressTemplate(handle, (mcspInt)m, mcsp_descr_A, csr_val_A, (mcspInt*)csr_row_A,
                                   (mcspInt*)nnz_per_row, (mcspInt*)nnz_C, tol);
}

mcspStatus_t mcspCuinCnnz_compress(mcspHandle_t handle, int m, const mcspMatDescr_t mcsp_descr_A,
                                 const mcspComplexFloat* csr_val_A, const int* csr_row_A, int* nnz_per_row, int* nnz_C,
                                 mcspComplexFloat tol) {
    return mcspNnzCompressTemplate(handle, (mcspInt)m, mcsp_descr_A, csr_val_A, (mcspInt*)csr_row_A,
                                   (mcspInt*)nnz_per_row, (mcspInt*)nnz_C, tol);
}

mcspStatus_t mcspCuinZnnz_compress(mcspHandle_t handle, int m, const mcspMatDescr_t mcsp_descr_A,
                                 const mcspComplexDouble* csr_val_A, const int* csr_row_A, int* nnz_per_row, int* nnz_C,
                                 mcspComplexDouble tol) {
    return mcspNnzCompressTemplate(handle, (mcspInt)m, mcsp_descr_A, csr_val_A, (mcspInt*)csr_row_A,
                                   (mcspInt*)nnz_per_row, (mcspInt*)nnz_C, tol);
}
#ifdef __cplusplus
}
#endif