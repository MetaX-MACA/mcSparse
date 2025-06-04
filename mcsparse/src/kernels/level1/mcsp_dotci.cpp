#include "block_reduce.hpp"
#include "common/mcsp_types.h"
#include "dotci_device.hpp"
#include "mcsp_handle.h"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE = 512, typename idxType, typename computeType, typename inoutType>
mcspStatus_t mcsp_dotci_template(mcspHandle_t handle, idxType nnz, const inoutType* x_val, const idxType* x_ind,
                                 const inoutType* y, computeType* result, mcsparseIndexBase_t idx_base) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (x_val == nullptr || x_ind == nullptr || y == nullptr || result == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    mcStream_t stream = mcspGetStreamInternal(handle);
    if (nnz == 0) {
        if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
            *result = 0;
        } else {
            MACA_ASSERT(mcMemsetAsync(result, 0, sizeof(computeType), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
        }
        return MCSP_STATUS_SUCCESS;
    }

    computeType* d_result = nullptr;
    if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
        MACA_ASSERT(mcMalloc(&d_result, sizeof(computeType)));
    } else {
        d_result = result;
    }

    unsigned int n_block = (nnz - 1) / BLOCKSIZE + 1;
    unsigned int elements = nnz;
    computeType* buffer = nullptr;
    bool use_buffer_pool = handle->mcspUsePoolBuffer((void**)&buffer, n_block * sizeof(*buffer));
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc((void**)&buffer, n_block * sizeof(*buffer)));
    }

    mcLaunchKernelGGL((dotci_block_kernel<BLOCKSIZE>), n_block, BLOCKSIZE, 0, stream, elements, x_val, x_ind, y,
                       buffer, idx_base);
    while (n_block > BLOCKSIZE) {
        elements = n_block;
        n_block = (n_block - 1) / BLOCKSIZE + 1;
        mcLaunchKernelGGL((mcprim::block_reduce_kernel<BLOCKSIZE>), n_block, BLOCKSIZE,
                           BLOCKSIZE * sizeof(computeType), stream, buffer, buffer, elements, computeType(0),
                           mcprim::plus<computeType>());
    }
    elements = n_block;
    n_block = (n_block - 1) / BLOCKSIZE + 1;
    mcLaunchKernelGGL((mcprim::block_reduce_kernel<BLOCKSIZE>), n_block, BLOCKSIZE, BLOCKSIZE * sizeof(computeType),
                       stream, buffer, d_result, elements, computeType(0), mcprim::plus<computeType>());
    MACA_ASSERT(mcStreamSynchronize(stream));

    if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
        MACA_ASSERT(mcMemcpyAsync(result, d_result, sizeof(computeType), mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        MACA_ASSERT(mcFree(d_result));
    }

    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(buffer));
    }

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspCdotci(mcspHandle_t handle, mcspInt nnz, const mcspComplexFloat* x_val, const mcspInt* x_ind,
                        const mcspComplexFloat* y, mcspComplexFloat* result, mcsparseIndexBase_t idx_base) {
    return mcsp_dotci_template(handle, nnz, x_val, x_ind, y, result, idx_base);
}

mcspStatus_t mcspZdotci(mcspHandle_t handle, mcspInt nnz, const mcspComplexDouble* x_val, const mcspInt* x_ind,
                        const mcspComplexDouble* y, mcspComplexDouble* result, mcsparseIndexBase_t idx_base) {
    return mcsp_dotci_template(handle, nnz, x_val, x_ind, y, result, idx_base);
}

#if defined(__MACA__)
mcspStatus_t mcspC16fC32fDotci(mcspHandle_t handle, mcspInt nnz, const __half2* x_val, const mcspInt* x_ind,
                               const __half2* y, mcspComplexFloat* result, mcsparseIndexBase_t idx_base) {
    return mcsp_dotci_template(handle, nnz, x_val, x_ind, y, result, idx_base);
}

mcspStatus_t mcspC16bfC32fDotci(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat162* x_val, const mcspInt* x_ind,
                                const mcsp_bfloat162* y, mcspComplexFloat* result, mcsparseIndexBase_t idx_base) {
    return mcsp_dotci_template(handle, nnz, x_val, x_ind, y, result, idx_base);
}
#endif

#ifdef __cplusplus
}
#endif
