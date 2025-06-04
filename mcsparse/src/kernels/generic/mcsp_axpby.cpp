#include "axpby_device.hpp"
#include "common/mcsp_types.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType, typename computeType, typename inoutType = computeType, unsigned int BLOCKSIZE = 512>
static mcspStatus_t mcsp_axpby_template(mcspHandle_t handle, const void* alpha, mcspSpVecDescr_t vecX, const void* beta,
                                        mcspDnVecDescr_t vecY) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (vecX == nullptr || vecY == nullptr || alpha == nullptr || beta == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (vecX->nnz < 0 || vecY->size < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (vecY->size == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    computeType h_beta = getScalarToHost((computeType*)beta, handle->ptr_mode);
    computeType h_alpha = getScalarToHost((computeType*)alpha, handle->ptr_mode);
    mcStream_t stream = mcspGetStreamInternal(handle);
    dim3 scale_blocks((vecY->size - 1) / BLOCKSIZE + 1);
    mcLaunchKernelGGL((axpby_scale_kernel<BLOCKSIZE>), scale_blocks, BLOCKSIZE, 0, stream, h_beta,
                       (inoutType*)vecY->values, (idxType)vecY->size);

    dim3 blocks((vecX->nnz - 1) / BLOCKSIZE + 1);
    dim3 block_size(BLOCKSIZE);
    mcLaunchKernelGGL((axpby_kernel<BLOCKSIZE>), blocks, block_size, 0, stream, (idxType)vecX->nnz, h_alpha,
                       (inoutType*)vecX->values, (idxType*)vecX->indices, (inoutType*)vecY->values, vecX->idxBase);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
static mcspStatus_t mcspAxpbyImpl(mcspHandle_t handle, const void* alpha, mcspSpVecDescr_t vecX, const void* beta,
                                  mcspDnVecDescr_t vecY) {
    switch (vecX->valueType) {
        case MACA_R_32F:
            return mcsp_axpby_template<idxType, float>(handle, alpha, vecX, beta, vecY);
        case MACA_R_64F:
            return mcsp_axpby_template<idxType, double>(handle, alpha, vecX, beta, vecY);
        case MACA_C_32F:
            return mcsp_axpby_template<idxType, mcspComplexFloat>(handle, alpha, vecX, beta, vecY);
        case MACA_C_64F:
            return mcsp_axpby_template<idxType, mcspComplexDouble>(handle, alpha, vecX, beta, vecY);
#if defined(__MACA__)
        case MACA_R_16F:
            return mcsp_axpby_template<idxType, float, __half>(handle, alpha, vecX, beta, vecY);
        case MACA_R_16BF:
            return mcsp_axpby_template<idxType, float, mcsp_bfloat16>(handle, alpha, vecX, beta, vecY);
        case MACA_C_16F:
            return mcsp_axpby_template<idxType, mcspComplexFloat, __half2>(handle, alpha, vecX, beta, vecY);
        case MACA_C_16BF:
            return mcsp_axpby_template<idxType, mcspComplexFloat, mcsp_bfloat162>(handle, alpha, vecX, beta, vecY);
#endif
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspAxpby(mcspHandle_t handle, const void* alpha, mcspSpVecDescr_t vecX, const void* beta,
                       mcspDnVecDescr_t vecY) {
    if (vecX->idxType == MCSPARSE_INDEX_32I) {
        return mcspAxpbyImpl<mcspInt>(handle, alpha, vecX, beta, vecY);
    } else if (vecX->idxType == MCSPARSE_INDEX_64I) {
        return mcspAxpbyImpl<int64_t>(handle, alpha, vecX, beta, vecY);
    } else {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

#ifdef __cplusplus
}
#endif