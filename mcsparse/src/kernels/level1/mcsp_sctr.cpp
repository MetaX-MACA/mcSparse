#include "common/mcsp_types.h"
#include "mcsp_handle.h"
#include "mcsp_runtime_wrapper.h"
#include "sctr_device.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspSctrTemplate(mcspHandle_t handle, idxType nnz, const valType *x_val, const idxType *x_ind, valType *y,
                              mcsparseIndexBase_t idx_base) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (y == nullptr || x_val == nullptr || x_ind == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    constexpr uint32_t block_size = 512;
    int n_block = (nnz + block_size - 1) / block_size;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL((mcspSctrKernel<block_size>), dim3(n_block), dim3(block_size), 0, stream, nnz, x_val, x_ind, y,
                       idx_base);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSsctr(mcspHandle_t handle, mcspInt nnz, const float *x_val, const mcspInt *x_ind, float *y,
                       mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, nnz, x_val, x_ind, y, idx_base);
}

mcspStatus_t mcspDsctr(mcspHandle_t handle, mcspInt nnz, const double *x_val, const mcspInt *x_ind, double *y,
                       mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, nnz, x_val, x_ind, y, idx_base);
}

mcspStatus_t mcspCsctr(mcspHandle_t handle, mcspInt nnz, const mcspComplexFloat *x_val, const mcspInt *x_ind,
                       mcspComplexFloat *y, mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, nnz, x_val, x_ind, y, idx_base);
}

mcspStatus_t mcspZsctr(mcspHandle_t handle, mcspInt nnz, const mcspComplexDouble *x_val, const mcspInt *x_ind,
                       mcspComplexDouble *y, mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, nnz, x_val, x_ind, y, idx_base);
}

#if defined(__MACA__)
mcspStatus_t mcspR16Fsctr(mcspHandle_t handle, mcspInt nnz, const __half *x_val, const mcspInt *x_ind, __half *y,
                          mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, nnz, x_val, x_ind, y, idx_base);
}
#endif

#ifdef __MACA__
mcspStatus_t mcspC16Fsctr(mcspHandle_t handle, mcspInt nnz, const __half2 *x_val, const mcspInt *x_ind, __half2 *y,
                          mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, nnz, x_val, x_ind, y, idx_base);
}

mcspStatus_t mcspR16BFsctr(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat16 *x_val, const mcspInt *x_ind,
                           mcsp_bfloat16 *y, mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, nnz, x_val, x_ind, y, idx_base);
}

mcspStatus_t mcspC16BFsctr(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat162 *x_val, const mcspInt *x_ind,
                           mcsp_bfloat162 *y, mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, nnz, x_val, x_ind, y, idx_base);
}
#endif

mcspStatus_t mcspCuinSsctr(mcspHandle_t handle, int nnz, const float *x_val, const int *x_ind, float *y,
                         mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, (mcspInt)nnz, x_val, (mcspInt *)x_ind, y, idx_base);
}

mcspStatus_t mcspCuinDsctr(mcspHandle_t handle, int nnz, const double *x_val, const int *x_ind, double *y,
                         mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, (mcspInt)nnz, x_val, (mcspInt *)x_ind, y, idx_base);
}

mcspStatus_t mcspCuinCsctr(mcspHandle_t handle, int nnz, const mcspComplexFloat *x_val, const int *x_ind,
                         mcspComplexFloat *y, mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, (mcspInt)nnz, x_val, (mcspInt *)x_ind, y, idx_base);
}

mcspStatus_t mcspCuinZsctr(mcspHandle_t handle, int nnz, const mcspComplexDouble *x_val, const int *x_ind,
                         mcspComplexDouble *y, mcsparseIndexBase_t idx_base) {
    return mcspSctrTemplate(handle, (mcspInt)nnz, x_val, (mcspInt *)x_ind, y, idx_base);
}

#ifdef __cplusplus
}
#endif