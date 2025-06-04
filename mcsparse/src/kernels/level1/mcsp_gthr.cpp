#include "common/mcsp_types.h"
#include "gthr_device.hpp"
#include "mcsp_handle.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType, typename valType>
mcspStatus_t mcspGthrTemplate(mcspHandle_t handle, idxType nnz, const valType *y, valType *x_val, const idxType *x_ind,
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
    mcLaunchKernelGGL((mcspGthrKernel<block_size>), dim3(n_block), dim3(block_size), 0, stream, nnz, y, x_val, x_ind,
                       idx_base);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSgthr(mcspHandle_t handle, mcspInt nnz, const float *y, float *x_val, const mcspInt *x_ind,
                       mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

mcspStatus_t mcspDgthr(mcspHandle_t handle, mcspInt nnz, const double *y, double *x_val, const mcspInt *x_ind,
                       mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

mcspStatus_t mcspCgthr(mcspHandle_t handle, mcspInt nnz, const mcspComplexFloat *y, mcspComplexFloat *x_val,
                       const mcspInt *x_ind, mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

mcspStatus_t mcspZgthr(mcspHandle_t handle, mcspInt nnz, const mcspComplexDouble *y, mcspComplexDouble *x_val,
                       const mcspInt *x_ind, mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

#if defined(__MACA__)
mcspStatus_t mcspR16Fgthr(mcspHandle_t handle, mcspInt nnz, const __half *y, __half *x_val, const mcspInt *x_ind,
                          mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}
#endif

#ifdef __MACA__
mcspStatus_t mcspC16Fgthr(mcspHandle_t handle, mcspInt nnz, const __half2 *y, __half2 *x_val, const mcspInt *x_ind,
                          mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

mcspStatus_t mcspR16BFgthr(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat16 *y, mcsp_bfloat16 *x_val,
                           const mcspInt *x_ind, mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

mcspStatus_t mcspC16BFgthr(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat162 *y, mcsp_bfloat162 *x_val,
                           const mcspInt *x_ind, mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}
#endif

mcspStatus_t mcspCuinSgthr(mcspHandle_t handle, int nnz, const float *y, float *x_val, const int *x_ind,
                         mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, (mcspInt)nnz, y, x_val, (mcspInt *)x_ind, idx_base);
}

mcspStatus_t mcspCuinDgthr(mcspHandle_t handle, int nnz, const double *y, double *x_val, const int *x_ind,
                         mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, (mcspInt)nnz, y, x_val, (mcspInt *)x_ind, idx_base);
}

mcspStatus_t mcspCuinCgthr(mcspHandle_t handle, int nnz, const mcspComplexFloat *y, mcspComplexFloat *x_val,
                         const int *x_ind, mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, (mcspInt)nnz, y, x_val, (mcspInt *)x_ind, idx_base);
}

mcspStatus_t mcspCuinZgthr(mcspHandle_t handle, int nnz, const mcspComplexDouble *y, mcspComplexDouble *x_val,
                         const int *x_ind, mcsparseIndexBase_t idx_base) {
    return mcspGthrTemplate(handle, (mcspInt)nnz, y, x_val, (mcspInt *)x_ind, idx_base);
}

#ifdef __cplusplus
}
#endif