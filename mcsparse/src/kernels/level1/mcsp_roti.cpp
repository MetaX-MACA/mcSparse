#include "common/mcsp_types.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_runtime_wrapper.h"
#include "roti_device.hpp"

template <typename idxType, typename computeType, typename inoutType>
mcspStatus_t mcspRotiTemplate(mcspHandle_t handle, idxType nnz, inoutType *x_val, const idxType *x_ind, inoutType *y,
                              const computeType *c, const computeType *s, mcsparseIndexBase_t idx_base) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (y == nullptr || x_val == nullptr || x_ind == nullptr || c == nullptr || s == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    computeType h_c = getScalarToHost(c, handle->ptr_mode);
    computeType h_s = getScalarToHost(s, handle->ptr_mode);
    if (h_c == static_cast<computeType>(1) && h_s == static_cast<computeType>(0)) {
        return MCSP_STATUS_SUCCESS;
    }

    constexpr uint32_t block_size = 512;
    int n_block = (nnz + block_size - 1) / block_size;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL((mcspRotiKernel<block_size>), dim3(n_block), dim3(block_size), 0, stream, nnz, x_val, x_ind, y,
                       h_c, h_s, idx_base);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSroti(mcspHandle_t handle, mcspInt nnz, float *x_val, const mcspInt *x_ind, float *y, const float *c,
                       const float *s, mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

mcspStatus_t mcspDroti(mcspHandle_t handle, mcspInt nnz, double *x_val, const mcspInt *x_ind, double *y,
                       const double *c, const double *s, mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

mcspStatus_t mcspCroti(mcspHandle_t handle, mcspInt nnz, mcspComplexFloat *x_val, const mcspInt *x_ind,
                       mcspComplexFloat *y, const mcspComplexFloat *c, const mcspComplexFloat *s,
                       mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

mcspStatus_t mcspZroti(mcspHandle_t handle, mcspInt nnz, mcspComplexDouble *x_val, const mcspInt *x_ind,
                       mcspComplexDouble *y, const mcspComplexDouble *c, const mcspComplexDouble *s,
                       mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

#if defined(__MACA__)
mcspStatus_t mcspR16fRoti(mcspHandle_t handle, mcspInt nnz, __half *x_val, const mcspInt *x_ind, __half *y,
                          const float *c, const float *s, mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}
#endif

#if defined(__MACA__)
mcspStatus_t mcspR16bfRoti(mcspHandle_t handle, mcspInt nnz, mcsp_bfloat16 *x_val, const mcspInt *x_ind,
                           mcsp_bfloat16 *y, const float *c, const float *s, mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

mcspStatus_t mcspC16fRoti(mcspHandle_t handle, mcspInt nnz, __half2 *x_val, const mcspInt *x_ind, __half2 *y,
                          const mcspComplexFloat *c, const mcspComplexFloat *s, mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}

mcspStatus_t mcspC16bfRoti(mcspHandle_t handle, mcspInt nnz, mcsp_bfloat162 *x_val, const mcspInt *x_ind,
                           mcsp_bfloat162 *y, const mcspComplexFloat *c, const mcspComplexFloat *s,
                           mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, nnz, x_val, x_ind, y, c, s, idx_base);
}
#endif

mcspStatus_t mcspCuinSroti(mcspHandle_t handle, int nnz, float *x_val, const int *x_ind, float *y, const float *c,
                         const float *s, mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, (mcspInt)nnz, x_val, (mcspInt *)x_ind, y, c, s, idx_base);
}

mcspStatus_t mcspCuinDroti(mcspHandle_t handle, int nnz, double *x_val, const int *x_ind, double *y, const double *c,
                         const double *s, mcsparseIndexBase_t idx_base) {
    return mcspRotiTemplate(handle, (mcspInt)nnz, x_val, (mcspInt *)x_ind, y, c, s, idx_base);
}

#ifdef __cplusplus
}
#endif