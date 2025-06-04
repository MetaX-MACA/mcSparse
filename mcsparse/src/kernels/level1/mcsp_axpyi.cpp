#include "axpyi_device.hpp"
#include "common/mcsp_types.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE = 512, typename idxType, typename valType>
mcspStatus_t mcsp_axpyi_template(mcspHandle_t handle, idxType nnz, const valType* alpha, const valType* x_val,
                                 const idxType* x_ind, valType* y, mcsparseIndexBase_t idx_base) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || x_val == nullptr || x_ind == nullptr || y == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    valType h_alpha = getScalarToHost((valType*)alpha, handle->ptr_mode);
    if (nnz == 0 || h_alpha == static_cast<valType>(0)) {
        return MCSP_STATUS_SUCCESS;
    }

    mcStream_t stream = mcspGetStreamInternal(handle);

    dim3 blocks((nnz - 1) / BLOCKSIZE + 1);
    dim3 block_size(BLOCKSIZE);
    mcLaunchKernelGGL((axpyi_kernel<BLOCKSIZE>), blocks, block_size, 0, stream, nnz, h_alpha, x_val, x_ind, y,
                       idx_base);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSaxpyi(mcspHandle_t handle, mcspInt nnz, const float* alpha, const float* x_val, const mcspInt* x_ind,
                        float* y, mcsparseIndexBase_t idx_base) {
    return mcsp_axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

mcspStatus_t mcspDaxpyi(mcspHandle_t handle, mcspInt nnz, const double* alpha, const double* x_val,
                        const mcspInt* x_ind, double* y, mcsparseIndexBase_t idx_base) {
    return mcsp_axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

mcspStatus_t mcspCaxpyi(mcspHandle_t handle, mcspInt nnz, const mcspComplexFloat* alpha, const mcspComplexFloat* x_val,
                        const mcspInt* x_ind, mcspComplexFloat* y, mcsparseIndexBase_t idx_base) {
    return mcsp_axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

mcspStatus_t mcspZaxpyi(mcspHandle_t handle, mcspInt nnz, const mcspComplexDouble* alpha,
                        const mcspComplexDouble* x_val, const mcspInt* x_ind, mcspComplexDouble* y,
                        mcsparseIndexBase_t idx_base) {
    return mcsp_axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base);
}

mcspStatus_t mcspCuinSaxpyi(mcspHandle_t handle, int nnz, const float* alpha, const float* x_val, const int* x_ind,
                          float* y, mcsparseIndexBase_t idx_base) {
    return mcsp_axpyi_template(handle, (mcspInt)nnz, alpha, x_val, (mcspInt*)x_ind, y, idx_base);
}

mcspStatus_t mcspCuinDaxpyi(mcspHandle_t handle, int nnz, const double* alpha, const double* x_val, const int* x_ind,
                          double* y, mcsparseIndexBase_t idx_base) {
    return mcsp_axpyi_template(handle, (mcspInt)nnz, alpha, x_val, (mcspInt*)x_ind, y, idx_base);
}

mcspStatus_t mcspCuinCaxpyi(mcspHandle_t handle, int nnz, const mcspComplexFloat* alpha, const mcspComplexFloat* x_val,
                          const int* x_ind, mcspComplexFloat* y, mcsparseIndexBase_t idx_base) {
    return mcsp_axpyi_template(handle, (mcspInt)nnz, alpha, x_val, (mcspInt*)x_ind, y, idx_base);
}

mcspStatus_t mcspCuinZaxpyi(mcspHandle_t handle, int nnz, const mcspComplexDouble* alpha, const mcspComplexDouble* x_val,
                          const int* x_ind, mcspComplexDouble* y, mcsparseIndexBase_t idx_base) {
    return mcsp_axpyi_template(handle, (mcspInt)nnz, alpha, x_val, (mcspInt*)x_ind, y, idx_base);
}

#ifdef __cplusplus
}
#endif
