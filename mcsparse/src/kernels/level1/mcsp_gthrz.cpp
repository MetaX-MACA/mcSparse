#include "common/mcsp_types.h"
#include "gthrz_device.hpp"
#include "mcsp_handle.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType, typename valType>
mcspStatus_t mcspGthrzTemplate(mcspHandle_t handle, idxType nnz, valType *y, valType *x_val, const idxType *x_ind,
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
    mcLaunchKernelGGL((mcspGthrzKernel<block_size>), dim3(n_block), dim3(block_size), 0, stream, nnz, y, x_val, x_ind,
                       idx_base);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSgthrz(mcspHandle_t handle, mcspInt nnz, float *y, float *x_val, const mcspInt *x_ind,
                        mcsparseIndexBase_t idx_base) {
    return mcspGthrzTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

mcspStatus_t mcspDgthrz(mcspHandle_t handle, mcspInt nnz, double *y, double *x_val, const mcspInt *x_ind,
                        mcsparseIndexBase_t idx_base) {
    return mcspGthrzTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

mcspStatus_t mcspCgthrz(mcspHandle_t handle, mcspInt nnz, mcspComplexFloat *y, mcspComplexFloat *x_val,
                        const mcspInt *x_ind, mcsparseIndexBase_t idx_base) {
    return mcspGthrzTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

mcspStatus_t mcspZgthrz(mcspHandle_t handle, mcspInt nnz, mcspComplexDouble *y, mcspComplexDouble *x_val,
                        const mcspInt *x_ind, mcsparseIndexBase_t idx_base) {
    return mcspGthrzTemplate(handle, nnz, y, x_val, x_ind, idx_base);
}

mcspStatus_t mcspCuinSgthrz(mcspHandle_t handle, int nnz, float *y, float *x_val, const int *x_ind,
                          mcsparseIndexBase_t idx_base) {
    return mcspGthrzTemplate(handle, (mcspInt)nnz, y, x_val, (mcspInt *)x_ind, idx_base);
}

mcspStatus_t mcspCuinDgthrz(mcspHandle_t handle, int nnz, double *y, double *x_val, const int *x_ind,
                          mcsparseIndexBase_t idx_base) {
    return mcspGthrzTemplate(handle, (mcspInt)nnz, y, x_val, (mcspInt *)x_ind, idx_base);
}

mcspStatus_t mcspCuinCgthrz(mcspHandle_t handle, int nnz, mcspComplexFloat *y, mcspComplexFloat *x_val, const int *x_ind,
                          mcsparseIndexBase_t idx_base) {
    return mcspGthrzTemplate(handle, (mcspInt)nnz, y, x_val, (mcspInt *)x_ind, idx_base);
}

mcspStatus_t mcspCuinZgthrz(mcspHandle_t handle, int nnz, mcspComplexDouble *y, mcspComplexDouble *x_val,
                          const int *x_ind, mcsparseIndexBase_t idx_base) {
    return mcspGthrzTemplate(handle, (mcspInt)nnz, y, x_val, (mcspInt *)x_ind, idx_base);
}

#ifdef __cplusplus
}
#endif