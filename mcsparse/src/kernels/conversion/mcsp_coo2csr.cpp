#include "common/mcsp_types.h"
#include "coo2csr_device.hpp"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType>
mcspStatus_t mcspCoo2CsrImpl(mcspHandle_t handle, const idxType *coo_rows, idxType nnz, idxType m, idxType *csr_rows,
                             mcsparseIndexBase_t idx_base) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (coo_rows == nullptr || csr_rows == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    int nElem = 512;
    int nBlock = ((nnz + 1) + nElem - 1) / nElem;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL(mcspCoo2CsrKernel, dim3(nBlock), dim3(nElem), 0, stream, nnz, m, coo_rows, csr_rows, idx_base);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCsr2CooImpl(mcspHandle_t handle, const idxType *csr_rows, idxType nnz, idxType m, idxType *coo_rows,
                             mcsparseIndexBase_t idx_base) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (coo_rows == nullptr || csr_rows == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    mcStream_t stream = mcspGetStreamInternal(handle);

    int nElem = 512;
    int nBlock = (m * WARP_SIZE + nElem - 1) / nElem;
    mcLaunchKernelGGL(mcspCsr2CooKernel, dim3(nBlock), dim3(nElem), 0, stream, m, csr_rows, coo_rows, idx_base);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspCoo2Csr(mcspHandle_t handle, const mcspInt *coo_rows, mcspInt nnz, mcspInt m, mcspInt *csr_rows,
                         mcsparseIndexBase_t idx_base) {
    return mcspCoo2CsrImpl(handle, coo_rows, nnz, m, csr_rows, idx_base);
}

mcspStatus_t mcspCoo2Csr64(mcspHandle_t handle, const int64_t *coo_rows, int64_t nnz, int64_t m, int64_t *csr_rows,
                           mcsparseIndexBase_t idx_base) {
    return mcspCoo2CsrImpl(handle, coo_rows, nnz, m, csr_rows, idx_base);
}

mcspStatus_t mcspCsr2Coo(mcspHandle_t handle, const mcspInt *csr_rows, mcspInt nnz, mcspInt m, mcspInt *coo_rows,
                         mcsparseIndexBase_t idx_base) {
    return mcspCsr2CooImpl(handle, csr_rows, nnz, m, coo_rows, idx_base);
}

mcspStatus_t mcspCsr2Coo64(mcspHandle_t handle, const int64_t *csr_rows, int64_t nnz, int64_t m, int64_t *coo_rows,
                           mcsparseIndexBase_t idx_base) {
    return mcspCsr2CooImpl(handle, csr_rows, nnz, m, coo_rows, idx_base);
}

mcspStatus_t mcspCuinXcoo2csr(mcspHandle_t handle, const int *coo_rows, int nnz, int m, int *csr_rows,
                            mcsparseIndexBase_t idx_base) {
    return mcspCoo2Csr(handle, (mcspInt *)coo_rows, (mcspInt)nnz, (mcspInt)m, (mcspInt *)csr_rows, idx_base);
}

mcspStatus_t mcspCuinXcsr2coo(mcspHandle_t handle, const int *csr_rows, int nnz, int m, int *coo_rows,
                            mcsparseIndexBase_t idx_base) {
    return mcspCsr2Coo(handle, (mcspInt *)csr_rows, (mcspInt)nnz, (mcspInt)m, (mcspInt *)coo_rows, idx_base);
}

#ifdef __cplusplus
}
#endif
