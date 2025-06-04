#include "common/mcsp_types.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"
#include "prim_types.h"
#include "sctr_device.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspCsr2csruTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz,
                                  const mcspMatDescr_t mcsp_descr_A, valType *csr_vals, const idxType *csr_rows,
                                  idxType *csr_cols, mcspCsru2csrInfo_t info, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (m == 0 || n == 0 || nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (csr_vals == nullptr || csr_rows == nullptr || csr_cols == nullptr || info == nullptr ||
        temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    void *buffer_head = temp_buffer;
    valType *sorted_csr_vals = reinterpret_cast<valType *>(buffer_head);
    buffer_head = (void *)(sorted_csr_vals + nnz);
    idxType *sorted_csr_cols = reinterpret_cast<idxType *>(buffer_head);
    idxType *perm = reinterpret_cast<idxType *>(info->perm);
    mcStream_t stream = mcspGetStreamInternal(handle);

    MACA_ASSERT(mcMemcpyAsync(sorted_csr_vals, csr_vals, nnz * sizeof(valType), mcMemcpyDeviceToDevice, stream));
    MACA_ASSERT(mcMemcpyAsync(sorted_csr_cols, csr_cols, nnz * sizeof(idxType), mcMemcpyDeviceToDevice, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    constexpr uint32_t block_size = 512;
    int n_block = (nnz + block_size - 1) / block_size;

    mcLaunchKernelGGL((mcspSctrKernel<block_size>), dim3(n_block), dim3(block_size), 0, stream, nnz, sorted_csr_vals,
                       perm, csr_vals, mcsp_descr_A->base);
    mcLaunchKernelGGL((mcspSctrKernel<block_size>), dim3(n_block), dim3(block_size), 0, stream, nnz, sorted_csr_cols,
                       perm, csr_cols, mcsp_descr_A->base);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsr2csru(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           float *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols, mcspCsru2csrInfo_t info,
                           void *temp_buffer) {
    return mcspCsr2csruTemplate(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
}

mcspStatus_t mcspDcsr2csru(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           double *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols, mcspCsru2csrInfo_t info,
                           void *temp_buffer) {
    return mcspCsr2csruTemplate(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCcsr2csru(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           mcspComplexFloat *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                           mcspCsru2csrInfo_t info, void *temp_buffer) {
    return mcspCsr2csruTemplate(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
}

mcspStatus_t mcspZcsr2csru(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           mcspComplexDouble *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                           mcspCsru2csrInfo_t info, void *temp_buffer) {
    return mcspCsr2csruTemplate(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCuinScsr2csru(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             float *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer) {
    return mcspCsr2csruTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, mcsp_descr_A, csr_vals,
                                (mcspInt *)csr_rows, (mcspInt *)csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCuinDcsr2csru(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             double *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer) {
    return mcspCsr2csruTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, mcsp_descr_A, csr_vals,
                                (mcspInt *)csr_rows, (mcspInt *)csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCuinCcsr2csru(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             mcspComplexFloat *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer) {
    return mcspCsr2csruTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, mcsp_descr_A, csr_vals,
                                (mcspInt *)csr_rows, (mcspInt *)csr_cols, info, temp_buffer);
}

mcspStatus_t mcspCuinZcsr2csru(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             mcspComplexDouble *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer) {
    return mcspCsr2csruTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, mcsp_descr_A, csr_vals,
                                (mcspInt *)csr_rows, (mcspInt *)csr_cols, info, temp_buffer);
}

#ifdef __cplusplus
}
#endif