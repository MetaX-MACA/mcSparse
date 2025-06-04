#include "common/mcsp_types.h"
#include "csr2ell_device.hpp"
#include "device_reduce.hpp"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType, typename valType>
mcspStatus_t mcspCsr2EllTemplate(mcspHandle_t handle, idxType m, const mcspMatDescr_t csr_descr,
                                 const valType *csr_vals, const idxType *csr_rows, const idxType *csr_cols,
                                 const mcspMatDescr_t ell_descr, idxType ell_width, valType *ell_vals,
                                 idxType *ell_cols) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if ((csr_descr == nullptr) || (ell_descr == nullptr) || (csr_rows == nullptr) || (csr_vals == nullptr) ||
        (csr_rows == nullptr) || (ell_vals == nullptr) || (ell_cols == nullptr)) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if ((csr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) || (ell_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    int n_elem = 512;
    int n_block = (m + n_elem / WARP_SIZE - 1) / (n_elem / WARP_SIZE);
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcLaunchKernelGGL(mcspCsr2EllKernel, dim3(n_block), dim3(n_elem), 0, stream, m, csr_vals, csr_rows, csr_cols,
                       csr_descr->base, ell_width, ell_vals, ell_cols, ell_descr->base);

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCsr2EllWidthTemplate(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr,
                                      const mcspInt *csr_rows, const mcspMatDescr_t ell_descr, mcspInt *ell_width) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if ((csr_descr == nullptr) || (ell_descr == nullptr) || (csr_rows == nullptr) || (ell_width == nullptr)) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if ((csr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) || (ell_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

#define CSR2ELL_BLOCKSIZE 512
#define CSR2ELL_LOOP_CNT 4
    mcspInt nElem = CSR2ELL_BLOCKSIZE;
    mcspInt nBlock = (m + (nElem * CSR2ELL_LOOP_CNT) - 1) / (nElem * CSR2ELL_LOOP_CNT);

    mcspInt buffersize;
    mcspInt *workspace;
    mcspInt *reduce_ws;
    mcspInt *reduce_buffer;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::reduce(nullptr, buffersize, workspace, workspace, nBlock, mcprim::maximum<mcspInt>(), stream);
    mcspInt total_size = buffersize + (nBlock + 1) * sizeof(mcspInt);
    void *tmp_buffer;
    bool use_buffer_pool = handle->mcspUsePoolBuffer(&tmp_buffer, total_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc(&tmp_buffer, total_size));
    }
    workspace = (mcspInt *)tmp_buffer;
    reduce_ws = workspace + nBlock;
    reduce_buffer = workspace + 1;

    mcLaunchKernelGGL((mcspCsr2EllWidthKernel<CSR2ELL_BLOCKSIZE, CSR2ELL_LOOP_CNT>), dim3(nBlock), dim3(nElem),
                       nElem * sizeof(*csr_rows), 0, m, csr_rows, workspace);
    MACA_ASSERT(mcStreamSynchronize(stream));
#undef CSR2ELL_LOOP_CNT
#undef CSR2ELL_BLOCKSIZE
    mcprim::reduce(reduce_buffer, buffersize, workspace, reduce_ws, nBlock, mcprim::maximum<mcspInt>(), stream);
    MACA_ASSERT(mcMemcpyAsync(ell_width, reduce_ws, sizeof(*ell_width), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(tmp_buffer));
    }

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspCsr2EllWidth(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr, const mcspInt *csr_rows,
                              const mcspMatDescr_t ell_descr, mcspInt *ell_width) {
    return mcspCsr2EllWidthTemplate(handle, m, csr_descr, csr_rows, ell_descr, ell_width);
}

mcspStatus_t mcspScsr2Ell(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr, const float *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspMatDescr_t ell_descr,
                          mcspInt ell_width, float *ell_vals, mcspInt *ell_cols) {
    return mcspCsr2EllTemplate(handle, m, csr_descr, csr_vals, csr_rows, csr_cols, ell_descr, ell_width, ell_vals,
                               ell_cols);
}

mcspStatus_t mcspDcsr2Ell(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr, const double *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspMatDescr_t ell_descr,
                          mcspInt ell_width, double *ell_vals, mcspInt *ell_cols) {
    return mcspCsr2EllTemplate(handle, m, csr_descr, csr_vals, csr_rows, csr_cols, ell_descr, ell_width, ell_vals,
                               ell_cols);
}

mcspStatus_t mcspCcsr2Ell(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr,
                          const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                          const mcspMatDescr_t ell_descr, mcspInt ell_width, mcspComplexFloat *ell_vals,
                          mcspInt *ell_cols) {
    return mcspCsr2EllTemplate(handle, m, csr_descr, csr_vals, csr_rows, csr_cols, ell_descr, ell_width, ell_vals,
                               ell_cols);
}

mcspStatus_t mcspZcsr2Ell(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr,
                          const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                          const mcspMatDescr_t ell_descr, mcspInt ell_width, mcspComplexDouble *ell_vals,
                          mcspInt *ell_cols) {
    return mcspCsr2EllTemplate(handle, m, csr_descr, csr_vals, csr_rows, csr_cols, ell_descr, ell_width, ell_vals,
                               ell_cols);
}

#ifdef __cplusplus
}
#endif
