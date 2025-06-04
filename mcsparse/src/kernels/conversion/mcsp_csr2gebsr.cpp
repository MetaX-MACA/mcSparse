#include "common/mcsp_types.h"
#include "csr2gebsr_device.hpp"
#include "device_radix_sort.hpp"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspCsr2gebsrBufferSizeTemplate(mcspHandle_t handle, mcsparseDirection_t dir, idxType m, idxType n,
                                             const mcspMatDescr_t csr_descr, const valType *csr_val,
                                             const idxType *csr_row, const idxType *csr_col, idxType row_block_dim,
                                             idxType col_block_dim, size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || row_block_dim < 0 || col_block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (csr_descr == nullptr || csr_val == nullptr || csr_row == nullptr || csr_col == nullptr ||
        buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    // Check matrix sorting mode
    if (csr_descr->storage_mode != MCSPARSE_STORAGE_MODE_SORTED) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (csr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (m == 0 || n == 0 || row_block_dim == 0 || col_block_dim == 0) {
        *buffer_size = ALIGNED_SIZE;
        return MCSP_STATUS_SUCCESS;
    }
    idxType bsr_mb = (m + row_block_dim - 1) / row_block_dim;
    mcStream_t stream = mcspGetStreamInternal(handle);
    // temporary csr row start indexes and end indexes for each process
    *buffer_size = 2 * bsr_mb * ALIGN(sizeof(idxType) * row_block_dim, ALIGNED_SIZE);
    *buffer_size +=
        bsr_mb * ALIGN(sizeof(idxType) * row_block_dim, ALIGNED_SIZE);  // temporary csr col indexes for each process
    *buffer_size +=
        bsr_mb * ALIGN(sizeof(valType) * row_block_dim, ALIGNED_SIZE);  // temporary csr values for each process
    idxType scan_buffer_size;
    mcprim::inclusive_scan(nullptr, scan_buffer_size, (idxType *)nullptr, (idxType *)nullptr, bsr_mb + 1, stream);
    *buffer_size += ALIGN(scan_buffer_size, ALIGNED_SIZE);                // scan output buffer
    *buffer_size += ALIGN(sizeof(idxType) * (bsr_mb + 1), ALIGNED_SIZE);  // temporary bsr rows buffer
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsr2gebsr_bufferSizeTemplate(mcspHandle_t handle, mcsparseDirection_t dir, idxType m, idxType n,
                                              const mcspMatDescr_t csr_descr, const valType *csr_val,
                                              const idxType *csr_row, const idxType *csr_col, idxType row_block_dim,
                                              idxType col_block_dim, int *buffer_size) {
    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    size_t temp_size = 0;
    mcspStatus_t ret = mcspCsr2gebsrBufferSizeTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col,
                                                       row_block_dim, col_block_dim, &temp_size);
    *buffer_size = (int)temp_size;
    return ret;
}

template <typename idxType>
mcspStatus_t mcspCsr2gebsrNnzTemplate(mcspHandle_t handle, mcsparseDirection_t dir, idxType m, idxType n,
                                      const mcspMatDescr_t csr_descr, const idxType *csr_row, const idxType *csr_col,
                                      const mcspMatDescr_t bsr_descr, idxType *bsr_row, idxType row_block_dim,
                                      idxType col_block_dim, idxType *nnzb, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || row_block_dim < 0 || col_block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (csr_descr == nullptr || csr_row == nullptr || csr_col == nullptr || bsr_descr == nullptr ||
        bsr_row == nullptr || nnzb == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    // Check matrix sorting mode
    if (csr_descr->storage_mode != MCSPARSE_STORAGE_MODE_SORTED) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (csr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL || bsr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    mcStream_t stream = mcspGetStreamInternal(handle);
    if (m == 0 || n == 0 || row_block_dim == 0 || col_block_dim == 0) {
        if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
            *nnzb = 0;
        } else {
            MACA_ASSERT(mcMemsetAsync(nnzb, 0, sizeof(*nnzb), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
        }
        return MCSP_STATUS_SUCCESS;
    }

    idxType bsr_mb = (m + row_block_dim - 1) / row_block_dim;
    idxType bsr_nb = (n + col_block_dim - 1) / col_block_dim;
    constexpr idxType block_size = 512;
    idxType blocks = (bsr_mb + block_size - 1) / block_size;
    mcLaunchKernelGGL((mcspCsr2BsrNnzKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, m, n, bsr_mb,
                       bsr_nb, csr_descr->base, csr_row, csr_col, bsr_descr->base, bsr_row, row_block_dim,
                       col_block_dim, temp_buffer);
    MACA_ASSERT(mcStreamSynchronize(stream));

    char *ptr = reinterpret_cast<char *>(temp_buffer);
    idxType *temp_bsr_row = reinterpret_cast<idxType *>(ptr);

    ptr += ALIGN(sizeof(idxType) * (bsr_mb + 1), ALIGNED_SIZE);
    idxType buffer_size;
    mcprim::inclusive_scan(nullptr, buffer_size, bsr_row, temp_bsr_row, bsr_mb + 1, stream);
    void *scan_buffer = ptr;
    mcprim::inclusive_scan(scan_buffer, buffer_size, bsr_row, temp_bsr_row, bsr_mb + 1, stream);
    MACA_ASSERT(mcMemcpyAsync(bsr_row, temp_bsr_row, sizeof(idxType) * (bsr_mb + 1), mcMemcpyDeviceToDevice, stream));
    idxType nnz, nnz_;
    MACA_ASSERT(mcMemcpyAsync(&nnz, &bsr_row[bsr_mb], sizeof(*bsr_row), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcMemcpyAsync(&nnz_, &bsr_row[0], sizeof(*bsr_row), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    nnz -= nnz_;

    if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
        *nnzb = nnz;
    } else {
        MACA_ASSERT(mcMemcpyAsync(nnzb, &nnz, sizeof(idxType), mcMemcpyHostToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsr2gebsrTemplate(mcspHandle_t handle, mcsparseDirection_t dir, idxType m, idxType n,
                                   const mcspMatDescr_t csr_descr, const valType *csr_val, const idxType *csr_row,
                                   const idxType *csr_col, const mcspMatDescr_t bsr_descr, valType *bsr_val,
                                   idxType *bsr_row, idxType *bsr_col, idxType row_block_dim, idxType col_block_dim,
                                   void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || row_block_dim < 0 || col_block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (csr_descr == nullptr || csr_val == nullptr || csr_row == nullptr || csr_col == nullptr ||
        bsr_descr == nullptr || bsr_val == nullptr || bsr_row == nullptr || bsr_col == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    // Check matrix sorting mode
    if (csr_descr->storage_mode != MCSPARSE_STORAGE_MODE_SORTED) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (csr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL || bsr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (m == 0 || n == 0 || row_block_dim == 0 || col_block_dim == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    idxType bsr_mb = (m + row_block_dim - 1) / row_block_dim;
    idxType bsr_nb = (n + col_block_dim - 1) / col_block_dim;
    idxType nnzb, nnzb_;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemcpyAsync(&nnzb, &bsr_row[bsr_mb], sizeof(*bsr_row), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcMemcpyAsync(&nnzb_, &bsr_row[0], sizeof(*bsr_row), mcMemcpyDeviceToHost, stream));
    nnzb -= nnzb_;
    MACA_ASSERT(mcMemsetAsync(bsr_val, 0, nnzb * row_block_dim * col_block_dim * sizeof(*bsr_val), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    constexpr idxType block_size = 512;
    idxType blocks = (bsr_mb + block_size - 1) / block_size;

    mcLaunchKernelGGL((mcspCsr2BsrKernel<block_size>), dim3(blocks), dim3(block_size), 0, stream, m, n, bsr_mb, bsr_nb,
                       csr_descr->base, csr_val, csr_row, csr_col, bsr_descr->base, bsr_val, bsr_row, bsr_col,
                       row_block_dim, col_block_dim, dir, temp_buffer);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCsr2bsrNnzTemplate(mcspHandle_t handle, mcsparseDirection_t dir, idxType m, idxType n,
                                    const mcspMatDescr_t csr_descr, const idxType *csr_row, const idxType *csr_col,
                                    idxType block_dim, const mcspMatDescr_t bsr_descr, idxType *bsr_row,
                                    idxType *nnzb) {
    if (block_dim == 0 || block_dim > n || block_dim > m) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    mcspStatus_t cur_stat = MCSP_STATUS_SUCCESS;
    size_t buffer_size = 0;

    cur_stat = mcspCsr2gebsrBufferSizeTemplate(handle, dir, m, n, csr_descr, (float *)csr_row, csr_row, csr_col,
                                               block_dim, block_dim, &buffer_size);
    if (cur_stat != MCSP_STATUS_SUCCESS) {
        return cur_stat;
    }

    void *temp_buffer = nullptr;
    bool use_buffer_pool = handle->mcspUsePoolBuffer((void **)&temp_buffer, buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc((void **)&temp_buffer, buffer_size));
    }

    cur_stat = mcspCsr2gebsrNnzTemplate(handle, dir, m, n, csr_descr, csr_row, csr_col, bsr_descr, bsr_row, block_dim,
                                        block_dim, nnzb, temp_buffer);

    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(temp_buffer));
    }

    return cur_stat;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsr2bsrTemplate(mcspHandle_t handle, mcsparseDirection_t dir, idxType m, idxType n,
                                 const mcspMatDescr_t csr_descr, const valType *csr_val, const idxType *csr_row,
                                 const idxType *csr_col, idxType block_dim, const mcspMatDescr_t bsr_descr,
                                 valType *bsr_val, idxType *bsr_row, idxType *bsr_col) {
    if (block_dim == 0 || block_dim > n || block_dim > m) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    mcspStatus_t cur_stat = MCSP_STATUS_SUCCESS;
    size_t buffer_size = 0;

    cur_stat = mcspCsr2gebsrBufferSizeTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, block_dim,
                                               block_dim, &buffer_size);
    if (cur_stat != MCSP_STATUS_SUCCESS) {
        return cur_stat;
    }

    void *temp_buffer = nullptr;
    bool use_buffer_pool = handle->mcspUsePoolBuffer((void **)&temp_buffer, buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc((void **)&temp_buffer, buffer_size));
    }

    cur_stat = mcspCsr2gebsrTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, bsr_descr, bsr_val,
                                     bsr_row, bsr_col, block_dim, block_dim, temp_buffer);

    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(temp_buffer));
    }

    return cur_stat;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsr2gebsrBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t csr_descr, const float *csr_val, const mcspInt *csr_row,
                                      const mcspInt *csr_col, mcspInt row_block_dim, mcspInt col_block_dim,
                                      size_t *buffer_size) {
    return mcspCsr2gebsrBufferSizeTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, row_block_dim,
                                           col_block_dim, buffer_size);
}

mcspStatus_t mcspDcsr2gebsrBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t csr_descr, const double *csr_val, const mcspInt *csr_row,
                                      const mcspInt *csr_col, mcspInt row_block_dim, mcspInt col_block_dim,
                                      size_t *buffer_size) {
    return mcspCsr2gebsrBufferSizeTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, row_block_dim,
                                           col_block_dim, buffer_size);
}

mcspStatus_t mcspCcsr2gebsrBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val,
                                      const mcspInt *csr_row, const mcspInt *csr_col, mcspInt row_block_dim,
                                      mcspInt col_block_dim, size_t *buffer_size) {
    return mcspCsr2gebsrBufferSizeTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, row_block_dim,
                                           col_block_dim, buffer_size);
}

mcspStatus_t mcspZcsr2gebsrBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val,
                                      const mcspInt *csr_row, const mcspInt *csr_col, mcspInt row_block_dim,
                                      mcspInt col_block_dim, size_t *buffer_size) {
    return mcspCsr2gebsrBufferSizeTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, row_block_dim,
                                           col_block_dim, buffer_size);
}

mcspStatus_t mcspCsr2gebsrNnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                              const mcspMatDescr_t csr_descr, const mcspInt *csr_row, const mcspInt *csr_col,
                              const mcspMatDescr_t bsr_descr, mcspInt *bsr_row, mcspInt row_block_dim,
                              mcspInt col_block_dim, mcspInt *nnzb, void *temp_buffer) {
    return mcspCsr2gebsrNnzTemplate(handle, dir, m, n, csr_descr, csr_row, csr_col, bsr_descr, bsr_row, row_block_dim,
                                    col_block_dim, nnzb, temp_buffer);
}

mcspStatus_t mcspScsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const float *csr_val, const mcspInt *csr_row,
                            const mcspInt *csr_col, const mcspMatDescr_t bsr_descr, float *bsr_val, mcspInt *bsr_row,
                            mcspInt *bsr_col, mcspInt row_block_dim, mcspInt col_block_dim, void *temp_buffer) {
    return mcspCsr2gebsrTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, bsr_descr, bsr_val, bsr_row,
                                 bsr_col, row_block_dim, col_block_dim, temp_buffer);
}

mcspStatus_t mcspDcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const double *csr_val, const mcspInt *csr_row,
                            const mcspInt *csr_col, const mcspMatDescr_t bsr_descr, double *bsr_val, mcspInt *bsr_row,
                            mcspInt *bsr_col, mcspInt row_block_dim, mcspInt col_block_dim, void *temp_buffer) {
    return mcspCsr2gebsrTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, bsr_descr, bsr_val, bsr_row,
                                 bsr_col, row_block_dim, col_block_dim, temp_buffer);
}

mcspStatus_t mcspCcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val, const mcspInt *csr_row,
                            const mcspInt *csr_col, const mcspMatDescr_t bsr_descr, mcspComplexFloat *bsr_val,
                            mcspInt *bsr_row, mcspInt *bsr_col, mcspInt row_block_dim, mcspInt col_block_dim,
                            void *temp_buffer) {
    return mcspCsr2gebsrTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, bsr_descr, bsr_val, bsr_row,
                                 bsr_col, row_block_dim, col_block_dim, temp_buffer);
}

mcspStatus_t mcspZcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val, const mcspInt *csr_row,
                            const mcspInt *csr_col, const mcspMatDescr_t bsr_descr, mcspComplexDouble *bsr_val,
                            mcspInt *bsr_row, mcspInt *bsr_col, mcspInt row_block_dim, mcspInt col_block_dim,
                            void *temp_buffer) {
    return mcspCsr2gebsrTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, bsr_descr, bsr_val, bsr_row,
                                 bsr_col, row_block_dim, col_block_dim, temp_buffer);
}

// csr2bsr
mcspStatus_t mcspCsr2bsrNnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const mcspInt *csr_row, const mcspInt *csr_col,
                            mcspInt block_dim, const mcspMatDescr_t bsr_descr, mcspInt *bsr_row, mcspInt *nnzb) {
    return mcspCsr2bsrNnzTemplate(handle, dir, m, n, csr_descr, csr_row, csr_col, block_dim, bsr_descr, bsr_row, nnzb);
}

mcspStatus_t mcspScsr2bsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                          const mcspMatDescr_t csr_descr, const float *csr_val, const mcspInt *csr_row,
                          const mcspInt *csr_col, mcspInt block_dim, const mcspMatDescr_t bsr_descr, float *bsr_val,
                          mcspInt *bsr_row, mcspInt *bsr_col) {
    return mcspCsr2bsrTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, block_dim, bsr_descr, bsr_val,
                               bsr_row, bsr_col);
}

mcspStatus_t mcspDcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                          const mcspMatDescr_t csr_descr, const double *csr_val, const mcspInt *csr_row,
                          const mcspInt *csr_col, mcspInt block_dim, const mcspMatDescr_t bsr_descr, double *bsr_val,
                          mcspInt *bsr_row, mcspInt *bsr_col) {
    return mcspCsr2bsrTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, block_dim, bsr_descr, bsr_val,
                               bsr_row, bsr_col);
}

mcspStatus_t mcspCcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                          const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val, const mcspInt *csr_row,
                          const mcspInt *csr_col, mcspInt block_dim, const mcspMatDescr_t bsr_descr,
                          mcspComplexFloat *bsr_val, mcspInt *bsr_row, mcspInt *bsr_col) {
    return mcspCsr2bsrTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, block_dim, bsr_descr, bsr_val,
                               bsr_row, bsr_col);
}

mcspStatus_t mcspZcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                          const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val, const mcspInt *csr_row,
                          const mcspInt *csr_col, mcspInt block_dim, const mcspMatDescr_t bsr_descr,
                          mcspComplexDouble *bsr_val, mcspInt *bsr_row, mcspInt *bsr_col) {
    return mcspCsr2bsrTemplate(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, block_dim, bsr_descr, bsr_val,
                               bsr_row, bsr_col);
}

mcspStatus_t mcspCuinScsr2gebsr_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                         const mcspMatDescr_t csr_descr, const float *csr_val, const int *csr_row,
                                         const int *csr_col, int row_block_dim, int col_block_dim, int *buffer_size) {
    return mcspCsr2gebsr_bufferSizeTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                            (mcspInt *)csr_col, (mcspInt)row_block_dim, (mcspInt)col_block_dim,
                                            buffer_size);
}

mcspStatus_t mcspCuinDcsr2gebsr_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                         const mcspMatDescr_t csr_descr, const double *csr_val, const int *csr_row,
                                         const int *csr_col, int row_block_dim, int col_block_dim, int *buffer_size) {
    return mcspCsr2gebsr_bufferSizeTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                            (mcspInt *)csr_col, (mcspInt)row_block_dim, (mcspInt)col_block_dim,
                                            buffer_size);
}

mcspStatus_t mcspCuinCcsr2gebsr_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                         const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val,
                                         const int *csr_row, const int *csr_col, int row_block_dim, int col_block_dim,
                                         int *buffer_size) {
    return mcspCsr2gebsr_bufferSizeTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                            (mcspInt *)csr_col, (mcspInt)row_block_dim, (mcspInt)col_block_dim,
                                            buffer_size);
}

mcspStatus_t mcspCuinZcsr2gebsr_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                         const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val,
                                         const int *csr_row, const int *csr_col, int row_block_dim, int col_block_dim,
                                         int *buffer_size) {
    return mcspCsr2gebsr_bufferSizeTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                            (mcspInt *)csr_col, (mcspInt)row_block_dim, (mcspInt)col_block_dim,
                                            buffer_size);
}

mcspStatus_t mcspCuinScsr2gebsr_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                            const mcspMatDescr_t csr_descr, const float *csr_val, const int *csr_row,
                                            const int *csr_col, int row_block_dim, int col_block_dim,
                                            size_t *buffer_size) {
    return mcspCsr2gebsrBufferSizeTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                           (mcspInt *)csr_col, (mcspInt)row_block_dim, (mcspInt)col_block_dim,
                                           buffer_size);
}

mcspStatus_t mcspCuinDcsr2gebsr_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                            const mcspMatDescr_t csr_descr, const double *csr_val, const int *csr_row,
                                            const int *csr_col, int row_block_dim, int col_block_dim,
                                            size_t *buffer_size) {
    return mcspCsr2gebsrBufferSizeTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                           (mcspInt *)csr_col, (mcspInt)row_block_dim, (mcspInt)col_block_dim,
                                           buffer_size);
}

mcspStatus_t mcspCuinCcsr2gebsr_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                            const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val,
                                            const int *csr_row, const int *csr_col, int row_block_dim,
                                            int col_block_dim, size_t *buffer_size) {
    return mcspCsr2gebsrBufferSizeTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                           (mcspInt *)csr_col, (mcspInt)row_block_dim, (mcspInt)col_block_dim,
                                           buffer_size);
}

mcspStatus_t mcspCuinZcsr2gebsr_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                            const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val,
                                            const int *csr_row, const int *csr_col, int row_block_dim,
                                            int col_block_dim, size_t *buffer_size) {
    return mcspCsr2gebsrBufferSizeTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                           (mcspInt *)csr_col, (mcspInt)row_block_dim, (mcspInt)col_block_dim,
                                           buffer_size);
}

mcspStatus_t mcspCuinXcsr2gebsrNnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                 const mcspMatDescr_t csr_descr, const int *csr_row, const int *csr_col,
                                 const mcspMatDescr_t bsr_descr, int *bsr_row, int row_block_dim, int col_block_dim,
                                 int *nnzb, void *temp_buffer) {
    return mcspCsr2gebsrNnzTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, (mcspInt *)csr_row,
                                    (mcspInt *)csr_col, bsr_descr, (mcspInt *)bsr_row, (mcspInt)row_block_dim,
                                    (mcspInt)col_block_dim, (mcspInt *)nnzb, temp_buffer);
}

mcspStatus_t mcspCuinScsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcspMatDescr_t csr_descr, const float *csr_val, const int *csr_row,
                              const int *csr_col, const mcspMatDescr_t bsr_descr, float *bsr_val, int *bsr_row,
                              int *bsr_col, int row_block_dim, int col_block_dim, void *temp_buffer) {
    return mcspCsr2gebsrTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                 (mcspInt *)csr_col, bsr_descr, bsr_val, (mcspInt *)bsr_row, (mcspInt *)bsr_col,
                                 (mcspInt)row_block_dim, (mcspInt)col_block_dim, temp_buffer);
}

mcspStatus_t mcspCuinDcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcspMatDescr_t csr_descr, const double *csr_val, const int *csr_row,
                              const int *csr_col, const mcspMatDescr_t bsr_descr, double *bsr_val, int *bsr_row,
                              int *bsr_col, int row_block_dim, int col_block_dim, void *temp_buffer) {
    return mcspCsr2gebsrTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                 (mcspInt *)csr_col, bsr_descr, bsr_val, (mcspInt *)bsr_row, (mcspInt *)bsr_col,
                                 (mcspInt)row_block_dim, (mcspInt)col_block_dim, temp_buffer);
}

mcspStatus_t mcspCuinCcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val, const int *csr_row,
                              const int *csr_col, const mcspMatDescr_t bsr_descr, mcspComplexFloat *bsr_val,
                              int *bsr_row, int *bsr_col, int row_block_dim, int col_block_dim, void *temp_buffer) {
    return mcspCsr2gebsrTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                 (mcspInt *)csr_col, bsr_descr, bsr_val, (mcspInt *)bsr_row, (mcspInt *)bsr_col,
                                 (mcspInt)row_block_dim, (mcspInt)col_block_dim, temp_buffer);
}

mcspStatus_t mcspCuinZcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val, const int *csr_row,
                              const int *csr_col, const mcspMatDescr_t bsr_descr, mcspComplexDouble *bsr_val,
                              int *bsr_row, int *bsr_col, int row_block_dim, int col_block_dim, void *temp_buffer) {
    return mcspCsr2gebsrTemplate(handle, dir, (mcspInt)m, (mcspInt)n, csr_descr, csr_val, (mcspInt *)csr_row,
                                 (mcspInt *)csr_col, bsr_descr, bsr_val, (mcspInt *)bsr_row, (mcspInt *)bsr_col,
                                 (mcspInt)row_block_dim, (mcspInt)col_block_dim, temp_buffer);
}

mcspStatus_t mcspCuinXcsr2bsrNnz(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                               const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                               const mcspMatDescr_t descrC, int *bsrSortedRowPtrC, int *nnzTotalDevHostPtr) {
    return mcspCsr2bsrNnz(handle, dirA, (mcspInt)m, (mcspInt)n, descrA, (mcspInt *)csrSortedRowPtrA,
                          (mcspInt *)csrSortedColIndA, (mcspInt)blockDim, descrC, (mcspInt *)bsrSortedRowPtrC,
                          (mcspInt *)nnzTotalDevHostPtr);
}

mcspStatus_t mcspCuinScsr2bsr(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                            const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                            int blockDim, const mcspMatDescr_t descrC, float *bsrSortedValC, int *bsrSortedRowPtrC,
                            int *bsrSortedColIndC) {
    return mcspScsr2bsr(handle, dirA, (mcspInt)m, (mcspInt)n, descrA, csrSortedValA, (mcspInt *)csrSortedRowPtrA,
                        (mcspInt *)csrSortedColIndA, (mcspInt)blockDim, descrC, bsrSortedValC,
                        (mcspInt *)bsrSortedRowPtrC, (mcspInt *)bsrSortedColIndC);
}

mcspStatus_t mcspCuinDcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                            const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                            int blockDim, const mcspMatDescr_t descrC, double *bsrSortedValC, int *bsrSortedRowPtrC,
                            int *bsrSortedColIndC) {
    return mcspDcsr2bsr(handle, dirA, (mcspInt)m, (mcspInt)n, descrA, csrSortedValA, (mcspInt *)csrSortedRowPtrA,
                        (mcspInt *)csrSortedColIndA, (mcspInt)blockDim, descrC, bsrSortedValC,
                        (mcspInt *)bsrSortedRowPtrC, (mcspInt *)bsrSortedColIndC);
}

mcspStatus_t mcspCuinCcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                            const mcFloatComplex *csrSortedValA, const int *csrSortedRowPtrA,
                            const int *csrSortedColIndA, int blockDim, const mcspMatDescr_t descrC,
                            mcFloatComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
    return mcspCcsr2bsr(handle, dirA, (mcspInt)m, (mcspInt)n, descrA, csrSortedValA, (mcspInt *)csrSortedRowPtrA,
                        (mcspInt *)csrSortedColIndA, (mcspInt)blockDim, descrC, bsrSortedValC,
                        (mcspInt *)bsrSortedRowPtrC, (mcspInt *)bsrSortedColIndC);
}

mcspStatus_t mcspCuinZcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                            const mcDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                            const int *csrSortedColIndA, int blockDim, const mcspMatDescr_t descrC,
                            mcDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
    return mcspZcsr2bsr(handle, dirA, (mcspInt)m, (mcspInt)n, descrA, csrSortedValA, (mcspInt *)csrSortedRowPtrA,
                        (mcspInt *)csrSortedColIndA, (mcspInt)blockDim, descrC, bsrSortedValC,
                        (mcspInt *)bsrSortedRowPtrC, (mcspInt *)bsrSortedColIndC);
}
#ifdef __cplusplus
}
#endif