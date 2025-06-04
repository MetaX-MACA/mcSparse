#include "common/mcsp_types.h"
#include "device_radix_sort.hpp"
#include "gebsr2gebsc_device.hpp"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType, typename valType>
mcspStatus_t mcspGebsr2gebscBufferSizeExtTemplate(mcspHandle_t handle, idxType mb, idxType nb, idxType nnzb,
                                                  const valType* bsr_vals, const idxType* bsr_rows,
                                                  const idxType* bsr_cols, idxType row_block_dim, idxType col_block_dim,
                                                  size_t* buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nb < 0 || mb < 0 || nnzb < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (bsr_rows == nullptr || bsr_cols == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (nb == 0 || mb == 0 || nnzb == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    constexpr unsigned int bsr2bsc_buffer_block = 4;
    idxType tmp_buffersize = 0;
    idxType* tmp_ptr = nullptr;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::radix_sort_pairs(nullptr, tmp_buffersize, bsr_cols, tmp_ptr, bsr_cols, tmp_ptr, nnzb, stream);
    *buffer_size = tmp_buffersize;
    *buffer_size += (bsr2bsc_buffer_block * nnzb * sizeof(*bsr_rows));

    *buffer_size = *buffer_size < MIN_BUFFER_SIZE ? MIN_BUFFER_SIZE : *buffer_size;

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGebsr2gebscBufferSizeTemplate(mcspHandle_t handle, idxType mb, idxType nb, idxType nnzb,
                                               const valType* bsr_vals, const idxType* bsr_rows,
                                               const idxType* bsr_cols, idxType row_block_dim, idxType col_block_dim,
                                               int* buffer_size) {
    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    size_t temp_size = MIN_BUFFER_SIZE;
    mcspStatus_t ret = mcspGebsr2gebscBufferSizeExtTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols,
                                                            row_block_dim, col_block_dim, &temp_size);
    *buffer_size = (int)temp_size;
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspGebsr2gebscTemplate(mcspHandle_t handle, idxType mb, idxType nb, idxType nnzb, const valType* bsr_vals,
                                     const idxType* bsr_rows, const idxType* bsr_cols, idxType row_block_dim,
                                     idxType col_block_dim, valType* bsc_vals, idxType* bsc_rows, idxType* bsc_cols,
                                     mcsparseAction_t copy_value, mcsparseIndexBase_t idx_base, void* temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nb < 0 || mb < 0 || nnzb < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (nb == 0 || mb == 0 || nnzb == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows == nullptr || bsr_cols == nullptr || bsc_vals == nullptr ||
        bsc_rows == nullptr || bsc_cols == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType* buffer_ptr = nullptr;
    idxType tmp_buffersize = 0;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::radix_sort_pairs(nullptr, tmp_buffersize, bsc_cols, bsc_rows, bsc_cols, bsc_rows, nnzb, stream);

    buffer_ptr = reinterpret_cast<idxType*>(temp_buffer);

    idxType* ibuffer = buffer_ptr;
    buffer_ptr += nnzb;

    idxType* sorted_ibuffer = buffer_ptr;
    buffer_ptr += nnzb;

    idxType* temp_coo_rows = buffer_ptr;
    buffer_ptr += nnzb;

    idxType* temp_bsr_cols = buffer_ptr;
    buffer_ptr += nnzb;

    mcspInt start_bit = 0;
    mcspInt end_bit = getHighBitLocOneBase(nb);

    if (copy_value == MCSPARSE_ACTION_NUMERIC) {
        mcspCreateIdentityPermutation(handle, nnzb, ibuffer);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, bsr_cols, temp_bsr_cols, ibuffer, sorted_ibuffer, nnzb,
                                 stream, start_bit, end_bit);
        mcspCoo2Csr(handle, temp_bsr_cols, nnzb, nb, bsc_cols, idx_base);
        mcspCsr2Coo(handle, bsr_rows, nnzb, mb, temp_coo_rows, idx_base);
        constexpr uint32_t n_elem = 512;
        uint32_t total_nnz = nnzb * row_block_dim * col_block_dim;
        uint32_t block_nnz = row_block_dim * col_block_dim;
        uint32_t n_block = (total_nnz + n_elem - 1) / n_elem;
        mcLaunchKernelGGL((mcspBsr2bscKernel<n_elem>), dim3(n_block), dim3(n_elem), 0, stream, total_nnz, block_nnz,
                           temp_coo_rows, bsr_vals, sorted_ibuffer, bsc_rows, bsc_vals);
    } else {
        mcspCsr2Coo(handle, bsr_rows, nnzb, mb, bsc_rows, idx_base);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, bsr_cols, temp_bsr_cols, bsc_rows, temp_coo_rows, nnzb,
                                 stream, start_bit, end_bit);
        mcspCoo2Csr(handle, temp_bsr_cols, nnzb, nb, bsc_cols, idx_base);
        MACA_ASSERT(mcMemcpyAsync(bsc_rows, temp_coo_rows, nnzb * sizeof(idxType), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSgebsr2gebsc_bufferSize(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                         const float* bsr_vals, const mcspInt* bsr_rows, const mcspInt* bsr_cols,
                                         mcspInt row_block_dim, mcspInt col_block_dim, int* buffer_size) {
    return mcspGebsr2gebscBufferSizeTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                             col_block_dim, buffer_size);
}

mcspStatus_t mcspDgebsr2gebsc_bufferSize(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                         const double* bsr_vals, const mcspInt* bsr_rows, const mcspInt* bsr_cols,
                                         mcspInt row_block_dim, mcspInt col_block_dim, int* buffer_size) {
    return mcspGebsr2gebscBufferSizeTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                             col_block_dim, buffer_size);
}

mcspStatus_t mcspCgebsr2gebsc_bufferSize(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                         const mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                                         const mcspInt* bsr_cols, mcspInt row_block_dim, mcspInt col_block_dim,
                                         int* buffer_size) {
    return mcspGebsr2gebscBufferSizeTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                             col_block_dim, buffer_size);
}

mcspStatus_t mcspZgebsr2gebsc_bufferSize(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                         const mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows,
                                         const mcspInt* bsr_cols, mcspInt row_block_dim, mcspInt col_block_dim,
                                         int* buffer_size) {
    return mcspGebsr2gebscBufferSizeTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                             col_block_dim, buffer_size);
}

mcspStatus_t mcspSgebsr2gebsc_bufferSizeExt(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                            const float* bsr_vals, const mcspInt* bsr_rows, const mcspInt* bsr_cols,
                                            mcspInt row_block_dim, mcspInt col_block_dim, size_t* buffer_size) {
    return mcspGebsr2gebscBufferSizeExtTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                                col_block_dim, buffer_size);
}

mcspStatus_t mcspDgebsr2gebsc_bufferSizeExt(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                            const double* bsr_vals, const mcspInt* bsr_rows, const mcspInt* bsr_cols,
                                            mcspInt row_block_dim, mcspInt col_block_dim, size_t* buffer_size) {
    return mcspGebsr2gebscBufferSizeExtTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                                col_block_dim, buffer_size);
}

mcspStatus_t mcspCgebsr2gebsc_bufferSizeExt(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                            const mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                                            const mcspInt* bsr_cols, mcspInt row_block_dim, mcspInt col_block_dim,
                                            size_t* buffer_size) {
    return mcspGebsr2gebscBufferSizeExtTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                                col_block_dim, buffer_size);
}

mcspStatus_t mcspZgebsr2gebsc_bufferSizeExt(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                            const mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows,
                                            const mcspInt* bsr_cols, mcspInt row_block_dim, mcspInt col_block_dim,
                                            size_t* buffer_size) {
    return mcspGebsr2gebscBufferSizeExtTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                                col_block_dim, buffer_size);
}

mcspStatus_t mcspSgebsr2gebsc(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb, const float* bsr_vals,
                              const mcspInt* bsr_rows, const mcspInt* bsr_cols, mcspInt row_block_dim,
                              mcspInt col_block_dim, float* bsc_vals, mcspInt* bsc_rows, mcspInt* bsc_cols,
                              mcsparseAction_t copy_value, mcsparseIndexBase_t idx_base, void* buffer) {
    return mcspGebsr2gebscTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim, col_block_dim,
                                   bsc_vals, bsc_rows, bsc_cols, copy_value, idx_base, buffer);
}

mcspStatus_t mcspDgebsr2gebsc(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb, const double* bsr_vals,
                              const mcspInt* bsr_rows, const mcspInt* bsr_cols, mcspInt row_block_dim,
                              mcspInt col_block_dim, double* bsc_vals, mcspInt* bsc_rows, mcspInt* bsc_cols,
                              mcsparseAction_t copy_value, mcsparseIndexBase_t idx_base, void* buffer) {
    return mcspGebsr2gebscTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim, col_block_dim,
                                   bsc_vals, bsc_rows, bsc_cols, copy_value, idx_base, buffer);
}

mcspStatus_t mcspCgebsr2gebsc(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb, const mcFloatComplex* bsr_vals,
                              const mcspInt* bsr_rows, const mcspInt* bsr_cols, mcspInt row_block_dim,
                              mcspInt col_block_dim, mcFloatComplex* bsc_vals, mcspInt* bsc_rows, mcspInt* bsc_cols,
                              mcsparseAction_t copy_value, mcsparseIndexBase_t idx_base, void* buffer) {
    return mcspGebsr2gebscTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim, col_block_dim,
                                   bsc_vals, bsc_rows, bsc_cols, copy_value, idx_base, buffer);
}

mcspStatus_t mcspZgebsr2gebsc(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                              const mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows, const mcspInt* bsr_cols,
                              mcspInt row_block_dim, mcspInt col_block_dim, mcDoubleComplex* bsc_vals,
                              mcspInt* bsc_rows, mcspInt* bsc_cols, mcsparseAction_t copy_value,
                              mcsparseIndexBase_t idx_base, void* buffer) {
    return mcspGebsr2gebscTemplate(handle, mb, nb, nnzb, bsr_vals, bsr_rows, bsr_cols, row_block_dim, col_block_dim,
                                   bsc_vals, bsc_rows, bsc_cols, copy_value, idx_base, buffer);
}

#ifdef __cplusplus
}
#endif