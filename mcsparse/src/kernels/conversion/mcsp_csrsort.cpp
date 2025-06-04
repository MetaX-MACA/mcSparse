#include "common/mcsp_types.h"
#include "create_identity_device.hpp"
#include "device_radix_sort.hpp"
#include "gthr_device.hpp"
#include "mcsp_config.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType>
mcspStatus_t mcspCsrSortBufferSizeTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz,
                                           const idxType *csr_rows, const idxType *csr_cols, size_t *buffer_size) {
    constexpr unsigned int csr_sort_buffer_block = 4;
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (csr_rows == nullptr || csr_cols == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (n == 0 || m == 0 || nnz == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }
    mcspInt tmp_buffersize = 0;
    mcspInt *tmp_ptr = nullptr;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::radix_sort_pairs(nullptr, tmp_buffersize, csr_cols, tmp_ptr, csr_cols, tmp_ptr, nnz, stream);
    *buffer_size = tmp_buffersize;
    *buffer_size += csr_sort_buffer_block * 2 * nnz * sizeof(*csr_cols);

    *buffer_size = *buffer_size < MIN_BUFFER_SIZE ? MIN_BUFFER_SIZE : *buffer_size;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCsrSortTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz, mcspMatDescr_t mcsp_descr_A,
                                 const idxType *csr_rows, idxType *csr_cols, idxType *perm, void *temp_buffer) {
    bool set_perm = true;
    bool use_buffer_pool = false;
    bool use_internal_mem = false;

    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (n == 0 || m == 0 || nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (csr_rows == nullptr || csr_cols == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType *buffer_ptr = reinterpret_cast<idxType *>(temp_buffer);
    idxType *temp_sorted_csr_cols = buffer_ptr;
    buffer_ptr += nnz;
    idxType *coo_rows = buffer_ptr;
    buffer_ptr += nnz;
    idxType *temp_sorted_coo_rows = buffer_ptr;
    buffer_ptr += nnz;

    mcspCsr2Coo(handle, csr_rows, nnz, m, coo_rows, mcsp_descr_A->base);

    idxType tmp_buffersize = 0;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::radix_sort_pairs(nullptr, tmp_buffersize, csr_cols, temp_sorted_csr_cols, perm, temp_sorted_coo_rows, nnz,
                             stream);
    mcspInt start_bit = 0;
    mcspInt end_bit;
    if (perm == nullptr) {
        end_bit = getHighBitLocOneBase(n);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, csr_cols, temp_sorted_csr_cols, coo_rows,
                                 temp_sorted_coo_rows, nnz, stream, start_bit, end_bit);
        end_bit = getHighBitLocOneBase(m);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, temp_sorted_coo_rows, coo_rows, temp_sorted_csr_cols,
                                 csr_cols, nnz, stream, start_bit, end_bit);
    } else {
        constexpr unsigned int n_elem = 512;
        int n_block = (nnz + n_elem - 1) / n_elem;
        idxType *temp_sorted_perm = buffer_ptr;
        buffer_ptr += nnz;
        idxType *identity_perm = buffer_ptr;
        buffer_ptr += nnz;

        mcLaunchKernelGGL(mcspCreateIdentityKernel, dim3(n_block), dim3(n_elem), 0, stream, nnz, identity_perm);

        end_bit = getHighBitLocOneBase(n);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, csr_cols, temp_sorted_csr_cols, identity_perm, temp_sorted_perm,
                                 nnz, stream, start_bit, end_bit);
        mcLaunchKernelGGL((mcspGthrKernel<n_elem>), dim3(n_block), dim3(n_elem), 0, stream, nnz, coo_rows,
                           temp_sorted_coo_rows, temp_sorted_perm, mcsp_descr_A->base);
        end_bit = getHighBitLocOneBase(m);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, temp_sorted_coo_rows, coo_rows, temp_sorted_perm, identity_perm,
                                 nnz, stream, start_bit, end_bit);
        mcLaunchKernelGGL((mcspGthrKernel<n_elem>), dim3(n_block), dim3(n_elem), 0, stream, nnz, csr_cols,
                           temp_sorted_csr_cols, identity_perm, mcsp_descr_A->base);
        mcLaunchKernelGGL((mcspGthrKernel<n_elem>), dim3(n_block), dim3(n_elem), 0, stream, nnz, perm,
                           temp_sorted_perm, identity_perm, mcsp_descr_A->base);
        MACA_ASSERT(
            mcMemcpyAsync(csr_cols, temp_sorted_csr_cols, nnz * sizeof(*csr_cols), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(
            mcMemcpyAsync(perm, temp_sorted_perm, nnz * sizeof(*csr_cols), mcMemcpyDeviceToDevice, stream));
    }

    MACA_ASSERT(mcStreamSynchronize(stream));

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspCsrSortBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspInt *csr_rows,
                                   const mcspInt *csr_cols, size_t *buffer_size) {
    return mcspCsrSortBufferSizeTemplate(handle, m, n, nnz, csr_rows, csr_cols, buffer_size);
}

mcspStatus_t mcspCsrSort(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspMatDescr_t mcsp_descr_A,
                         const mcspInt *csr_rows, mcspInt *csr_cols, mcspInt *perm, void *temp_buffer) {
    return mcspCsrSortTemplate(handle, m, n, nnz, mcsp_descr_A, csr_rows, csr_cols, perm, temp_buffer);
}

mcspStatus_t mcspCuinXcsrsort_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, const int *csr_rows,
                                          const int *csr_cols, size_t *buffer_size) {
    return mcspCsrSortBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (mcspInt *)csr_rows,
                                         (mcspInt *)csr_cols, buffer_size);
}

mcspStatus_t mcspCuinXcsrsort(mcspHandle_t handle, int m, int n, int nnz, mcspMatDescr_t mcsp_descr_A,
                            const int *csr_rows, int *csr_cols, int *perm, void *temp_buffer) {
    return mcspCsrSortTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, mcsp_descr_A, (mcspInt *)csr_rows,
                               (mcspInt *)csr_cols, (mcspInt *)perm, temp_buffer);
}

#ifdef __cplusplus
}
#endif