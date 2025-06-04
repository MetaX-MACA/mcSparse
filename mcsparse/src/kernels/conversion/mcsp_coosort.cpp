#include "create_identity_device.hpp"
#include "device_radix_sort.hpp"
#include "gthr_device.hpp"
#include "mcsp_config.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_interface.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType>
mcspStatus_t mcspCooSortBufferSizeTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz,
                                           const idxType *coo_rows, const idxType *coo_cols, size_t *buffer_size) {
    constexpr unsigned int coo_sort_buffer_block = 3;
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (coo_rows == nullptr || coo_cols == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (n == 0 || m == 0 || nnz == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }
    mcspInt tmp_buffersize = 0;
    mcspInt *tmp_ptr = nullptr;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::radix_sort_pairs(nullptr, tmp_buffersize, coo_cols, tmp_ptr, coo_cols, tmp_ptr, nnz, stream);
    *buffer_size = tmp_buffersize;
    *buffer_size += (coo_sort_buffer_block + 1) * nnz * sizeof(*coo_cols);

    *buffer_size = *buffer_size < MIN_BUFFER_SIZE ? MIN_BUFFER_SIZE : *buffer_size;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCooSortByRowTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz, idxType *coo_rows,
                                      idxType *coo_cols, idxType *perm, void *temp_buffer) {
    bool set_perm = true;

    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (n == 0 || m == 0 || nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (coo_rows == nullptr || coo_cols == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType *buffer_ptr = reinterpret_cast<idxType *>(temp_buffer);
    idxType *temp_sorted_coo_cols = buffer_ptr;
    buffer_ptr += nnz;
    idxType *temp_sorted_coo_rows = buffer_ptr;
    buffer_ptr += nnz;

    idxType tmp_buffersize = 0;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::radix_sort_pairs(nullptr, tmp_buffersize, coo_cols, temp_sorted_coo_cols, perm, temp_sorted_coo_rows, nnz,
                             stream);
    mcspInt start_bit = 0;
    mcspInt end_bit;
    if (perm == nullptr) {
        end_bit = getHighBitLocOneBase(n);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, coo_cols, temp_sorted_coo_cols, coo_rows,
                                 temp_sorted_coo_rows, nnz, stream, start_bit, end_bit);
        end_bit = getHighBitLocOneBase(m);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, temp_sorted_coo_rows, coo_rows, temp_sorted_coo_cols,
                                 coo_cols, nnz, stream, start_bit, end_bit);
    } else {
        idxType *temp_sorted_perm = buffer_ptr;
        buffer_ptr += nnz;
        idxType *temp_perm = buffer_ptr;
        buffer_ptr += nnz;
        constexpr unsigned int n_elem = 512;
        int n_block = (nnz + n_elem - 1) / n_elem;
        mcLaunchKernelGGL(mcspCreateIdentityKernel, dim3(n_block), dim3(n_elem), 0, stream, nnz, temp_perm);

        end_bit = getHighBitLocOneBase(n);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, coo_cols, temp_sorted_coo_cols, temp_perm,
                                 temp_sorted_perm, nnz, stream, start_bit, end_bit);
        mcLaunchKernelGGL((mcspGthrKernel<n_elem>), dim3(n_block), dim3(n_elem), 0, stream, nnz, coo_rows,
                           temp_sorted_coo_rows, temp_sorted_perm, MCSPARSE_INDEX_BASE_ZERO);
        end_bit = getHighBitLocOneBase(m);
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, temp_sorted_coo_rows, coo_rows, temp_sorted_perm,
                                 temp_perm, nnz, stream, start_bit, end_bit);
        mcLaunchKernelGGL((mcspGthrKernel<n_elem>), dim3(n_block), dim3(n_elem), 0, stream, nnz, coo_cols,
                           temp_sorted_coo_cols, temp_perm, MCSPARSE_INDEX_BASE_ZERO);
        mcLaunchKernelGGL((mcspGthrKernel<n_elem>), dim3(n_block), dim3(n_elem), 0, stream, nnz, perm,
                           temp_sorted_perm, temp_perm, MCSPARSE_INDEX_BASE_ZERO);
        MACA_ASSERT(
            mcMemcpyAsync(coo_cols, temp_sorted_coo_cols, nnz * sizeof(*coo_cols), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcMemcpyAsync(perm, temp_sorted_perm, nnz * sizeof(*coo_cols), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCooSortByColumnTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz, idxType *coo_rows,
                                         idxType *coo_cols, idxType *perm, void *temp_buffer) {
    return mcspCooSortByRowTemplate(handle, n, m, nnz, coo_cols, coo_rows, perm, temp_buffer);
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspCooSortBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspInt *coo_rows,
                                   const mcspInt *coo_cols, size_t *buffer_size) {
    return mcspCooSortBufferSizeTemplate(handle, m, n, nnz, coo_rows, coo_cols, buffer_size);
}

mcspStatus_t mcspCooSortByRow(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspInt *coo_rows,
                              mcspInt *coo_cols, mcspInt *perm, void *temp_buffer) {
    return mcspCooSortByRowTemplate(handle, m, n, nnz, coo_rows, coo_cols, perm, temp_buffer);
}

mcspStatus_t mcspCooSortByColumn(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspInt *coo_rows,
                                 mcspInt *coo_cols, mcspInt *perm, void *temp_buffer) {
    return mcspCooSortByColumnTemplate(handle, m, n, nnz, coo_rows, coo_cols, perm, temp_buffer);
}

mcspStatus_t mcspCuinXcoosort_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, const int *coo_rows,
                                          const int *coo_cols, size_t *buffer_size) {
    return mcspCooSortBufferSizeTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (mcspInt *)coo_rows,
                                         (mcspInt *)coo_cols, buffer_size);
}

mcspStatus_t mcspCuinXcoosortByRow(mcspHandle_t handle, int m, int n, int nnz, int *coo_rows, int *coo_cols, int *perm,
                                 void *temp_buffer) {
    return mcspCooSortByRowTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (mcspInt *)coo_rows,
                                    (mcspInt *)coo_cols, (mcspInt *)perm, temp_buffer);
}

mcspStatus_t mcspCuinXcoosortByColumn(mcspHandle_t handle, int m, int n, int nnz, int *coo_rows, int *coo_cols, int *perm,
                                    void *temp_buffer) {
    return mcspCooSortByColumnTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (mcspInt *)coo_rows,
                                       (mcspInt *)coo_cols, (mcspInt *)perm, temp_buffer);
}

#ifdef __cplusplus
}
#endif