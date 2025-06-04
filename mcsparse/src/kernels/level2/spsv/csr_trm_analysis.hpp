#include "common/mcsp_types.h"
#include "csr_spsv_device.hpp"
#include "device_radix_sort.hpp"
#include "device_reduce.hpp"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"

template <typename idxType, typename valType>
mcspStatus_t mcspCsrTrmAnalysis_template(mcspHandle_t handle, idxType row_num, idxType nnz, const mcspMatDescr_t descr,
                                         const valType *csr_vals, const idxType *csr_rows, const idxType *csr_cols,
                                         mcspTrmInfo_t trm_info, int &zero_pivot_lead, void *tmp_buffer,
                                         bool lower_flag) {
    void *buffer_head = tmp_buffer;
    idxType *row_nnz_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(row_nnz_buffer + row_num);
    idxType *perm_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(perm_buffer + row_num);
    idxType *depth_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(depth_buffer + row_num);
    idxType *reduce_output_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(reduce_output_buffer + 4);
    void *prim_buffer = buffer_head;
    mcStream_t stream = mcspGetStreamInternal(handle);

    idxType prim_buffersize;
    mcprim::reduce(nullptr, prim_buffersize, (idxType *)nullptr, (idxType *)nullptr, row_num,
                   mcprim::minimum<idxType>(), stream);

    MACA_ASSERT(mcMalloc(&(trm_info->row_map), row_num * sizeof(idxType)));
    MACA_ASSERT(mcMalloc(&(trm_info->trm_diag_ind), row_num * sizeof(idxType)));

    MACA_ASSERT(mcMalloc(&(trm_info->zero_pivot_array), row_num * sizeof(idxType)));
    MACA_ASSERT(mcMalloc(&(trm_info->zero_pivot_lead), sizeof(idxType)));
    MACA_ASSERT(mcMemsetAsync((void *)(trm_info->zero_pivot_array), 0xFFFFFFFF, row_num * sizeof(idxType), stream));
    MACA_ASSERT(mcMemsetAsync((void *)(trm_info->zero_pivot_lead), 0xFFFFFFFF, sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    idxType *zero_pivot_info = (idxType *)(trm_info->zero_pivot_array);
    idxType *zero_pivot_lead_device = (idxType *)(trm_info->zero_pivot_lead);

    int n_elem = 256;
    int n_block = (row_num + n_elem / WARP_SIZE - 1) / (n_elem / WARP_SIZE);
    MACA_ASSERT(mcMemsetAsync(trm_info->trm_diag_ind, 0, row_num * sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    mcLaunchKernelGGL(mcspCsrTrmDiagRowNnzKernel, dim3(n_block), dim3(n_elem), 0, stream, row_num, csr_vals, csr_rows,
                       csr_cols, (idxType *)(trm_info->trm_diag_ind), zero_pivot_info, row_nnz_buffer, descr->base);

    MACA_ASSERT(mcStreamSynchronize(stream));

    mcprim::reduce(prim_buffer, prim_buffersize, row_nnz_buffer, reduce_output_buffer, row_num,
                   mcprim::maximum<idxType>(), stream);
    MACA_ASSERT(
        mcMemcpyAsync(&(trm_info->max_row_nnz), reduce_output_buffer, sizeof(idxType), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    mcprim::reduce(prim_buffer, prim_buffersize, zero_pivot_info, zero_pivot_lead_device, row_num,
                   mcprim::minimum<idxType>(), stream);
    MACA_ASSERT(mcMemcpyAsync(&zero_pivot_lead, zero_pivot_lead_device, sizeof(int), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    mcspCreateIdentityPermutation(handle, row_num, perm_buffer);
    MACA_ASSERT(mcMemsetAsync((void *)(depth_buffer), 0, row_num * sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    if (lower_flag) {
        mcLaunchKernelGGL((mcspCsrTrmDepthKernel<true>), dim3(n_block), dim3(n_elem), 0, stream, row_num, csr_vals,
                           csr_rows, csr_cols, depth_buffer, descr->base);
    } else {
        mcLaunchKernelGGL((mcspCsrTrmDepthKernel<false>), dim3(n_block), dim3(n_elem), 0, stream, row_num, csr_vals,
                           csr_rows, csr_cols, depth_buffer, descr->base);
    }
    MACA_ASSERT(mcStreamSynchronize(stream));
    idxType *depth_out_buffer = row_nnz_buffer;

    mcprim::radix_sort_pairs(nullptr, prim_buffersize, depth_buffer, depth_out_buffer, perm_buffer,
                             (idxType *)(trm_info->row_map), row_num, stream);
    mcspInt start_bit = 0;
    mcspInt end_bit = getHighBitLocOneBase(row_num);
    mcprim::radix_sort_pairs(prim_buffer, prim_buffersize, depth_buffer, depth_out_buffer, perm_buffer,
                             (idxType *)(trm_info->row_map), row_num, stream, start_bit, end_bit);

    return MCSP_STATUS_SUCCESS;
}
