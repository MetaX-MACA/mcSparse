#include "bsric0_device.hpp"
#include "common/mcsp_types.h"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "spsv/csr_trm_analysis.hpp"

mcspStatus_t mcspBsric02_zeroPivot(mcspHandle_t handle, mcspBsric02Info_t info, int* position) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (info == nullptr || info->bsric0_mat == nullptr || position == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *position = info->bsric0_mat->zero_pivot_lead;
    if (*position != -1) {
        return MCSP_STATUS_ZERO_PIVOT;
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsric02_bufferSizeImpl(mcspHandle_t handle, mcsparseDirection_t dir, idxType mb, idxType nnzb,
                                        const mcspMatDescr_t descr, valType* bsr_vals, const idxType* bsr_rows,
                                        const idxType* bsr_cols, idxType block_dim, mcspBsric02Info_t info,
                                        int* buffer_size) {
    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    size_t temp_size = 0;
    mcspStatus_t ret = mcspBsric02_bufferSizeExtImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols,
                                                     block_dim, info, &temp_size);
    *buffer_size = (int)temp_size;
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsric02_bufferSizeExtImpl(mcspHandle_t handle, mcsparseDirection_t dir, idxType mb, idxType nnzb,
                                           const mcspMatDescr_t descr, valType* bsr_vals, const idxType* bsr_rows,
                                           const idxType* bsr_cols, idxType block_dim, mcspBsric02Info_t info,
                                           size_t* buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (dir != MCSPARSE_DIRECTION_ROW && dir != MCSPARSE_DIRECTION_COLUMN) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (mb < 0 || nnzb < 0 || block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (descr == nullptr || info == nullptr || info->bsric0_mat == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (mb == 0 || nnzb == 0 || block_dim == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows == nullptr || bsr_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    size_t reduce_buffersize;
    size_t radix_sort_buffersize;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::reduce(nullptr, reduce_buffersize, (idxType*)nullptr, (idxType*)nullptr, (size_t)mb,
                   mcprim::minimum<idxType>(), stream);
    mcprim::radix_sort_pairs(nullptr, radix_sort_buffersize, (idxType*)nullptr, (idxType*)nullptr, (idxType*)nullptr,
                             (idxType*)nullptr, (size_t)mb, stream);
    *buffer_size = ALIGN((4 + 3 * mb) * sizeof(idxType), ALIGNED_SIZE) + ALIGN(reduce_buffersize, ALIGNED_SIZE) +
                   ALIGN(radix_sort_buffersize, ALIGNED_SIZE);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsric02_analysisImpl(mcspHandle_t handle, mcsparseDirection_t dir, idxType mb, idxType nnzb,
                                      const mcspMatDescr_t descr, const valType* bsr_vals, const idxType* bsr_rows,
                                      const idxType* bsr_cols, idxType block_dim, mcspBsric02Info_t info,
                                      mcsparseSolvePolicy_t policy, void* buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (dir != MCSPARSE_DIRECTION_ROW && dir != MCSPARSE_DIRECTION_COLUMN) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (mb < 0 || nnzb < 0 || block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (descr == nullptr || info == nullptr || info->bsric0_mat == nullptr || buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (mb == 0 || nnzb == 0 || block_dim == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows == nullptr || bsr_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (mcspCreateTrmInfo(&(info->bsric0_mat->bsric0_info)) != MCSP_STATUS_SUCCESS) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCsrTrmAnalysis_template(handle, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols,
                                       info->bsric0_mat->bsric0_info, info->bsric0_mat->zero_pivot_lead, buffer, true);
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsric02Impl(mcspHandle_t handle, mcsparseDirection_t dir, idxType mb, idxType nnzb,
                             const mcspMatDescr_t descr, valType* bsr_vals, const idxType* bsr_rows,
                             const idxType* bsr_cols, idxType block_dim, mcspBsric02Info_t info,
                             mcsparseSolvePolicy_t policy, void* buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (dir != MCSPARSE_DIRECTION_ROW && dir != MCSPARSE_DIRECTION_COLUMN) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (mb < 0 || nnzb < 0 || block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (descr == nullptr || info == nullptr || info->bsric0_mat == nullptr ||
        info->bsric0_mat->bsric0_info == nullptr || buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (mb == 0 || nnzb == 0 || block_dim == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows == nullptr || bsr_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    int n_elem = WARP_SIZE;
    int n_block = mb;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemsetAsync(buffer, 0, mb * sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    idxType max_row_nnz = info->bsric0_mat->bsric0_info->max_row_nnz;
    if (max_row_nnz < 256) {
        mcLaunchKernelGGL((mcspBsric0HashTableKernel<WARP_SIZE, 256>), dim3(n_block), dim3(n_elem),
                           256 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsric0_mat->bsric0_info->row_map),
                           (idxType*)(info->bsric0_mat->bsric0_info->trm_diag_ind),
                           (idxType*)info->bsric0_mat->bsric0_info->zero_pivot_lead, (idxType*)buffer, descr->base);
    } else if (max_row_nnz < 512) {
        mcLaunchKernelGGL((mcspBsric0HashTableKernel<WARP_SIZE, 512>), dim3(n_block), dim3(n_elem),
                           512 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsric0_mat->bsric0_info->row_map),
                           (idxType*)(info->bsric0_mat->bsric0_info->trm_diag_ind),
                           (idxType*)info->bsric0_mat->bsric0_info->zero_pivot_lead, (idxType*)buffer, descr->base);
    } else if (max_row_nnz < 1024) {
        mcLaunchKernelGGL((mcspBsric0HashTableKernel<WARP_SIZE, 1024>), dim3(n_block), dim3(n_elem),
                           1024 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsric0_mat->bsric0_info->row_map),
                           (idxType*)(info->bsric0_mat->bsric0_info->trm_diag_ind),
                           (idxType*)info->bsric0_mat->bsric0_info->zero_pivot_lead, (idxType*)buffer, descr->base);

    } else if (max_row_nnz < 2048) {
        mcLaunchKernelGGL((mcspBsric0HashTableKernel<WARP_SIZE, 2048>), dim3(n_block), dim3(n_elem),
                           2048 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsric0_mat->bsric0_info->row_map),
                           (idxType*)(info->bsric0_mat->bsric0_info->trm_diag_ind),
                           (idxType*)info->bsric0_mat->bsric0_info->zero_pivot_lead, (idxType*)buffer, descr->base);

    } else if (max_row_nnz < 4096) {
        mcLaunchKernelGGL((mcspBsric0HashTableKernel<WARP_SIZE, 4096>), dim3(n_block), dim3(n_elem),
                           4096 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsric0_mat->bsric0_info->row_map),
                           (idxType*)(info->bsric0_mat->bsric0_info->trm_diag_ind),
                           (idxType*)info->bsric0_mat->bsric0_info->zero_pivot_lead, (idxType*)buffer, descr->base);

    } else {
        mcLaunchKernelGGL((mcspBsric0BsearchKernel), dim3(n_block), dim3(n_elem), 0, stream, dir, mb, block_dim,
                           bsr_vals, bsr_rows, bsr_cols, (idxType*)(info->bsric0_mat->bsric0_info->row_map),
                           (idxType*)(info->bsric0_mat->bsric0_info->trm_diag_ind),
                           (idxType*)info->bsric0_mat->bsric0_info->zero_pivot_lead, (idxType*)buffer, descr->base);
    }
    MACA_ASSERT(mcMemcpyAsync(&info->bsric0_mat->zero_pivot_lead, info->bsric0_mat->bsric0_info->zero_pivot_lead,
                              sizeof(idxType), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspXbsric02_zeroPivot(mcspHandle_t handle, mcspBsric02Info_t info, int* position) {
    return mcspBsric02_zeroPivot(handle, info, position);
}

mcspStatus_t mcspSbsric02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                     const mcspMatDescr_t descr, float* bsr_vals, const mcspInt* bsr_rows,
                                     const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                     int* buffer_size) {
    return mcspBsric02_bufferSizeImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info,
                                      buffer_size);
}

mcspStatus_t mcspDbsric02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                     const mcspMatDescr_t descr, double* bsr_vals, const mcspInt* bsr_rows,
                                     const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                     int* buffer_size) {
    return mcspBsric02_bufferSizeImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info,
                                      buffer_size);
}

mcspStatus_t mcspCbsric02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                     const mcspMatDescr_t descr, mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                                     const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                     int* buffer_size) {
    return mcspBsric02_bufferSizeImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info,
                                      buffer_size);
}

mcspStatus_t mcspZbsric02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                     const mcspMatDescr_t descr, mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows,
                                     const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                     int* buffer_size) {
    return mcspBsric02_bufferSizeImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info,
                                      buffer_size);
}

mcspStatus_t mcspSbsric02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                        const mcspMatDescr_t descr, float* bsr_vals, const mcspInt* bsr_rows,
                                        const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                        size_t* buffer_size) {
    return mcspBsric02_bufferSizeExtImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info,
                                         buffer_size);
}

mcspStatus_t mcspDbsric02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                        const mcspMatDescr_t descr, double* bsr_vals, const mcspInt* bsr_rows,
                                        const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                        size_t* buffer_size) {
    return mcspBsric02_bufferSizeExtImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info,
                                         buffer_size);
}

mcspStatus_t mcspCbsric02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                        const mcspMatDescr_t descr, mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                                        const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                        size_t* buffer_size) {
    return mcspBsric02_bufferSizeExtImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info,
                                         buffer_size);
}

mcspStatus_t mcspZbsric02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                        const mcspMatDescr_t descr, mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows,
                                        const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                        size_t* buffer_size) {
    return mcspBsric02_bufferSizeExtImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info,
                                         buffer_size);
}

mcspStatus_t mcspSbsric02_analysis(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                   const mcspMatDescr_t descr, const float* bsr_vals, const mcspInt* bsr_rows,
                                   const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsric02_analysisImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info, policy,
                                    buffer);
}

mcspStatus_t mcspDbsric02_analysis(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                   const mcspMatDescr_t descr, const double* bsr_vals, const mcspInt* bsr_rows,
                                   const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsric02_analysisImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info, policy,
                                    buffer);
}

mcspStatus_t mcspCbsric02_analysis(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                   const mcspMatDescr_t descr, const mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                                   const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsric02_analysisImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info, policy,
                                    buffer);
}

mcspStatus_t mcspZbsric02_analysis(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                                   const mcspMatDescr_t descr, const mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows,
                                   const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsric02_analysisImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info, policy,
                                    buffer);
}

mcspStatus_t mcspSbsric02(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                          const mcspMatDescr_t descr, float* bsr_vals, const mcspInt* bsr_rows, const mcspInt* bsr_cols,
                          mcspInt block_dim, mcspBsric02Info_t info, mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsric02Impl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info, policy, buffer);
}

mcspStatus_t mcspDbsric02(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                          const mcspMatDescr_t descr, double* bsr_vals, const mcspInt* bsr_rows,
                          const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                          mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsric02Impl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info, policy, buffer);
}

mcspStatus_t mcspCbsric02(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                          const mcspMatDescr_t descr, mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                          const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                          mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsric02Impl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info, policy, buffer);
}

mcspStatus_t mcspZbsric02(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt mb, mcspInt nnzb,
                          const mcspMatDescr_t descr, mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows,
                          const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                          mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsric02Impl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols, block_dim, info, policy, buffer);
}

#ifdef __cplusplus
}
#endif