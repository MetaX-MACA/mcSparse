#include "bsrilu0_device.hpp"
#include "common/mcsp_types.h"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "spsv/csr_trm_analysis.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspBsrilu02_numericBoostImpl(mcspHandle_t handle, mcspBsrilu02Info_t info, idxType enable_boost,
                                           double* tol, valType* boost_val) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (enable_boost == 0) {
        info->config.boost_enable = 0;
        return MCSP_STATUS_SUCCESS;
    }

    if (tol == nullptr || boost_val == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    info->config.boost_enable = 1;
    info->config.boost_tol = (void*)tol;
    info->config.boost_val = (void*)boost_val;

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspBsrilu02_zeroPivot(mcspHandle_t handle, mcspBsrilu02Info_t info, int* position) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (info == nullptr || info->bsrilu0_mat == nullptr || position == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *position = info->bsrilu0_mat->zero_pivot_lead;
    if (*position == -1) {
        return MCSP_STATUS_SUCCESS;
    } else {
        return MCSP_STATUS_ZERO_PIVOT;
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrilu02_bufferSizeExtImpl(mcspHandle_t handle, mcsparseDirection_t dir, idxType mb, idxType nnzb,
                                            const mcspMatDescr_t descr, valType* bsr_vals, const idxType* bsr_rows,
                                            const idxType* bsr_cols, idxType block_dim, mcspBsrilu02Info_t info,
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

    if (descr == nullptr || info == nullptr || info->bsrilu0_mat == nullptr || buffer_size == nullptr) {
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
    *buffer_size = (4 + 3 * mb) * sizeof(idxType) + ALIGN(reduce_buffersize, ALIGNED_SIZE) +
                   ALIGN(radix_sort_buffersize, ALIGNED_SIZE);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrilu02_bufferSizeImpl(mcspHandle_t handle, mcsparseDirection_t dir, idxType mb, idxType nnzb,
                                         const mcspMatDescr_t descr, valType* bsr_vals, const idxType* bsr_rows,
                                         const idxType* bsr_cols, idxType block_dim, mcspBsrilu02Info_t info,
                                         int* buffer_size) {
    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    size_t temp_size = 0;
    mcspStatus_t ret = mcspBsrilu02_bufferSizeExtImpl(handle, dir, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols,
                                                      block_dim, info, &temp_size);
    *buffer_size = (int)temp_size;
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrilu02_analysisImpl(mcspHandle_t handle, mcsparseDirection_t dir, idxType mb, idxType nnzb,
                                       const mcspMatDescr_t descr, valType* bsr_vals, const idxType* bsr_rows,
                                       const idxType* bsr_cols, idxType block_dim, mcspBsrilu02Info_t info,
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

    if (descr == nullptr || info == nullptr || info->bsrilu0_mat == nullptr || buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (mb == 0 || nnzb == 0 || block_dim == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows == nullptr || bsr_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (mcspCreateTrmInfo(&(info->bsrilu0_mat->bsrilu0_info)) != MCSP_STATUS_SUCCESS) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCsrTrmAnalysis_template(handle, mb, nnzb, descr, bsr_vals, bsr_rows, bsr_cols,
                                       info->bsrilu0_mat->bsrilu0_info, info->bsrilu0_mat->zero_pivot_lead, buffer,
                                       true);
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrilu02Impl(mcspHandle_t handle, mcsparseDirection_t dir, idxType mb, idxType nnzb,
                              const mcspMatDescr_t descr, valType* bsr_vals, const idxType* bsr_rows,
                              const idxType* bsr_cols, idxType block_dim, mcspBsrilu02Info_t info,
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

    if (descr == nullptr || info == nullptr || info->bsrilu0_mat == nullptr ||
        info->bsrilu0_mat->bsrilu0_info == nullptr || buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (mb == 0 || nnzb == 0 || block_dim == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows == nullptr || bsr_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    constexpr uint32_t block_size = WARP_SIZE;
    uint32_t blocks = mb;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemsetAsync(buffer, 0, mb * sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    idxType max_row_nnz = info->bsrilu0_mat->bsrilu0_info->max_row_nnz;
    double boost_tol;
    valType boost_val;
    if (info->config.boost_enable) {
        if (info->config.boost_tol == nullptr || info->config.boost_val == nullptr) {
            return MCSP_STATUS_INVALID_POINTER;
        }
        boost_tol = getScalarToHost(reinterpret_cast<const double*>(info->config.boost_tol), handle->ptr_mode);
        boost_val = getScalarToHost(reinterpret_cast<const valType*>(info->config.boost_val), handle->ptr_mode);
    } else {
        boost_tol = static_cast<double>(0);
        boost_val = static_cast<valType>(0);
    }
    if (max_row_nnz < 256) {
        mcLaunchKernelGGL((mcspBsrilu0HashTableKernel<WARP_SIZE, 256>), dim3(blocks), dim3(block_size),
                           256 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->row_map),
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->trm_diag_ind), (idxType*)buffer, descr->base,
                           (idxType*)info->bsrilu0_mat->bsrilu0_info->zero_pivot_lead, info->config.boost_enable,
                           boost_tol, boost_val);
    } else if (max_row_nnz < 512) {
        mcLaunchKernelGGL((mcspBsrilu0HashTableKernel<WARP_SIZE, 512>), dim3(blocks), dim3(block_size),
                           512 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->row_map),
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->trm_diag_ind), (idxType*)buffer, descr->base,
                           (idxType*)info->bsrilu0_mat->bsrilu0_info->zero_pivot_lead, info->config.boost_enable,
                           boost_tol, boost_val);
    } else if (max_row_nnz < 1024) {
        mcLaunchKernelGGL((mcspBsrilu0HashTableKernel<WARP_SIZE, 1024>), dim3(blocks), dim3(block_size),
                           1024 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->row_map),
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->trm_diag_ind), (idxType*)buffer, descr->base,
                           (idxType*)info->bsrilu0_mat->bsrilu0_info->zero_pivot_lead, info->config.boost_enable,
                           boost_tol, boost_val);
    } else if (max_row_nnz < 2048) {
        mcLaunchKernelGGL((mcspBsrilu0HashTableKernel<WARP_SIZE, 2048>), dim3(blocks), dim3(block_size),
                           2048 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->row_map),
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->trm_diag_ind), (idxType*)buffer, descr->base,
                           (idxType*)info->bsrilu0_mat->bsrilu0_info->zero_pivot_lead, info->config.boost_enable,
                           boost_tol, boost_val);
    } else if (max_row_nnz < 4096) {
        mcLaunchKernelGGL((mcspBsrilu0HashTableKernel<WARP_SIZE, 4096>), dim3(blocks), dim3(block_size),
                           4096 * 2 * sizeof(idxType), stream, dir, mb, block_dim, bsr_vals, bsr_rows, bsr_cols,
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->row_map),
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->trm_diag_ind), (idxType*)buffer, descr->base,
                           (idxType*)info->bsrilu0_mat->bsrilu0_info->zero_pivot_lead, info->config.boost_enable,
                           boost_tol, boost_val);
    } else {
        mcLaunchKernelGGL((mcspBsrilu0BsearchKernel), dim3(blocks), dim3(block_size), 0, stream, dir, mb, block_dim,
                           bsr_vals, bsr_rows, bsr_cols, (idxType*)(info->bsrilu0_mat->bsrilu0_info->row_map),
                           (idxType*)(info->bsrilu0_mat->bsrilu0_info->trm_diag_ind), (idxType*)buffer, descr->base,
                           (idxType*)info->bsrilu0_mat->bsrilu0_info->zero_pivot_lead, info->config.boost_enable,
                           boost_tol, boost_val);
    }
    MACA_ASSERT(mcStreamSynchronize(stream));
    MACA_ASSERT(mcMemcpyAsync(&info->bsrilu0_mat->zero_pivot_lead, info->bsrilu0_mat->bsrilu0_info->zero_pivot_lead,
                              sizeof(idxType), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspSbsrilu02_numericBoost(mcspHandle_t handle, mcspBsrilu02Info_t info, mcspInt enable_boost, double* tol,
                                        float* boost_val) {
    return mcspBsrilu02_numericBoostImpl(handle, info, enable_boost, tol, boost_val);
}

mcspStatus_t mcspDbsrilu02_numericBoost(mcspHandle_t handle, mcspBsrilu02Info_t info, mcspInt enable_boost, double* tol,
                                        double* boost_val) {
    return mcspBsrilu02_numericBoostImpl(handle, info, enable_boost, tol, boost_val);
}

mcspStatus_t mcspCbsrilu02_numericBoost(mcspHandle_t handle, mcspBsrilu02Info_t info, mcspInt enable_boost, double* tol,
                                        mcFloatComplex* boost_val) {
    return mcspBsrilu02_numericBoostImpl(handle, info, enable_boost, tol, boost_val);
}

mcspStatus_t mcspZbsrilu02_numericBoost(mcspHandle_t handle, mcspBsrilu02Info_t info, mcspInt enable_boost, double* tol,
                                        mcDoubleComplex* boost_val) {
    return mcspBsrilu02_numericBoostImpl(handle, info, enable_boost, tol, boost_val);
}

mcspStatus_t mcspXbsrilu02_zeroPivot(mcspHandle_t handle, mcspBsrilu02Info_t info, int* position) {
    return mcspBsrilu02_zeroPivot(handle, info, position);
}

mcspStatus_t mcspSbsrilu02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                      const mcspMatDescr_t descrA, float* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                      const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                                      int* pBufferSizeInBytes) {
    return mcspBsrilu02_bufferSizeImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                       blockDim, info, pBufferSizeInBytes);
}

mcspStatus_t mcspDbsrilu02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                      const mcspMatDescr_t descrA, double* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                      const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                                      int* pBufferSizeInBytes) {
    return mcspBsrilu02_bufferSizeImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                       blockDim, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCbsrilu02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                      const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                      const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockDim,
                                      mcspBsrilu02Info_t info, int* pBufferSizeInBytes) {
    return mcspBsrilu02_bufferSizeImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                       blockDim, info, pBufferSizeInBytes);
}

mcspStatus_t mcspZbsrilu02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                      const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                      const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockDim,
                                      mcspBsrilu02Info_t info, int* pBufferSizeInBytes) {
    return mcspBsrilu02_bufferSizeImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                       blockDim, info, pBufferSizeInBytes);
}

mcspStatus_t mcspSbsrilu02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                         const mcspMatDescr_t descrA, float* bsrSortedVal,
                                         const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                         mcspInt blockSize, mcspBsrilu02Info_t info, size_t* pBufferSize) {
    return mcspBsrilu02_bufferSizeExtImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                          bsrSortedColInd, blockSize, info, pBufferSize);
}

mcspStatus_t mcspDbsrilu02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                         const mcspMatDescr_t descrA, double* bsrSortedVal,
                                         const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                         mcspInt blockSize, mcspBsrilu02Info_t info, size_t* pBufferSize) {
    return mcspBsrilu02_bufferSizeExtImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                          bsrSortedColInd, blockSize, info, pBufferSize);
}

mcspStatus_t mcspCbsrilu02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                         const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                         const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                         mcspInt blockSize, mcspBsrilu02Info_t info, size_t* pBufferSize) {
    return mcspBsrilu02_bufferSizeExtImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                          bsrSortedColInd, blockSize, info, pBufferSize);
}

mcspStatus_t mcspZbsrilu02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                         const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                         const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                         mcspInt blockSize, mcspBsrilu02Info_t info, size_t* pBufferSize) {
    return mcspBsrilu02_bufferSizeExtImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                          bsrSortedColInd, blockSize, info, pBufferSize);
}

mcspStatus_t mcspSbsrilu02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, float* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                    const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                                    mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrilu02_analysisImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                     blockDim, info, policy, pBuffer);
}

mcspStatus_t mcspDbsrilu02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, double* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                    const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                                    mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrilu02_analysisImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                     blockDim, info, policy, pBuffer);
}

mcspStatus_t mcspCbsrilu02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                    const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockDim,
                                    mcspBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrilu02_analysisImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                     blockDim, info, policy, pBuffer);
}

mcspStatus_t mcspZbsrilu02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                    const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockDim,
                                    mcspBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrilu02_analysisImpl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                     blockDim, info, policy, pBuffer);
}

mcspStatus_t mcspSbsrilu02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                           const mcspMatDescr_t descrA, float* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                           const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                           mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrilu02Impl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim,
                            info, policy, pBuffer);
}

mcspStatus_t mcspDbsrilu02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                           const mcspMatDescr_t descrA, double* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                           const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                           mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrilu02Impl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim,
                            info, policy, pBuffer);
}

mcspStatus_t mcspCbsrilu02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                           const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                           const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                           mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrilu02Impl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim,
                            info, policy, pBuffer);
}

mcspStatus_t mcspZbsrilu02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                           const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                           const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                           mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrilu02Impl(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim,
                            info, policy, pBuffer);
}
#ifdef __cplusplus
}
#endif