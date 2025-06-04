#include "bsrsv_device.hpp"
#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_transpose_sparse.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"
#include "spsv/csr_trm_analysis.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspBsrsvBufferSizeTemplate(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans,
                                         idxType mb, idxType nnzb, const mcspMatDescr_t bsr_descr,
                                         const valType* bsr_vals, const idxType* bsr_rows_ind,
                                         const idxType* bsr_cols_ind, idxType block_dim, mcspBsrsv2Info_t info,
                                         size_t* buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (mb < 0 || nnzb < 0 || block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (bsr_descr == nullptr || buffer_size == nullptr || info == nullptr || info->mat_info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (bsr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (bsr_descr->fill_mode == MCSPARSE_FILL_MODE_FULL) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE && trans != MCSPARSE_OPERATION_TRANSPOSE &&
        trans != MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (trans == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double>) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    if (mb == 0 || nnzb == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows_ind == nullptr || bsr_cols_ind == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *buffer_size = 0;
    size_t reduce_buffersize = 0;
    size_t radix_sort_buffersize = 0;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::reduce(nullptr, reduce_buffersize, (idxType*)nullptr, (idxType*)nullptr, (size_t)mb,
                   mcprim::minimum<idxType>(), stream);
    mcprim::radix_sort_pairs(nullptr, radix_sort_buffersize, (idxType*)nullptr, (idxType*)nullptr, (idxType*)nullptr,
                             (idxType*)nullptr, (size_t)mb, stream);
    *buffer_size += ALIGN(reduce_buffersize, ALIGNED_SIZE);
    *buffer_size += ALIGN(radix_sort_buffersize, ALIGNED_SIZE);
    *buffer_size += ALIGN(4 * sizeof(idxType), ALIGNED_SIZE);   // reduce output buffer
    *buffer_size += ALIGN(mb * sizeof(idxType), ALIGNED_SIZE);  // row nnz buffer with length of mb
    *buffer_size += ALIGN(mb * sizeof(idxType), ALIGNED_SIZE);  // permutation buffer with length of mb
    *buffer_size += ALIGN(mb * sizeof(idxType), ALIGNED_SIZE);  // row depth buffer with length of mb
    if (block_dim * WARP_SIZE > 512) {
        *buffer_size += ALIGN(mb * block_dim * sizeof(idxType), ALIGNED_SIZE);  // solved row buffer
    }
    mcspStatus_t ret = MCSP_STATUS_SUCCESS;
    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        size_t trans_buffer_size = 0;
        info->mat_info->bsrt_dir = dir;
        mcspStatus_t ret =
            mcspGetBsrTransExBuffersize<valType>(handle, mb, mb, nnzb, block_dim, block_dim, bsr_rows_ind, bsr_cols_ind,
                                                 bsr_vals, info->mat_info, trans_buffer_size);
        if (ret == MCSP_STATUS_SUCCESS) {
            *buffer_size = info->mat_info->fixed_length_buffer_size + std::max(*buffer_size, trans_buffer_size);
        }
    }
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrsv_bufferSizeTemplate(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans,
                                          idxType mb, idxType nnzb, const mcspMatDescr_t bsr_descr,
                                          const valType* bsr_vals, const idxType* bsr_rows_ind,
                                          const idxType* bsr_cols_ind, idxType block_dim, mcspBsrsv2Info_t info,
                                          int* buffer_size) {
    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    size_t temp_size = 0;
    mcspStatus_t ret = mcspBsrsvBufferSizeTemplate(handle, dir, trans, mb, nnzb, bsr_descr, bsr_vals, bsr_rows_ind,
                                                   bsr_cols_ind, block_dim, info, &temp_size);
    *buffer_size = (int)temp_size;
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrsvAnalysisTemplate(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans,
                                       idxType mb, idxType nnzb, const mcspMatDescr_t bsr_descr,
                                       const valType* bsr_vals, const idxType* bsr_rows_ind,
                                       const idxType* bsr_cols_ind, idxType block_dim, mcspBsrsv2Info_t info,
                                       mcsparseSolvePolicy_t policy, void* buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (mb < 0 || nnzb < 0 || block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (bsr_descr == nullptr || buffer == nullptr || info == nullptr || info->mat_info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (bsr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (bsr_descr->fill_mode == MCSPARSE_FILL_MODE_FULL) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE && trans != MCSPARSE_OPERATION_TRANSPOSE &&
        trans != MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (trans == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double>) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    if (mb == 0 || nnzb == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows_ind == nullptr || bsr_cols_ind == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcspStatus_t ret = MCSP_STATUS_SUCCESS;
    const valType* working_bsr_vals;
    const idxType* working_bsr_rows;
    const idxType* working_bsr_cols;
    mcsparseFillMode_t working_fill_mode = bsr_descr->fill_mode;

    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        ret = mcspRawTransposeBsr<valType>(handle, trans, mb, mb, nnzb, block_dim, block_dim, bsr_rows_ind,
                                           bsr_cols_ind, bsr_vals, bsr_descr->base, info->mat_info, buffer);
        if (ret != MCSP_STATUS_SUCCESS) {
            return ret;
        }
        working_bsr_vals = (valType*)info->mat_info->bsrt_vals;
        working_bsr_rows = (idxType*)info->mat_info->bsrt_rows;
        working_bsr_cols = (idxType*)info->mat_info->bsrt_cols;
        working_fill_mode =
            (bsr_descr->fill_mode == MCSPARSE_FILL_MODE_LOWER) ? MCSPARSE_FILL_MODE_UPPER : MCSPARSE_FILL_MODE_LOWER;
    } else {
        working_bsr_vals = bsr_vals;
        working_bsr_rows = bsr_rows_ind;
        working_bsr_cols = bsr_cols_ind;
    }

    buffer = reinterpret_cast<void*>(reinterpret_cast<char*>(buffer) + info->mat_info->fixed_length_buffer_size);
    if (working_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        if (info->mat_info->bsrsv_lower_info == nullptr) {
            ret = mcspCreateTrmInfo(&(info->mat_info->bsrsv_lower_info));
            if (ret != MCSP_STATUS_SUCCESS) {
                return ret;
            }
            ret = mcspCsrTrmAnalysis_template(handle, mb, nnzb, bsr_descr, working_bsr_vals, working_bsr_rows,
                                              working_bsr_cols, info->mat_info->bsrsv_lower_info,
                                              info->mat_info->zero_pivot_lead, buffer, true);
            if (ret != MCSP_STATUS_SUCCESS) {
                return ret;
            }
        }
    } else if (working_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        if (info->mat_info->bsrsv_upper_info == nullptr) {
            ret = mcspCreateTrmInfo(&(info->mat_info->bsrsv_upper_info));
            if (ret != MCSP_STATUS_SUCCESS) {
                return ret;
            }
            ret = mcspCsrTrmAnalysis_template(handle, mb, nnzb, bsr_descr, working_bsr_vals, working_bsr_rows,
                                              working_bsr_cols, info->mat_info->bsrsv_upper_info,
                                              info->mat_info->zero_pivot_lead, buffer, false);
            if (ret != MCSP_STATUS_SUCCESS) {
                return ret;
            }
        }
    }
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrsvSolveTemplate(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, idxType mb,
                                    idxType nnzb, const valType* alpha, const mcspMatDescr_t bsr_descr,
                                    const valType* bsr_vals, const idxType* bsr_rows_ind, const idxType* bsr_cols_ind,
                                    idxType block_dim, mcspBsrsv2Info_t info, const valType* x, valType* y,
                                    mcsparseSolvePolicy_t policy, void* buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (mb < 0 || nnzb < 0 || block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (bsr_descr == nullptr || buffer == nullptr || info == nullptr || info->mat_info == nullptr || alpha == nullptr ||
        x == nullptr || y == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (bsr_descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (bsr_descr->fill_mode == MCSPARSE_FILL_MODE_FULL) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE && trans != MCSPARSE_OPERATION_TRANSPOSE &&
        trans != MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (trans == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double>) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    if (mb == 0 || nnzb == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows_ind == nullptr || bsr_cols_ind == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    const valType* working_bsr_vals;
    const idxType* working_bsr_rows;
    const idxType* working_bsr_cols;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcsparseFillMode_t working_fill_mode = bsr_descr->fill_mode;
    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        working_bsr_vals = (valType*)info->mat_info->bsrt_vals;
        working_bsr_rows = (idxType*)info->mat_info->bsrt_rows;
        working_bsr_cols = (idxType*)info->mat_info->bsrt_cols;
        working_fill_mode =
            (bsr_descr->fill_mode == MCSPARSE_FILL_MODE_LOWER) ? MCSPARSE_FILL_MODE_UPPER : MCSPARSE_FILL_MODE_LOWER;
    } else {
        working_bsr_vals = bsr_vals;
        working_bsr_rows = bsr_rows_ind;
        working_bsr_cols = bsr_cols_ind;
    }

    mcspTrmInfo_t trm_info;
    if (working_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        trm_info = info->mat_info->bsrsv_lower_info;
    } else if (working_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        trm_info = info->mat_info->bsrsv_upper_info;
    }

    if (trm_info == nullptr || trm_info->row_map == nullptr || trm_info->trm_diag_ind == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    buffer = reinterpret_cast<void*>(reinterpret_cast<char*>(buffer) + info->mat_info->fixed_length_buffer_size);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    MACA_ASSERT(mcMemsetAsync(trm_info->zero_pivot_lead, -1, sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    if (block_dim * WARP_SIZE <= 512) {
        MACA_ASSERT(mcMemsetAsync(buffer, 0, mb * sizeof(idxType), stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        idxType block_size = block_dim * WARP_SIZE;
        idxType blocks = mb;
        if (working_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
            if (bsr_descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
                mcLaunchKernelGGL((mcspBsrsvSolveSharedKernel<true, false>), dim3(blocks), dim3(block_size),
                                   (WARP_SIZE + 1) * block_dim * sizeof(valType), stream, mb, block_dim,
                                   bsr_descr->base, dir, h_alpha, working_bsr_vals, working_bsr_rows, working_bsr_cols,
                                   (idxType*)(trm_info->row_map), (idxType*)(trm_info->trm_diag_ind), x, y,
                                   (idxType*)buffer, (idxType*)trm_info->zero_pivot_lead);
            } else if (bsr_descr->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
                mcLaunchKernelGGL((mcspBsrsvSolveSharedKernel<true, true>), dim3(blocks), dim3(block_size),
                                   (WARP_SIZE + 1) * block_dim * sizeof(valType), stream, mb, block_dim,
                                   bsr_descr->base, dir, h_alpha, working_bsr_vals, working_bsr_rows, working_bsr_cols,
                                   (idxType*)(trm_info->row_map), (idxType*)(trm_info->trm_diag_ind), x, y,
                                   (idxType*)buffer, (idxType*)trm_info->zero_pivot_lead);
            }
        } else if (working_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
            if (bsr_descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
                mcLaunchKernelGGL((mcspBsrsvSolveSharedKernel<false, false>), dim3(blocks), dim3(block_size),
                                   (WARP_SIZE + 1) * block_dim * sizeof(valType), stream, mb, block_dim,
                                   bsr_descr->base, dir, h_alpha, working_bsr_vals, working_bsr_rows, working_bsr_cols,
                                   (idxType*)(trm_info->row_map), (idxType*)(trm_info->trm_diag_ind), x, y,
                                   (idxType*)buffer, (idxType*)trm_info->zero_pivot_lead);
            } else if (bsr_descr->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
                mcLaunchKernelGGL((mcspBsrsvSolveSharedKernel<false, true>), dim3(blocks), dim3(block_size),
                                   (WARP_SIZE + 1) * block_dim * sizeof(valType), stream, mb, block_dim,
                                   bsr_descr->base, dir, h_alpha, working_bsr_vals, working_bsr_rows, working_bsr_cols,
                                   (idxType*)(trm_info->row_map), (idxType*)(trm_info->trm_diag_ind), x, y,
                                   (idxType*)buffer, (idxType*)trm_info->zero_pivot_lead);
            }
        }
    } else {
        MACA_ASSERT(mcMemsetAsync(buffer, 0, (mb + mb * block_dim) * sizeof(idxType), stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        constexpr idxType block_size = WARP_SIZE;
        idxType blocks = mb * block_dim;
        if (working_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
            if (bsr_descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
                mcLaunchKernelGGL((mcspBsrsvSolveGlobalKernel<true, false>), dim3(blocks), dim3(block_size), 0, stream,
                                   mb, block_dim, bsr_descr->base, dir, h_alpha, working_bsr_vals, working_bsr_rows,
                                   working_bsr_cols, (idxType*)(trm_info->row_map), (idxType*)(trm_info->trm_diag_ind),
                                   x, y, (idxType*)buffer, (idxType*)((char*)buffer + mb * sizeof(idxType)),
                                   (idxType*)trm_info->zero_pivot_lead);
            } else if (bsr_descr->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
                mcLaunchKernelGGL((mcspBsrsvSolveGlobalKernel<true, true>), dim3(blocks), dim3(block_size), 0, stream,
                                   mb, block_dim, bsr_descr->base, dir, h_alpha, working_bsr_vals, working_bsr_rows,
                                   working_bsr_cols, (idxType*)(trm_info->row_map), (idxType*)(trm_info->trm_diag_ind),
                                   x, y, (idxType*)buffer, (idxType*)((char*)buffer + mb * sizeof(idxType)),
                                   (idxType*)trm_info->zero_pivot_lead);
            }
        } else if (working_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
            if (bsr_descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
                mcLaunchKernelGGL(
                    (mcspBsrsvSolveGlobalKernel<false, false>), dim3(blocks), dim3(block_size), 0, stream, mb,
                    block_dim, bsr_descr->base, dir, h_alpha, working_bsr_vals, working_bsr_rows, working_bsr_cols,
                    (idxType*)(trm_info->row_map), (idxType*)(trm_info->trm_diag_ind), x, y, (idxType*)buffer,
                    (idxType*)((char*)buffer + mb * sizeof(idxType)), (idxType*)trm_info->zero_pivot_lead);
            } else if (bsr_descr->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
                mcLaunchKernelGGL((mcspBsrsvSolveGlobalKernel<false, true>), dim3(blocks), dim3(block_size), 0, stream,
                                   mb, block_dim, bsr_descr->base, dir, h_alpha, working_bsr_vals, working_bsr_rows,
                                   working_bsr_cols, (idxType*)(trm_info->row_map), (idxType*)(trm_info->trm_diag_ind),
                                   x, y, (idxType*)buffer, (idxType*)((char*)buffer + mb * sizeof(idxType)),
                                   (idxType*)trm_info->zero_pivot_lead);
            }
        }
    }
    MACA_ASSERT(mcMemcpyAsync(&info->mat_info->zero_pivot_lead, trm_info->zero_pivot_lead, sizeof(idxType),
                              mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspBsrsvZeroPivotTemplate(mcspHandle_t handle, mcspBsrsv2Info_t info, idxType* position) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (info == nullptr || info->mat_info == nullptr || position == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *position = (mcspInt)(info->mat_info->zero_pivot_lead);
    if (info->mat_info->zero_pivot_lead != -1) {
        return MCSP_STATUS_ZERO_PIVOT;
    }
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSbsrsvBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                  mcspInt nnzb, const mcspMatDescr_t bsr_descr, const float* bsr_vals,
                                  const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                                  mcspBsrsv2Info_t info, size_t* buffer_size) {
    return mcspBsrsvBufferSizeTemplate(handle, dir, trans, mb, nnzb, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                       block_dim, info, buffer_size);
}

mcspStatus_t mcspDbsrsvBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                  mcspInt nnzb, const mcspMatDescr_t bsr_descr, const double* bsr_vals,
                                  const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                                  mcspBsrsv2Info_t info, size_t* buffer_size) {
    return mcspBsrsvBufferSizeTemplate(handle, dir, trans, mb, nnzb, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                       block_dim, info, buffer_size);
}

mcspStatus_t mcspCbsrsvBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                  mcspInt nnzb, const mcspMatDescr_t bsr_descr, const mcspComplexFloat* bsr_vals,
                                  const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                                  mcspBsrsv2Info_t info, size_t* buffer_size) {
    return mcspBsrsvBufferSizeTemplate(handle, dir, trans, mb, nnzb, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                       block_dim, info, buffer_size);
}

mcspStatus_t mcspZbsrsvBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                  mcspInt nnzb, const mcspMatDescr_t bsr_descr, const mcspComplexDouble* bsr_vals,
                                  const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                                  mcspBsrsv2Info_t info, size_t* buffer_size) {
    return mcspBsrsvBufferSizeTemplate(handle, dir, trans, mb, nnzb, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                       block_dim, info, buffer_size);
}

mcspStatus_t mcspSbsrsvAnalysis(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                mcspInt nnzb, const mcspMatDescr_t bsr_descr, const float* bsr_vals,
                                const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                                mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsrsvAnalysisTemplate(handle, dir, trans, mb, nnzb, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                     block_dim, info, policy, buffer);
}

mcspStatus_t mcspDbsrsvAnalysis(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                mcspInt nnzb, const mcspMatDescr_t bsr_descr, const double* bsr_vals,
                                const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                                mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsrsvAnalysisTemplate(handle, dir, trans, mb, nnzb, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                     block_dim, info, policy, buffer);
}

mcspStatus_t mcspCbsrsvAnalysis(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                mcspInt nnzb, const mcspMatDescr_t bsr_descr, const mcspComplexFloat* bsr_vals,
                                const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                                mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsrsvAnalysisTemplate(handle, dir, trans, mb, nnzb, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                     block_dim, info, policy, buffer);
}

mcspStatus_t mcspZbsrsvAnalysis(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                mcspInt nnzb, const mcspMatDescr_t bsr_descr, const mcspComplexDouble* bsr_vals,
                                const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                                mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsrsvAnalysisTemplate(handle, dir, trans, mb, nnzb, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                     block_dim, info, policy, buffer);
}

mcspStatus_t mcspSbsrsvSolve(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                             mcspInt nnzb, const float* alpha, const mcspMatDescr_t bsr_descr, const float* bsr_vals,
                             const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                             mcspBsrsv2Info_t info, const float* x, float* y, mcsparseSolvePolicy_t policy,
                             void* buffer) {
    return mcspBsrsvSolveTemplate(handle, dir, trans, mb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                  block_dim, info, x, y, policy, buffer);
}

mcspStatus_t mcspDbsrsvSolve(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                             mcspInt nnzb, const double* alpha, const mcspMatDescr_t bsr_descr, const double* bsr_vals,
                             const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind, mcspInt block_dim,
                             mcspBsrsv2Info_t info, const double* x, double* y, mcsparseSolvePolicy_t policy,
                             void* buffer) {
    return mcspBsrsvSolveTemplate(handle, dir, trans, mb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                  block_dim, info, x, y, policy, buffer);
}

mcspStatus_t mcspCbsrsvSolve(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                             mcspInt nnzb, const mcspComplexFloat* alpha, const mcspMatDescr_t bsr_descr,
                             const mcspComplexFloat* bsr_vals, const mcspInt* bsr_rows_ind, const mcspInt* bsr_cols_ind,
                             mcspInt block_dim, mcspBsrsv2Info_t info, const mcspComplexFloat* x, mcspComplexFloat* y,
                             mcsparseSolvePolicy_t policy, void* buffer) {
    return mcspBsrsvSolveTemplate(handle, dir, trans, mb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                  block_dim, info, x, y, policy, buffer);
}

mcspStatus_t mcspZbsrsvSolve(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                             mcspInt nnzb, const mcspComplexDouble* alpha, const mcspMatDescr_t bsr_descr,
                             const mcspComplexDouble* bsr_vals, const mcspInt* bsr_rows_ind,
                             const mcspInt* bsr_cols_ind, mcspInt block_dim, mcspBsrsv2Info_t info,
                             const mcspComplexDouble* x, mcspComplexDouble* y, mcsparseSolvePolicy_t policy,
                             void* buffer) {
    return mcspBsrsvSolveTemplate(handle, dir, trans, mb, nnzb, alpha, bsr_descr, bsr_vals, bsr_rows_ind, bsr_cols_ind,
                                  block_dim, info, x, y, policy, buffer);
}

mcspStatus_t mcspBsrsvZeroPivot(mcspHandle_t handle, mcspBsrsv2Info_t info, int* position) {
    return mcspBsrsvZeroPivotTemplate(handle, info, position);
}

mcspStatus_t mcspCuinXbsrsv2_zeroPivot(mcspHandle_t handle, mcspBsrsv2Info_t info, int* position) {
    return mcspBsrsvZeroPivotTemplate(handle, info, (mcspInt*)position);
}

mcspStatus_t mcspCuinSbsrsv2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                      int nnzb, const mcspMatDescr_t descrA, float* bsrSortedValA,
                                      const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                      mcspBsrsv2Info_t info, int* pBufferSizeInBytes) {
    return mcspBsrsv_bufferSizeTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                        (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info,
                                        pBufferSizeInBytes);
}

mcspStatus_t mcspCuinDbsrsv2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                      int nnzb, const mcspMatDescr_t descrA, double* bsrSortedValA,
                                      const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                      mcspBsrsv2Info_t info, int* pBufferSizeInBytes) {
    return mcspBsrsv_bufferSizeTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                        (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info,
                                        pBufferSizeInBytes);
}

mcspStatus_t mcspCuinCbsrsv2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                      int nnzb, const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedValA,
                                      const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                      mcspBsrsv2Info_t info, int* pBufferSizeInBytes) {
    return mcspBsrsv_bufferSizeTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                        (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info,
                                        pBufferSizeInBytes);
}

mcspStatus_t mcspCuinZbsrsv2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                      int nnzb, const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedValA,
                                      const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                      mcspBsrsv2Info_t info, int* pBufferSizeInBytes) {
    return mcspBsrsv_bufferSizeTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                        (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info,
                                        pBufferSizeInBytes);
}

mcspStatus_t mcspCuinSbsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                         int mb, int nnzb, const mcspMatDescr_t descrA, float* bsrSortedValA,
                                         const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize,
                                         mcspBsrsv2Info_t info, size_t* pBufferSize) {
    return mcspBsrsvBufferSizeTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                       (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockSize, info,
                                       pBufferSize);
}

mcspStatus_t mcspCuinDbsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                         int mb, int nnzb, const mcspMatDescr_t descrA, double* bsrSortedValA,
                                         const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize,
                                         mcspBsrsv2Info_t info, size_t* pBufferSize) {
    return mcspBsrsvBufferSizeTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                       (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockSize, info,
                                       pBufferSize);
}

mcspStatus_t mcspCuinCbsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                         int mb, int nnzb, const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedValA,
                                         const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize,
                                         mcspBsrsv2Info_t info, size_t* pBufferSize) {
    return mcspBsrsvBufferSizeTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                       (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockSize, info,
                                       pBufferSize);
}

mcspStatus_t mcspCuinZbsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                         int mb, int nnzb, const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedValA,
                                         const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize,
                                         mcspBsrsv2Info_t info, size_t* pBufferSize) {
    return mcspBsrsvBufferSizeTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                       (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockSize, info,
                                       pBufferSize);
}

mcspStatus_t mcspCuinSbsrsv2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                    int nnzb, const mcspMatDescr_t descrA, const float* bsrSortedValA,
                                    const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                    mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsvAnalysisTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                     (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info,
                                     policy, pBuffer);
}

mcspStatus_t mcspCuinDbsrsv2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                    int nnzb, const mcspMatDescr_t descrA, const double* bsrSortedValA,
                                    const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                    mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsvAnalysisTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                     (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info,
                                     policy, pBuffer);
}

mcspStatus_t mcspCuinCbsrsv2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                    int nnzb, const mcspMatDescr_t descrA, const mcFloatComplex* bsrSortedValA,
                                    const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                    mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsvAnalysisTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                     (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info,
                                     policy, pBuffer);
}

mcspStatus_t mcspCuinZbsrsv2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                    int nnzb, const mcspMatDescr_t descrA, const mcDoubleComplex* bsrSortedValA,
                                    const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                    mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsvAnalysisTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedValA,
                                     (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info,
                                     policy, pBuffer);
}

mcspStatus_t mcspCuinSbsrsv2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                 int nnzb, const float* alpha, const mcspMatDescr_t descrA, const float* bsrSortedValA,
                                 const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                 mcspBsrsv2Info_t info, const float* f, float* x, mcsparseSolvePolicy_t policy,
                                 void* pBuffer) {
    return mcspBsrsvSolveTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, alpha, descrA, bsrSortedValA,
                                  (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info, f, x,
                                  policy, pBuffer);
}

mcspStatus_t mcspCuinDbsrsv2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                 int nnzb, const double* alpha, const mcspMatDescr_t descrA,
                                 const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA,
                                 int blockDim, mcspBsrsv2Info_t info, const double* f, double* x,
                                 mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsvSolveTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, alpha, descrA, bsrSortedValA,
                                  (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info, f, x,
                                  policy, pBuffer);
}

mcspStatus_t mcspCuinCbsrsv2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                 int nnzb, const mcFloatComplex* alpha, const mcspMatDescr_t descrA,
                                 const mcFloatComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                 const int* bsrSortedColIndA, int blockDim, mcspBsrsv2Info_t info,
                                 const mcFloatComplex* f, mcFloatComplex* x, mcsparseSolvePolicy_t policy,
                                 void* pBuffer) {
    return mcspBsrsvSolveTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, alpha, descrA, bsrSortedValA,
                                  (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info, f, x,
                                  policy, pBuffer);
}

mcspStatus_t mcspCuinZbsrsv2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                 int nnzb, const mcDoubleComplex* alpha, const mcspMatDescr_t descrA,
                                 const mcDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                 const int* bsrSortedColIndA, int blockDim, mcspBsrsv2Info_t info,
                                 const mcDoubleComplex* f, mcDoubleComplex* x, mcsparseSolvePolicy_t policy,
                                 void* pBuffer) {
    return mcspBsrsvSolveTemplate(handle, dirA, transA, (mcspInt)mb, (mcspInt)nnzb, alpha, descrA, bsrSortedValA,
                                  (mcspInt*)bsrSortedRowPtrA, (mcspInt*)bsrSortedColIndA, (mcspInt)blockDim, info, f, x,
                                  policy, pBuffer);
}
#ifdef __cplusplus
}
#endif