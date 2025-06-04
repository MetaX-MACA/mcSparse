#include "bsrsm_device.hpp"
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

mcspStatus_t mcspBsrsmZeroPivot(mcspHandle_t handle, mcspBsrsm2Info_t info, int* position) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || position == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *position = (mcspInt)(info->mat_info->zero_pivot_lead);
    if (info->mat_info->zero_pivot_lead != -1) {
        return MCSP_STATUS_ZERO_PIVOT;
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrsmBufferSizeTemplate(mcspHandle_t handle, mcsparseDirection_t dir_A, mcsparseOperation_t transA,
                                         mcsparseOperation_t transXY, idxType mb, idxType n, idxType nnzb,
                                         const mcspMatDescr_t descrA, valType* bsr_vals, const idxType* bsr_rows,
                                         const idxType* bsr_cols, idxType block_dim, mcspBsrsm2Info_t info,
                                         size_t* buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if ((dir_A != MCSPARSE_DIRECTION_ROW && dir_A != MCSPARSE_DIRECTION_COLUMN) ||
        (transA != MCSPARSE_OPERATION_NON_TRANSPOSE && transA != MCSPARSE_OPERATION_TRANSPOSE &&
         transA != MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) ||
        (transXY != MCSPARSE_OPERATION_NON_TRANSPOSE && transXY != MCSPARSE_OPERATION_TRANSPOSE)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (transA == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double>) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    if (mb < 0 || n < 0 || nnzb < 0 || block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (descrA == nullptr || info == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descrA->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (descrA->fill_mode == MCSPARSE_FILL_MODE_FULL) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (mb == 0 || n == 0 || nnzb == 0 || block_dim == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows == nullptr || bsr_cols == nullptr) {
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
    *buffer_size += 4 * sizeof(idxType);                         // reduce output buffer
    *buffer_size += mb * sizeof(idxType);                        // row nnz buffer with length of mb
    *buffer_size += mb * sizeof(idxType);                        // permutation buffer with length of mb
    *buffer_size += mb * sizeof(idxType);                        // row depth buffer with length of mb
    *buffer_size += n * (block_dim + 1) * mb * sizeof(idxType);  // solved row buffer

    mcspStatus_t ret = MCSP_STATUS_SUCCESS;
    if (transA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        size_t trans_buffer_size = 0;
        mcspStatus_t ret = mcspGetBsrTransExBuffersize<valType>(handle, mb, mb, nnzb, block_dim, block_dim, bsr_rows,
                                                                bsr_cols, bsr_vals, info->mat_info, trans_buffer_size);
        if (ret == MCSP_STATUS_SUCCESS) {
            *buffer_size = info->mat_info->fixed_length_buffer_size + std::max(*buffer_size, trans_buffer_size);
        }
    }
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrsm_bufferSizeTemplate(mcspHandle_t handle, mcsparseDirection_t dir_A, mcsparseOperation_t transA,
                                          mcsparseOperation_t transXY, idxType mb, idxType n, idxType nnzb,
                                          const mcspMatDescr_t descrA, valType* bsr_vals, const idxType* bsr_rows,
                                          const idxType* bsr_cols, idxType block_dim, mcspBsrsm2Info_t info,
                                          int* buffer_size) {
    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    size_t temp_size = 0;
    mcspStatus_t ret = mcspBsrsmBufferSizeTemplate(handle, dir_A, transA, transXY, mb, n, nnzb, descrA, bsr_vals,
                                                   bsr_rows, bsr_cols, block_dim, info, &temp_size);
    *buffer_size = temp_size;
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrsmAnalysisTemplate(mcspHandle_t handle, mcsparseDirection_t dir_A, mcsparseOperation_t transA,
                                       mcsparseOperation_t transXY, idxType mb, idxType n, idxType nnzb,
                                       const mcspMatDescr_t descrA, const valType* bsr_vals, const idxType* bsr_rows,
                                       const idxType* bsr_cols, idxType block_dim, mcspBsrsm2Info_t info,
                                       mcsparseSolvePolicy_t policy, void* buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if ((dir_A != MCSPARSE_DIRECTION_ROW && dir_A != MCSPARSE_DIRECTION_COLUMN) ||
        (transA != MCSPARSE_OPERATION_NON_TRANSPOSE && transA != MCSPARSE_OPERATION_TRANSPOSE &&
         transA != MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) ||
        (transXY != MCSPARSE_OPERATION_NON_TRANSPOSE && transXY != MCSPARSE_OPERATION_TRANSPOSE)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (transA == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double>) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    if (mb < 0 || n < 0 || nnzb < 0 || block_dim < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (descrA == nullptr || info == nullptr || buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descrA->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (descrA->fill_mode == MCSPARSE_FILL_MODE_FULL) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (mb == 0 || n == 0 || nnzb == 0 || block_dim == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows == nullptr || bsr_cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcspStatus_t ret = MCSP_STATUS_SUCCESS;
    const valType* working_bsr_vals;
    const idxType* working_bsr_rows;
    const idxType* working_bsr_cols;
    mcsparseFillMode_t working_fill_mode = descrA->fill_mode;

    if (transA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        ret = mcspRawTransposeBsr<valType>(handle, transA, mb, mb, nnzb, block_dim, block_dim, bsr_rows, bsr_cols,
                                           bsr_vals, descrA->base, info->mat_info, buffer);
        if (ret != MCSP_STATUS_SUCCESS) {
            return ret;
        }
        info->mat_info->bsrt_dir = dir_A;
        working_bsr_vals = (valType*)info->mat_info->bsrt_vals;
        working_bsr_rows = (idxType*)info->mat_info->bsrt_rows;
        working_bsr_cols = (idxType*)info->mat_info->bsrt_cols;
        working_fill_mode =
            (descrA->fill_mode == MCSPARSE_FILL_MODE_LOWER) ? MCSPARSE_FILL_MODE_UPPER : MCSPARSE_FILL_MODE_LOWER;
    } else {
        working_bsr_vals = bsr_vals;
        working_bsr_rows = bsr_rows;
        working_bsr_cols = bsr_cols;
    }

    buffer = reinterpret_cast<void*>(reinterpret_cast<char*>(buffer) + info->mat_info->fixed_length_buffer_size);
    if (working_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        if (info->mat_info->bsrsm_lower_info == nullptr) {
            ret = mcspCreateTrmInfo(&(info->mat_info->bsrsm_lower_info));
            if (ret != MCSP_STATUS_SUCCESS) {
                return ret;
            }
            ret = mcspCsrTrmAnalysis_template(handle, mb, nnzb, descrA, working_bsr_vals, working_bsr_rows,
                                              working_bsr_cols, info->mat_info->bsrsm_lower_info,
                                              info->mat_info->zero_pivot_lead, buffer, true);
        }
    } else if (working_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        if (info->mat_info->bsrsm_upper_info == nullptr) {
            ret = mcspCreateTrmInfo(&(info->mat_info->bsrsm_upper_info));
            if (ret != MCSP_STATUS_SUCCESS) {
                return ret;
            }
            ret = mcspCsrTrmAnalysis_template(handle, mb, nnzb, descrA, working_bsr_vals, working_bsr_rows,
                                              working_bsr_cols, info->mat_info->bsrsm_upper_info,
                                              info->mat_info->zero_pivot_lead, buffer, false);
        }
    }

    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspBsrsmSolveTemplate(mcspHandle_t handle, mcsparseDirection_t dir_A, mcsparseOperation_t transA,
                                    mcsparseOperation_t transXY, idxType mb, idxType n, idxType nnzb,
                                    const valType* alpha, const mcspMatDescr_t descrA, const valType* bsr_vals,
                                    const idxType* bsr_rows, const idxType* bsr_cols, idxType block_dim,
                                    mcspBsrsm2Info_t info, const valType* B, idxType ldb, valType* X, idxType ldx,
                                    mcsparseSolvePolicy_t policy, void* buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if ((dir_A != MCSPARSE_DIRECTION_ROW && dir_A != MCSPARSE_DIRECTION_COLUMN) ||
        (transA != MCSPARSE_OPERATION_NON_TRANSPOSE && transA != MCSPARSE_OPERATION_TRANSPOSE &&
         transA != MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) ||
        (transXY != MCSPARSE_OPERATION_NON_TRANSPOSE && transXY != MCSPARSE_OPERATION_TRANSPOSE)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (transA == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double>) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }
    if (mb < 0 || n < 0 || nnzb < 0 || block_dim < 0 || ldb < 0 || ldx < 0 ||
        (transXY == MCSPARSE_OPERATION_NON_TRANSPOSE && (ldx < mb * block_dim || ldb < mb * block_dim)) ||
        (transXY != MCSPARSE_OPERATION_NON_TRANSPOSE && (ldx < n || ldb < n))) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (descrA == nullptr || info == nullptr || buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descrA->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (descrA->fill_mode == MCSPARSE_FILL_MODE_FULL) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (mb == 0 || n == 0 || nnzb == 0 || block_dim == 0 || ldb == 0 || ldx == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (bsr_vals == nullptr || bsr_rows == nullptr || bsr_cols == nullptr || B == nullptr || X == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    const valType* working_bsr_vals;
    const idxType* working_bsr_rows;
    const idxType* working_bsr_cols;
    mcsparseFillMode_t working_fill_mode = descrA->fill_mode;
    if (transA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        working_bsr_vals = (valType*)info->mat_info->bsrt_vals;
        working_bsr_rows = (idxType*)info->mat_info->bsrt_rows;
        working_bsr_cols = (idxType*)info->mat_info->bsrt_cols;
        working_fill_mode =
            (descrA->fill_mode == MCSPARSE_FILL_MODE_LOWER) ? MCSPARSE_FILL_MODE_UPPER : MCSPARSE_FILL_MODE_LOWER;
    } else {
        working_bsr_vals = bsr_vals;
        working_bsr_rows = bsr_rows;
        working_bsr_cols = bsr_cols;
    }

    mcspTrmInfo_t trm_info;
    if (working_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        trm_info = info->mat_info->bsrsm_lower_info;
    } else if (working_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        trm_info = info->mat_info->bsrsm_upper_info;
    }

    if (trm_info == nullptr || trm_info->row_map == nullptr || trm_info->trm_diag_ind == nullptr ||
        trm_info->zero_pivot_lead == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    buffer = reinterpret_cast<void*>(reinterpret_cast<char*>(buffer) + info->mat_info->fixed_length_buffer_size);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemsetAsync((void*)trm_info->zero_pivot_lead, 0xFFFFFFFF, sizeof(idxType), stream));
    char* ptr = reinterpret_cast<char*>(buffer);
    idxType* bsr_done_buffer = reinterpret_cast<idxType*>(ptr);
    MACA_ASSERT(mcMemsetAsync(bsr_done_buffer, 0, n * mb * sizeof(idxType), stream));
    idxType* done_buffer = reinterpret_cast<idxType*>(ptr + n * mb * sizeof(idxType));
    MACA_ASSERT(mcMemsetAsync(done_buffer, 0, n * mb * block_dim * sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    constexpr uint32_t block_size = 512;
    constexpr uint32_t n_col = block_size / WARP_SIZE;
    dim3 blocks(mb * block_dim, (n - 1) / n_col + 1);
    if (working_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        if (descrA->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
            mcLaunchKernelGGL((mcspBsrsmSolveKernel<true, false, n_col, block_size>), blocks, dim3(block_size), 0,
                               stream, mb, n, block_dim, descrA->base, dir_A, h_alpha, working_bsr_vals,
                               working_bsr_rows, working_bsr_cols, (idxType*)(trm_info->row_map),
                               (idxType*)(trm_info->trm_diag_ind), X, ldx, B, ldb, bsr_done_buffer, done_buffer,
                               (idxType*)trm_info->zero_pivot_lead, transXY);
        } else if (descrA->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
            mcLaunchKernelGGL((mcspBsrsmSolveKernel<true, true, n_col, block_size>), blocks, dim3(block_size), 0,
                               stream, mb, n, block_dim, descrA->base, dir_A, h_alpha, working_bsr_vals,
                               working_bsr_rows, working_bsr_cols, (idxType*)(trm_info->row_map),
                               (idxType*)(trm_info->trm_diag_ind), X, ldx, B, ldb, bsr_done_buffer, done_buffer,
                               (idxType*)trm_info->zero_pivot_lead, transXY);
        }
    } else if (working_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        if (descrA->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
            mcLaunchKernelGGL((mcspBsrsmSolveKernel<false, false, n_col, block_size>), blocks, dim3(block_size), 0,
                               stream, mb, n, block_dim, descrA->base, dir_A, h_alpha, working_bsr_vals,
                               working_bsr_rows, working_bsr_cols, (idxType*)(trm_info->row_map),
                               (idxType*)(trm_info->trm_diag_ind), X, ldx, B, ldb, bsr_done_buffer, done_buffer,
                               (idxType*)trm_info->zero_pivot_lead, transXY);
        } else if (descrA->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
            mcLaunchKernelGGL((mcspBsrsmSolveKernel<false, true, n_col, block_size>), blocks, dim3(block_size), 0,
                               stream, mb, n, block_dim, descrA->base, dir_A, h_alpha, working_bsr_vals,
                               working_bsr_rows, working_bsr_cols, (idxType*)(trm_info->row_map),
                               (idxType*)(trm_info->trm_diag_ind), X, ldx, B, ldb, bsr_done_buffer, done_buffer,
                               (idxType*)trm_info->zero_pivot_lead, transXY);
        }
    }
    MACA_ASSERT(mcStreamSynchronize(stream));

    MACA_ASSERT(mcMemcpyAsync(&info->mat_info->zero_pivot_lead, trm_info->zero_pivot_lead, sizeof(idxType),
                              mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspXbsrsm2_zeroPivot(mcspHandle_t handle, mcspBsrsm2Info_t info, int* position) {
    return mcspBsrsmZeroPivot(handle, info, position);
}

mcspStatus_t mcspSbsrsm2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                    mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, float* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                    const mcspInt* bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                                    int* pBufferSizeInBytes) {
    return mcspBsrsm_bufferSizeTemplate(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                                        bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes);
}

mcspStatus_t mcspDbsrsm2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                    mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, double* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                    const mcspInt* bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                                    int* pBufferSizeInBytes) {
    return mcspBsrsm_bufferSizeTemplate(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                                        bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCbsrsm2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                    mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                    const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockSize,
                                    mcspBsrsm2Info_t info, int* pBufferSizeInBytes) {
    return mcspBsrsm_bufferSizeTemplate(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                                        bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes);
}

mcspStatus_t mcspZbsrsm2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                    mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                    const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockSize,
                                    mcspBsrsm2Info_t info, int* pBufferSizeInBytes) {
    return mcspBsrsm_bufferSizeTemplate(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                                        bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes);
}

mcspStatus_t mcspSbsrsm2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, mcspInt mb, mcspInt n, mcspInt nnzb,
                                       const mcspMatDescr_t descrA, float* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                       const mcspInt* bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                                       size_t* pBufferSize) {
    return mcspBsrsmBufferSizeTemplate(handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                       bsrSortedColInd, blockSize, info, pBufferSize);
}

mcspStatus_t mcspDbsrsm2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, mcspInt mb, mcspInt n, mcspInt nnzb,
                                       const mcspMatDescr_t descrA, double* bsrSortedVal,
                                       const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                       mcspInt blockSize, mcspBsrsm2Info_t info, size_t* pBufferSize) {
    return mcspBsrsmBufferSizeTemplate(handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                       bsrSortedColInd, blockSize, info, pBufferSize);
}

mcspStatus_t mcspCbsrsm2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, mcspInt mb, mcspInt n, mcspInt nnzb,
                                       const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                       const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                       mcspInt blockSize, mcspBsrsm2Info_t info, size_t* pBufferSize) {
    return mcspBsrsmBufferSizeTemplate(handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                       bsrSortedColInd, blockSize, info, pBufferSize);
}

mcspStatus_t mcspZbsrsm2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, mcspInt mb, mcspInt n, mcspInt nnzb,
                                       const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                       const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                       mcspInt blockSize, mcspBsrsm2Info_t info, size_t* pBufferSize) {
    return mcspBsrsmBufferSizeTemplate(handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                       bsrSortedColInd, blockSize, info, pBufferSize);
}

mcspStatus_t mcspSbsrsm2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                  mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                  const mcspMatDescr_t descrA, const float* bsrSortedVal,
                                  const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockSize,
                                  mcspBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsmAnalysisTemplate(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                     bsrSortedColInd, blockSize, info, policy, pBuffer);
}

mcspStatus_t mcspDbsrsm2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                  mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                  const mcspMatDescr_t descrA, const double* bsrSortedVal,
                                  const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockSize,
                                  mcspBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsmAnalysisTemplate(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                     bsrSortedColInd, blockSize, info, policy, pBuffer);
}

mcspStatus_t mcspCbsrsm2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                  mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                  const mcspMatDescr_t descrA, const mcFloatComplex* bsrSortedVal,
                                  const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockSize,
                                  mcspBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsmAnalysisTemplate(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                     bsrSortedColInd, blockSize, info, policy, pBuffer);
}

mcspStatus_t mcspZbsrsm2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                  mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                  const mcspMatDescr_t descrA, const mcDoubleComplex* bsrSortedVal,
                                  const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockSize,
                                  mcspBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsmAnalysisTemplate(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                                     bsrSortedColInd, blockSize, info, policy, pBuffer);
}

mcspStatus_t mcspSbsrsm2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                               mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb, const float* alpha,
                               const mcspMatDescr_t descrA, const float* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                               const mcspInt* bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info, const float* B,
                               mcspInt ldb, float* X, mcspInt ldx, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsmSolveTemplate(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal,
                                  bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer);
}

mcspStatus_t mcspDbsrsm2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                               mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb, const double* alpha,
                               const mcspMatDescr_t descrA, const double* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                               const mcspInt* bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                               const double* B, mcspInt ldb, double* X, mcspInt ldx, mcsparseSolvePolicy_t policy,
                               void* pBuffer) {
    return mcspBsrsmSolveTemplate(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal,
                                  bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer);
}

mcspStatus_t mcspCbsrsm2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                               mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                               const mcFloatComplex* alpha, const mcspMatDescr_t descrA,
                               const mcFloatComplex* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                               const mcspInt* bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                               const mcFloatComplex* B, mcspInt ldb, mcFloatComplex* X, mcspInt ldx,
                               mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsmSolveTemplate(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal,
                                  bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer);
}

mcspStatus_t mcspZbsrsm2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                               mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                               const mcDoubleComplex* alpha, const mcspMatDescr_t descrA,
                               const mcDoubleComplex* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                               const mcspInt* bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                               const mcDoubleComplex* B, mcspInt ldb, mcDoubleComplex* X, mcspInt ldx,
                               mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspBsrsmSolveTemplate(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal,
                                  bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer);
}
#ifdef __cplusplus
}
#endif