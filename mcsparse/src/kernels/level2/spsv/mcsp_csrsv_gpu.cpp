#include "common/mcsp_types.h"
#include "csr_spsv_device.hpp"
#include "csr_trm_analysis.hpp"
#include "device_reduce.hpp"
#include "internal_interface/mcsp_internal_conversion.h"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_transpose_sparse.hpp"
#include "mcsp_internal_types.h"

template <typename idxType, typename valType>
mcspStatus_t mcspCsrSpsvBuffersize_template(mcspHandle_t handle, mcsparseOperation_t trans, idxType row_num,
                                            idxType nnz, const mcspMatDescr_t descr, const valType *csr_vals,
                                            const idxType *csr_rows, const idxType *csr_cols, mcspMatInfo_t info,
                                            size_t *buffer_size) {
    size_t size = 0;
    size_t reduce_buffersize;
    size_t radix_sort_buffersize;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::reduce(nullptr, reduce_buffersize, (idxType *)nullptr, (idxType *)nullptr, (size_t)row_num,
                   mcprim::minimum<idxType>(), stream);
    size += reduce_buffersize;
    mcprim::radix_sort_pairs(nullptr, radix_sort_buffersize, (idxType *)nullptr, (idxType *)nullptr, (idxType *)nullptr,
                             (idxType *)nullptr, (size_t)row_num, stream);
    size += radix_sort_buffersize;
    size += 4 * sizeof(idxType);        // reduce output buffer
    size += row_num * sizeof(idxType);  // row nnz buffer with length of row_num
    size += row_num * sizeof(idxType);  // permutation buffer with length of row_num
    size += row_num * sizeof(idxType);  // row depth buffer with length of row_num

    *buffer_size = size;

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsrSpsvAnalysis_template(mcspHandle_t handle, mcsparseOperation_t trans, idxType row_num, idxType nnz,
                                          const mcspMatDescr_t descr, const valType *csr_vals, const idxType *csr_rows,
                                          const idxType *csr_cols, mcspMatInfo_t info,
                                          mcsparseAnalysisPolicy_t analysis_policy, mcsparseSolvePolicy_t solve_policy,
                                          void *buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (row_num < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (descr == nullptr || csr_vals == nullptr || csr_rows == nullptr || csr_cols == nullptr || info == nullptr ||
        buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (descr->fill_mode == MCSPARSE_FILL_MODE_FULL) {
        return MCSP_STATUS_INVALID_VALUE;
    }
    if (analysis_policy == MCSPARSE_ANALYSIS_POLICY_REUSE) {
        return MCSP_STATUS_NOT_IMPLEMENTED;  // TO DO
    }

    mcsparseFillMode_t work_fill_mode = descr->fill_mode;
    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        work_fill_mode =
            (descr->fill_mode == MCSPARSE_FILL_MODE_LOWER) ? MCSPARSE_FILL_MODE_UPPER : MCSPARSE_FILL_MODE_LOWER;
    }
    if (work_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        if (info->csr_spsv_lower_info == nullptr) {
            mcspCreateTrmInfo(&(info->csr_spsv_lower_info));
            mcspCsrTrmAnalysis_template(handle, row_num, nnz, descr, csr_vals, csr_rows, csr_cols,
                                        info->csr_spsv_lower_info, info->zero_pivot_lead, buffer, true);
        }

    } else if (work_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        if (info->csr_spsv_upper_info == nullptr) {
            mcspCreateTrmInfo(&(info->csr_spsv_upper_info));
            mcspCsrTrmAnalysis_template(handle, row_num, nnz, descr, csr_vals, csr_rows, csr_cols,
                                        info->csr_spsv_upper_info, info->zero_pivot_lead, buffer, false);
        }
    }
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsrSpsvSolve_template(mcspHandle_t handle, mcsparseOperation_t trans, idxType row_num, idxType nnz,
                                       const valType *alpha, const mcspMatDescr_t descr, const valType *csr_vals,
                                       const idxType *csr_rows, const idxType *csr_cols, mcspMatInfo_t info,
                                       const valType *x, valType *y, mcsparseSolvePolicy_t solve_policy, void *buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (row_num < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (descr == nullptr || csr_vals == nullptr || csr_rows == nullptr || csr_cols == nullptr || info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
    if (descr->fill_mode == MCSPARSE_FILL_MODE_FULL) {
        return MCSP_STATUS_INVALID_VALUE;
    }
    if (descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT && info->zero_pivot_lead != -1) {
        return MCSP_STATUS_ZERO_PIVOT;
    }

    mcsparseFillMode_t work_fill_mode = descr->fill_mode;
    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        work_fill_mode =
            (descr->fill_mode == MCSPARSE_FILL_MODE_LOWER) ? MCSPARSE_FILL_MODE_UPPER : MCSPARSE_FILL_MODE_LOWER;
    }
    mcspTrmInfo_t trm_info;
    if (work_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        trm_info = info->csr_spsv_lower_info;
    } else if (work_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        trm_info = info->csr_spsv_upper_info;
    }

    if (trm_info == nullptr || trm_info->row_map == nullptr || trm_info->trm_diag_ind == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    int n_elem = 256;
    int n_block = (row_num + n_elem / WARP_SIZE - 1) / (n_elem / WARP_SIZE);
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemsetAsync(buffer, 0, row_num * sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    if (work_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        if (descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
            mcLaunchKernelGGL((mcspCsrSpsvSolveKernel<true, false>), dim3(n_block), dim3(n_elem),
                               n_elem * sizeof(valType), stream, row_num, h_alpha, csr_vals, csr_rows, csr_cols,
                               (idxType *)(trm_info->row_map), (idxType *)(trm_info->trm_diag_ind), x, y,
                               (idxType *)buffer, (idxType *)trm_info->zero_pivot_lead, descr->base);
        } else if (descr->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
            mcLaunchKernelGGL((mcspCsrSpsvSolveKernel<true, true>), dim3(n_block), dim3(n_elem),
                               n_elem * sizeof(valType), stream, row_num, h_alpha, csr_vals, csr_rows, csr_cols,
                               (idxType *)(trm_info->row_map), (idxType *)(trm_info->trm_diag_ind), x, y,
                               (idxType *)buffer, (idxType *)trm_info->zero_pivot_lead, descr->base);
        }
    } else if (work_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        if (descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
            mcLaunchKernelGGL((mcspCsrSpsvSolveKernel<false, false>), dim3(n_block), dim3(n_elem),
                               n_elem * sizeof(valType), stream, row_num, h_alpha, csr_vals, csr_rows, csr_cols,
                               (idxType *)(trm_info->row_map), (idxType *)(trm_info->trm_diag_ind), x, y,
                               (idxType *)buffer, (idxType *)trm_info->zero_pivot_lead, descr->base);
        } else if (descr->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
            mcLaunchKernelGGL((mcspCsrSpsvSolveKernel<false, true>), dim3(n_block), dim3(n_elem),
                               n_elem * sizeof(valType), stream, row_num, h_alpha, csr_vals, csr_rows, csr_cols,
                               (idxType *)(trm_info->row_map), (idxType *)(trm_info->trm_diag_ind), x, y,
                               (idxType *)buffer, (idxType *)trm_info->zero_pivot_lead, descr->base);
        }
    }
    MACA_ASSERT(mcStreamSynchronize(stream));
    MACA_ASSERT(mcMemcpyAsync(&info->zero_pivot_lead, trm_info->zero_pivot_lead, sizeof(idxType), mcMemcpyDeviceToHost,
                              stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCsrSpsvClearImpl(mcspHandle_t handle, const mcspMatDescr_t descr, mcspMatInfo_t info,
                                  mcsparseOperation_t opA) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (descr == nullptr || info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcsparseFillMode_t work_fill_mode = descr->fill_mode;
    if (opA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        work_fill_mode =
            (descr->fill_mode == MCSPARSE_FILL_MODE_LOWER) ? MCSPARSE_FILL_MODE_UPPER : MCSPARSE_FILL_MODE_LOWER;
    }
    mcspStatus_t status;
    if (work_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        status = mcspDestroyTrmInfo(info->csr_spsv_lower_info);
        info->csr_spsv_lower_info = nullptr;
    } else if (work_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        status = mcspDestroyTrmInfo(info->csr_spsv_upper_info);
        info->csr_spsv_upper_info = nullptr;
    }
    return status;
}

mcspStatus_t mcspCsrSpsvZeroPivotImpl(mcspHandle_t handle, const mcspMatDescr_t descr, mcspMatInfo_t info,
                                      mcspInt *position) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (descr == nullptr || info == nullptr || position == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *position = (mcspInt)(info->zero_pivot_lead);
    if (info->zero_pivot_lead != -1) {
        return MCSP_STATUS_ZERO_PIVOT;
    }
    return MCSP_STATUS_SUCCESS;
}

// buffer structure: ->csc_cols ->csc_rows ->csc_vals ->max_of(csc_buffer, spsv_analysis_buffer)
template <typename valType, typename sizeType>
mcspStatus_t mcspCuinXcsrsv2_bufferSize_template(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                               const mcspMatDescr_t descrA, const valType *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                               mcspCsrsv2Info_t info, sizeType *pBufferSizeInBytes) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    mcspStatus_t status;
    size_t spsv_buffer_size;
    if (pBufferSizeInBytes == nullptr || info == nullptr || info->mat_info == nullptr || descrA == nullptr ||
        csrSortedValA == nullptr || csrSortedRowPtrA == nullptr || csrSortedColIndA == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    status = mcspCsrSpsvBuffersize_template(handle, transA, (mcspInt)m, (mcspInt)nnz, descrA, csrSortedValA,
                                            (const mcspInt *)csrSortedRowPtrA, (const mcspInt *)csrSortedColIndA,
                                            info->mat_info, &spsv_buffer_size);
    spsv_buffer_size = ALIGN(spsv_buffer_size, ALIGNED_SIZE);
    *pBufferSizeInBytes = (sizeType)spsv_buffer_size;
    if (status != MCSP_STATUS_SUCCESS) {
        return status;
    }

    if (transA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        size_t trans_buffer_size = 0;
        status = mcspGetCsrTransExBuffersize<valType>(handle, (mcspInt)m, (mcspInt)m, (mcspInt)nnz,
                                                      (mcspInt *)csrSortedRowPtrA, (mcspInt *)csrSortedColIndA,
                                                      info->mat_info, trans_buffer_size);
        if (status == MCSP_STATUS_SUCCESS) {
            *pBufferSizeInBytes =
                info->mat_info->fixed_length_buffer_size + std::max(spsv_buffer_size, trans_buffer_size);
        }
    }
    return status;
}

template <typename valType>
mcspStatus_t mcspCuinXcsrsv2_analysis_template(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                             const mcspMatDescr_t descrA, const valType *csrSortedValA,
                                             const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                             mcspCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || info->mat_info == nullptr || descrA == nullptr || csrSortedValA == nullptr ||
        csrSortedRowPtrA == nullptr || csrSortedColIndA == nullptr || pBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (transA == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double>) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    if (transA != MCSPARSE_OPERATION_NON_TRANSPOSE &&
        (info->mat_info->assist_index_buffer_size == 0 || info->mat_info->fixed_length_buffer_size == 0)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    void *working_rows = (void *)csrSortedRowPtrA;
    void *working_cols = (void *)csrSortedColIndA;
    void *working_vals = (void *)csrSortedValA;
    void *analysis_buffer = pBuffer;
    if (transA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        mcspStatus_t stat = mcspTransposeSpMatForRawAPI(
            handle, transA, (mcspInt)m, (mcspInt)m, (mcspInt)nnz, csrSortedValA, (mcspInt *)csrSortedRowPtrA,
            (mcspInt *)csrSortedColIndA, descrA->base, info->mat_info, pBuffer);
        if (stat != MCSP_STATUS_SUCCESS) {
            return stat;
        }

        working_rows = info->mat_info->to_csc_cols;
        working_cols = info->mat_info->to_csc_rows;
        working_vals = info->mat_info->to_csc_vals;
        analysis_buffer = (void *)(reinterpret_cast<char *>(pBuffer) + info->mat_info->fixed_length_buffer_size);
    }

    return mcspCsrSpsvAnalysis_template(handle, transA, (mcspInt)m, (mcspInt)nnz, descrA, (valType *)working_vals,
                                        (const mcspInt *)working_rows, (const mcspInt *)working_cols, info->mat_info,
                                        MCSPARSE_ANALYSIS_POLICY_FORCE, policy, analysis_buffer);
}

template <typename valType>
mcspStatus_t mcspCuinScsrsv2_solve_template(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                          const valType *alpha, const mcspMatDescr_t descrA,
                                          const valType *csrSortedValA, const int *csrSortedRowPtrA,
                                          const int *csrSortedColIndA, mcspCsrsv2Info_t info, const valType *f,
                                          valType *x, mcsparseSolvePolicy_t policy, void *pBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || info->mat_info == nullptr || pBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (transA == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        if constexpr (std::is_same_v<valType, float> || std::is_same_v<valType, double>) {
            return MCSP_STATUS_INVALID_VALUE;
        }
    }

    void *working_rows = nullptr;
    void *working_cols = nullptr;
    void *working_vals = nullptr;
    void *solve_buffer = nullptr;
    if (transA == MCSPARSE_OPERATION_NON_TRANSPOSE) {
        working_rows = (void *)csrSortedRowPtrA;
        working_cols = (void *)csrSortedColIndA;
        working_vals = (void *)csrSortedValA;
        solve_buffer = pBuffer;
    } else {
        working_rows = info->mat_info->to_csc_cols;
        working_cols = info->mat_info->to_csc_rows;
        working_vals = info->mat_info->to_csc_vals;
        solve_buffer = (void *)(reinterpret_cast<char *>(pBuffer) + info->mat_info->fixed_length_buffer_size);
    }

    return mcspCsrSpsvSolve_template(handle, transA, (mcspInt)m, (mcspInt)nnz, alpha, descrA, (valType *)working_vals,
                                     (const mcspInt *)working_rows, (const mcspInt *)working_cols, info->mat_info, f, x,
                                     policy, solve_buffer);
}

mcspStatus_t mcspCuinCsrSpsvZeroPivotImpl(mcspHandle_t handle, mcspCsrsv2Info_t info, int *position) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (info == nullptr || info->mat_info == nullptr || position == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *position = info->mat_info->zero_pivot_lead;
    if (*position != -1) {
        return MCSP_STATUS_ZERO_PIVOT;
    }
    return MCSP_STATUS_SUCCESS;
}

// buffer structure: ->csr_rows ->csc_cols ->csc_rows ->csc_vals ->max_of(csc_buffer, spsv_analysis_buffer)
template <typename idxType>
mcspStatus_t mcspCuinSpSV_bufferSize_impl(mcspHandle_t handle, mcsparseOperation_t opA, const void *alpha,
                                        mcspSpMatDescr_t matA, mcspDnVecDescr_t vecX, mcspDnVecDescr_t vecY,
                                        macaDataType computeType, mcsparseSpSVAlg_t alg, mcspSpSVDescr_t spsvDescr,
                                        size_t *bufferSize) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (matA == nullptr || spsvDescr == nullptr || spsvDescr->mat_info == nullptr || bufferSize == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcspStatus_t status = MCSP_STATUS_SUCCESS;
    size_t spsv_buffer_size = 0;
    switch (computeType) {
        case MACA_R_32F:
            status = mcspCsrSpsvBuffersize_template(handle, opA, (idxType)matA->row_num, (idxType)matA->nnz,
                                                    matA->mat_descr, (float *)matA->vals, (idxType *)matA->rows,
                                                    (idxType *)matA->cols, spsvDescr->mat_info, &spsv_buffer_size);
            break;
        case MACA_R_64F:
            status = mcspCsrSpsvBuffersize_template(handle, opA, (idxType)matA->row_num, (idxType)matA->nnz,
                                                    matA->mat_descr, (double *)matA->vals, (idxType *)matA->rows,
                                                    (idxType *)matA->cols, spsvDescr->mat_info, &spsv_buffer_size);
            break;
        case MACA_C_32F:
            status =
                mcspCsrSpsvBuffersize_template(handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, matA->mat_descr,
                                               (mcspComplexFloat *)matA->vals, (idxType *)matA->rows,
                                               (idxType *)matA->cols, spsvDescr->mat_info, &spsv_buffer_size);
            break;
        case MACA_C_64F:
            status =
                mcspCsrSpsvBuffersize_template(handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, matA->mat_descr,
                                               (mcspComplexDouble *)matA->vals, (idxType *)matA->rows,
                                               (idxType *)matA->cols, spsvDescr->mat_info, &spsv_buffer_size);
            break;
        default:
            status = MCSP_STATUS_NOT_IMPLEMENTED;
    }
    if (status != MCSP_STATUS_SUCCESS) {
        return status;
    }

    spsv_buffer_size = ALIGN(spsv_buffer_size, ALIGNED_SIZE);
    *bufferSize = spsv_buffer_size;

    size_t trans_buffer_size = 0;
    status = mcspGetGenericTransExBuffersize(handle, opA, computeType, (idxType *)matA->rows, (idxType *)matA->cols,
                                             matA, trans_buffer_size);
    if (status != MCSP_STATUS_SUCCESS) {
        return status;
    }
    *bufferSize = matA->fixed_length_buffer_size + std::max(spsv_buffer_size, trans_buffer_size);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCuinSpSV_analysis_impl(mcspHandle_t handle, mcsparseOperation_t opA, const void *alpha,
                                      mcspSpMatDescr_t matA, mcspDnVecDescr_t vecX, mcspDnVecDescr_t vecY,
                                      macaDataType computeType, mcsparseSpSVAlg_t alg, mcspSpSVDescr_t spsvDescr,
                                      void *externalBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (matA == nullptr || matA->mat_descr == nullptr || spsvDescr == nullptr || spsvDescr->mat_info == nullptr ||
        externalBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (opA == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE && (computeType == MACA_R_32F || computeType == MACA_R_64F)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (matA->row_num != matA->col_num) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (opA != MCSPARSE_OPERATION_NON_TRANSPOSE &&
        (matA->assist_index_buffer_size == 0 || matA->fixed_length_buffer_size == 0)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    matA->to_csr_rows = matA->rows;
    mcspStatus_t stat = mcspTransposeSpMatForGenericAPI<idxType>(handle, opA, computeType, matA, externalBuffer);
    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }

    void *working_rows = (void *)matA->to_csr_rows;
    void *working_cols = (void *)matA->cols;
    void *working_vals = (void *)matA->vals;
    if (opA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        working_rows = matA->to_csc_cols;
        working_cols = matA->to_csc_rows;
        working_vals = matA->to_csc_vals;
    }

    void *analysis_buffer = (void *)(reinterpret_cast<char *>(externalBuffer) + matA->fixed_length_buffer_size);
    spsvDescr->external_buffer = analysis_buffer;
    mcStream_t stream = mcspGetStreamInternal(handle);
    switch (computeType) {
        case MACA_R_32F:
            stat = mcspCsrSpsvAnalysis_template(
                handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, matA->mat_descr, (float *)working_vals,
                (idxType *)working_rows, (idxType *)working_cols, spsvDescr->mat_info, MCSPARSE_ANALYSIS_POLICY_FORCE,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, analysis_buffer);
            break;
        case MACA_R_64F:
            stat = mcspCsrSpsvAnalysis_template(
                handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, matA->mat_descr, (double *)working_vals,
                (idxType *)working_rows, (idxType *)working_cols, spsvDescr->mat_info, MCSPARSE_ANALYSIS_POLICY_FORCE,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, analysis_buffer);
            break;
        case MACA_C_32F:
            stat = mcspCsrSpsvAnalysis_template(
                handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, matA->mat_descr,
                (mcspComplexFloat *)working_vals, (idxType *)working_rows, (idxType *)working_cols, spsvDescr->mat_info,
                MCSPARSE_ANALYSIS_POLICY_FORCE, MCSPARSE_SOLVE_POLICY_NO_LEVEL, analysis_buffer);
            break;
        case MACA_C_64F:
            stat = mcspCsrSpsvAnalysis_template(
                handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, matA->mat_descr,
                (mcspComplexDouble *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                spsvDescr->mat_info, MCSPARSE_ANALYSIS_POLICY_FORCE, MCSPARSE_SOLVE_POLICY_NO_LEVEL, analysis_buffer);
            break;
        default:
            stat = MCSP_STATUS_NOT_IMPLEMENTED;
    }
    if ((matA->mat_descr->diag_type != MCSPARSE_DIAG_TYPE_UNIT) && (spsvDescr->mat_info->zero_pivot_lead != -1)) {
        stat = MCSP_STATUS_ZERO_PIVOT;
        MACA_ASSERT(mcMemcpyAsync(externalBuffer, &(spsvDescr->mat_info->zero_pivot_lead),
                                  sizeof(spsvDescr->mat_info->zero_pivot_lead), mcMemcpyHostToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }
    return stat;
}

template <typename idxType>
mcspStatus_t mcspCuinSpSV_solve_impl(mcspHandle_t handle, mcsparseOperation_t opA, const void *alpha,
                                   mcspSpMatDescr_t matA, mcspDnVecDescr_t vecX, mcspDnVecDescr_t vecY,
                                   macaDataType computeType, mcsparseSpSVAlg_t alg, mcspSpSVDescr_t spsvDescr) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (matA == nullptr || alpha == nullptr || vecX == nullptr || vecY == nullptr || spsvDescr == nullptr ||
        spsvDescr->mat_info == nullptr || spsvDescr->external_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    void *working_rows = nullptr;
    void *working_cols = nullptr;
    void *working_vals = nullptr;
    if (opA == MCSPARSE_OPERATION_NON_TRANSPOSE) {
        working_rows = matA->to_csr_rows;
        working_cols = matA->cols;
        working_vals = matA->vals;
    } else {
        working_rows = matA->to_csc_cols;
        working_cols = matA->to_csc_rows;
        working_vals = matA->to_csc_vals;
    }
    switch (computeType) {
        case MACA_R_32F:
            return mcspCsrSpsvSolve_template(handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, (float *)alpha,
                                             matA->mat_descr, (float *)working_vals, (idxType *)working_rows,
                                             (idxType *)working_cols, spsvDescr->mat_info, (float *)vecX->values,
                                             (float *)vecY->values, MCSPARSE_SOLVE_POLICY_NO_LEVEL,
                                             spsvDescr->external_buffer);
        case MACA_R_64F:
            return mcspCsrSpsvSolve_template(handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, (double *)alpha,
                                             matA->mat_descr, (double *)working_vals, (idxType *)working_rows,
                                             (idxType *)working_cols, spsvDescr->mat_info, (double *)vecX->values,
                                             (double *)vecY->values, MCSPARSE_SOLVE_POLICY_NO_LEVEL,
                                             spsvDescr->external_buffer);
        case MACA_C_32F:
            return mcspCsrSpsvSolve_template(
                handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr,
                (mcspComplexFloat *)working_vals, (idxType *)working_rows, (idxType *)working_cols, spsvDescr->mat_info,
                (mcspComplexFloat *)vecX->values, (mcspComplexFloat *)vecY->values, MCSPARSE_SOLVE_POLICY_NO_LEVEL,
                spsvDescr->external_buffer);
        case MACA_C_64F:
            return mcspCsrSpsvSolve_template(
                handle, opA, (idxType)matA->row_num, (idxType)matA->nnz, (mcspComplexDouble *)alpha, matA->mat_descr,
                (mcspComplexDouble *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                spsvDescr->mat_info, (mcspComplexDouble *)vecX->values, (mcspComplexDouble *)vecY->values,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, spsvDescr->external_buffer);
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsrSpsvBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                    const mcspMatDescr_t descr, const float *csr_vals, const mcspInt *csr_rows,
                                    const mcspInt *csr_cols, mcspMatInfo_t info, size_t *buffer_size) {
    return mcspCsrSpsvBuffersize_template(handle, trans, row_num, nnz, descr, csr_vals, csr_rows, csr_cols, info,
                                          buffer_size);
}

mcspStatus_t mcspDcsrSpsvBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                    const mcspMatDescr_t descr, const double *csr_vals, const mcspInt *csr_rows,
                                    const mcspInt *csr_cols, mcspMatInfo_t info, size_t *buffer_size) {
    return mcspCsrSpsvBuffersize_template(handle, trans, row_num, nnz, descr, csr_vals, csr_rows, csr_cols, info,
                                          buffer_size);
}

mcspStatus_t mcspCcsrSpsvBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                    const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals,
                                    const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info,
                                    size_t *buffer_size) {
    return mcspCsrSpsvBuffersize_template(handle, trans, row_num, nnz, descr, csr_vals, csr_rows, csr_cols, info,
                                          buffer_size);
}

mcspStatus_t mcspZcsrSpsvBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                    const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                                    const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info,
                                    size_t *buffer_size) {
    return mcspCsrSpsvBuffersize_template(handle, trans, row_num, nnz, descr, csr_vals, csr_rows, csr_cols, info,
                                          buffer_size);
}

mcspStatus_t mcspScsrSpsvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                  const mcspMatDescr_t descr, const float *csr_vals, const mcspInt *csr_rows,
                                  const mcspInt *csr_cols, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void *buffer) {
    return mcspCsrSpsvAnalysis_template(handle, trans, row_num, nnz, descr, csr_vals, csr_rows, csr_cols, info,
                                        analysis_policy, solve_policy, buffer);
}

mcspStatus_t mcspDcsrSpsvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                  const mcspMatDescr_t descr, const double *csr_vals, const mcspInt *csr_rows,
                                  const mcspInt *csr_cols, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void *buffer) {
    return mcspCsrSpsvAnalysis_template(handle, trans, row_num, nnz, descr, csr_vals, csr_rows, csr_cols, info,
                                        analysis_policy, solve_policy, buffer);
}

mcspStatus_t mcspCcsrSpsvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                  const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals, const mcspInt *csr_rows,
                                  const mcspInt *csr_cols, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void *buffer) {
    return mcspCsrSpsvAnalysis_template(handle, trans, row_num, nnz, descr, csr_vals, csr_rows, csr_cols, info,
                                        analysis_policy, solve_policy, buffer);
}

mcspStatus_t mcspZcsrSpsvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                  const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info,
                                  mcsparseAnalysisPolicy_t analysis_policy, mcsparseSolvePolicy_t solve_policy,
                                  void *buffer) {
    return mcspCsrSpsvAnalysis_template(handle, trans, row_num, nnz, descr, csr_vals, csr_rows, csr_cols, info,
                                        analysis_policy, solve_policy, buffer);
}

mcspStatus_t mcspScsrSpsvSolve(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                               const float *alpha, const mcspMatDescr_t descr, const float *csr_vals,
                               const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info, const float *x,
                               float *y, mcsparseSolvePolicy_t solve_policy, void *buffer) {
    return mcspCsrSpsvSolve_template(handle, trans, row_num, nnz, alpha, descr, csr_vals, csr_rows, csr_cols, info, x,
                                     y, solve_policy, buffer);
}

mcspStatus_t mcspDcsrSpsvSolve(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                               const double *alpha, const mcspMatDescr_t descr, const double *csr_vals,
                               const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info, const double *x,
                               double *y, mcsparseSolvePolicy_t solve_policy, void *buffer) {
    return mcspCsrSpsvSolve_template(handle, trans, row_num, nnz, alpha, descr, csr_vals, csr_rows, csr_cols, info, x,
                                     y, solve_policy, buffer);
}

mcspStatus_t mcspCcsrSpsvSolve(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                               const mcspComplexFloat *alpha, const mcspMatDescr_t descr,
                               const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                               mcspMatInfo_t info, const mcspComplexFloat *x, mcspComplexFloat *y,
                               mcsparseSolvePolicy_t solve_policy, void *buffer) {
    return mcspCsrSpsvSolve_template(handle, trans, row_num, nnz, alpha, descr, csr_vals, csr_rows, csr_cols, info, x,
                                     y, solve_policy, buffer);
}

mcspStatus_t mcspZcsrSpsvSolve(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                               const mcspComplexDouble *alpha, const mcspMatDescr_t descr,
                               const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                               mcspMatInfo_t info, const mcspComplexDouble *x, mcspComplexDouble *y,
                               mcsparseSolvePolicy_t solve_policy, void *buffer) {
    return mcspCsrSpsvSolve_template(handle, trans, row_num, nnz, alpha, descr, csr_vals, csr_rows, csr_cols, info, x,
                                     y, solve_policy, buffer);
}

mcspStatus_t mcspCsrSpsvClear(mcspHandle_t handle, const mcspMatDescr_t descr, mcspMatInfo_t info,
                              mcsparseOperation_t opA) {
    return mcspCsrSpsvClearImpl(handle, descr, info, opA);
}

mcspStatus_t mcspCsrSpsvZeroPivot(mcspHandle_t handle, const mcspMatDescr_t descr, mcspMatInfo_t info,
                                  mcspInt *position) {
    return mcspCsrSpsvZeroPivotImpl(handle, descr, info, position);
}

// cusaprse wrapper of Xcsrsv2
mcspStatus_t mcspCuinXcsrsv2_zeroPivot(mcspHandle_t handle, mcspCsrsv2Info_t info, int *position) {
    return mcspCuinCsrSpsvZeroPivotImpl(handle, info, position);
}

mcspStatus_t mcspCuinScsrsv2_bufferSize(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcspMatDescr_t descrA, float *csrSortedValA, const int *csrSortedRowPtrA,
                                      const int *csrSortedColIndA, mcspCsrsv2Info_t info, int *pBufferSizeInBytes) {
    return mcspCuinXcsrsv2_bufferSize_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                             csrSortedColIndA, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinDcsrsv2_bufferSize(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcspMatDescr_t descrA, double *csrSortedValA, const int *csrSortedRowPtrA,
                                      const int *csrSortedColIndA, mcspCsrsv2Info_t info, int *pBufferSizeInBytes) {
    return mcspCuinXcsrsv2_bufferSize_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                             csrSortedColIndA, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinCcsrsv2_bufferSize(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcspMatDescr_t descrA, mcFloatComplex *csrSortedValA,
                                      const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                      int *pBufferSizeInBytes) {
    return mcspCuinXcsrsv2_bufferSize_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                             csrSortedColIndA, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinZcsrsv2_bufferSize(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcspMatDescr_t descrA, mcDoubleComplex *csrSortedValA,
                                      const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                      int *pBufferSizeInBytes) {
    return mcspCuinXcsrsv2_bufferSize_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                             csrSortedColIndA, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinScsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                         const mcspMatDescr_t descrA, float *csrSortedValA, const int *csrSortedRowPtrA,
                                         const int *csrSortedColIndA, mcspCsrsv2Info_t info, size_t *pBufferSize) {
    return mcspCuinXcsrsv2_bufferSize_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                             csrSortedColIndA, info, pBufferSize);
}

mcspStatus_t mcspCuinDcsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                         const mcspMatDescr_t descrA, double *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                         mcspCsrsv2Info_t info, size_t *pBufferSize) {
    return mcspCuinXcsrsv2_bufferSize_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                             csrSortedColIndA, info, pBufferSize);
}

mcspStatus_t mcspCuinCcsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                         const mcspMatDescr_t descrA, mcFloatComplex *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                         mcspCsrsv2Info_t info, size_t *pBufferSize) {
    return mcspCuinXcsrsv2_bufferSize_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                             csrSortedColIndA, info, pBufferSize);
}

mcspStatus_t mcspCuinZcsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                         const mcspMatDescr_t descrA, mcDoubleComplex *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                         mcspCsrsv2Info_t info, size_t *pBufferSize) {
    return mcspCuinXcsrsv2_bufferSize_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                             csrSortedColIndA, info, pBufferSize);
}

mcspStatus_t mcspCuinScsrsv2_analysis(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                    const mcspMatDescr_t descrA, const float *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                    mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsv2_analysis_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinDcsrsv2_analysis(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                    const mcspMatDescr_t descrA, const double *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                    mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsv2_analysis_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinCcsrsv2_analysis(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                    const mcspMatDescr_t descrA, const mcFloatComplex *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                    mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsv2_analysis_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinZcsrsv2_analysis(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                    const mcspMatDescr_t descrA, const mcDoubleComplex *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                    mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsv2_analysis_template(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinScsrsv2_solve(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz, const float *alpha,
                                 const mcspMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspCsrsv2Info_t info, const float *f, float *x,
                                 mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinScsrsv2_solve_template(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, info, f, x, policy, pBuffer);
}

mcspStatus_t mcspCuinDcsrsv2_solve(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz, const double *alpha,
                                 const mcspMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspCsrsv2Info_t info, const double *f, double *x,
                                 mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinScsrsv2_solve_template(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, info, f, x, policy, pBuffer);
}

mcspStatus_t mcspCuinCcsrsv2_solve(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                 const mcFloatComplex *alpha, const mcspMatDescr_t descrA,
                                 const mcFloatComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspCsrsv2Info_t info, const mcFloatComplex *f,
                                 mcFloatComplex *x, mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinScsrsv2_solve_template(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, info, f, x, policy, pBuffer);
}

mcspStatus_t mcspCuinZcsrsv2_solve(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                 const mcDoubleComplex *alpha, const mcspMatDescr_t descrA,
                                 const mcDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspCsrsv2Info_t info, const mcDoubleComplex *f,
                                 mcDoubleComplex *x, mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinScsrsv2_solve_template(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                        csrSortedColIndA, info, f, x, policy, pBuffer);
}

// cusaprse wrapper of generic SpSV
mcspStatus_t mcspCuinSpSV_bufferSize(mcspHandle_t handle, mcsparseOperation_t opA, const void *alpha,
                                   mcspSpMatDescr_t matA, mcspDnVecDescr_t vecX, mcspDnVecDescr_t vecY,
                                   macaDataType computeType, mcsparseSpSVAlg_t alg, mcspSpSVDescr_t spsvDescr,
                                   size_t *bufferSize) {
    return mcspCuinSpSV_bufferSize_impl<mcspInt>(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr,
                                               bufferSize);
}

mcspStatus_t mcspCuinSpSV_analysis(mcspHandle_t handle, mcsparseOperation_t opA, const void *alpha, mcspSpMatDescr_t matA,
                                 mcspDnVecDescr_t vecX, mcspDnVecDescr_t vecY, macaDataType computeType,
                                 mcsparseSpSVAlg_t alg, mcspSpSVDescr_t spsvDescr, void *externalBuffer) {
    return mcspCuinSpSV_analysis_impl<mcspInt>(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr,
                                             externalBuffer);
}

mcspStatus_t mcspCuinSpSV_solve(mcspHandle_t handle, mcsparseOperation_t opA, const void *alpha, mcspSpMatDescr_t matA,
                              mcspDnVecDescr_t vecX, mcspDnVecDescr_t vecY, macaDataType computeType,
                              mcsparseSpSVAlg_t alg, mcspSpSVDescr_t spsvDescr) {
    return mcspCuinSpSV_solve_impl<mcspInt>(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr);
}

#ifdef __cplusplus
}
#endif
