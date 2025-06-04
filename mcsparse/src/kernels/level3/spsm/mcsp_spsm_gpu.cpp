#include <iostream>

#include "common/mcsp_types.h"
#include "csr_spsm_device.hpp"
#include "device_reduce.hpp"
#include "internal_interface/mcsp_internal_conversion.h"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_debug.h"
#include "mcsp_dense_transpose_device.hpp"
#include "mcsp_handle.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_transpose_dense.hpp"
#include "mcsp_internal_transpose_sparse.hpp"
#include "mcsp_internal_types.h"
#include "spsv/csr_trm_analysis.hpp"

constexpr int kTileSize = 256;
template <typename idxType, typename valType>
mcspStatus_t mcspCsrSpsmBuffersize_template(mcspHandle_t handle, mcsparseOperation_t trans_A,
                                            mcsparseOperation_t trans_B, idxType m, idxType nrhs, idxType nnz,
                                            const valType *alpha, const mcspMatDescr_t descr, const valType *csr_vals,
                                            const idxType *csr_rows, const idxType *csr_cols, const valType *B,
                                            idxType ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy,
                                            size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (buffer_size == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    size_t size = 0;
    size_t reduce_buffersize;
    size_t radix_sort_buffersize;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::reduce(nullptr, reduce_buffersize, (idxType *)nullptr, (idxType *)nullptr, (size_t)m,
                   mcprim::minimum<idxType>(), stream);
    size += reduce_buffersize;
    mcprim::radix_sort_pairs(nullptr, radix_sort_buffersize, (idxType *)nullptr, (idxType *)nullptr, (idxType *)nullptr,
                             (idxType *)nullptr, (size_t)m, stream);
    size += radix_sort_buffersize;
    size += 4 * sizeof(idxType);  // reduce output buffer
    size += m * sizeof(idxType);  // row nnz buffer with length of m
    size += m * sizeof(idxType);  // permutation buffer with length of m
    size += m * sizeof(idxType);  // row depth buffer with length of m
    size = ALIGN(size, kTileSize);

    int tile_n = (nrhs + kTileSize - 1) / kTileSize;         // tile_m = 1
    size += ALIGN(m * tile_n, kTileSize) * sizeof(idxType);  // for done array
    if (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE) {
        size += ALIGN(m * nrhs, kTileSize) * sizeof(valType);  // for non transposed B matrix
    }

    *buffer_size = size;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsrSpsmAnalysis_template(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                          idxType m, idxType nrhs, idxType nnz, const valType *alpha,
                                          const mcspMatDescr_t descr, const valType *csr_vals, const idxType *csr_rows,
                                          const idxType *csr_cols, const valType *B, idxType ldb, mcspMatInfo_t info,
                                          mcsparseAnalysisPolicy_t analysis, mcsparseSolvePolicy_t solve,
                                          void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || nnz < 0 || nrhs < 0 || ldb < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || B == nullptr || descr == nullptr || csr_vals == nullptr || csr_rows == nullptr ||
        csr_cols == nullptr || info == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {  // TODO: mode
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (descr->fill_mode == MCSPARSE_FILL_MODE_FULL) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (analysis == MCSPARSE_ANALYSIS_POLICY_REUSE) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    mcsparseFillMode_t work_fill_mode = descr->fill_mode;
    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        work_fill_mode =
            (descr->fill_mode == MCSPARSE_FILL_MODE_LOWER) ? MCSPARSE_FILL_MODE_UPPER : MCSPARSE_FILL_MODE_LOWER;
    }
    if (work_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        if (info->csr_spsm_lower_info == nullptr) {
            mcspCreateTrmInfo(&(info->csr_spsm_lower_info));
            mcspCsrTrmAnalysis_template(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info->csr_spsm_lower_info,
                                        info->zero_pivot_lead, temp_buffer, true);
        }
    } else if (work_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        if (info->csr_spsm_upper_info == nullptr) {
            mcspCreateTrmInfo(&(info->csr_spsm_upper_info));
            mcspCsrTrmAnalysis_template(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info->csr_spsm_upper_info,
                                        info->zero_pivot_lead, temp_buffer, false);
        }
    }
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsrSpsmSolve_template(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                       idxType m, idxType nrhs, idxType nnz, const valType *alpha,
                                       const mcspMatDescr_t descr, const valType *csr_vals, const idxType *csr_rows,
                                       const idxType *csr_cols, valType *B, idxType ldb, mcspMatInfo_t info,
                                       mcsparseSolvePolicy_t policy, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || nnz < 0 || nrhs < 0 || ldb < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || B == nullptr || descr == nullptr || csr_vals == nullptr || csr_rows == nullptr ||
        csr_cols == nullptr || info == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (trans_B == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (descr->fill_mode == MCSPARSE_FILL_MODE_FULL || ((trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE) && (ldb < m)) ||
        ((trans_B == MCSPARSE_OPERATION_TRANSPOSE) && (ldb < nrhs))) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT && info->zero_pivot_lead != -1) {
        return MCSP_STATUS_ZERO_PIVOT;
    }

    mcsparseFillMode_t work_fill_mode = descr->fill_mode;
    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        work_fill_mode =
            (descr->fill_mode == MCSPARSE_FILL_MODE_LOWER) ? MCSPARSE_FILL_MODE_UPPER : MCSPARSE_FILL_MODE_LOWER;
    }
    mcspTrmInfo_t trm_info;
    if (work_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        trm_info = info->csr_spsm_lower_info;
    } else if (work_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        trm_info = info->csr_spsm_upper_info;
    }
    if (trm_info == nullptr || trm_info->row_map == nullptr || trm_info->zero_pivot_array == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    // temp_buffer used for done array
    const int tile_n = (nrhs + kTileSize - 1) / kTileSize;  // tile_m = 1
    char *ptr = reinterpret_cast<char *>(temp_buffer);
    idxType *done_array = reinterpret_cast<idxType *>(ptr);
    ptr += ALIGN(m * tile_n, kTileSize) * sizeof(idxType);
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemsetAsync(done_array, 0, ALIGN(m * tile_n, kTileSize) * sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    // Transpose B to improve matrix loading cache hit rate if B was not initially transposed
    const int trans_dimX = 32;
    const int trans_dimY = 8;
    dim3 trans_blocks((m - 1) / trans_dimX + 1);
    dim3 trans_threads(trans_dimX * trans_dimY);
    idxType ldbt = ldb;
    valType *Bt = B;
    if (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE) {
        ldbt = nrhs;
        Bt = reinterpret_cast<valType *>(ptr);
        ptr += ALIGN(m * nrhs, kTileSize) * sizeof(idxType);
        MACA_ASSERT(mcMemsetAsync(Bt, 0, ALIGN(m * nrhs, kTileSize) * sizeof(valType), stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        mcLaunchKernelGGL((mcspDenseTransposeKernel<trans_dimX, trans_dimY>), trans_blocks, trans_threads, 0, stream, m,
                           nrhs, B, ldb, Bt, ldbt);
    }

    // each block calculate kTileSize=1x256 of m x nrhs matrix C
    dim3 block(kTileSize);
    dim3 grid(tile_n * m);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    if (work_fill_mode == MCSPARSE_FILL_MODE_LOWER) {
        if (descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
            mcLaunchKernelGGL((mcspCsrSpsmSolveKernel<kTileSize, true, false>), grid, block, 0, stream, m, nrhs,
                               h_alpha, csr_vals, csr_rows, csr_cols, (idxType *)(trm_info->row_map),
                               (idxType *)(trm_info->zero_pivot_array), Bt, ldbt, done_array, descr->base);
        } else if (descr->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
            mcLaunchKernelGGL((mcspCsrSpsmSolveKernel<kTileSize, true, true>), grid, block, 0, stream, m, nrhs,
                               h_alpha, csr_vals, csr_rows, csr_cols, (idxType *)(trm_info->row_map),
                               (idxType *)(trm_info->zero_pivot_array), Bt, ldbt, done_array, descr->base);
        }
    } else if (work_fill_mode == MCSPARSE_FILL_MODE_UPPER) {
        if (descr->diag_type == MCSPARSE_DIAG_TYPE_NON_UNIT) {
            mcLaunchKernelGGL((mcspCsrSpsmSolveKernel<kTileSize, false, false>), grid, block, 0, stream, m, nrhs,
                               h_alpha, csr_vals, csr_rows, csr_cols, (idxType *)(trm_info->row_map),
                               (idxType *)(trm_info->zero_pivot_array), Bt, ldbt, done_array, descr->base);
        } else if (descr->diag_type == MCSPARSE_DIAG_TYPE_UNIT) {
            mcLaunchKernelGGL((mcspCsrSpsmSolveKernel<kTileSize, false, true>), grid, block, 0, stream, m, nrhs,
                               h_alpha, csr_vals, csr_rows, csr_cols, (idxType *)(trm_info->row_map),
                               (idxType *)(trm_info->zero_pivot_array), Bt, ldbt, done_array, descr->base);
        }
    }
    MACA_ASSERT(mcStreamSynchronize(stream));

    // Transpose B back if B was not initially transposed
    if (trans_B == MCSPARSE_OPERATION_NON_TRANSPOSE) {
        mcLaunchKernelGGL((mcspDenseTransposeBackKernel<trans_dimX, trans_dimY>), trans_blocks, trans_threads, 0,
                           stream, m, nrhs, Bt, ldbt, B, ldb);
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCsrSpsmClearImpl(mcspHandle_t handle, mcspMatInfo_t info) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcspStatus_t stat;
    if (info->csr_spsm_lower_info != nullptr) {
        stat = mcspDestroyTrmInfo(info->csr_spsm_lower_info);
        info->csr_spsm_lower_info = nullptr;
    } else if (info->csr_spsm_upper_info != nullptr) {
        stat = mcspDestroyTrmInfo(info->csr_spsm_upper_info);
        info->csr_spsm_upper_info = nullptr;
    }
    return stat;
}

mcspStatus_t mcspCsrSpsmZeroPivotImpl(mcspHandle_t handle, mcspMatInfo_t info, int *position) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (info == nullptr || position == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *position = (mcspInt)(info->zero_pivot_lead);
    if (info->zero_pivot_lead != -1) {
        return MCSP_STATUS_ZERO_PIVOT;
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCuinCsrSpsmZeroPivotImpl(mcspHandle_t handle, mcspCsrsm2Info_t info, int *position) {
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

template <typename valType>
mcspStatus_t mcspCuinXcsrsm2_bufferSize_template(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                               mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                               const valType *alpha, const mcspMatDescr_t descrA,
                                               const valType *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, const valType *B, int ldb,
                                               mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                               size_t *pBufferSize) {
    mcspStatus_t status;
    size_t spsm_buffer_size;
    if (pBufferSize == nullptr || info == nullptr || info->mat_info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    status = mcspCsrSpsmBuffersize_template(handle, transA, transB, (mcspInt)m, (mcspInt)nrhs, (mcspInt)nnz, alpha,
                                            descrA, csrSortedValA, (const mcspInt *)csrSortedRowPtrA,
                                            (const mcspInt *)csrSortedColIndA, B, (mcspInt)ldb, info->mat_info, policy,
                                            &spsm_buffer_size);
    spsm_buffer_size = ALIGN(spsm_buffer_size, ALIGNED_SIZE);
    *pBufferSize = spsm_buffer_size;
    if (status != MCSP_STATUS_SUCCESS) {
        return status;
    }

    if (transA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        size_t trans_buffer_size = 0;
        status = mcspGetCsrTransExBuffersize<valType>(handle, (mcspInt)m, (mcspInt)m, (mcspInt)nnz,
                                                      (mcspInt *)csrSortedRowPtrA, (mcspInt *)csrSortedColIndA,
                                                      info->mat_info, trans_buffer_size);
        if (status == MCSP_STATUS_SUCCESS) {
            *pBufferSize = info->mat_info->fixed_length_buffer_size + std::max(spsm_buffer_size, trans_buffer_size);
        }
    }
    return status;
}

template <typename valType>
mcspStatus_t mcspCuinXcsrsm2_analysis_template(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                             mcsparseOperation_t transB, int m, int nrhs, int nnz, const valType *alpha,
                                             const mcspMatDescr_t descrA, const valType *csrSortedValA,
                                             const int *csrSortedRowPtrA, const int *csrSortedColIndA, const valType *B,
                                             int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                             void *pBuffer) {
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

    return mcspCsrSpsmAnalysis_template(handle, transA, transB, (mcspInt)m, (mcspInt)nrhs, (mcspInt)nnz, alpha, descrA,
                                        (valType *)working_vals, (const mcspInt *)working_rows,
                                        (const mcspInt *)working_cols, B, (mcspInt)ldb, info->mat_info,
                                        MCSPARSE_ANALYSIS_POLICY_AUTO, policy, analysis_buffer);
}

template <typename valType>
mcspStatus_t mcspCuinXcsrsm2_solve_template(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                          mcsparseOperation_t transB, int m, int nrhs, int nnz, const valType *alpha,
                                          const mcspMatDescr_t descrA, const valType *csrSortedValA,
                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA, valType *B, int ldb,
                                          mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer) {
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

    return mcspCsrSpsmSolve_template(handle, transA, transB, (mcspInt)m, (mcspInt)nrhs, (mcspInt)nnz, alpha, descrA,
                                     (valType *)working_vals, (const mcspInt *)working_rows,
                                     (const mcspInt *)working_cols, B, (mcspInt)ldb, info->mat_info, policy,
                                     solve_buffer);
}

template <typename idxType>
mcspStatus_t mcspSpSM_bufferSize_impl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                      const void *alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB,
                                      mcspDnMatDescr_t matC, macaDataType computeType, mcsparseSpSMAlg_t alg,
                                      mcspSpSMDescr_t spsmDescr, size_t *bufferSize) {
    if (matA == nullptr || matB == nullptr || matC == nullptr || spsmDescr == nullptr || bufferSize == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (!(matA->format == MCSPARSE_FORMAT_CSR || matA->format == MCSPARSE_FORMAT_COO)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
    // NOTE: for generic API, operation_B does not influence matC storage, which is different from raw csrsm2
    if (!((opB == MCSPARSE_OPERATION_NON_TRANSPOSE && matB->row_num == matA->col_num &&
           matB->row_num == matC->row_num && matB->col_num == matC->col_num && matB->ld >= matB->row_num &&
           matC->ld >= matC->row_num) ||
          (opB != MCSPARSE_OPERATION_NON_TRANSPOSE && matB->col_num == matA->col_num &&
           matB->col_num == matC->row_num && matB->ld >= matB->row_num && matC->ld >= matC->row_num))) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    idxType working_nrhs = (idxType)matB->col_num;
    if (opB != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        working_nrhs = (idxType)matB->row_num;
    }
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    size_t spsm_buffer_size = 0;
    switch (computeType) {
        case MACA_R_32F: {
            stat = mcspCsrSpsmBuffersize_template(handle, opA, opB, (idxType)matA->row_num, working_nrhs,
                                                  (idxType)matA->nnz, (float *)alpha, matA->mat_descr,
                                                  (float *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                  (float *)matB->values, (idxType)matB->ld, spsmDescr->mat_info,
                                                  MCSPARSE_SOLVE_POLICY_NO_LEVEL, &spsm_buffer_size);
            break;
        }
        case MACA_R_64F: {
            stat = mcspCsrSpsmBuffersize_template(handle, opA, opB, (idxType)matA->row_num, working_nrhs,
                                                  (idxType)matA->nnz, (double *)alpha, matA->mat_descr,
                                                  (double *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                  (double *)matB->values, (idxType)matB->ld, spsmDescr->mat_info,
                                                  MCSPARSE_SOLVE_POLICY_NO_LEVEL, &spsm_buffer_size);
            break;
        }
        case MACA_C_32F: {
            stat = mcspCsrSpsmBuffersize_template(
                handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz, (mcspComplexFloat *)alpha,
                matA->mat_descr, (mcspComplexFloat *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                (mcspComplexFloat *)matB->values, (idxType)matB->ld, spsmDescr->mat_info,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, &spsm_buffer_size);
            break;
        }
        case MACA_C_64F: {
            stat = mcspCsrSpsmBuffersize_template(
                handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz, (mcspComplexDouble *)alpha,
                matA->mat_descr, (mcspComplexDouble *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                (mcspComplexDouble *)matB->values, (idxType)matB->ld, spsmDescr->mat_info,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, &spsm_buffer_size);
            break;
        }
        default:
            stat = MCSP_STATUS_NOT_IMPLEMENTED;
    }
    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }

    spsm_buffer_size = ALIGN(spsm_buffer_size, ALIGNED_SIZE);
    *bufferSize = spsm_buffer_size;

    size_t trans_buffer_size = 0;
    stat = mcspGetGenericTransExBuffersize(handle, opA, computeType, (idxType *)matA->rows, (idxType *)matA->cols, matA,
                                           trans_buffer_size);
    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }
    *bufferSize = matA->fixed_length_buffer_size + std::max(spsm_buffer_size, trans_buffer_size);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspSpSM_analysis_impl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                    const void *alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB,
                                    mcspDnMatDescr_t matC, macaDataType computeType, mcsparseSpSMAlg_t alg,
                                    mcspSpSMDescr_t spsmDescr, void *externalBuffer) {
    if (matA == nullptr || matA->mat_descr == nullptr || matB == nullptr || matC == nullptr || spsmDescr == nullptr ||
        externalBuffer == nullptr || matA->to_csr_rows != nullptr || matA->to_csr_cols != nullptr ||
        matA->to_csr_vals != nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (!(matA->format == MCSPARSE_FORMAT_CSR || matA->format == MCSPARSE_FORMAT_COO)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
    if (matA->valueType != computeType) {
        return MCSP_STATUS_INVALID_VALUE;
    }
    // NOTE: for generic API, operation_B does not influence matC storage, which is different from raw csrsm2
    if (!((opB == MCSPARSE_OPERATION_NON_TRANSPOSE && matB->row_num == matA->col_num &&
           matB->row_num == matC->row_num && matB->col_num == matC->col_num && matB->ld >= matB->row_num &&
           matC->ld >= matC->row_num) ||
          (opB != MCSPARSE_OPERATION_NON_TRANSPOSE && matB->col_num == matA->col_num &&
           matB->col_num == matC->row_num && matB->ld >= matB->row_num && matC->ld >= matC->row_num))) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (opA != MCSPARSE_OPERATION_NON_TRANSPOSE &&
        (matA->assist_index_buffer_size == 0 || matA->fixed_length_buffer_size == 0)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    idxType working_nrhs = (idxType)matB->col_num;
    if (opB != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        working_nrhs = (idxType)matB->row_num;
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

    void *spsm_buffer = (void *)(reinterpret_cast<char *>(externalBuffer) + matA->fixed_length_buffer_size);
    spsmDescr->external_buffer = spsm_buffer;
    mcStream_t stream = mcspGetStreamInternal(handle);
    switch (computeType) {
        case MACA_R_32F:
            stat = mcspCsrSpsmAnalysis_template(
                handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz, (float *)alpha,
                matA->mat_descr, (float *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                (float *)matB->values, (idxType)matB->ld, spsmDescr->mat_info, MCSPARSE_ANALYSIS_POLICY_AUTO,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, spsmDescr->external_buffer);
            break;
        case MACA_R_64F:
            stat = mcspCsrSpsmAnalysis_template(
                handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz, (double *)alpha,
                matA->mat_descr, (double *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                (double *)matB->values, (idxType)matB->ld, spsmDescr->mat_info, MCSPARSE_ANALYSIS_POLICY_AUTO,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, spsmDescr->external_buffer);
            break;
        case MACA_C_32F:
            stat = mcspCsrSpsmAnalysis_template(
                handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz, (mcspComplexFloat *)alpha,
                matA->mat_descr, (mcspComplexFloat *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                (mcspComplexFloat *)matB->values, (idxType)matB->ld, spsmDescr->mat_info, MCSPARSE_ANALYSIS_POLICY_AUTO,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, spsmDescr->external_buffer);
            break;
        case MACA_C_64F:
            stat = mcspCsrSpsmAnalysis_template(
                handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz, (mcspComplexDouble *)alpha,
                matA->mat_descr, (mcspComplexDouble *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                (mcspComplexDouble *)matB->values, (idxType)matB->ld, spsmDescr->mat_info,
                MCSPARSE_ANALYSIS_POLICY_AUTO, MCSPARSE_SOLVE_POLICY_NO_LEVEL, spsmDescr->external_buffer);
            break;
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
    if ((matA->mat_descr->diag_type != MCSPARSE_DIAG_TYPE_UNIT) && (spsmDescr->mat_info->zero_pivot_lead != -1)) {
        stat = MCSP_STATUS_ZERO_PIVOT;
        MACA_ASSERT(mcMemcpyAsync(externalBuffer, &(spsmDescr->mat_info->zero_pivot_lead),
                                  sizeof(spsmDescr->mat_info->zero_pivot_lead), mcMemcpyHostToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }
    return stat;
}

template <typename idxType>
mcspStatus_t mcspSpSM_solve_impl(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void *alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, mcspDnMatDescr_t matC,
                                 macaDataType computeType, mcsparseSpSMAlg_t alg, mcspSpSMDescr_t spsmDescr) {
    if (matA == nullptr || matB == nullptr || matC == nullptr || spsmDescr == nullptr || matC->values == nullptr ||
        matB->values == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (!(matB->valueType == matC->valueType && matB->order == matC->order)) {
        return MCSP_STATUS_INVALID_VALUE;
    }
    // NOTE: for generic API, operation_B does not influence matC storage, which is different from raw csrsm2
    if (!((opB == MCSPARSE_OPERATION_NON_TRANSPOSE && matB->row_num == matA->col_num &&
           matB->row_num == matC->row_num && matB->col_num == matC->col_num && matB->ld >= matB->row_num &&
           matC->ld >= matC->row_num) ||
          (opB != MCSPARSE_OPERATION_NON_TRANSPOSE && matB->col_num == matA->col_num &&
           matB->col_num == matC->row_num && matB->ld >= matB->row_num && matC->ld >= matC->row_num))) {
        return MCSP_STATUS_INVALID_SIZE;
    }
    if (!(matA->format == MCSPARSE_FORMAT_CSR || matA->format == MCSPARSE_FORMAT_COO)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    idxType working_nrhs = (idxType)matB->col_num;
    if (opB != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        working_nrhs = (idxType)matB->row_num;
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

    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    switch (computeType) {
        case MACA_R_32F: {
            stat = mcspCsrSpsmSolve_template(handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz,
                                             (float *)alpha, matA->mat_descr, (float *)working_vals,
                                             (idxType *)working_rows, (idxType *)working_cols, (float *)matB->values,
                                             (idxType)matB->ld, spsmDescr->mat_info, MCSPARSE_SOLVE_POLICY_NO_LEVEL,
                                             spsmDescr->external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }
            stat = mcspTransferDeviceDenseMat(handle, opB, (idxType)matB->row_num, (idxType)matB->col_num,
                                              (idxType)matB->ld, (idxType)matC->row_num, (idxType)matC->col_num,
                                              (idxType)matC->ld, (float *)matB->values, (float *)matC->values);
            break;
        }
        case MACA_R_64F: {
            stat = mcspCsrSpsmSolve_template(handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz,
                                             (double *)alpha, matA->mat_descr, (double *)working_vals,
                                             (idxType *)working_rows, (idxType *)working_cols, (double *)matB->values,
                                             (idxType)matB->ld, spsmDescr->mat_info, MCSPARSE_SOLVE_POLICY_NO_LEVEL,
                                             spsmDescr->external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }
            stat = mcspTransferDeviceDenseMat(handle, opB, (idxType)matB->row_num, (idxType)matB->col_num,
                                              (idxType)matB->ld, (idxType)matC->row_num, (idxType)matC->col_num,
                                              (idxType)matC->ld, (double *)matB->values, (double *)matC->values);
            break;
        }
        case MACA_C_32F: {
            stat = mcspCsrSpsmSolve_template(
                handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz, (mcspComplexFloat *)alpha,
                matA->mat_descr, (mcspComplexFloat *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                (mcspComplexFloat *)matB->values, (idxType)matB->ld, spsmDescr->mat_info,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, spsmDescr->external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }
            stat = mcspTransferDeviceDenseMat(handle, opB, (idxType)matB->row_num, (idxType)matB->col_num,
                                              (idxType)matB->ld, (idxType)matC->row_num, (idxType)matC->col_num,
                                              (idxType)matC->ld, (mcspComplexFloat *)matB->values,
                                              (mcspComplexFloat *)matC->values);
            break;
        }
        case MACA_C_64F: {
            stat = mcspCsrSpsmSolve_template(
                handle, opA, opB, (idxType)matA->row_num, working_nrhs, (idxType)matA->nnz, (mcspComplexDouble *)alpha,
                matA->mat_descr, (mcspComplexDouble *)working_vals, (idxType *)working_rows, (idxType *)working_cols,
                (mcspComplexDouble *)matB->values, (idxType)matB->ld, spsmDescr->mat_info,
                MCSPARSE_SOLVE_POLICY_NO_LEVEL, spsmDescr->external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }
            stat = mcspTransferDeviceDenseMat(handle, opB, (idxType)matB->row_num, (idxType)matB->col_num,
                                              (idxType)matB->ld, (idxType)matC->row_num, (idxType)matC->col_num,
                                              (idxType)matC->ld, (mcspComplexDouble *)matB->values,
                                              (mcspComplexDouble *)matC->values);
            break;
        }
        default:
            stat = MCSP_STATUS_NOT_IMPLEMENTED;
    }

    return stat;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsrSpsmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt nrhs, mcspInt nnz, const float *alpha,
                                    const mcspMatDescr_t descr, const float *csr_vals, const mcspInt *csr_rows,
                                    const mcspInt *csr_cols, const float *B, mcspInt ldb, mcspMatInfo_t info,
                                    mcsparseSolvePolicy_t policy, size_t *buffer_size) {
    return mcspCsrSpsmBuffersize_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows,
                                          csr_cols, B, ldb, info, policy, buffer_size);
}

mcspStatus_t mcspDcsrSpsmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt nrhs, mcspInt nnz, const double *alpha,
                                    const mcspMatDescr_t descr, const double *csr_vals, const mcspInt *csr_rows,
                                    const mcspInt *csr_cols, const double *B, mcspInt ldb, mcspMatInfo_t info,
                                    mcsparseSolvePolicy_t policy, size_t *buffer_size) {
    return mcspCsrSpsmBuffersize_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows,
                                          csr_cols, B, ldb, info, policy, buffer_size);
}

mcspStatus_t mcspCcsrSpsmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt nrhs, mcspInt nnz, const mcspComplexFloat *alpha,
                                    const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals,
                                    const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspComplexFloat *B,
                                    mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy,
                                    size_t *buffer_size) {
    return mcspCsrSpsmBuffersize_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows,
                                          csr_cols, B, ldb, info, policy, buffer_size);
}

mcspStatus_t mcspZcsrSpsmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt nrhs, mcspInt nnz, const mcspComplexDouble *alpha,
                                    const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                                    const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspComplexDouble *B,
                                    mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy,
                                    size_t *buffer_size) {
    return mcspCsrSpsmBuffersize_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows,
                                          csr_cols, B, ldb, info, policy, buffer_size);
}

mcspStatus_t mcspScsrSpsmAnalysis(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                  mcspInt m, mcspInt nrhs, mcspInt nnz, const float *alpha, const mcspMatDescr_t descr,
                                  const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                  const float *B, mcspInt ldb, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis,
                                  mcsparseSolvePolicy_t solve, void *temp_buffer) {
    return mcspCsrSpsmAnalysis_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows,
                                        csr_cols, B, ldb, info, analysis, solve, temp_buffer);
}

mcspStatus_t mcspDcsrSpsmAnalysis(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                  mcspInt m, mcspInt nrhs, mcspInt nnz, const double *alpha, const mcspMatDescr_t descr,
                                  const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                  const double *B, mcspInt ldb, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis,
                                  mcsparseSolvePolicy_t solve, void *temp_buffer) {
    return mcspCsrSpsmAnalysis_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows,
                                        csr_cols, B, ldb, info, analysis, solve, temp_buffer);
}

mcspStatus_t mcspCcsrSpsmAnalysis(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                  mcspInt m, mcspInt nrhs, mcspInt nnz, const mcspComplexFloat *alpha,
                                  const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals, const mcspInt *csr_rows,
                                  const mcspInt *csr_cols, const mcspComplexFloat *B, mcspInt ldb, mcspMatInfo_t info,
                                  mcsparseAnalysisPolicy_t analysis, mcsparseSolvePolicy_t solve, void *temp_buffer) {
    return mcspCsrSpsmAnalysis_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows,
                                        csr_cols, B, ldb, info, analysis, solve, temp_buffer);
}

mcspStatus_t mcspZcsrSpsmAnalysis(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                  mcspInt m, mcspInt nrhs, mcspInt nnz, const mcspComplexDouble *alpha,
                                  const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspComplexDouble *B,
                                  mcspInt ldb, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis,
                                  mcsparseSolvePolicy_t solve, void *temp_buffer) {
    return mcspCsrSpsmAnalysis_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows,
                                        csr_cols, B, ldb, info, analysis, solve, temp_buffer);
}

mcspStatus_t mcspScsrSpsmSolve(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                               mcspInt nrhs, mcspInt nnz, const float *alpha, const mcspMatDescr_t descr,
                               const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols, float *B,
                               mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy, void *temp_buffer) {
    return mcspCsrSpsmSolve_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows, csr_cols,
                                     B, ldb, info, policy, temp_buffer);
}

mcspStatus_t mcspDcsrSpsmSolve(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                               mcspInt nrhs, mcspInt nnz, const double *alpha, const mcspMatDescr_t descr,
                               const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols, double *B,
                               mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy, void *temp_buffer) {
    return mcspCsrSpsmSolve_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows, csr_cols,
                                     B, ldb, info, policy, temp_buffer);
}

mcspStatus_t mcspCcsrSpsmSolve(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                               mcspInt nrhs, mcspInt nnz, const mcspComplexFloat *alpha, const mcspMatDescr_t descr,
                               const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                               mcspComplexFloat *B, mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy,
                               void *temp_buffer) {
    return mcspCsrSpsmSolve_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows, csr_cols,
                                     B, ldb, info, policy, temp_buffer);
}

mcspStatus_t mcspZcsrSpsmSolve(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                               mcspInt nrhs, mcspInt nnz, const mcspComplexDouble *alpha, const mcspMatDescr_t descr,
                               const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                               mcspComplexDouble *B, mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy,
                               void *temp_buffer) {
    return mcspCsrSpsmSolve_template(handle, trans_A, trans_B, m, nrhs, nnz, alpha, descr, csr_vals, csr_rows, csr_cols,
                                     B, ldb, info, policy, temp_buffer);
}

mcspStatus_t mcspCsrSpsmClear(mcspHandle_t handle, mcspMatInfo_t info) {
    return mcspCsrSpsmClearImpl(handle, info);
}

mcspStatus_t mcspCsrSpsmZeroPivot(mcspHandle_t handle, mcspMatInfo_t info, int *position) {
    return mcspCsrSpsmZeroPivotImpl(handle, info, position);
}

mcspStatus_t mcspCuinXcsrsm2_zeroPivot(mcspHandle_t handle, mcspCsrsm2Info_t info, int *position) {
    return mcspCuinCsrSpsmZeroPivotImpl(handle, info, position);
}

mcspStatus_t mcspCuinScsrsm2_bufferSizeExt(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                         mcsparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
                                         const mcspMatDescr_t descrA, const float *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B,
                                         int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                         size_t *pBufferSize) {
    return mcspCuinXcsrsm2_bufferSize_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}

mcspStatus_t mcspCuinDcsrsm2_bufferSizeExt(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                         mcsparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
                                         const mcspMatDescr_t descrA, const double *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *B,
                                         int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                         size_t *pBufferSize) {
    return mcspCuinXcsrsm2_bufferSize_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}

mcspStatus_t mcspCuinCcsrsm2_bufferSizeExt(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                         mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                         const mcspComplexFloat *alpha, const mcspMatDescr_t descrA,
                                         const mcspComplexFloat *csrSortedValA, const int *csrSortedRowPtrA,
                                         const int *csrSortedColIndA, const mcspComplexFloat *B, int ldb,
                                         mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, size_t *pBufferSize) {
    return mcspCuinXcsrsm2_bufferSize_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}

mcspStatus_t mcspCuinZcsrsm2_bufferSizeExt(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                         mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                         const mcspComplexDouble *alpha, const mcspMatDescr_t descrA,
                                         const mcspComplexDouble *csrSortedValA, const int *csrSortedRowPtrA,
                                         const int *csrSortedColIndA, const mcspComplexDouble *B, int ldb,
                                         mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, size_t *pBufferSize) {
    return mcspCuinXcsrsm2_bufferSize_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}

mcspStatus_t mcspCuinScsrsm2_analysis(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                    mcsparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
                                    const mcspMatDescr_t descrA, const float *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B, int ldb,
                                    mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsm2_analysis_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

mcspStatus_t mcspCuinDcsrsm2_analysis(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                    mcsparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
                                    const mcspMatDescr_t descrA, const double *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *B, int ldb,
                                    mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsm2_analysis_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

mcspStatus_t mcspCuinCcsrsm2_analysis(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                    mcsparseOperation_t transB, int m, int nrhs, int nnz, const mcspComplexFloat *alpha,
                                    const mcspMatDescr_t descrA, const mcspComplexFloat *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const mcspComplexFloat *B,
                                    int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsm2_analysis_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

mcspStatus_t mcspCuinZcsrsm2_analysis(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                    mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                    const mcspComplexDouble *alpha, const mcspMatDescr_t descrA,
                                    const mcspComplexDouble *csrSortedValA, const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA, const mcspComplexDouble *B, int ldb,
                                    mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsm2_analysis_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

mcspStatus_t mcspCuinScsrsm2_solve(mcspHandle_t handle, int algo, mcsparseOperation_t transA, mcsparseOperation_t transB,
                                 int m, int nrhs, int nnz, const float *alpha, const mcspMatDescr_t descrA,
                                 const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                 float *B, int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                 void *pBuffer) {
    return mcspCuinXcsrsm2_solve_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                        csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

mcspStatus_t mcspCuinDcsrsm2_solve(mcspHandle_t handle, int algo, mcsparseOperation_t transA, mcsparseOperation_t transB,
                                 int m, int nrhs, int nnz, const double *alpha, const mcspMatDescr_t descrA,
                                 const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                 double *B, int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                 void *pBuffer) {
    return mcspCuinXcsrsm2_solve_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                        csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

mcspStatus_t mcspCuinCcsrsm2_solve(mcspHandle_t handle, int algo, mcsparseOperation_t transA, mcsparseOperation_t transB,
                                 int m, int nrhs, int nnz, const mcspComplexFloat *alpha, const mcspMatDescr_t descrA,
                                 const mcspComplexFloat *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspComplexFloat *B, int ldb, mcspCsrsm2Info_t info,
                                 mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsm2_solve_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                        csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

mcspStatus_t mcspCuinZcsrsm2_solve(mcspHandle_t handle, int algo, mcsparseOperation_t transA, mcsparseOperation_t transB,
                                 int m, int nrhs, int nnz, const mcspComplexDouble *alpha, const mcspMatDescr_t descrA,
                                 const mcspComplexDouble *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspComplexDouble *B, int ldb, mcspCsrsm2Info_t info,
                                 mcsparseSolvePolicy_t policy, void *pBuffer) {
    return mcspCuinXcsrsm2_solve_template(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                        csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

mcspStatus_t mcspSpSM_bufferSize(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void *alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, mcspDnMatDescr_t matC,
                                 macaDataType computeType, mcsparseSpSMAlg_t alg, mcspSpSMDescr_t spsmDescr,
                                 size_t *bufferSize) {
    return mcspSpSM_bufferSize_impl<mcspInt>(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr,
                                             bufferSize);
}

mcspStatus_t mcspSpSM_analysis(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void *alpha,
                               mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, mcspDnMatDescr_t matC,
                               macaDataType computeType, mcsparseSpSMAlg_t alg, mcspSpSMDescr_t spsmDescr,
                               void *externalBuffer) {
    return mcspSpSM_analysis_impl<mcspInt>(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr,
                                           externalBuffer);
}

mcspStatus_t mcspSpSM_solve(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void *alpha,
                            mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, mcspDnMatDescr_t matC,
                            macaDataType computeType, mcsparseSpSMAlg_t alg, mcspSpSMDescr_t spsmDescr) {
    return mcspSpSM_solve_impl<mcspInt>(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr);
}

#ifdef __cplusplus
}
#endif
