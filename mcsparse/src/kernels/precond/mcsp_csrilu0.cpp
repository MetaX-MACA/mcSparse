#include <algorithm>
#include <vector>

#include "common/mcsp_types.h"
#include "csrilu0_device.hpp"
#include "device_reduce.hpp"
#include "internal_interface/mcsp_internal_conversion.h"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "spsv/csr_trm_analysis.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspCsrilu0BuffersizeImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                                       const valType* csr_vals, const idxType* csr_rows, const idxType* csr_cols,
                                       mcspMatInfo_t info, size_t* buffersize) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descr == nullptr || csr_vals == nullptr || csr_rows == nullptr || csr_cols == nullptr ||
        buffersize == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    size_t reduce_buffersize;
    size_t radix_sort_buffersize;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::reduce(nullptr, reduce_buffersize, (idxType*)nullptr, (idxType*)nullptr, (size_t)m,
                   mcprim::minimum<idxType>(), stream);
    mcprim::radix_sort_pairs(nullptr, radix_sort_buffersize, (idxType*)nullptr, (idxType*)nullptr, (idxType*)nullptr,
                             (idxType*)nullptr, (size_t)m, stream);
    *buffersize = ALIGN((4 + 3 * m) * sizeof(idxType) + reduce_buffersize + radix_sort_buffersize, ALIGNED_SIZE);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsrilu0AnalysisImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                                     const valType* csr_vals, const idxType* csr_rows, const idxType* csr_cols,
                                     mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                     mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descr == nullptr || csr_vals == nullptr || csr_rows == nullptr || csr_cols == nullptr ||
        temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (info->csrilu0_info == nullptr && mcspCreateTrmInfo(&(info->csrilu0_info)) != MCSP_STATUS_SUCCESS) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }
    return mcspCsrTrmAnalysis_template(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info->csrilu0_info,
                                       info->zero_pivot_lead, temp_buffer, true);
}

mcspStatus_t mcspCsrilu0ZeroPivotImpl(mcspHandle_t handle, mcspMatInfo_t info, int* position) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || position == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *position = info->zero_pivot_lead;
    if (*position == -1) {
        return MCSP_STATUS_SUCCESS;
    } else {
        return MCSP_STATUS_ZERO_PIVOT;
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsrilu0Impl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr, valType* csr_vals,
                             const idxType* csr_rows, const idxType* csr_cols, mcspMatInfo_t info,
                             mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descr == nullptr || csr_vals == nullptr || csr_rows == nullptr || csr_cols == nullptr ||
        temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    int n_elem = WARP_SIZE;
    int n_block = (m + n_elem / WARP_SIZE - 1) / (n_elem / WARP_SIZE);
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemsetAsync(temp_buffer, 0, m * sizeof(idxType), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    idxType max_row_nnz = info->csrilu0_info->max_row_nnz;
    if (max_row_nnz < 256) {
        mcLaunchKernelGGL((mcspCsrilu0HashTableKernel<WARP_SIZE, 256>), dim3(n_block), dim3(n_elem),
                           256 * 2 * sizeof(idxType), stream, m, csr_vals, csr_rows, csr_cols,
                           (idxType*)(info->csrilu0_info->row_map), (idxType*)(info->csrilu0_info->trm_diag_ind),
                           (idxType*)temp_buffer, descr->base);
    } else if (max_row_nnz < 512) {
        mcLaunchKernelGGL((mcspCsrilu0HashTableKernel<WARP_SIZE, 512>), dim3(n_block), dim3(n_elem),
                           512 * 2 * sizeof(idxType), stream, m, csr_vals, csr_rows, csr_cols,
                           (idxType*)(info->csrilu0_info->row_map), (idxType*)(info->csrilu0_info->trm_diag_ind),
                           (idxType*)temp_buffer, descr->base);
    } else if (max_row_nnz < 1024) {
        mcLaunchKernelGGL((mcspCsrilu0HashTableKernel<WARP_SIZE, 1024>), dim3(n_block), dim3(n_elem),
                           1024 * 2 * sizeof(idxType), stream, m, csr_vals, csr_rows, csr_cols,
                           (idxType*)(info->csrilu0_info->row_map), (idxType*)(info->csrilu0_info->trm_diag_ind),
                           (idxType*)temp_buffer, descr->base);
    } else if (max_row_nnz < 2048) {
        mcLaunchKernelGGL((mcspCsrilu0HashTableKernel<WARP_SIZE, 2048>), dim3(n_block), dim3(n_elem),
                           2048 * 2 * sizeof(idxType), stream, m, csr_vals, csr_rows, csr_cols,
                           (idxType*)(info->csrilu0_info->row_map), (idxType*)(info->csrilu0_info->trm_diag_ind),
                           (idxType*)temp_buffer, descr->base);
    } else if (max_row_nnz < 4096) {
        mcLaunchKernelGGL((mcspCsrilu0HashTableKernel<WARP_SIZE, 4096>), dim3(n_block), dim3(n_elem),
                           4096 * 2 * sizeof(idxType), stream, m, csr_vals, csr_rows, csr_cols,
                           (idxType*)(info->csrilu0_info->row_map), (idxType*)(info->csrilu0_info->trm_diag_ind),
                           (idxType*)temp_buffer, descr->base);
    } else {
        mcLaunchKernelGGL((mcspCsrilu0BsearchKernel), dim3(n_block), dim3(n_elem), 0, stream, m, csr_vals, csr_rows,
                           csr_cols, (idxType*)(info->csrilu0_info->row_map),
                           (idxType*)(info->csrilu0_info->trm_diag_ind), (idxType*)temp_buffer, descr->base);
    }

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDcsrilu0ClearImpl(mcspHandle_t handle, mcspMatInfo_t info) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (info->csrilu0_info != nullptr) {
        return mcspDestroyTrmInfo(info->csrilu0_info);
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspXcsrilu02_zeroPivotImpl(mcspHandle_t handle, mcspCsrilu02Info_t info, int* position) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || position == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    return mcspCsrilu0ZeroPivotImpl(handle, info->csrilu0_mat, position);
}

template <typename idxType, typename valType>
mcspStatus_t mcspXcsrilu02_bufferSizeImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descrA,
                                          valType* csrValA, const idxType* csrRowPtrA, const idxType* csrColIndA,
                                          mcspCsrilu02Info_t info, int* pBufferSizeInBytes) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descrA == nullptr || csrValA == nullptr || csrRowPtrA == nullptr || csrColIndA == nullptr ||
        pBufferSizeInBytes == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }
    size_t temp_size = 0;
    mcspStatus_t ret = mcspCsrilu0BuffersizeImpl(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
                                                 info->csrilu0_mat, &temp_size);
    *pBufferSizeInBytes = (int)temp_size;
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspXcsrilu02_bufferSizeExtImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descrA,
                                             valType* csrSortedVal, const idxType* csrSortedRowPtr,
                                             const idxType* csrSortedColInd, mcspCsrilu02Info_t info,
                                             size_t* pBufferSize) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descrA == nullptr || csrSortedVal == nullptr || csrSortedRowPtr == nullptr ||
        csrSortedColInd == nullptr || pBufferSize == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    return mcspCsrilu0BuffersizeImpl(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                                     info->csrilu0_mat, pBufferSize);
}

template <typename valType>
mcspStatus_t mcspXcsrilu02_boostImpl(mcspHandle_t handle, mcspCsrilu02Info_t info, int enable_boost, double* tol,
                                     valType* boost_val) {
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspXcsrilu02_analysisImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descrA,
                                        const valType* csrValA, const idxType* csrRowPtrA, const idxType* csrColIndA,
                                        mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descrA == nullptr || csrValA == nullptr || csrRowPtrA == nullptr || csrColIndA == nullptr ||
        pBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    return mcspCsrilu0AnalysisImpl(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info->csrilu0_mat,
                                   MCSPARSE_ANALYSIS_POLICY_AUTO, policy, pBuffer);
}

template <typename idxType, typename valType>
mcspStatus_t mcspXcsrilu02Impl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descrA,
                               valType* csrValA_valM, const idxType* csrRowPtrA, const idxType* csrColIndA,
                               mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descrA == nullptr || csrValA_valM == nullptr || csrRowPtrA == nullptr ||
        csrColIndA == nullptr || pBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    return mcspCsrilu0Impl(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info->csrilu0_mat, policy,
                           pBuffer);
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsrilu0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                    const float* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                    mcspMatInfo_t info, size_t* buffersize) {
    return mcspCsrilu0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
}

mcspStatus_t mcspDcsrilu0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                    const double* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                    mcspMatInfo_t info, size_t* buffersize) {
    return mcspCsrilu0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
}

mcspStatus_t mcspCcsrilu0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                    const mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                    mcspMatInfo_t info, size_t* buffersize) {
    return mcspCsrilu0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
}

mcspStatus_t mcspZcsrilu0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                    const mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                    mcspMatInfo_t info, size_t* buffersize) {
    return mcspCsrilu0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
}

mcspStatus_t mcspScsrilu0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                  const float* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                  mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsrilu0AnalysisImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, analysis_policy,
                                   solve_policy, temp_buffer);
}

mcspStatus_t mcspDcsrilu0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                  const double* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                  mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsrilu0AnalysisImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, analysis_policy,
                                   solve_policy, temp_buffer);
}

mcspStatus_t mcspCcsrilu0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                  const mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                  mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsrilu0AnalysisImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, analysis_policy,
                                   solve_policy, temp_buffer);
}

mcspStatus_t mcspZcsrilu0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                  const mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                  mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsrilu0AnalysisImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, analysis_policy,
                                   solve_policy, temp_buffer);
}

mcspStatus_t mcspXcsrilu0ZeroPivot(mcspHandle_t handle, mcspMatInfo_t info, int* position) {
    return mcspCsrilu0ZeroPivotImpl(handle, info, position);
}

mcspStatus_t mcspScsrilu0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, float* csr_vals,
                          const mcspInt* csr_rows, const mcspInt* csr_cols, mcspMatInfo_t info,
                          mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsrilu0Impl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspDcsrilu0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, double* csr_vals,
                          const mcspInt* csr_rows, const mcspInt* csr_cols, mcspMatInfo_t info,
                          mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsrilu0Impl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCcsrilu0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                          mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                          mcspMatInfo_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsrilu0Impl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspZcsrilu0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                          mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                          mcspMatInfo_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsrilu0Impl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCsrilu0Clear(mcspHandle_t handle, mcspMatInfo_t info) {
    return mcspDcsrilu0ClearImpl(handle, info);
}

mcspStatus_t mcspCuinXcsrilu02_zeroPivot(mcspHandle_t handle, mcspCsrilu02Info_t info, int* position) {
    return mcspXcsrilu02_zeroPivotImpl(handle, info, position);
}

mcspStatus_t mcspCuinScsrilu02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                        float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                        mcspCsrilu02Info_t info, int* pBufferSizeInBytes) {
    return mcspXcsrilu02_bufferSizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA, (mcspInt*)csrRowPtrA,
                                        (mcspInt*)csrColIndA, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinDcsrilu02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                        double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                        mcspCsrilu02Info_t info, int* pBufferSizeInBytes) {
    return mcspXcsrilu02_bufferSizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA, (mcspInt*)csrRowPtrA,
                                        (mcspInt*)csrColIndA, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinCcsrilu02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                        mcspComplexFloat* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                        mcspCsrilu02Info_t info, int* pBufferSizeInBytes) {
    return mcspXcsrilu02_bufferSizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA, (mcspInt*)csrRowPtrA,
                                        (mcspInt*)csrColIndA, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinZcsrilu02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                        mcspComplexDouble* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                        mcspCsrilu02Info_t info, int* pBufferSizeInBytes) {
    return mcspXcsrilu02_bufferSizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA, (mcspInt*)csrRowPtrA,
                                        (mcspInt*)csrColIndA, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinScsrilu02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                           float* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd,
                                           mcspCsrilu02Info_t info, size_t* pBufferSize) {
    return mcspXcsrilu02_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrSortedVal,
                                           (mcspInt*)csrSortedRowPtr, (mcspInt*)csrSortedColInd, info, pBufferSize);
}

mcspStatus_t mcspCuinDcsrilu02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                           double* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd,
                                           mcspCsrilu02Info_t info, size_t* pBufferSize) {
    return mcspXcsrilu02_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrSortedVal,
                                           (mcspInt*)csrSortedRowPtr, (mcspInt*)csrSortedColInd, info, pBufferSize);
}

mcspStatus_t mcspCuinCcsrilu02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                           mcFloatComplex* csrSortedVal, const int* csrSortedRowPtr,
                                           const int* csrSortedColInd, mcspCsrilu02Info_t info, size_t* pBufferSize) {
    return mcspXcsrilu02_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrSortedVal,
                                           (mcspInt*)csrSortedRowPtr, (mcspInt*)csrSortedColInd, info, pBufferSize);
}

mcspStatus_t mcspCuinZcsrilu02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                           mcDoubleComplex* csrSortedVal, const int* csrSortedRowPtr,
                                           const int* csrSortedColInd, mcspCsrilu02Info_t info, size_t* pBufferSize) {
    return mcspXcsrilu02_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrSortedVal,
                                           (mcspInt*)csrSortedRowPtr, (mcspInt*)csrSortedColInd, info, pBufferSize);
}

mcspStatus_t mcspCuinScsrilu02_numericBoost(mcspHandle_t handle, mcspCsrilu02Info_t info, int enable_boost, double* tol,
                                          float* boost_val) {
    return mcspXcsrilu02_boostImpl(handle, info, enable_boost, tol, boost_val);
}

mcspStatus_t mcspCuinDcsrilu02_numericBoost(mcspHandle_t handle, mcspCsrilu02Info_t info, int enable_boost, double* tol,
                                          double* boost_val) {
    return mcspXcsrilu02_boostImpl(handle, info, enable_boost, tol, boost_val);
}

mcspStatus_t mcspCuinCcsrilu02_numericBoost(mcspHandle_t handle, mcspCsrilu02Info_t info, int enable_boost, double* tol,
                                          mcspComplexFloat* boost_val) {
    return mcspXcsrilu02_boostImpl(handle, info, enable_boost, tol, boost_val);
}

mcspStatus_t mcspCuinZcsrilu02_numericBoost(mcspHandle_t handle, mcspCsrilu02Info_t info, int enable_boost, double* tol,
                                          mcspComplexDouble* boost_val) {
    return mcspXcsrilu02_boostImpl(handle, info, enable_boost, tol, boost_val);
}

mcspStatus_t mcspCuinScsrilu02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                      const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                      mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspXcsrilu02_analysisImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA, (mcspInt*)csrRowPtrA,
                                      (mcspInt*)csrColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinDcsrilu02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                      const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                      mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspXcsrilu02_analysisImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA, (mcspInt*)csrRowPtrA,
                                      (mcspInt*)csrColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinCcsrilu02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                      const mcspComplexFloat* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                      mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspXcsrilu02_analysisImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA, (mcspInt*)csrRowPtrA,
                                      (mcspInt*)csrColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinZcsrilu02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                      const mcspComplexDouble* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                      mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspXcsrilu02_analysisImpl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA, (mcspInt*)csrRowPtrA,
                                      (mcspInt*)csrColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinScsrilu02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA, float* csrValA_valM,
                             const int* csrRowPtrA, const int* csrColIndA, mcspCsrilu02Info_t info,
                             mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspXcsrilu02Impl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA_valM, (mcspInt*)csrRowPtrA,
                             (mcspInt*)csrColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinDcsrilu02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA, double* csrValA_valM,
                             const int* csrRowPtrA, const int* csrColIndA, mcspCsrilu02Info_t info,
                             mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspXcsrilu02Impl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA_valM, (mcspInt*)csrRowPtrA,
                             (mcspInt*)csrColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinCcsrilu02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                             mcspComplexFloat* csrValA_valM, const int* csrRowPtrA, const int* csrColIndA,
                             mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspXcsrilu02Impl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA_valM, (mcspInt*)csrRowPtrA,
                             (mcspInt*)csrColIndA, info, policy, pBuffer);
}

mcspStatus_t mcspCuinZcsrilu02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                             mcspComplexDouble* csrValA_valM, const int* csrRowPtrA, const int* csrColIndA,
                             mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return mcspXcsrilu02Impl(handle, (mcspInt)m, (mcspInt)nnz, descrA, csrValA_valM, (mcspInt*)csrRowPtrA,
                             (mcspInt*)csrColIndA, info, policy, pBuffer);
}

#ifdef __cplusplus
}
#endif
