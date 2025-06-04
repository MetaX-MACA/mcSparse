#include "common/mcsp_types.h"
#include "csric0_device.hpp"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "spsv/csr_trm_analysis.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspCsric0BuffersizeImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                                      const valType* csr_vals, const idxType* csr_rows, const idxType* csr_cols,
                                      mcspMatInfo_t info, size_t* buffersize) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descr == nullptr || buffersize == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m == 0) {
        *buffersize = ALIGNED_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    if (csr_rows == nullptr || (nnz != 0 && (csr_vals == nullptr || csr_cols == nullptr))) {
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
mcspStatus_t mcspCsric0AnalysisImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                                    const valType* csr_vals, const idxType* csr_rows, const idxType* csr_cols,
                                    mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                    mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descr == nullptr || temp_buffer == nullptr || csr_rows == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if ((nnz != 0 && (csr_vals == nullptr || csr_cols == nullptr))) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (info->csric0_info == nullptr && mcspCreateTrmInfo(&(info->csric0_info)) != MCSP_STATUS_SUCCESS) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCsrTrmAnalysis_template(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info->csric0_info,
                                       info->zero_pivot_lead, temp_buffer, true);
}

mcspStatus_t mcspCsric0ZeroPivotImpl(mcspHandle_t handle, mcspMatInfo_t info, int* position) {
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
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsric0Impl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr, valType* csr_vals,
                            const idxType* csr_rows, const idxType* csr_cols, mcspMatInfo_t info,
                            mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (info == nullptr || descr == nullptr || csr_rows == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (nnz != 0 && (csr_vals == nullptr || csr_cols == nullptr)) {
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

    idxType max_row_nnz = info->csric0_info->max_row_nnz;
    if (max_row_nnz < 256) {
        mcLaunchKernelGGL((mcspCsric0HashTableKernel<WARP_SIZE, 256>), dim3(n_block), dim3(n_elem),
                           256 * 2 * sizeof(idxType) + WARP_SIZE * sizeof(valType), stream, m, csr_vals, csr_rows,
                           csr_cols, (idxType*)(info->csric0_info->row_map),
                           (idxType*)(info->csric0_info->trm_diag_ind), (idxType*)info->csric0_info->zero_pivot_lead,
                           (idxType*)temp_buffer, descr->base);
    } else if (max_row_nnz < 512) {
        mcLaunchKernelGGL((mcspCsric0HashTableKernel<WARP_SIZE, 512>), dim3(n_block), dim3(n_elem),
                           512 * 2 * sizeof(idxType) + WARP_SIZE * sizeof(valType), stream, m, csr_vals, csr_rows,
                           csr_cols, (idxType*)(info->csric0_info->row_map),
                           (idxType*)(info->csric0_info->trm_diag_ind), (idxType*)info->csric0_info->zero_pivot_lead,
                           (idxType*)temp_buffer, descr->base);
    } else if (max_row_nnz < 1024) {
        mcLaunchKernelGGL((mcspCsric0HashTableKernel<WARP_SIZE, 1024>), dim3(n_block), dim3(n_elem),
                           1024 * 2 * sizeof(idxType) + WARP_SIZE * sizeof(valType), stream, m, csr_vals, csr_rows,
                           csr_cols, (idxType*)(info->csric0_info->row_map),
                           (idxType*)(info->csric0_info->trm_diag_ind), (idxType*)info->csric0_info->zero_pivot_lead,
                           (idxType*)temp_buffer, descr->base);

    } else if (max_row_nnz < 2048) {
        mcLaunchKernelGGL((mcspCsric0HashTableKernel<WARP_SIZE, 2048>), dim3(n_block), dim3(n_elem),
                           2048 * 2 * sizeof(idxType) + WARP_SIZE * sizeof(valType), stream, m, csr_vals, csr_rows,
                           csr_cols, (idxType*)(info->csric0_info->row_map),
                           (idxType*)(info->csric0_info->trm_diag_ind), (idxType*)info->csric0_info->zero_pivot_lead,
                           (idxType*)temp_buffer, descr->base);

    } else if (max_row_nnz < 4096) {
        mcLaunchKernelGGL((mcspCsric0HashTableKernel<WARP_SIZE, 4096>), dim3(n_block), dim3(n_elem),
                           4096 * 2 * sizeof(idxType) + WARP_SIZE * sizeof(valType), stream, m, csr_vals, csr_rows,
                           csr_cols, (idxType*)(info->csric0_info->row_map),
                           (idxType*)(info->csric0_info->trm_diag_ind), (idxType*)info->csric0_info->zero_pivot_lead,
                           (idxType*)temp_buffer, descr->base);

    } else {
        mcLaunchKernelGGL((mcspCsric0BsearchKernel), dim3(n_block), dim3(n_elem), WARP_SIZE * sizeof(valType), stream,
                           m, csr_vals, csr_rows, csr_cols, (idxType*)(info->csric0_info->row_map),
                           (idxType*)(info->csric0_info->trm_diag_ind), (idxType*)info->csric0_info->zero_pivot_lead,
                           (idxType*)temp_buffer, descr->base);
    }
    MACA_ASSERT(mcMemcpyAsync(&info->zero_pivot_lead, info->csric0_info->zero_pivot_lead, sizeof(idxType),
                              mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCsric0ClearImpl(mcspHandle_t handle, mcspMatInfo_t info) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (info->csric0_info != nullptr) {
        return mcspDestroyTrmInfo(info->csric0_info);
    }
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCuinCsric0BuffersizeImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                                        const valType* csr_vals, const idxType* csr_rows, const idxType* csr_cols,
                                        mcspCsric02Info_t info, int* buffersize) {
    if (info == nullptr || buffersize == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    size_t temp_size = 0;
    mcspStatus_t ret =
        mcspCsric0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info->csric0_mat, &temp_size);
    *buffersize = (int)temp_size;
    return ret;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCuinCsric0_buffersizeImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                                         const valType* csr_vals, const idxType* csr_rows, const idxType* csr_cols,
                                         mcspCsric02Info_t info, size_t* buffersize) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    return mcspCsric0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info->csric0_mat, buffersize);
}

template <typename idxType, typename valType>
mcspStatus_t mcspCuinCsric0AnalysisImpl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                                      const valType* csr_vals, const idxType* csr_rows, const idxType* csr_cols,
                                      mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    return mcspCsric0AnalysisImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info->csric0_mat,
                                  MCSPARSE_ANALYSIS_POLICY_AUTO, solve_policy, temp_buffer);
}

mcspStatus_t mcspCuinCsric0ZeroPivotImpl(mcspHandle_t handle, mcspCsric02Info_t info, int* position) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    return mcspCsric0ZeroPivotImpl(handle, info->csric0_mat, position);
}

template <typename idxType, typename valType>
mcspStatus_t mcspCuinCsric0Impl(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                              valType* csr_vals, const idxType* csr_rows, const idxType* csr_cols,
                              mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    return mcspCsric0Impl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info->csric0_mat, solve_policy,
                          temp_buffer);
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsric0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                   const float* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                   mcspMatInfo_t info, size_t* buffersize) {
    return mcspCsric0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
}

mcspStatus_t mcspDcsric0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                   const double* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                   mcspMatInfo_t info, size_t* buffersize) {
    return mcspCsric0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
}

mcspStatus_t mcspCcsric0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                   const mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                   mcspMatInfo_t info, size_t* buffersize) {
    return mcspCsric0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
}

mcspStatus_t mcspZcsric0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                   const mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                   mcspMatInfo_t info, size_t* buffersize) {
    return mcspCsric0BuffersizeImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
}

mcspStatus_t mcspScsric0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                 const float* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                 mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                 mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsric0AnalysisImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, analysis_policy,
                                  solve_policy, temp_buffer);
}

mcspStatus_t mcspDcsric0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                 const double* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                 mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                 mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsric0AnalysisImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, analysis_policy,
                                  solve_policy, temp_buffer);
}

mcspStatus_t mcspCcsric0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                 const mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                 mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                 mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsric0AnalysisImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, analysis_policy,
                                  solve_policy, temp_buffer);
}

mcspStatus_t mcspZcsric0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                 const mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                 mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                 mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsric0AnalysisImpl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, analysis_policy,
                                  solve_policy, temp_buffer);
}

mcspStatus_t mcspXcsric0ZeroPivot(mcspHandle_t handle, mcspMatInfo_t info, int* position) {
    return mcspCsric0ZeroPivotImpl(handle, info, position);
}

mcspStatus_t mcspScsric0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, float* csr_vals,
                         const mcspInt* csr_rows, const mcspInt* csr_cols, mcspMatInfo_t info,
                         mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsric0Impl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspDcsric0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, double* csr_vals,
                         const mcspInt* csr_rows, const mcspInt* csr_cols, mcspMatInfo_t info,
                         mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsric0Impl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCcsric0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                         mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                         mcspMatInfo_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsric0Impl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspZcsric0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                         mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                         mcspMatInfo_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCsric0Impl(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCsric0Clear(mcspHandle_t handle, mcspMatInfo_t info) {
    return mcspCsric0ClearImpl(handle, info);
}

mcspStatus_t mcspCuinScsric02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                       const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                       mcspCsric02Info_t info, int* buffersize) {
    return mcspCuinCsric0BuffersizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                      (mcspInt*)csr_cols, info, buffersize);
}

mcspStatus_t mcspCuinDcsric02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                       const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                       mcspCsric02Info_t info, int* buffersize) {
    return mcspCuinCsric0BuffersizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                      (mcspInt*)csr_cols, info, buffersize);
}

mcspStatus_t mcspCuinCcsric02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                       const mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                       mcspCsric02Info_t info, int* buffersize) {
    return mcspCuinCsric0BuffersizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                      (mcspInt*)csr_cols, info, buffersize);
}

mcspStatus_t mcspCuinZcsric02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                       const mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                       mcspCsric02Info_t info, int* buffersize) {
    return mcspCuinCsric0BuffersizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                      (mcspInt*)csr_cols, info, buffersize);
}

mcspStatus_t mcspCuinScsric02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                          const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                          mcspCsric02Info_t info, size_t* buffersize) {
    return mcspCuinCsric0_buffersizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                       (mcspInt*)csr_cols, info, buffersize);
}

mcspStatus_t mcspCuinDcsric02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                          const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                          mcspCsric02Info_t info, size_t* buffersize) {
    return mcspCuinCsric0_buffersizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                       (mcspInt*)csr_cols, info, buffersize);
}

mcspStatus_t mcspCuinCcsric02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                          const mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                          mcspCsric02Info_t info, size_t* buffersize) {
    return mcspCuinCsric0_buffersizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                       (mcspInt*)csr_cols, info, buffersize);
}

mcspStatus_t mcspCuinZcsric02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                          const mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                          mcspCsric02Info_t info, size_t* buffersize) {
    return mcspCuinCsric0_buffersizeImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                       (mcspInt*)csr_cols, info, buffersize);
}

mcspStatus_t mcspCuinScsric02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                     const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                     mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCuinCsric0AnalysisImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                    (mcspInt*)csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCuinDcsric02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                     const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                     mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCuinCsric0AnalysisImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                    (mcspInt*)csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCuinCcsric02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                     const mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                     mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCuinCsric0AnalysisImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                    (mcspInt*)csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCuinZcsric02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                     const mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                     mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCuinCsric0AnalysisImpl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows,
                                    (mcspInt*)csr_cols, info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCuinXcsric02_zeroPivot(mcspHandle_t handle, mcspCsric02Info_t info, int* position) {
    return mcspCuinCsric0ZeroPivotImpl(handle, info, position);
}

mcspStatus_t mcspCuinScsric02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, float* csr_vals,
                            const int* csr_rows, const int* csr_cols, mcspCsric02Info_t info,
                            mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCuinCsric0Impl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows, (mcspInt*)csr_cols,
                            info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCuinDcsric02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, double* csr_vals,
                            const int* csr_rows, const int* csr_cols, mcspCsric02Info_t info,
                            mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCuinCsric0Impl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows, (mcspInt*)csr_cols,
                            info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCuinCcsric02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, mcspComplexFloat* csr_vals,
                            const int* csr_rows, const int* csr_cols, mcspCsric02Info_t info,
                            mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCuinCsric0Impl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows, (mcspInt*)csr_cols,
                            info, solve_policy, temp_buffer);
}

mcspStatus_t mcspCuinZcsric02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                            mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                            mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    return mcspCuinCsric0Impl(handle, (mcspInt)m, (mcspInt)nnz, descr, csr_vals, (mcspInt*)csr_rows, (mcspInt*)csr_cols,
                            info, solve_policy, temp_buffer);
}

#ifdef __cplusplus
}
#endif