#include <assert.h>
#include <stdio.h>

#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "coo_flat_device.hpp"
#include "csr_scalar_device.hpp"
#include "csr_vector_device.hpp"
#include "ell_scalar_device.hpp"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_transpose_sparse.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "utils/mcsp_logger.h"

static constexpr mcspInt kRowAverageElementNumThreshold = 2;
template <typename idxType, typename computeType, typename sparseType, typename inputDenseType, typename outputType>
mcspStatus_t mcspSpmvCooFlatTemplate(mcspHandle_t handle, mcsparseOperation_t trans, idxType row_num, idxType col_num,
                                     idxType nnz, const computeType *alpha, const mcspMatDescr_t descr,
                                     const sparseType *coo_vals, const idxType *coo_rows, const idxType *coo_cols,
                                     const inputDenseType *vec_x, const computeType *beta, outputType *vec_y) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz < 0 || row_num < 0 || col_num < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || beta == nullptr || coo_rows == nullptr || coo_cols == nullptr || coo_vals == nullptr ||
        vec_x == nullptr || vec_y == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    computeType h_beta = getScalarToHost(beta, handle->ptr_mode);
    computeType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    if (nnz == 0 || (IsZero(h_alpha) && IsZero(h_beta))) {
        return MCSP_STATUS_SUCCESS;  // @TODO: alpha=0, beta!=0 can be optimized
    }

    idxType *carries_rows;
    computeType *carries;
    mcStream_t stream = mcspGetStreamInternal(handle);

#ifdef __MACA__
    idxType thread_per_block = 512;
#else
    idxType thread_per_block = 1024;
#endif
    idxType n_block_scaling = (row_num + thread_per_block - 1) / thread_per_block;
    idxType n_per_thread = 16;
    idxType n_per_block = n_per_thread * thread_per_block;
    idxType tail_start = (nnz / n_per_block) * n_per_block;
    idxType n_in_tail = nnz - tail_start;
    idxType n_block = tail_start / n_per_block;
    idxType interval_len = n_per_thread * WARP_SIZE;
    idxType n_warp = n_block * thread_per_block / WARP_SIZE;

    size_t buffer_size_req = n_warp * (sizeof(*carries_rows) + sizeof(*carries));
    void *tmp_buffer;
    bool use_buffer_pool = handle->mcspUsePoolBuffer(&tmp_buffer, buffer_size_req);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc(&tmp_buffer, buffer_size_req));
    }
    carries_rows = (idxType *)tmp_buffer;
    carries = (computeType *)(carries_rows + n_warp);

    if constexpr (std::is_same_v<computeType, __half2> || std::is_same_v<computeType, mcsp_bfloat162>) {
        mcLaunchKernelGGL(mcspCooScalingKernelLowPrecisionComplex, dim3(n_block_scaling), dim3(thread_per_block), 0,
                           stream, row_num, h_beta, vec_y);
        if (n_block > 0) {
            mcLaunchKernelGGL(mcspSpmvCooBodyKernelLowPrecisionComplex, dim3(n_block), dim3(thread_per_block), 0,
                               stream, interval_len, nnz, h_alpha, coo_rows, coo_cols, coo_vals, vec_x, vec_y,
                               carries_rows, carries, descr->base);
        }
        mcLaunchKernelGGL(mcspSpmvCooCarriesReduceKernel, dim3(1), dim3(thread_per_block), 0, stream, n_warp,
                           carries_rows, carries, vec_y);
        mcLaunchKernelGGL(mcspSpmvCooTailKernelLowPrecisionComplex, dim3(1), dim3(thread_per_block), 0, stream,
                           n_in_tail, h_alpha, &coo_rows[tail_start], &coo_cols[tail_start], &coo_vals[tail_start],
                           vec_x, vec_y, descr->base);
    } else {
        mcLaunchKernelGGL(mcspCooScalingKernel, dim3(n_block_scaling), dim3(thread_per_block), 0, stream, row_num,
                           h_beta, vec_y);
        if (n_block > 0) {
            if constexpr ((std::is_same_v<computeType, mcspComplexFloat> && std::is_same_v<sparseType, float>) ||
                          (std::is_same_v<computeType, mcspComplexDouble> && std::is_same_v<sparseType, double>)) {
                mcLaunchKernelGGL(mcspSpmvCooBodyMixedRealComplexKernel, dim3(n_block), dim3(thread_per_block), 0,
                                   stream, interval_len, nnz, h_alpha, coo_rows, coo_cols, coo_vals, vec_x, vec_y,
                                   carries_rows, carries, descr->base);
            } else {
                mcLaunchKernelGGL(mcspSpmvCooBodyKernel, dim3(n_block), dim3(thread_per_block), 0, stream,
                                   interval_len, nnz, h_alpha, coo_rows, coo_cols, coo_vals, vec_x, vec_y, carries_rows,
                                   carries, descr->base);
            }
        }
        mcLaunchKernelGGL(mcspSpmvCooCarriesReduceKernel, dim3(1), dim3(thread_per_block), 0, stream, n_warp,
                           carries_rows, carries, vec_y);
        if constexpr ((std::is_same_v<computeType, mcspComplexFloat> && std::is_same_v<sparseType, float>) ||
                      (std::is_same_v<computeType, mcspComplexDouble> && std::is_same_v<sparseType, double>)) {
            mcLaunchKernelGGL(mcspSpmvCooTailMixedRealComplexKernel, dim3(1), dim3(thread_per_block), 0, stream,
                               n_in_tail, h_alpha, &coo_rows[tail_start], &coo_cols[tail_start], &coo_vals[tail_start],
                               vec_x, vec_y, descr->base);
        } else {
            mcLaunchKernelGGL(mcspSpmvCooTailKernel, dim3(1), dim3(thread_per_block), 0, stream, n_in_tail, h_alpha,
                               &coo_rows[tail_start], &coo_cols[tail_start], &coo_vals[tail_start], vec_x, vec_y,
                               descr->base);
        }
    }

    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(tmp_buffer));
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename computeType>
mcspStatus_t mcspSpmvCsrAnalysisTemplate(mcspHandle_t handle, mcsparseOperation_t trans, idxType row_num,
                                         idxType col_num, idxType nnz, const mcspMatDescr_t descr,
                                         const computeType *csr_vals, const idxType *csr_rows, const idxType *csr_cols,
                                         mcspMatInfo_t info) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz < 0 || row_num < 0 || col_num < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (csr_rows == nullptr || csr_cols == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    // @TODO: Add analysis of matrix for kernel performance enhancement
    mcspStatus_t ret = mcspCreateMatInfo(&info);
    return ret;
}

template <typename idxType, typename computeType>
mcspStatus_t mcspSpmvCsrScalarTemplate(mcspHandle_t handle, mcsparseOperation_t trans, idxType row_num, idxType col_num,
                                       idxType nnz, const computeType *alpha, const mcspMatDescr_t descr,
                                       const computeType *csr_vals, const idxType *csr_rows, const idxType *csr_cols,
                                       mcspMatInfo_t info, const computeType *vec_x, const computeType *beta,
                                       computeType *vec_y) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz < 0 || row_num < 0 || col_num < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || beta == nullptr || csr_rows == nullptr || csr_cols == nullptr || csr_vals == nullptr ||
        vec_x == nullptr || vec_y == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (trans != MCSPARSE_OPERATION_NON_TRANSPOSE || descr->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr->fill_mode != MCSPARSE_FILL_MODE_FULL || descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT ||
        descr->base != MCSPARSE_INDEX_BASE_ZERO) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    computeType h_beta = getScalarToHost(beta, handle->ptr_mode);
    computeType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    mcStream_t stream = mcspGetStreamInternal(handle);
    if (nnz == 0 || (h_alpha == static_cast<computeType>(0) && h_beta == static_cast<computeType>(1))) {
        return MCSP_STATUS_SUCCESS;  // @TODO: alpha=0, beta!=0 can be optimized
    }

    int nElem = 1024;
#ifdef __MACA__
    nElem = 512;
#endif
    int nBlock = (row_num + nElem - 1) / nElem;
    mcLaunchKernelGGL(mcspSpmvCsrScalarKernel, dim3(nBlock), dim3(nElem), 0, stream, row_num, h_alpha, h_beta,
                       csr_rows, csr_cols, csr_vals, vec_x, vec_y);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename computeType, typename sparseType, typename inputDenseType, typename outputType>
mcspStatus_t mcspSpmvCsrVectorTemplate(mcspHandle_t handle, mcsparseOperation_t trans, idxType row_num, idxType col_num,
                                       idxType nnz, const computeType *alpha, const mcspMatDescr_t descr,
                                       const sparseType *csr_vals, const idxType *csr_rows, const idxType *csr_cols,
                                       mcspMatInfo_t info, const inputDenseType *vec_x, const computeType *beta,
                                       outputType *vec_y) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (nnz < 0 || row_num < 0 || col_num < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (alpha == nullptr || beta == nullptr || csr_rows == nullptr || csr_cols == nullptr || csr_vals == nullptr ||
        vec_x == nullptr || vec_y == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    computeType h_beta = getScalarToHost(beta, handle->ptr_mode);
    computeType h_alpha = getScalarToHost(alpha, handle->ptr_mode);

    mcStream_t stream = mcspGetStreamInternal(handle);
    if constexpr (std::is_same_v<computeType, __half2>) {
        printf("alpha: %f %f\n", __low2float(h_alpha), __high2float(h_alpha));
        printf("beta: %f %f\n", __low2float(h_beta), __high2float(h_beta));
    }
    if (nnz == 0 || (IsZero(h_alpha) && IsZero(h_beta))) {
        return MCSP_STATUS_SUCCESS;  // @TODO: alpha=0, beta!=0 can be optimized
    }
    int nElem = 512;
    int gElem = nElem / WARP_SIZE;
    int nBlock = (row_num + gElem - 1) / gElem;
    if constexpr (std::is_same_v<computeType, __half2> || std::is_same_v<computeType, mcsp_bfloat162>) {
        mcLaunchKernelGGL(mcspSpmvCsrVectorKernelLowPrecisionComplex, dim3(nBlock), dim3(nElem),
                           nElem * sizeof(*csr_vals), stream, row_num, h_alpha, h_beta, csr_rows, csr_cols, csr_vals,
                           vec_x, vec_y, descr->base);
    } else {
        size_t compute_type_size = GetMacaDataTypeSize(GetMacaDataTypeFromTypename<computeType>());
        if constexpr ((std::is_same_v<computeType, mcspComplexFloat> && std::is_same_v<sparseType, float>) ||
                      (std::is_same_v<computeType, mcspComplexDouble> && std::is_same_v<sparseType, double>)) {
            mcLaunchKernelGGL(mcspSpmvCsrVectorMixedRealComplexKernel, dim3(nBlock), dim3(nElem),
                               nElem * compute_type_size, stream, row_num, h_alpha, h_beta, csr_rows, csr_cols,
                               csr_vals, vec_x, vec_y, descr->base);
        } else {
            mcLaunchKernelGGL(mcspSpmvCsrVectorKernel, dim3(nBlock), dim3(nElem), nElem * compute_type_size, stream,
                               row_num, h_alpha, h_beta, csr_rows, csr_cols, csr_vals, vec_x, vec_y, descr->base);
        }
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename computeType>
mcspStatus_t mcspSpmvEllAdaptiveTemplate(mcspHandle_t handle, mcsparseOperation_t trans, idxType row_num,
                                         idxType col_num, const computeType *alpha, const mcspMatDescr_t descr,
                                         const computeType *ell_vals, const idxType *ell_cols, idxType ell_k,
                                         const computeType *vec_x, const computeType *beta, computeType *vec_y) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (ell_k < 0 || row_num < 0 || col_num < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (ell_cols == nullptr || ell_vals == nullptr || vec_x == nullptr || vec_y == nullptr || descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT || descr->base != MCSPARSE_INDEX_BASE_ZERO) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    computeType h_beta = getScalarToHost(beta, handle->ptr_mode);
    computeType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    mcStream_t stream = mcspGetStreamInternal(handle);
    if (ell_k == 0 || (h_alpha == static_cast<computeType>(0) && h_beta == static_cast<computeType>(1))) {
        return MCSP_STATUS_SUCCESS;
    }

#ifdef __MACA__
    constexpr int nElem = 512;
#else
    constexpr int nElem = 1024;
#endif
    if (ell_k < WARP_SIZE / 2) {
        int nBlock = (row_num + nElem - 1) / nElem;
        mcLaunchKernelGGL(mcspSpmvEllScalarKernel, dim3(nBlock), dim3(nElem), 0, stream, row_num, col_num, ell_k,
                           h_alpha, h_beta, ell_cols, ell_vals, vec_x, vec_y);
    } else {
        int gElem = nElem / WARP_SIZE;
        int nBlock = (row_num + gElem - 1) / gElem;
        mcLaunchKernelGGL((mcspSpmvEllVectorKernel<nElem>), dim3(nBlock), dim3(nElem), nElem * sizeof(*ell_vals),
                           stream, row_num, col_num, ell_k, h_alpha, h_beta, ell_cols, ell_vals, vec_x, vec_y);
    }
    return MCSP_STATUS_SUCCESS;
}

// buffer structure: ->csr_rows ->csc_cols ->csc_rows ->csc_vals ->max_of(csc_buffer, 0)
template <typename idxType>
mcspStatus_t mcspSpMV_bufferSizeImpl(mcspHandle_t handle, mcsparseOperation_t op_a, const void *alpha,
                                     mcspSpMatDescr_t matA, mcspDnVecDescr_t vec_x, const void *beta,
                                     mcspDnVecDescr_t vec_y, macaDataType compute_type, mcsparseSpMVAlg_t alg,
                                     size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (alpha == nullptr || beta == nullptr || matA == nullptr || vec_x == nullptr || vec_y == nullptr ||
        buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (matA->row_num >> 32 != 0 || matA->col_num >> 32 != 0 || matA->nnz >> 32 != 0 || vec_x->size >> 32 != 0 ||
        vec_y->size >> 32 != 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (!(matA->format == MCSPARSE_FORMAT_CSR || matA->format == MCSPARSE_FORMAT_COO ||
          matA->format == MCSPARSE_FORMAT_COO_AOS || matA->format == MCSPARSE_FORMAT_CSC)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (!((op_a == MCSPARSE_OPERATION_NON_TRANSPOSE && vec_x->size == matA->col_num && vec_y->size == matA->row_num) ||
          (op_a != MCSPARSE_OPERATION_NON_TRANSPOSE && vec_x->size == matA->row_num && vec_y->size == matA->col_num))) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    *buffer_size = MIN_BUFFER_SIZE;
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    macaDataType a_type = matA->valueType;
    if ((op_a != MCSPARSE_OPERATION_NON_TRANSPOSE &&
         (matA->format == MCSPARSE_FORMAT_CSR || matA->format == MCSPARSE_FORMAT_COO ||
          matA->format == MCSPARSE_FORMAT_COO_AOS)) ||
        (op_a == MCSPARSE_OPERATION_NON_TRANSPOSE && matA->format == MCSPARSE_FORMAT_CSC)) {
        stat = CalculateAssistBufferSizeForTranspose<idxType>(handle, matA, a_type, buffer_size);
    }
    if (matA->format == MCSPARSE_FORMAT_COO_AOS) {
        // buffer for Aos Coo
        *buffer_size += 2 * ALIGN(matA->nnz * sizeof(idxType), ALIGNED_SIZE);
    }
    return stat;
}

template <typename idxType>
mcspStatus_t mcspUnifiedSpMVImpl(mcspHandle_t handle, mcsparseOperation_t working_a_op, idxType working_m,
                                 idxType working_n, mcspSpMatDescr_t matA, void *working_vals, idxType *working_rows,
                                 idxType *working_cols, mcsparseFormat_t working_format, const void *alpha,
                                 mcspMatInfo_t mat_info, mcspDnVecDescr_t vec_x, const void *beta,
                                 mcspDnVecDescr_t vec_y, macaDataType compute_type) {
    mcStream_t stream = mcspGetStreamInternal(handle);
    switch (working_format) {
        case MCSPARSE_FORMAT_CSR: {
            switch (compute_type) {
                case MACA_R_32F: {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (float *)alpha,
                                                     matA->mat_descr, (float *)working_vals, (idxType *)working_rows,
                                                     (idxType *)working_cols, mat_info, (float *)vec_x->values,
                                                     (float *)beta, (float *)vec_y->values);
                }
                case MACA_R_64F: {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (double *)alpha,
                                                     matA->mat_descr, (double *)working_vals, (idxType *)working_rows,
                                                     (idxType *)working_cols, mat_info, (double *)vec_x->values,
                                                     (double *)beta, (double *)vec_y->values);
                }
                case MACA_C_32F: {
                    return mcspSpmvCsrVectorTemplate(
                        handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m, (idxType)working_n,
                        (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr,
                        (mcspComplexFloat *)working_vals, (idxType *)working_rows, (idxType *)working_cols, mat_info,
                        (mcspComplexFloat *)vec_x->values, (mcspComplexFloat *)beta, (mcspComplexFloat *)vec_y->values);
                }
                case MACA_C_64F: {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (mcspComplexDouble *)alpha,
                                                     matA->mat_descr, (mcspComplexDouble *)working_vals,
                                                     (idxType *)working_rows, (idxType *)working_cols, mat_info,
                                                     (mcspComplexDouble *)vec_x->values, (mcspComplexDouble *)beta,
                                                     (mcspComplexDouble *)vec_y->values);
                }
#ifdef __MACA__
                case MACA_R_16F: {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (__half *)alpha,
                                                     matA->mat_descr, (__half *)working_vals, (idxType *)working_rows,
                                                     (idxType *)working_cols, mat_info, (__half *)vec_x->values,
                                                     (__half *)beta, (__half *)vec_y->values);
                }
                case MACA_C_16F: {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (__half2 *)alpha,
                                                     matA->mat_descr, (__half2 *)working_vals, (idxType *)working_rows,
                                                     (idxType *)working_cols, mat_info, (__half2 *)vec_x->values,
                                                     (__half2 *)beta, (__half2 *)vec_y->values);
                }
                case MACA_R_16BF: {
                    return mcspSpmvCsrVectorTemplate(
                        handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m, (idxType)working_n,
                        (idxType)matA->nnz, (mcsp_bfloat16 *)alpha, matA->mat_descr, (mcsp_bfloat16 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, mat_info, (mcsp_bfloat16 *)vec_x->values,
                        (mcsp_bfloat16 *)beta, (mcsp_bfloat16 *)vec_y->values);
                }
                case MACA_C_16BF: {
                    return mcspSpmvCsrVectorTemplate(
                        handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m, (idxType)working_n,
                        (idxType)matA->nnz, (mcsp_bfloat162 *)alpha, matA->mat_descr, (mcsp_bfloat162 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, mat_info, (mcsp_bfloat162 *)vec_x->values,
                        (mcsp_bfloat162 *)beta, (mcsp_bfloat162 *)vec_y->values);
                }
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_COO: {
            switch (compute_type) {
                case MACA_R_32F: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (float *)alpha, matA->mat_descr,
                                                   (float *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (float *)vec_x->values, (float *)beta, (float *)vec_y->values);
                }
                case MACA_R_64F: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (double *)alpha, matA->mat_descr,
                                                   (double *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (double *)vec_x->values, (double *)beta, (double *)vec_y->values);
                }
                case MACA_C_32F: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr,
                                                   (mcspComplexFloat *)matA->vals, (idxType *)matA->rows,
                                                   (idxType *)matA->cols, (mcspComplexFloat *)vec_x->values,
                                                   (mcspComplexFloat *)beta, (mcspComplexFloat *)vec_y->values);
                }
                case MACA_C_64F: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (mcspComplexDouble *)alpha, matA->mat_descr,
                                                   (mcspComplexDouble *)matA->vals, (idxType *)matA->rows,
                                                   (idxType *)matA->cols, (mcspComplexDouble *)vec_x->values,
                                                   (mcspComplexDouble *)beta, (mcspComplexDouble *)vec_y->values);
                }
#ifdef __MACA__
                case MACA_R_16F: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (__half *)alpha, matA->mat_descr,
                                                   (__half *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (__half *)vec_x->values, (__half *)beta, (__half *)vec_y->values);
                }
                case MACA_C_16F: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (__half2 *)alpha, matA->mat_descr,
                                                   (__half2 *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (__half2 *)vec_x->values, (__half2 *)beta, (__half2 *)vec_y->values);
                }
                case MACA_R_16BF: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (mcsp_bfloat16 *)alpha, matA->mat_descr,
                                                   (mcsp_bfloat16 *)matA->vals, (idxType *)matA->rows,
                                                   (idxType *)matA->cols, (mcsp_bfloat16 *)vec_x->values,
                                                   (mcsp_bfloat16 *)beta, (mcsp_bfloat16 *)vec_y->values);
                }
                case MACA_C_16BF: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (mcsp_bfloat162 *)alpha, matA->mat_descr,
                                                   (mcsp_bfloat162 *)matA->vals, (idxType *)matA->rows,
                                                   (idxType *)matA->cols, (mcsp_bfloat162 *)vec_x->values,
                                                   (mcsp_bfloat162 *)beta, (mcsp_bfloat162 *)vec_y->values);
                }
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

template <typename idxType>
mcspStatus_t mcspMixedPrecisionSpMVImpl(mcspHandle_t handle, mcsparseOperation_t working_a_op, idxType working_m,
                                        idxType working_n, mcspSpMatDescr_t matA, void *working_vals,
                                        idxType *working_rows, idxType *working_cols, mcsparseFormat_t working_format,
                                        const void *alpha, mcspMatInfo_t mat_info, mcspDnVecDescr_t vec_x,
                                        const void *beta, mcspDnVecDescr_t vec_y, macaDataType compute_type) {
    macaDataType inputType = vec_x->valueType;
    macaDataType outputType = vec_y->valueType;
    uint64_t mixedType = GetMixedDataType(inputType, outputType);
    mcStream_t stream = mcspGetStreamInternal(handle);
    switch (working_format) {
        case MCSPARSE_FORMAT_CSR: {
            switch (mixedType) {
#if defined(__MACA__)
                case GetMixedDataType(MACA_R_16F, MACA_R_32F): {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (float *)alpha,
                                                     matA->mat_descr, (__half *)working_vals, (idxType *)working_rows,
                                                     (idxType *)working_cols, mat_info, (__half *)vec_x->values,
                                                     (float *)beta, (float *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_16BF, MACA_R_32F): {
                    return mcspSpmvCsrVectorTemplate(
                        handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m, (idxType)working_n,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (mcsp_bfloat16 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, mat_info, (mcsp_bfloat16 *)vec_x->values,
                        (float *)beta, (float *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_8I, MACA_R_32F): {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (float *)alpha,
                                                     matA->mat_descr, (int8_t *)working_vals, (idxType *)working_rows,
                                                     (idxType *)working_cols, mat_info, (int8_t *)vec_x->values,
                                                     (float *)beta, (float *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_16F, MACA_R_16F): {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (float *)alpha,
                                                     matA->mat_descr, (__half *)working_vals, (idxType *)working_rows,
                                                     (idxType *)working_cols, mat_info, (__half *)vec_x->values,
                                                     (float *)beta, (__half *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_16BF, MACA_R_16BF): {
                    return mcspSpmvCsrVectorTemplate(
                        handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m, (idxType)working_n,
                        (idxType)matA->nnz, (float *)alpha, matA->mat_descr, (mcsp_bfloat16 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, mat_info, (mcsp_bfloat16 *)vec_x->values,
                        (float *)beta, (mcsp_bfloat16 *)vec_y->values);
                }
                case GetMixedDataType(MACA_C_16F, MACA_C_16F): {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (mcspComplexFloat *)alpha,
                                                     matA->mat_descr, (__half2 *)working_vals, (idxType *)working_rows,
                                                     (idxType *)working_cols, mat_info, (__half2 *)vec_x->values,
                                                     (mcspComplexFloat *)beta, (__half2 *)vec_y->values);
                }
                case GetMixedDataType(MACA_C_16BF, MACA_C_16BF): {
                    return mcspSpmvCsrVectorTemplate(
                        handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m, (idxType)working_n,
                        (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr, (mcsp_bfloat162 *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, mat_info, (mcsp_bfloat162 *)vec_x->values,
                        (mcspComplexFloat *)beta, (mcsp_bfloat162 *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_8I, MACA_R_32I): {
                    return mcspSpmvCsrVectorTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m,
                                                     (idxType)working_n, (idxType)matA->nnz, (int32_t *)alpha,
                                                     matA->mat_descr, (int8_t *)working_vals, (idxType *)working_rows,
                                                     (idxType *)working_cols, mat_info, (int8_t *)vec_x->values,
                                                     (int32_t *)beta, (int32_t *)vec_y->values);
                }
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_COO: {
            switch (mixedType) {
#if defined(__MACA__)
                case GetMixedDataType(MACA_R_16F, MACA_R_32F): {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (float *)alpha, matA->mat_descr,
                                                   (__half *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (__half *)vec_x->values, (float *)beta, (float *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_16BF, MACA_R_32F): {
                    return mcspSpmvCooFlatTemplate(
                        handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num, (idxType)matA->nnz,
                        (float *)alpha, matA->mat_descr, (mcsp_bfloat16 *)matA->vals, (idxType *)matA->rows,
                        (idxType *)matA->cols, (mcsp_bfloat16 *)vec_x->values, (float *)beta, (float *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_8I, MACA_R_32F): {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (float *)alpha, matA->mat_descr,
                                                   (int8_t *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (int8_t *)vec_x->values, (float *)beta, (float *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_16F, MACA_R_16F): {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (float *)alpha, matA->mat_descr,
                                                   (__half *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (__half *)vec_x->values, (float *)beta, (__half *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_16BF, MACA_R_16BF): {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (float *)alpha, matA->mat_descr,
                                                   (mcsp_bfloat16 *)matA->vals, (idxType *)matA->rows,
                                                   (idxType *)matA->cols, (mcsp_bfloat16 *)vec_x->values, (float *)beta,
                                                   (mcsp_bfloat16 *)vec_y->values);
                }
                case GetMixedDataType(MACA_C_16F, MACA_C_16F): {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr,
                                                   (__half2 *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (__half2 *)vec_x->values, (mcspComplexFloat *)beta,
                                                   (__half2 *)vec_y->values);
                }
                case GetMixedDataType(MACA_C_16BF, MACA_C_16BF): {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr,
                                                   (mcsp_bfloat162 *)matA->vals, (idxType *)matA->rows,
                                                   (idxType *)matA->cols, (mcsp_bfloat162 *)vec_x->values,
                                                   (mcspComplexFloat *)beta, (mcsp_bfloat162 *)vec_y->values);
                }
                case GetMixedDataType(MACA_R_8I, MACA_R_32I): {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (int32_t *)alpha, matA->mat_descr,
                                                   (int8_t *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (int8_t *)vec_x->values, (int32_t *)beta, (int32_t *)vec_y->values);
                }
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
    return MCSP_STATUS_NOT_IMPLEMENTED;
}

template <typename idxType>
mcspStatus_t mcspMixedRealComplexSpMVImpl(mcspHandle_t handle, mcsparseOperation_t working_a_op, idxType working_m,
                                          idxType working_n, mcspSpMatDescr_t matA, void *working_vals,
                                          idxType *working_rows, idxType *working_cols, mcsparseFormat_t working_format,
                                          const void *alpha, mcspMatInfo_t mat_info, mcspDnVecDescr_t vec_x,
                                          const void *beta, mcspDnVecDescr_t vec_y, macaDataType compute_type) {
    mcStream_t stream = mcspGetStreamInternal(handle);
    switch (working_format) {
        case MCSPARSE_FORMAT_CSR: {
            switch (compute_type) {
                case MACA_C_32F: {
                    return mcspSpmvCsrVectorTemplate(
                        handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m, (idxType)working_n,
                        (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr, (float *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, mat_info, (mcspComplexFloat *)vec_x->values,
                        (mcspComplexFloat *)beta, (mcspComplexFloat *)vec_y->values);
                }
                case MACA_C_64F: {
                    return mcspSpmvCsrVectorTemplate(
                        handle, MCSPARSE_OPERATION_NON_TRANSPOSE, (idxType)working_m, (idxType)working_n,
                        (idxType)matA->nnz, (mcspComplexDouble *)alpha, matA->mat_descr, (double *)working_vals,
                        (idxType *)working_rows, (idxType *)working_cols, mat_info, (mcspComplexDouble *)vec_x->values,
                        (mcspComplexDouble *)beta, (mcspComplexDouble *)vec_y->values);
                }
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_COO: {
            switch (compute_type) {
                case MACA_C_32F: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (mcspComplexFloat *)alpha, matA->mat_descr,
                                                   (float *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (mcspComplexFloat *)vec_x->values, (mcspComplexFloat *)beta,
                                                   (mcspComplexFloat *)vec_y->values);
                }
                case MACA_C_64F: {
                    return mcspSpmvCooFlatTemplate(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                                   (idxType)matA->nnz, (mcspComplexDouble *)alpha, matA->mat_descr,
                                                   (double *)matA->vals, (idxType *)matA->rows, (idxType *)matA->cols,
                                                   (mcspComplexDouble *)vec_x->values, (mcspComplexDouble *)beta,
                                                   (mcspComplexDouble *)vec_y->values);
                }
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
    return MCSP_STATUS_NOT_IMPLEMENTED;
}

template <typename idxType>
mcspStatus_t mcspSpMVImpl(mcspHandle_t handle, mcsparseOperation_t op_a, const void *alpha, mcspSpMatDescr_t matA,
                          mcspDnVecDescr_t vec_x, const void *beta, mcspDnVecDescr_t vec_y, macaDataType compute_type,
                          mcsparseSpMVAlg_t alg, void *external_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (alpha == nullptr || beta == nullptr || matA == nullptr || vec_x == nullptr || vec_y == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (matA->row_num >> 32 != 0 || matA->col_num >> 32 != 0 || matA->nnz >> 32 != 0 || vec_x->size >> 32 != 0 ||
        vec_y->size >> 32 != 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (external_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (!(matA->format == MCSPARSE_FORMAT_CSR || matA->format == MCSPARSE_FORMAT_COO ||
          matA->format == MCSPARSE_FORMAT_COO_AOS || matA->format == MCSPARSE_FORMAT_CSC)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (!((op_a == MCSPARSE_OPERATION_NON_TRANSPOSE && vec_x->size == matA->col_num && vec_y->size == matA->row_num) ||
          (op_a != MCSPARSE_OPERATION_NON_TRANSPOSE && vec_x->size == matA->row_num && vec_y->size == matA->col_num))) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if ((compute_type == MACA_R_32F || compute_type == MACA_R_64F || compute_type == MACA_R_16F ||
         compute_type == MACA_R_16BF) &&
        op_a == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    void *working_rows = nullptr;
    void *working_cols = nullptr;
    void *working_vals = nullptr;
    void *helper_rows = nullptr;
    idxType working_m = 0;
    idxType working_n = 0;
    mcsparseFormat_t working_format = matA->format;
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    mcsparseOperation_t working_a_op = op_a;
    macaDataType sparseType = matA->valueType;
    macaDataType outputType = vec_y->valueType;

    if (working_format == MCSPARSE_FORMAT_CSC) {
        std::swap(matA->rows, matA->cols);
        std::swap(matA->row_num, matA->col_num);
        working_a_op = (working_a_op == MCSPARSE_OPERATION_NON_TRANSPOSE) ? MCSPARSE_OPERATION_TRANSPOSE
                                                                          : MCSPARSE_OPERATION_NON_TRANSPOSE;
        working_format = MCSPARSE_FORMAT_CSR;

        if (op_a == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
            constexpr uint32_t block_size = 512;
            uint32_t n_blocks = ((idxType)matA->nnz - 1) / block_size + 1;
            mcStream_t stream = mcspGetStreamInternal(handle);
            switch (sparseType) {
                case MACA_C_32F: {
                    mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                       (idxType)matA->nnz, (mcspComplexFloat *)matA->vals);
                    break;
                }
                case MACA_C_64F: {
                    mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                       (idxType)matA->nnz, (mcspComplexDouble *)matA->vals);
                    break;
                }
#if defined(__MACA__)
                case MACA_C_16F: {
                    mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                       (idxType)matA->nnz, (__half2 *)matA->vals);
                    break;
                }
                case MACA_C_16BF: {
                    mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                       (idxType)matA->nnz, (mcsp_bfloat162 *)matA->vals);
                    break;
                }
#endif
            }
            MACA_ASSERT(mcStreamSynchronize(stream));
        }
    }

    // Aos Coo to Coo
    if (working_format == MCSPARSE_FORMAT_COO_AOS) {
        idxType *working_coo_rows = nullptr;
        idxType *working_coo_cols = nullptr;
        working_coo_rows = (idxType *)external_buffer;
        external_buffer = (char *)external_buffer + ALIGN(matA->nnz * sizeof(idxType), ALIGNED_SIZE);
        working_coo_cols = (idxType *)external_buffer;
        external_buffer = (char *)external_buffer + ALIGN(matA->nnz * sizeof(idxType), ALIGNED_SIZE);
        mcStream_t stream = mcspGetStreamInternal(handle);
        idxType nBlock = (matA->nnz + COO_BLOCK_SIZE - 1) / COO_BLOCK_SIZE;
        mcLaunchKernelGGL(mcspSpmvAosCoo2CooKernel, dim3(nBlock), dim3(COO_BLOCK_SIZE), 0, stream, matA->nnz,
                           (idxType *)matA->coo_aos_ind, working_coo_rows, working_coo_cols);
        MACA_ASSERT(mcStreamSynchronize(stream));
        matA->rows = working_coo_rows;
        matA->cols = working_coo_cols;
        working_format = MCSPARSE_FORMAT_COO;
    }

    if (working_a_op == MCSPARSE_OPERATION_NON_TRANSPOSE) {
        working_rows = matA->rows;
        working_cols = matA->cols;
        working_vals = matA->vals;
        working_m = matA->row_num;
        working_n = matA->col_num;
    } else {
        if (external_buffer != nullptr && matA->is_buffersize_called == 0) {
            size_t total_buffer_size = 0;
            mcspStatus_t stat = mcspSpMV_bufferSizeImpl<idxType>(handle, working_a_op, alpha, matA, vec_x, beta, vec_y,
                                                                 compute_type, alg, &total_buffer_size);
            if (stat != MCSP_STATUS_SUCCESS) {
                return MCSP_STATUS_INTERNAL_ERROR;
            }
            LOG_FS_WARN("Running SpMV without calling SpMV_buffersize for matrix A ahead.\n");
            LOG_FS_WARN("Enough buffer with buffersize %d bytes should be guaranteed or the program may crash.\n",
                        total_buffer_size);
            mcStream_t stream = mcspGetStreamInternal(handle);
            MACA_ASSERT(mcMemsetAsync(external_buffer, 0, total_buffer_size, stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
        }

        helper_rows = matA->rows;
        idxType *buffer_head = reinterpret_cast<idxType *>(external_buffer);
        matA->to_csr_rows = (void *)buffer_head;
        buffer_head += matA->row_num + 1;
        if (working_format == MCSPARSE_FORMAT_COO) {
            stat = mcspCallCoo2Csr(handle, (idxType *)matA->rows, (idxType)matA->nnz, (idxType)matA->row_num,
                                   (idxType *)matA->to_csr_rows, matA->idxBase);
            if (stat != MCSP_STATUS_SUCCESS) {
                return stat;
            }
            working_format = MCSPARSE_FORMAT_CSR;
            helper_rows = matA->to_csr_rows;
        }

        matA->to_csc_cols = (void *)buffer_head;
        buffer_head += matA->col_num + 1;
        matA->to_csc_rows = (void *)buffer_head;
        matA->to_csc_vals = (void *)((reinterpret_cast<char *>(external_buffer) + matA->assist_index_buffer_size));
        void *csr2csc_buffer = (void *)(reinterpret_cast<char *>(external_buffer) + matA->fixed_length_buffer_size);
        stat = mcspTransposeSparseByCsr2Csc(handle, working_a_op, (idxType)matA->row_num, (idxType)matA->col_num,
                                            (idxType)matA->nnz, matA->vals, (idxType *)helper_rows,
                                            (idxType *)matA->cols, matA->to_csc_vals, (idxType *)matA->to_csc_rows,
                                            (idxType *)matA->to_csc_cols, matA->idxBase, sparseType, csr2csc_buffer);
        if (stat != MCSP_STATUS_SUCCESS) {
            return stat;
        }

        working_rows = matA->to_csc_cols;
        working_cols = matA->to_csc_rows;
        working_vals = matA->to_csc_vals;
        working_m = matA->col_num;
        working_n = matA->row_num;
    }

    mcspMatInfo_t mat_info;
    if (compute_type == sparseType && compute_type == outputType) {
        stat = mcspUnifiedSpMVImpl<idxType>(handle, working_a_op, working_m, working_n, matA, working_vals,
                                            (idxType *)working_rows, (idxType *)working_cols, working_format, alpha,
                                            mat_info, vec_x, beta, vec_y, compute_type);
    } else if (sparseType == MACA_R_32F || sparseType == MACA_R_64F) {
        stat = mcspMixedRealComplexSpMVImpl<idxType>(handle, working_a_op, working_m, working_n, matA, working_vals,
                                                     (idxType *)working_rows, (idxType *)working_cols, working_format,
                                                     alpha, mat_info, vec_x, beta, vec_y, compute_type);
    } else {
        stat = mcspMixedPrecisionSpMVImpl<idxType>(handle, working_a_op, working_m, working_n, matA, working_vals,
                                                   (idxType *)working_rows, (idxType *)working_cols, working_format,
                                                   alpha, mat_info, vec_x, beta, vec_y, compute_type);
    }

    if (matA->format == MCSPARSE_FORMAT_COO_AOS) {
        matA->rows = nullptr;
        matA->cols = nullptr;
    }

    if (matA->format == MCSPARSE_FORMAT_CSC) {
        std::swap(matA->rows, matA->cols);
        std::swap(matA->row_num, matA->col_num);
        if (op_a == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
            constexpr uint32_t block_size = 512;
            uint32_t n_blocks = ((idxType)matA->nnz - 1) / block_size + 1;
            mcStream_t stream = mcspGetStreamInternal(handle);
            switch (sparseType) {
                case MACA_C_32F: {
                    mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                       (idxType)matA->nnz, (mcspComplexFloat *)matA->vals);
                    break;
                }
                case MACA_C_64F: {
                    mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                       (idxType)matA->nnz, (mcspComplexDouble *)matA->vals);
                    break;
                }
#if defined(__MACA__)
                case MACA_C_16F: {
                    mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                       (idxType)matA->nnz, (__half2 *)matA->vals);
                    break;
                }
                case MACA_C_16BF: {
                    mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                       (idxType)matA->nnz, (mcsp_bfloat162 *)matA->vals);
                    break;
                }
#endif
            }
            MACA_ASSERT(mcStreamSynchronize(stream));
        }
    }

    return stat;
}

mcspStatus_t mcspCsrmvExImpl(mcspHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, mcspInt m,
                             mcspInt n, mcspInt nnz, const void *alpha, macaDataType alphatype,
                             const mcspMatDescr_t descrA, const void *csrValA, macaDataType csrValAtype,
                             const mcspInt *csrRowPtrA, const mcspInt *csrColIndA, const void *x, macaDataType xtype,
                             const void *beta, macaDataType betatype, void *y, macaDataType outType,
                             macaDataType executiontype, void *buffer) {
    if (!(alphatype == xtype && csrValAtype == xtype && betatype == xtype && outType == xtype &&
          executiontype == xtype)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    mcspMatInfo_t mat_info;
    switch (csrValAtype) {
        case MACA_R_32F: {
            return mcspSpmvCsrVectorTemplate(handle, transA, m, n, nnz, (float *)alpha, descrA, (float *)csrValA,
                                             (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, mat_info, (float *)x,
                                             (float *)beta, (float *)y);
        }
        case MACA_R_64F: {
            return mcspSpmvCsrVectorTemplate(handle, transA, m, n, nnz, (double *)alpha, descrA, (double *)csrValA,
                                             (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, mat_info, (double *)x,
                                             (double *)beta, (double *)y);
        }
        case MACA_C_32F: {
            return mcspSpmvCsrVectorTemplate(handle, transA, m, n, nnz, (mcspComplexFloat *)alpha, descrA,
                                             (mcspComplexFloat *)csrValA, (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA,
                                             mat_info, (mcspComplexFloat *)x, (mcspComplexFloat *)beta,
                                             (mcspComplexFloat *)y);
        }
        case MACA_C_64F: {
            return mcspSpmvCsrVectorTemplate(handle, transA, m, n, nnz, (mcspComplexDouble *)alpha, descrA,
                                             (mcspComplexDouble *)csrValA, (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA,
                                             mat_info, (mcspComplexDouble *)x, (mcspComplexDouble *)beta,
                                             (mcspComplexDouble *)y);
        }
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

template <typename computeType>
mcspStatus_t mcspCsrmvImpl(mcspHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                           const computeType *alpha, const mcspMatDescr_t descrA, const computeType *csrValA,
                           const int *csrRowPtrA, const int *csrColIndA, const computeType *x, const computeType *beta,
                           computeType *y) {
    mcspMatInfo_t mat_info;
    return mcspSpmvCsrVectorTemplate(handle, transA, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, alpha, descrA, csrValA,
                                     (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, mat_info, x, beta, y);
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScooSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const float *alpha, const mcspMatDescr_t descr, const float *coo_vals,
                          const mcspInt *coo_rows, const mcspInt *coo_cols, const float *vec_x, const float *beta,
                          float *vec_y) {
    return mcspSpmvCooFlatTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, coo_vals, coo_rows, coo_cols,
                                   vec_x, beta, vec_y);
}

mcspStatus_t mcspDcooSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const double *alpha, const mcspMatDescr_t descr, const double *coo_vals,
                          const mcspInt *coo_rows, const mcspInt *coo_cols, const double *vec_x, const double *beta,
                          double *vec_y) {
    return mcspSpmvCooFlatTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, coo_vals, coo_rows, coo_cols,
                                   vec_x, beta, vec_y);
}

mcspStatus_t mcspCcooSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const mcspComplexFloat *alpha, const mcspMatDescr_t descr, const mcspComplexFloat *coo_vals,
                          const mcspInt *coo_rows, const mcspInt *coo_cols, const mcspComplexFloat *vec_x,
                          const mcspComplexFloat *beta, mcspComplexFloat *vec_y) {
    return mcspSpmvCooFlatTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, coo_vals, coo_rows, coo_cols,
                                   vec_x, beta, vec_y);
}

mcspStatus_t mcspZcooSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const mcspComplexDouble *alpha, const mcspMatDescr_t descr, const mcspComplexDouble *coo_vals,
                          const mcspInt *coo_rows, const mcspInt *coo_cols, const mcspComplexDouble *vec_x,
                          const mcspComplexDouble *beta, mcspComplexDouble *vec_y) {
    return mcspSpmvCooFlatTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, coo_vals, coo_rows, coo_cols,
                                   vec_x, beta, vec_y);
}

mcspStatus_t mcspScsrSpmvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                                  mcspInt nnz, const mcspMatDescr_t descr, const float *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info) {
    return mcspSpmvCsrAnalysisTemplate<mcspInt, float>(handle, trans, row_num, col_num, nnz, descr, csr_vals, csr_rows,
                                                       csr_cols, info);
}
mcspStatus_t mcspDcsrSpmvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                                  mcspInt nnz, const mcspMatDescr_t descr, const double *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info) {
    return mcspSpmvCsrAnalysisTemplate<mcspInt, double>(handle, trans, row_num, col_num, nnz, descr, csr_vals, csr_rows,
                                                        csr_cols, info);
}
mcspStatus_t mcspCcsrSpmvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                                  mcspInt nnz, const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info) {
    return mcspSpmvCsrAnalysisTemplate<mcspInt, mcspComplexFloat>(handle, trans, row_num, col_num, nnz, descr, csr_vals,
                                                                  csr_rows, csr_cols, info);
}
mcspStatus_t mcspZcsrSpmvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                                  mcspInt nnz, const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info) {
    return mcspSpmvCsrAnalysisTemplate<mcspInt, mcspComplexDouble>(handle, trans, row_num, col_num, nnz, descr,
                                                                   csr_vals, csr_rows, csr_cols, info);
}

mcspStatus_t mcspCsrSpmvAnalysisClear(mcspHandle_t handle, mcspMatInfo_t info) {
    mcspStatus_t ret = mcspDestroyMatInfo(info);
    return ret;
}

/**
 * compute CSR-based SpMV in single and double precision in GPU, csr_vector algorithm
 * y = alpha * A * x + beta * y
 */
mcspStatus_t mcspScsrSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const float *alpha, const mcspMatDescr_t descr, const float *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info, const float *vec_x,
                          const float *beta, float *vec_y) {
    if ((nnz + row_num - 1) / row_num > kRowAverageElementNumThreshold) {
        return mcspSpmvCsrVectorTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, csr_vals, csr_rows,
                                         csr_cols, info, vec_x, beta, vec_y);
    } else {
        return mcspSpmvCsrScalarTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, csr_vals, csr_rows,
                                         csr_cols, info, vec_x, beta, vec_y);
    }
}

mcspStatus_t mcspDcsrSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const double *alpha, const mcspMatDescr_t descr, const double *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info, const double *vec_x,
                          const double *beta, double *vec_y) {
    if ((nnz + row_num - 1) / row_num > kRowAverageElementNumThreshold) {
        return mcspSpmvCsrVectorTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, csr_vals, csr_rows,
                                         csr_cols, info, vec_x, beta, vec_y);
    } else {
        return mcspSpmvCsrScalarTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, csr_vals, csr_rows,
                                         csr_cols, info, vec_x, beta, vec_y);
    }
}

mcspStatus_t mcspCcsrSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const mcspComplexFloat *alpha, const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info,
                          const mcspComplexFloat *vec_x, const mcspComplexFloat *beta, mcspComplexFloat *vec_y) {
    if ((nnz + row_num - 1) / row_num > kRowAverageElementNumThreshold) {
        return mcspSpmvCsrVectorTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, csr_vals, csr_rows,
                                         csr_cols, info, vec_x, beta, vec_y);
    } else {
        return mcspSpmvCsrScalarTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, csr_vals, csr_rows,
                                         csr_cols, info, vec_x, beta, vec_y);
    }
}

mcspStatus_t mcspZcsrSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const mcspComplexDouble *alpha, const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info,
                          const mcspComplexDouble *vec_x, const mcspComplexDouble *beta, mcspComplexDouble *vec_y) {
    if ((nnz + row_num - 1) / row_num > kRowAverageElementNumThreshold) {
        return mcspSpmvCsrVectorTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, csr_vals, csr_rows,
                                         csr_cols, info, vec_x, beta, vec_y);
    } else {
        return mcspSpmvCsrScalarTemplate(handle, trans, row_num, col_num, nnz, alpha, descr, csr_vals, csr_rows,
                                         csr_cols, info, vec_x, beta, vec_y);
    }
}

mcspStatus_t mcspSellSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                          const float *alpha, const mcspMatDescr_t descr, const float *ell_vals,
                          const mcspInt *ell_cols, mcspInt ell_k, const float *vec_x, const float *beta, float *vec_y) {
    return mcspSpmvEllAdaptiveTemplate(handle, trans, row_num, col_num, alpha, descr, ell_vals, ell_cols, ell_k, vec_x,
                                       beta, vec_y);
}

mcspStatus_t mcspDellSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                          const double *alpha, const mcspMatDescr_t descr, const double *ell_vals,
                          const mcspInt *ell_cols, mcspInt ell_k, const double *vec_x, const double *beta,
                          double *vec_y) {
    return mcspSpmvEllAdaptiveTemplate(handle, trans, row_num, col_num, alpha, descr, ell_vals, ell_cols, ell_k, vec_x,
                                       beta, vec_y);
}

mcspStatus_t mcspCellSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                          const mcspComplexFloat *alpha, const mcspMatDescr_t descr, const mcspComplexFloat *ell_vals,
                          const mcspInt *ell_cols, mcspInt ell_k, const mcspComplexFloat *vec_x,
                          const mcspComplexFloat *beta, mcspComplexFloat *vec_y) {
    return mcspSpmvEllAdaptiveTemplate(handle, trans, row_num, col_num, alpha, descr, ell_vals, ell_cols, ell_k, vec_x,
                                       beta, vec_y);
}

mcspStatus_t mcspZellSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                          const mcspComplexDouble *alpha, const mcspMatDescr_t descr, const mcspComplexDouble *ell_vals,
                          const mcspInt *ell_cols, mcspInt ell_k, const mcspComplexDouble *vec_x,
                          const mcspComplexDouble *beta, mcspComplexDouble *vec_y) {
    return mcspSpmvEllAdaptiveTemplate(handle, trans, row_num, col_num, alpha, descr, ell_vals, ell_cols, ell_k, vec_x,
                                       beta, vec_y);
}

mcspStatus_t mcspSpMV_bufferSize_native(mcspHandle_t handle, mcsparseOperation_t op_a, const void *alpha,
                                        mcspSpMatDescr_t matA, mcspDnVecDescr_t vec_x, const void *beta,
                                        mcspDnVecDescr_t vec_y, macaDataType compute_type, mcsparseSpMVAlg_t alg,
                                        size_t *buffer_size) {
    if (matA->rowIdxType == MCSPARSE_INDEX_32I && matA->colIdxType == MCSPARSE_INDEX_32I) {
        return mcspSpMV_bufferSizeImpl<mcspInt>(handle, op_a, alpha, matA, vec_x, beta, vec_y, compute_type, alg,
                                                buffer_size);
    } else if (matA->rowIdxType == MCSPARSE_INDEX_64I && matA->colIdxType == MCSPARSE_INDEX_64I) {
        return mcspSpMV_bufferSizeImpl<int64_t>(handle, op_a, alpha, matA, vec_x, beta, vec_y, compute_type, alg,
                                                buffer_size);
    } else {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

mcspStatus_t mcspSpMV_native(mcspHandle_t handle, mcsparseOperation_t op_a, const void *alpha, mcspSpMatDescr_t matA,
                             mcspDnVecDescr_t vec_x, const void *beta, mcspDnVecDescr_t vec_y,
                             macaDataType compute_type, mcsparseSpMVAlg_t alg, void *external_buffer) {
    if (matA->rowIdxType == MCSPARSE_INDEX_32I && matA->colIdxType == MCSPARSE_INDEX_32I) {
        return mcspSpMVImpl<mcspInt>(handle, op_a, alpha, matA, vec_x, beta, vec_y, compute_type, alg, external_buffer);
    } else if (matA->rowIdxType == MCSPARSE_INDEX_64I && matA->colIdxType == MCSPARSE_INDEX_64I) {
        return mcspSpMVImpl<int64_t>(handle, op_a, alpha, matA, vec_x, beta, vec_y, compute_type, alg, external_buffer);
    } else {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

mcspStatus_t mcspSpMV_bufferSize(mcspHandle_t handle, mcsparseOperation_t op_a, const void *alpha,
                                 mcspSpMatDescr_t matA, mcspDnVecDescr_t vec_x, const void *beta,
                                 mcspDnVecDescr_t vec_y, macaDataType compute_type, mcsparseSpMVAlg_t alg,
                                 size_t *buffer_size) {
    return mcspSpMV_bufferSize_native(handle, op_a, alpha, matA, vec_x, beta, vec_y, compute_type, alg,
                                          buffer_size);
}

mcspStatus_t mcspSpMV(mcspHandle_t handle, mcsparseOperation_t op_a, const void *alpha, mcspSpMatDescr_t matA,
                      mcspDnVecDescr_t vec_x, const void *beta, mcspDnVecDescr_t vec_y, macaDataType compute_type,
                      mcsparseSpMVAlg_t alg, void *external_buffer) {
    return mcspSpMV_native(handle, op_a, alpha, matA, vec_x, beta, vec_y, compute_type, alg, external_buffer);
}

mcspStatus_t mcspCsrmvEx_bufferSize(mcspHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, mcspInt m,
                                    mcspInt n, mcspInt nnz, const void *alpha, macaDataType alphatype,
                                    const mcspMatDescr_t descrA, const void *csrValA, macaDataType csrValAtype,
                                    const mcspInt *csrRowPtrA, const mcspInt *csrColIndA, const void *x,
                                    macaDataType xtype, const void *beta, macaDataType betatype, void *y,
                                    macaDataType outType, macaDataType executiontype, size_t *bufferSizeInBytes) {
    *bufferSizeInBytes = MIN_BUFFER_SIZE;
    return MCSP_STATUS_SUCCESS;
}

// csrmv
mcspStatus_t mcspCsrmvEx(mcspHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, mcspInt m, mcspInt n,
                         mcspInt nnz, const void *alpha, macaDataType alphatype, const mcspMatDescr_t descrA,
                         const void *csrValA, macaDataType csrValAtype, const mcspInt *csrRowPtrA,
                         const mcspInt *csrColIndA, const void *x, macaDataType xtype, const void *beta,
                         macaDataType betatype, void *y, macaDataType outType, macaDataType executiontype,
                         void *buffer) {
    return mcspCsrmvExImpl(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA,
                           csrColIndA, x, xtype, beta, betatype, y, outType, executiontype, buffer);
}

mcspStatus_t mcspCuinCsrmvEx_bufferSize(mcspHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, int m,
                                      int n, int nnz, const void *alpha, macaDataType alphatype,
                                      const mcspMatDescr_t descrA, const void *csrValA, macaDataType csrValAtype,
                                      const int *csrRowPtrA, const int *csrColIndA, const void *x, macaDataType xtype,
                                      const void *beta, macaDataType betatype, void *y, macaDataType outType,
                                      macaDataType executiontype, size_t *bufferSizeInBytes) {
    *bufferSizeInBytes = MIN_BUFFER_SIZE;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCuinCsrmvEx(mcspHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, int m, int n,
                           int nnz, const void *alpha, macaDataType alphatype, const mcspMatDescr_t descrA,
                           const void *csrValA, macaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA,
                           const void *x, macaDataType xtype, const void *beta, macaDataType betatype, void *y,
                           macaDataType outType, macaDataType executiontype, void *buffer) {
    return mcspCsrmvExImpl(handle, alg, transA, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, alpha, alphatype, descrA, csrValA,
                           csrValAtype, (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, x, xtype, beta, betatype, y,
                           outType, executiontype, buffer);
}

mcspStatus_t mcspCuinScsrmv(mcspHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz, const float *alpha,
                          const mcspMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA,
                          const int *csrColIndA, const float *x, const float *beta, float *y) {
    return mcspCsrmvImpl(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
}

mcspStatus_t mcspCuinDcsrmv(mcspHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz, const double *alpha,
                          const mcspMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA,
                          const int *csrColIndA, const double *x, const double *beta, double *y) {
    return mcspCsrmvImpl(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
}

mcspStatus_t mcspCuinCcsrmv(mcspHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                          const mcspComplexFloat *alpha, const mcspMatDescr_t descrA, const mcspComplexFloat *csrValA,
                          const int *csrRowPtrA, const int *csrColIndA, const mcspComplexFloat *x,
                          const mcspComplexFloat *beta, mcspComplexFloat *y) {
    return mcspCsrmvImpl(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
}

mcspStatus_t mcspCuinZcsrmv(mcspHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                          const mcspComplexDouble *alpha, const mcspMatDescr_t descrA, const mcspComplexDouble *csrValA,
                          const int *csrRowPtrA, const int *csrColIndA, const mcspComplexDouble *x,
                          const mcspComplexDouble *beta, mcspComplexDouble *y) {
    return mcspCsrmvImpl(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
}

#ifdef __cplusplus
}
#endif
