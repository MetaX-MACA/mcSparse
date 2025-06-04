#include "common/mcsp_types.h"
#include "csr_spgemm_device.hpp"
#include "csr_spgemm_host.hpp"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

mcspStatus_t mcspSpGEMM_createDescrImpl(mcspSpGEMMDescr_t *spgemm_descr) {
    if (spgemm_descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *spgemm_descr = new mcspSpGEMMDescr();
    if (*spgemm_descr == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    if (mcspCreateMatInfo(&((*spgemm_descr)->mat_info)) != MCSP_STATUS_SUCCESS ||
        mcspCreateMatDescr(&((*spgemm_descr)->mat_descr)) != MCSP_STATUS_SUCCESS) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpGEMM_destroyDescrImpl(mcspSpGEMMDescr_t spgemm_descr) {
    if (spgemm_descr == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }

    if (spgemm_descr->rows != nullptr && !spgemm_descr->is_reused_spgemm) {
        MACA_ASSERT(mcFree(spgemm_descr->rows));
    }
    spgemm_descr->rows = nullptr;
    if (spgemm_descr->cols != nullptr && !spgemm_descr->is_reused_spgemm) {
        MACA_ASSERT(mcFree(spgemm_descr->cols));
    }
    spgemm_descr->cols = nullptr;
    if (spgemm_descr->vals != nullptr && !spgemm_descr->is_reused_spgemm) {
        MACA_ASSERT(mcFree(spgemm_descr->vals));
    }
    spgemm_descr->vals = nullptr;
    mcspStatus_t stat;
    if (spgemm_descr->mat_info != nullptr) {
        stat = mcspDestroyMatInfo(spgemm_descr->mat_info);
        if (stat != MCSP_STATUS_SUCCESS) {
            return MCSP_STATUS_INTERNAL_ERROR;
        }
    }
    if (spgemm_descr->mat_descr != nullptr) {
        stat = mcspDestroyMatDescr(spgemm_descr->mat_descr);
        if (stat != MCSP_STATUS_SUCCESS) {
            return MCSP_STATUS_INTERNAL_ERROR;
        }
    }

    delete spgemm_descr;
    spgemm_descr = nullptr;
    return MCSP_STATUS_SUCCESS;
}

// Calculate buffer size and result nnz, store them into mcspSpGEMMDescr_t
template <typename idxType>
mcspStatus_t mcspSpGEMM_workEstimationImpl(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                           const void *alpha, mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B,
                                           const void *beta, mcspSpMatDescr_t mat_C, macaDataType compute_type,
                                           mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr, size_t *buffer_size,
                                           void *external_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (spgemm_descr == nullptr || mat_A == nullptr || mat_B == nullptr || mat_C == nullptr || buffer_size == nullptr ||
        alpha == nullptr || beta == nullptr || mat_A->rows == nullptr || mat_A->cols == nullptr ||
        mat_B->rows == nullptr || mat_B->cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (!(mat_A->format == MCSPARSE_FORMAT_CSR && mat_B->format == MCSPARSE_FORMAT_CSR &&
          mat_C->format == MCSPARSE_FORMAT_CSR &&
          (op_A == MCSPARSE_OPERATION_NON_TRANSPOSE && op_B == MCSPARSE_OPERATION_NON_TRANSPOSE))) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mat_A->row_num >> 32 != 0 || mat_A->col_num >> 32 != 0 || mat_A->nnz >> 32 != 0 || mat_B->row_num >> 32 != 0 ||
        mat_B->col_num >> 32 != 0 || mat_C->nnz >> 32 != 0 || mat_C->row_num >> 32 != 0 || mat_C->col_num >> 32 != 0 ||
        mat_C->nnz >> 32 != 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mat_A->row_num < 0 || mat_A->col_num < 0 || mat_A->nnz < 0 || mat_B->row_num < 0 || mat_B->col_num < 0 ||
        mat_B->nnz < 0 || mat_C->row_num < 0 || mat_C->col_num < 0 || mat_C->nnz < 0 ||
        (!(mat_A->row_num == mat_C->row_num && mat_B->col_num == mat_C->col_num && mat_A->col_num == mat_B->row_num))) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    mcspStatus_t stat = mcspCsrgemmBuffersizeTemplate(
        handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, mat_A->mat_descr,
        (idxType)mat_A->nnz, (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
        (idxType *)mat_B->rows, (idxType *)mat_B->cols, mat_C->mat_descr, (idxType)mat_C->nnz, (idxType *)mat_C->rows,
        (idxType *)mat_C->cols, spgemm_descr->mat_info, buffer_size);

    // If called for the first time, just calculate buffersize1
    if (external_buffer == nullptr) {
        return stat;
    }

    spgemm_descr->row_num = mat_C->row_num;
    spgemm_descr->col_num = mat_C->col_num;
    spgemm_descr->compute_type = compute_type;
    MACA_ASSERT(mcMalloc(&(spgemm_descr->rows), (spgemm_descr->row_num + 1) * sizeof(idxType)));

    // only buffer calculated by mcspXgemmBuffersize is used in Xgemm calculation
    bool include_addition = false;
    spgemm_descr->buff = external_buffer;
    spgemm_descr->mat_descr->base = mat_C->mat_descr->base;
    return mcspCsrgemmNnzTemplate(
        handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, mat_A->mat_descr,
        (idxType)mat_A->nnz, (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
        (idxType *)mat_B->rows, (idxType *)mat_B->cols, mat_C->mat_descr, (idxType)mat_C->nnz, (idxType *)mat_C->rows,
        (idxType *)mat_C->cols, spgemm_descr->mat_descr, (idxType *)spgemm_descr->rows, (idxType *)&(spgemm_descr->nnz),
        spgemm_descr->mat_info, external_buffer, include_addition);
}

template <typename idxType>
mcspStatus_t mcspSpGEMM_computeImpl(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                    const void *alpha, mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, const void *beta,
                                    mcspSpMatDescr_t mat_C, macaDataType compute_type, mcsparseSpGEMMAlg_t alg,
                                    mcspSpGEMMDescr_t spgemm_descr, size_t *buffer_size, void *external_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (spgemm_descr == nullptr || mat_A == nullptr || mat_B == nullptr || mat_C == nullptr || buffer_size == nullptr ||
        alpha == nullptr || beta == nullptr || mat_A->rows == nullptr || mat_A->cols == nullptr ||
        mat_A->vals == nullptr || mat_B->rows == nullptr || mat_B->cols == nullptr || mat_B->vals == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (!(mat_A->format == MCSPARSE_FORMAT_CSR && mat_B->format == MCSPARSE_FORMAT_CSR &&
          mat_C->format == MCSPARSE_FORMAT_CSR &&
          (op_A == MCSPARSE_OPERATION_NON_TRANSPOSE && op_B == MCSPARSE_OPERATION_NON_TRANSPOSE))) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mat_A->row_num >> 32 != 0 || mat_A->col_num >> 32 != 0 || mat_A->nnz >> 32 != 0 || mat_B->row_num >> 32 != 0 ||
        mat_B->col_num >> 32 != 0 || mat_C->nnz >> 32 != 0 || mat_C->row_num >> 32 != 0 || mat_C->col_num >> 32 != 0 ||
        mat_C->nnz >> 32 != 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mat_A->row_num < 0 || mat_A->col_num < 0 || mat_A->nnz < 0 || mat_B->row_num < 0 || mat_B->col_num < 0 ||
        mat_B->nnz < 0 || mat_C->row_num < 0 || mat_C->col_num < 0 || mat_C->nnz < 0 ||
        (!(mat_A->row_num == mat_C->row_num && mat_B->col_num == mat_C->col_num && mat_A->col_num == mat_B->row_num))) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    // If called for the first time, just return a pseudo buffersize2
    if (external_buffer == nullptr) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    if (spgemm_descr->nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    mat_C->nnz = spgemm_descr->nnz;
    MACA_ASSERT(mcMalloc(&(spgemm_descr->cols), spgemm_descr->nnz * sizeof(idxType)));
    mcspStatus_t stat;
    mcsparsePointerMode_t ptr_mode = handle->ptr_mode;
    bool include_addition = false;
    switch (compute_type) {
        case MACA_R_32F: {
            MACA_ASSERT(mcMalloc(&(spgemm_descr->vals), spgemm_descr->nnz * sizeof(float)));
            float h_beta = getScalarToHost((float *)beta, handle->ptr_mode);
            float h_alpha = 1;
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            // only buffer calculated by mcspXgemmBuffersize is used in Xgemm calculation
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, &h_alpha,
                mat_A->mat_descr, (idxType)mat_A->nnz, (float *)mat_A->vals, (idxType *)mat_A->rows,
                (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (float *)mat_B->vals,
                (idxType *)mat_B->rows, (idxType *)mat_B->cols, &h_beta, mat_C->mat_descr, (idxType)mat_C->nnz,
                (float *)mat_C->vals, (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (float *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_R_64F: {
            MACA_ASSERT(mcMalloc(&(spgemm_descr->vals), spgemm_descr->nnz * sizeof(double)));
            double h_beta = getScalarToHost((double *)beta, handle->ptr_mode);
            double h_alpha = static_cast<double>(1);
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            // only buffer calculated by mcspXgemmBuffersize is used in Xgemm calculation
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, &h_alpha,
                mat_A->mat_descr, (idxType)mat_A->nnz, (double *)mat_A->vals, (idxType *)mat_A->rows,
                (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (double *)mat_B->vals,
                (idxType *)mat_B->rows, (idxType *)mat_B->cols, &h_beta, mat_C->mat_descr, (idxType)mat_C->nnz,
                (double *)mat_C->vals, (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (double *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_C_32F: {
            MACA_ASSERT(mcMalloc(&(spgemm_descr->vals), spgemm_descr->nnz * sizeof(mcspComplexFloat)));
            mcspComplexFloat h_beta = getScalarToHost((mcspComplexFloat *)beta, handle->ptr_mode);
            mcspComplexFloat h_alpha = static_cast<mcspComplexFloat>(1);
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            // only buffer calculated by mcspXgemmBuffersize is used in Xgemm calculation
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, &h_alpha,
                mat_A->mat_descr, (idxType)mat_A->nnz, (mcspComplexFloat *)mat_A->vals, (idxType *)mat_A->rows,
                (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (mcspComplexFloat *)mat_B->vals,
                (idxType *)mat_B->rows, (idxType *)mat_B->cols, &h_beta, mat_C->mat_descr, (idxType)mat_C->nnz,
                (mcspComplexFloat *)mat_C->vals, (idxType *)mat_C->rows, (idxType *)mat_C->cols,
                spgemm_descr->mat_descr, (mcspComplexFloat *)spgemm_descr->vals, (idxType *)spgemm_descr->rows,
                (idxType *)spgemm_descr->cols, spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_C_64F: {
            MACA_ASSERT(mcMalloc(&(spgemm_descr->vals), spgemm_descr->nnz * sizeof(mcspComplexDouble)));
            mcspComplexDouble h_beta = getScalarToHost((mcspComplexDouble *)beta, handle->ptr_mode);
            mcspComplexDouble h_alpha = static_cast<mcspComplexDouble>(1);
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            // only buffer calculated by mcspXgemmBuffersize is used in Xgemm calculation
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, &h_alpha,
                mat_A->mat_descr, (idxType)mat_A->nnz, (mcspComplexDouble *)mat_A->vals, (idxType *)mat_A->rows,
                (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (mcspComplexDouble *)mat_B->vals,
                (idxType *)mat_B->rows, (idxType *)mat_B->cols, &h_beta, mat_C->mat_descr, (idxType)mat_C->nnz,
                (mcspComplexDouble *)mat_C->vals, (idxType *)mat_C->rows, (idxType *)mat_C->cols,
                spgemm_descr->mat_descr, (mcspComplexDouble *)spgemm_descr->vals, (idxType *)spgemm_descr->rows,
                (idxType *)spgemm_descr->cols, spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
#ifdef __MACA__
        case MACA_R_16F: {
            MACA_ASSERT(mcMalloc(&(spgemm_descr->vals), spgemm_descr->nnz * sizeof(__half)));
            __half h_beta = getScalarToHost((__half *)beta, handle->ptr_mode);
            __half h_alpha = 1.0;
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            // only buffer calculated by mcspXgemmBuffersize is used in Xgemm calculation
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, &h_alpha,
                mat_A->mat_descr, (idxType)mat_A->nnz, (__half *)mat_A->vals, (idxType *)mat_A->rows,
                (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (__half *)mat_B->vals,
                (idxType *)mat_B->rows, (idxType *)mat_B->cols, &h_beta, mat_C->mat_descr, (idxType)mat_C->nnz,
                (__half *)mat_C->vals, (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (__half *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_R_16BF: {
            MACA_ASSERT(mcMalloc(&(spgemm_descr->vals), spgemm_descr->nnz * sizeof(mcsp_bfloat16)));
            mcsp_bfloat16 h_beta = getScalarToHost((mcsp_bfloat16 *)beta, handle->ptr_mode);
            mcsp_bfloat16 h_alpha = 1.0;
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            // only buffer calculated by mcspXgemmBuffersize is used in Xgemm calculation
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, &h_alpha,
                mat_A->mat_descr, (idxType)mat_A->nnz, (mcsp_bfloat16 *)mat_A->vals, (idxType *)mat_A->rows,
                (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (mcsp_bfloat16 *)mat_B->vals,
                (idxType *)mat_B->rows, (idxType *)mat_B->cols, &h_beta, mat_C->mat_descr, (idxType)mat_C->nnz,
                (mcsp_bfloat16 *)mat_C->vals, (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (mcsp_bfloat16 *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_C_16F: {
            MACA_ASSERT(mcMalloc(&(spgemm_descr->vals), spgemm_descr->nnz * sizeof(__half2)));
            __half2 h_beta = getScalarToHost((__half2 *)beta, handle->ptr_mode);
            __half2 h_alpha = {1.0, 0.0};
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            // only buffer calculated by mcspXgemmBuffersize is used in Xgemm calculation
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, &h_alpha,
                mat_A->mat_descr, (idxType)mat_A->nnz, (__half2 *)mat_A->vals, (idxType *)mat_A->rows,
                (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (__half2 *)mat_B->vals,
                (idxType *)mat_B->rows, (idxType *)mat_B->cols, &h_beta, mat_C->mat_descr, (idxType)mat_C->nnz,
                (__half2 *)mat_C->vals, (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (__half2 *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_C_16BF: {
            MACA_ASSERT(mcMalloc(&(spgemm_descr->vals), spgemm_descr->nnz * sizeof(mcsp_bfloat162)));
            mcsp_bfloat162 h_beta = getScalarToHost((mcsp_bfloat162 *)beta, handle->ptr_mode);
            mcsp_bfloat162 h_alpha = {1.0, 0.0};
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            // only buffer calculated by mcspXgemmBuffersize is used in Xgemm calculation
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, &h_alpha,
                mat_A->mat_descr, (idxType)mat_A->nnz, (mcsp_bfloat162 *)mat_A->vals, (idxType *)mat_A->rows,
                (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (mcsp_bfloat162 *)mat_B->vals,
                (idxType *)mat_B->rows, (idxType *)mat_B->cols, &h_beta, mat_C->mat_descr, (idxType)mat_C->nnz,
                (mcsp_bfloat162 *)mat_C->vals, (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (mcsp_bfloat162 *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
#endif
        default:
            stat = MCSP_STATUS_NOT_IMPLEMENTED;
    }
    handle->ptr_mode = ptr_mode;
    return stat;
}

template <typename idxType>
mcspStatus_t mcspSpGEMM_copyImpl(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                 const void *alpha, mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, const void *beta,
                                 mcspSpMatDescr_t mat_C, macaDataType compute_type, mcsparseSpGEMMAlg_t alg,
                                 mcspSpGEMMDescr_t spgemm_descr) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (spgemm_descr == nullptr || mat_A == nullptr || mat_B == nullptr || mat_C == nullptr || alpha == nullptr ||
        beta == nullptr || mat_C->rows == nullptr || mat_C->cols == nullptr || mat_C->vals == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (!(mat_A->format == MCSPARSE_FORMAT_CSR && mat_B->format == MCSPARSE_FORMAT_CSR &&
          mat_C->format == MCSPARSE_FORMAT_CSR &&
          (op_A == MCSPARSE_OPERATION_NON_TRANSPOSE && op_B == MCSPARSE_OPERATION_NON_TRANSPOSE))) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mat_A->row_num >> 32 != 0 || mat_A->col_num >> 32 != 0 || mat_A->nnz >> 32 != 0 || mat_B->row_num >> 32 != 0 ||
        mat_B->col_num >> 32 != 0 || mat_C->nnz >> 32 != 0 || mat_C->row_num >> 32 != 0 || mat_C->col_num >> 32 != 0 ||
        mat_C->nnz >> 32 != 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mat_A->row_num < 0 || mat_A->col_num < 0 || mat_A->nnz < 0 || mat_B->row_num < 0 || mat_B->col_num < 0 ||
        mat_B->nnz < 0 || mat_C->row_num < 0 || mat_C->col_num < 0 || mat_C->nnz < 0 ||
        (!(mat_A->row_num == mat_C->row_num && mat_B->col_num == mat_C->col_num && mat_A->col_num == mat_B->row_num))) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemcpyAsync(mat_C->rows, spgemm_descr->rows, (mat_C->row_num + 1) * sizeof(idxType),
                              mcMemcpyDeviceToDevice, stream));
    MACA_ASSERT(mcMemcpyAsync(mat_C->cols, spgemm_descr->cols, spgemm_descr->nnz * sizeof(idxType),
                              mcMemcpyDeviceToDevice, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
#define SPGEMMAXPBY_BLOCK_SIZE 512
    int n_block = (spgemm_descr->nnz + SPGEMMAXPBY_BLOCK_SIZE - 1) / SPGEMMAXPBY_BLOCK_SIZE;

    switch (compute_type) {
        case MACA_R_32F: {
            float h_beta = getScalarToHost((float *)beta, handle->ptr_mode);
            float h_alpha = getScalarToHost((float *)alpha, handle->ptr_mode);
            denseAxpby<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, spgemm_descr->nnz, h_alpha, (float *)spgemm_descr->vals,
                                               h_beta, (float *)mat_C->vals);
            break;
        }
        case MACA_R_64F: {
            double h_beta = getScalarToHost((double *)beta, handle->ptr_mode);
            double h_alpha = getScalarToHost((double *)alpha, handle->ptr_mode);
            denseAxpby<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                               (double *)spgemm_descr->vals, h_beta, (double *)mat_C->vals);
            break;
        }
        case MACA_C_32F: {
            mcspComplexFloat h_beta = getScalarToHost((mcspComplexFloat *)beta, handle->ptr_mode);
            mcspComplexFloat h_alpha = getScalarToHost((mcspComplexFloat *)alpha, handle->ptr_mode);
            denseAxpby<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                               (mcspComplexFloat *)spgemm_descr->vals, h_beta,
                                               (mcspComplexFloat *)mat_C->vals);
            break;
        }
        case MACA_C_64F: {
            mcspComplexDouble h_beta = getScalarToHost((mcspComplexDouble *)beta, handle->ptr_mode);
            mcspComplexDouble h_alpha = getScalarToHost((mcspComplexDouble *)alpha, handle->ptr_mode);
            denseAxpby<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                               (mcspComplexDouble *)spgemm_descr->vals, h_beta,
                                               (mcspComplexDouble *)mat_C->vals);
            break;
        }
#ifdef __MACA__
        case MACA_R_16F: {
            __half h_beta = getScalarToHost((__half *)beta, handle->ptr_mode);
            __half h_alpha = getScalarToHost((__half *)alpha, handle->ptr_mode);
            denseAxpby<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                               (__half *)spgemm_descr->vals, h_beta, (__half *)mat_C->vals);
            break;
        }
        case MACA_R_16BF: {
            mcsp_bfloat16 h_beta = getScalarToHost((mcsp_bfloat16 *)beta, handle->ptr_mode);
            mcsp_bfloat16 h_alpha = getScalarToHost((mcsp_bfloat16 *)alpha, handle->ptr_mode);
            denseAxpby<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                               (mcsp_bfloat16 *)spgemm_descr->vals, h_beta,
                                               (mcsp_bfloat16 *)mat_C->vals);
            break;
        }
        case MACA_C_16F: {
            __half2 h_beta = getScalarToHost((__half2 *)beta, handle->ptr_mode);
            __half2 h_alpha = getScalarToHost((__half2 *)alpha, handle->ptr_mode);
            denseAxpbyLowPrecisionComplex<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                                                  (__half2 *)spgemm_descr->vals, h_beta,
                                                                  (__half2 *)mat_C->vals);
            break;
        }
        case MACA_C_16BF: {
            mcsp_bfloat162 h_beta = getScalarToHost((mcsp_bfloat162 *)beta, handle->ptr_mode);
            mcsp_bfloat162 h_alpha = getScalarToHost((mcsp_bfloat162 *)alpha, handle->ptr_mode);
            denseAxpbyLowPrecisionComplex<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                                                  (mcsp_bfloat162 *)spgemm_descr->vals, h_beta,
                                                                  (mcsp_bfloat162 *)mat_C->vals);
            break;
        }
#endif
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
#undef SPGEMMAXPBY_BLOCK_SIZE
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSpGEMM_createDescr(mcspSpGEMMDescr_t *spgemm_descr) {
    return mcspSpGEMM_createDescrImpl(spgemm_descr);
}

mcspStatus_t mcspSpGEMM_destroyDescr(mcspSpGEMMDescr_t spgemm_descr) {
    return mcspSpGEMM_destroyDescrImpl(spgemm_descr);
}

mcspStatus_t mcspSpGEMM_workEstimation(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                       const void *alpha, mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B,
                                       const void *beta, mcspSpMatDescr_t mat_C, macaDataType compute_type,
                                       mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr, size_t *buffer_size1,
                                       void *external_buffer1) {
    if (mat_A->rowIdxType == MCSPARSE_INDEX_32I && mat_A->rowIdxType == mat_A->colIdxType &&
        mat_A->rowIdxType == mat_B->rowIdxType && mat_A->rowIdxType == mat_B->colIdxType &&
        mat_A->rowIdxType == mat_C->rowIdxType && mat_A->rowIdxType == mat_C->colIdxType) {
        return mcspSpGEMM_workEstimationImpl<mcspInt>(handle, op_A, op_B, alpha, mat_A, mat_B, beta, mat_C,
                                                      compute_type, alg, spgemm_descr, buffer_size1, external_buffer1);
    } else {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

mcspStatus_t mcspSpGEMM_compute(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                const void *alpha, mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, const void *beta,
                                mcspSpMatDescr_t mat_C, macaDataType compute_type, mcsparseSpGEMMAlg_t alg,
                                mcspSpGEMMDescr_t spgemm_descr, size_t *buffer_size2, void *external_buffer2) {
    if (mat_A->rowIdxType == MCSPARSE_INDEX_32I && mat_A->rowIdxType == mat_A->colIdxType &&
        mat_A->rowIdxType == mat_B->rowIdxType && mat_A->rowIdxType == mat_B->colIdxType &&
        mat_A->rowIdxType == mat_C->rowIdxType && mat_A->rowIdxType == mat_C->colIdxType) {
        return mcspSpGEMM_computeImpl<mcspInt>(handle, op_A, op_B, alpha, mat_A, mat_B, beta, mat_C, compute_type, alg,
                                               spgemm_descr, buffer_size2, external_buffer2);
    } else {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

mcspStatus_t mcspSpGEMM_copy(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B, const void *alpha,
                             mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, const void *beta, mcspSpMatDescr_t mat_C,
                             macaDataType compute_type, mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr) {
    if (mat_A->rowIdxType == MCSPARSE_INDEX_32I && mat_A->rowIdxType == mat_A->colIdxType &&
        mat_A->rowIdxType == mat_B->rowIdxType && mat_A->rowIdxType == mat_B->colIdxType &&
        mat_A->rowIdxType == mat_C->rowIdxType && mat_A->rowIdxType == mat_C->colIdxType) {
        return mcspSpGEMM_copyImpl<mcspInt>(handle, op_A, op_B, alpha, mat_A, mat_B, beta, mat_C, compute_type, alg,
                                            spgemm_descr);
    } else {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

#ifdef __cplusplus
}
#endif
