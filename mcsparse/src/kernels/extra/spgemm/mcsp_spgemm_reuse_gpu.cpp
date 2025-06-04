#include "common/mcsp_types.h"
#include "csr_spgemm_device.hpp"
#include "csr_spgemm_host.hpp"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "spgemm_reuse_device.hpp"

template <typename idxType>
mcspStatus_t mcspSpGEMMreuse_workEstimationImpl(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                                mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, mcspSpMatDescr_t mat_C,
                                                mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr,
                                                size_t *buffer_size1, void *external_buffer1) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (spgemm_descr == nullptr || mat_A == nullptr || mat_B == nullptr || mat_C == nullptr ||
        buffer_size1 == nullptr || mat_A->rows == nullptr || mat_A->cols == nullptr || mat_B->rows == nullptr ||
        mat_B->cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    // mcsparse only support MCSPARSE_OPERATION_NON_TRANSPOSE and csr format at the present
    if (!(mat_A->format == MCSPARSE_FORMAT_CSR && mat_B->format == MCSPARSE_FORMAT_CSR &&
          mat_C->format == MCSPARSE_FORMAT_CSR &&
          (op_A == MCSPARSE_OPERATION_NON_TRANSPOSE && op_B == MCSPARSE_OPERATION_NON_TRANSPOSE))) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mat_A->row_num < 0 || mat_A->col_num < 0 || mat_A->nnz < 0 || mat_B->row_num < 0 || mat_B->col_num < 0 ||
        mat_B->nnz < 0 || !mat_A->col_num == mat_B->row_num) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    spgemm_descr->is_reused_spgemm = true;

    if (external_buffer1 == nullptr) {
        *buffer_size1 = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspSpGEMMreuse_nnzImpl(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                     mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, mcspSpMatDescr_t mat_C,
                                     mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr, size_t *buffer_size2,
                                     void *external_buffer2, size_t *nnz_buffer_size, void *nnz_external_buffer,
                                     size_t *row_buffer_size, void *row_external_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (spgemm_descr == nullptr || mat_A == nullptr || mat_B == nullptr || mat_C == nullptr ||
        buffer_size2 == nullptr || nnz_buffer_size == nullptr || row_buffer_size == nullptr || mat_A->rows == nullptr ||
        mat_A->cols == nullptr || mat_B->rows == nullptr || mat_B->cols == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    // mcsparse only support MCSPARSE_OPERATION_NON_TRANSPOSE at the present
    if (!(mat_A->format == MCSPARSE_FORMAT_CSR && mat_B->format == MCSPARSE_FORMAT_CSR &&
          mat_C->format == MCSPARSE_FORMAT_CSR &&
          (op_A == MCSPARSE_OPERATION_NON_TRANSPOSE && op_B == MCSPARSE_OPERATION_NON_TRANSPOSE))) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mat_A->row_num < 0 || mat_A->col_num < 0 || mat_A->nnz < 0 || mat_B->row_num < 0 || mat_B->col_num < 0 ||
        mat_B->nnz < 0 || !mat_A->col_num == mat_B->row_num) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    mcspStatus_t stat;
    if (external_buffer2 == nullptr && nnz_external_buffer == nullptr && row_external_buffer == nullptr) {
        stat = mcspZcsrgemmBuffersize(
            handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num,
            (mcspComplexDouble *)nullptr, mat_A->mat_descr, (idxType)mat_A->nnz, (idxType *)mat_A->rows,
            (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (idxType *)mat_B->rows,
            (idxType *)mat_B->cols, (mcspComplexDouble *)nullptr, mat_C->mat_descr, 0, (idxType *)mat_C->rows,
            (idxType *)mat_C->cols, spgemm_descr->mat_info, nnz_buffer_size);
        *buffer_size2 = MIN_BUFFER_SIZE;
        *row_buffer_size = (mat_A->row_num + 1) * sizeof(mat_A->row_num);
        return stat;
    }
    if (external_buffer2 == nullptr || nnz_external_buffer == nullptr || row_external_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    spgemm_descr->row_num = mat_C->row_num;
    spgemm_descr->col_num = mat_C->col_num;
    spgemm_descr->rows = row_external_buffer;

    bool include_addition = false;
    spgemm_descr->buff = nnz_external_buffer;
    spgemm_descr->mat_descr->base = mat_C->mat_descr->base;
    stat = mcspCsrgemmNnzTemplate(
        handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num, mat_A->mat_descr,
        (idxType)mat_A->nnz, (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
        (idxType *)mat_B->rows, (idxType *)mat_B->cols, mat_C->mat_descr, (idxType)mat_C->nnz, (idxType *)mat_C->rows,
        (idxType *)mat_C->cols, spgemm_descr->mat_descr, (idxType *)spgemm_descr->rows,
        (idxType *)(&(spgemm_descr->nnz)), spgemm_descr->mat_info, nnz_external_buffer, include_addition);

    mat_C->row_num = mat_A->row_num;
    mat_C->col_num = mat_B->col_num;
    mat_C->nnz = spgemm_descr->nnz;
    return stat;
}

template <typename idxType>
mcspStatus_t mcspSpGEMMreuse_copyImpl(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                      mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, mcspSpMatDescr_t mat_C,
                                      mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr,
                                      size_t *col_val_buffer_size, void *col_val_external_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (spgemm_descr == nullptr || mat_A == nullptr || mat_B == nullptr || mat_C == nullptr ||
        col_val_buffer_size == nullptr || mat_A->rows == nullptr || mat_A->cols == nullptr || mat_A->vals == nullptr ||
        mat_B->rows == nullptr || mat_B->cols == nullptr || mat_B->vals == nullptr || mat_C->rows == nullptr ||
        mat_C->cols == nullptr || mat_C->vals == nullptr || spgemm_descr->rows == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (!(mat_A->format == MCSPARSE_FORMAT_CSR && mat_B->format == MCSPARSE_FORMAT_CSR &&
          mat_C->format == MCSPARSE_FORMAT_CSR &&
          (op_A == MCSPARSE_OPERATION_NON_TRANSPOSE && op_B == MCSPARSE_OPERATION_NON_TRANSPOSE))) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (mat_A->row_num < 0 || mat_A->col_num < 0 || mat_A->nnz < 0 || mat_B->row_num < 0 || mat_B->col_num < 0 ||
        mat_B->nnz < 0 || !mat_A->col_num == mat_B->row_num || spgemm_descr->nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }
    macaDataType compute_type = mat_A->valueType;
    if (col_val_external_buffer == nullptr) {
        *col_val_buffer_size = ALIGN(spgemm_descr->nnz * sizeof(mat_A->col_num), ALIGNED_SIZE);
        switch (compute_type) {
            case MACA_R_32F: {
                *col_val_buffer_size += ALIGN(spgemm_descr->nnz * sizeof(float), ALIGNED_SIZE);
                return MCSP_STATUS_SUCCESS;
            }
            case MACA_R_64F: {
                *col_val_buffer_size += ALIGN(spgemm_descr->nnz * sizeof(double), ALIGNED_SIZE);
                return MCSP_STATUS_SUCCESS;
            }
            case MACA_C_32F: {
                *col_val_buffer_size += ALIGN(spgemm_descr->nnz * sizeof(mcspComplexFloat), ALIGNED_SIZE);
                return MCSP_STATUS_SUCCESS;
            }
            case MACA_C_64F: {
                *col_val_buffer_size += ALIGN(spgemm_descr->nnz * sizeof(mcspComplexDouble), ALIGNED_SIZE);
                return MCSP_STATUS_SUCCESS;
            }
#ifdef __MACA__
            case MACA_R_16F: {
                *col_val_buffer_size += ALIGN(spgemm_descr->nnz * sizeof(__half), ALIGNED_SIZE);
                return MCSP_STATUS_SUCCESS;
            }
            case MACA_C_16F: {
                *col_val_buffer_size += ALIGN(spgemm_descr->nnz * sizeof(__half2), ALIGNED_SIZE);
                return MCSP_STATUS_SUCCESS;
            }
            case MACA_R_16BF: {
                *col_val_buffer_size += ALIGN(spgemm_descr->nnz * sizeof(mcsp_bfloat16), ALIGNED_SIZE);
                return MCSP_STATUS_SUCCESS;
            }
            case MACA_C_16BF: {
                *col_val_buffer_size += ALIGN(spgemm_descr->nnz * sizeof(mcsp_bfloat162), ALIGNED_SIZE);
                return MCSP_STATUS_SUCCESS;
            }
#endif
            default: {
                return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
    }
    spgemm_descr->cols = col_val_external_buffer;
    spgemm_descr->vals =
        (char *)col_val_external_buffer + ALIGN(spgemm_descr->nnz * sizeof(mat_A->col_num), ALIGNED_SIZE);

    mat_C->nnz = spgemm_descr->nnz;
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    mcsparsePointerMode_t ptr_mode = handle->ptr_mode;
    handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
    bool include_addition = false;
    // TODO need a new function to obtain only A*B's column index
    switch (compute_type) {
        case MACA_R_32F: {
            float h_beta = 1;
            float h_alpha = 1;
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num,
                (float *)&h_alpha, mat_A->mat_descr, (idxType)mat_A->nnz, (float *)mat_A->vals, (idxType *)mat_A->rows,
                (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz, (float *)mat_B->vals,
                (idxType *)mat_B->rows, (idxType *)mat_B->cols, (float *)&h_beta, mat_C->mat_descr, (idxType)mat_C->nnz,
                (float *)mat_C->vals, (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (float *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_R_64F: {
            double h_beta = 1;
            double h_alpha = 1;
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num,
                (double *)&h_alpha, mat_A->mat_descr, (idxType)mat_A->nnz, (double *)mat_A->vals,
                (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
                (double *)mat_B->vals, (idxType *)mat_B->rows, (idxType *)mat_B->cols, (double *)&h_beta,
                mat_C->mat_descr, (idxType)mat_C->nnz, (double *)mat_C->vals, (idxType *)mat_C->rows,
                (idxType *)mat_C->cols, spgemm_descr->mat_descr, (double *)spgemm_descr->vals,
                (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols, spgemm_descr->mat_info,
                spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_C_32F: {
            mcspComplexFloat h_beta = static_cast<mcspComplexFloat>(1);
            mcspComplexFloat h_alpha = static_cast<mcspComplexFloat>(1);
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num,
                (mcspComplexFloat *)&h_alpha, mat_A->mat_descr, (idxType)mat_A->nnz, (mcspComplexFloat *)mat_A->vals,
                (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
                (mcspComplexFloat *)mat_B->vals, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                (mcspComplexFloat *)&h_beta, mat_C->mat_descr, (idxType)mat_C->nnz, (mcspComplexFloat *)mat_C->vals,
                (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (mcspComplexFloat *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_C_64F: {
            mcspComplexDouble h_beta = static_cast<mcspComplexDouble>(1);
            mcspComplexDouble h_alpha = static_cast<mcspComplexDouble>(1);
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num,
                (mcspComplexDouble *)&h_alpha, mat_A->mat_descr, (idxType)mat_A->nnz, (mcspComplexDouble *)mat_A->vals,
                (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
                (mcspComplexDouble *)mat_B->vals, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                (mcspComplexDouble *)&h_beta, mat_C->mat_descr, (idxType)mat_C->nnz, (mcspComplexDouble *)mat_C->vals,
                (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (mcspComplexDouble *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
#ifdef __MACA__
        case MACA_R_16F: {
            __half h_beta = 1.f;
            __half h_alpha = 1.f;
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num,
                (__half *)&h_alpha, mat_A->mat_descr, (idxType)mat_A->nnz, (__half *)mat_A->vals,
                (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
                (__half *)mat_B->vals, (idxType *)mat_B->rows, (idxType *)mat_B->cols, (__half *)&h_beta,
                mat_C->mat_descr, (idxType)mat_C->nnz, (__half *)mat_C->vals, (idxType *)mat_C->rows,
                (idxType *)mat_C->cols, spgemm_descr->mat_descr, (__half *)spgemm_descr->vals,
                (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols, spgemm_descr->mat_info,
                spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_R_16BF: {
            mcsp_bfloat16 h_beta = 1.f;
            mcsp_bfloat16 h_alpha = 1.f;
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num,
                (mcsp_bfloat16 *)&h_alpha, mat_A->mat_descr, (idxType)mat_A->nnz, (mcsp_bfloat16 *)mat_A->vals,
                (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
                (mcsp_bfloat16 *)mat_B->vals, (idxType *)mat_B->rows, (idxType *)mat_B->cols, (mcsp_bfloat16 *)&h_beta,
                mat_C->mat_descr, (idxType)mat_C->nnz, (mcsp_bfloat16 *)mat_C->vals, (idxType *)mat_C->rows,
                (idxType *)mat_C->cols, spgemm_descr->mat_descr, (mcsp_bfloat16 *)spgemm_descr->vals,
                (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols, spgemm_descr->mat_info,
                spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_C_16F: {
            __half2 h_beta = {1.f, 0.f};
            __half2 h_alpha = {1.f, 0.f};
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num,
                (__half2 *)&h_alpha, mat_A->mat_descr, (idxType)mat_A->nnz, (__half2 *)mat_A->vals,
                (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
                (__half2 *)mat_B->vals, (idxType *)mat_B->rows, (idxType *)mat_B->cols, (__half2 *)&h_beta,
                mat_C->mat_descr, (idxType)mat_C->nnz, (__half2 *)mat_C->vals, (idxType *)mat_C->rows,
                (idxType *)mat_C->cols, spgemm_descr->mat_descr, (__half2 *)spgemm_descr->vals,
                (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols, spgemm_descr->mat_info,
                spgemm_descr->buff, include_addition);
            break;
        }
        case MACA_C_16BF: {
            mcsp_bfloat162 h_beta = {1.f, 0.f};
            mcsp_bfloat162 h_alpha = {1.f, 0.f};
            handle->ptr_mode = MCSPARSE_POINTER_MODE_HOST;
            stat = mcspCsrgemmTemplate(
                handle, op_A, op_B, (idxType)mat_A->row_num, (idxType)mat_B->col_num, (idxType)mat_A->col_num,
                (mcsp_bfloat162 *)&h_alpha, mat_A->mat_descr, (idxType)mat_A->nnz, (mcsp_bfloat162 *)mat_A->vals,
                (idxType *)mat_A->rows, (idxType *)mat_A->cols, mat_B->mat_descr, (idxType)mat_B->nnz,
                (mcsp_bfloat162 *)mat_B->vals, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                (mcsp_bfloat162 *)&h_beta, mat_C->mat_descr, (idxType)mat_C->nnz, (mcsp_bfloat162 *)mat_C->vals,
                (idxType *)mat_C->rows, (idxType *)mat_C->cols, spgemm_descr->mat_descr,
                (mcsp_bfloat162 *)spgemm_descr->vals, (idxType *)spgemm_descr->rows, (idxType *)spgemm_descr->cols,
                spgemm_descr->mat_info, spgemm_descr->buff, include_addition);
            break;
        }
#endif
        default:
            stat = MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (stat == MCSP_STATUS_SUCCESS) {
        mcStream_t stream = mcspGetStreamInternal(handle);
        MACA_ASSERT(mcMemcpyAsync(mat_C->rows, spgemm_descr->rows, (mat_C->row_num + 1) * sizeof(idxType),
                                  mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcMemcpyAsync(mat_C->cols, spgemm_descr->cols, spgemm_descr->nnz * sizeof(idxType),
                                  mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }
    handle->ptr_mode = ptr_mode;

    return stat;
}

template <typename idxType>
mcspStatus_t mcspSpGEMMreuse_computeImpl(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                         const void *alpha, mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B,
                                         const void *beta, mcspSpMatDescr_t mat_C, macaDataType compute_type,
                                         mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (spgemm_descr == nullptr || mat_A == nullptr || mat_B == nullptr || mat_C == nullptr || mat_A->rows == nullptr ||
        mat_A->cols == nullptr || mat_A->vals == nullptr || mat_B->rows == nullptr || mat_B->cols == nullptr ||
        mat_B->vals == nullptr || mat_C->rows == nullptr || mat_C->cols == nullptr || mat_C->vals == nullptr ||
        spgemm_descr->rows == nullptr || spgemm_descr->cols == nullptr || spgemm_descr->vals == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (mat_A->row_num < 0 || mat_A->col_num < 0 || mat_A->nnz < 0 || mat_B->row_num < 0 || mat_B->col_num < 0 ||
        mat_B->nnz < 0 || mat_C->row_num < 0 || mat_C->col_num < 0 || mat_C->nnz < 0 || spgemm_descr->row_num < 0 ||
        spgemm_descr->col_num < 0 || spgemm_descr->nnz < 0 || !mat_A->col_num == mat_B->row_num) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    macaDataType data_type = mat_A->valueType;

    constexpr int32_t block_size = 512;
    int32_t n_block = mat_C->row_num;
    constexpr uint32_t sege_size = block_size / WARP_SIZE;
    mcStream_t stream = mcspGetStreamInternal(handle);
    switch (data_type) {
        case MACA_R_32F: {
            MACA_ASSERT(mcMemsetAsync(spgemm_descr->vals, 0, spgemm_descr->nnz * sizeof(float), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
            mcLaunchKernelGGL((mcspSpgemmReuseKernel<block_size, sege_size>), dim3(n_block), dim3(block_size), 0,
                               stream, (idxType *)mat_A->rows, (idxType *)mat_A->cols, (float *)mat_A->vals,
                               mat_A->mat_descr->base, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                               (float *)mat_B->vals, mat_B->mat_descr->base, (idxType *)mat_C->rows,
                               (idxType *)mat_C->cols, (float *)spgemm_descr->vals, mat_C->mat_descr->base);
            break;
        }
        case MACA_R_64F: {
            MACA_ASSERT(mcMemsetAsync(spgemm_descr->vals, 0, spgemm_descr->nnz * sizeof(double), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
            mcLaunchKernelGGL((mcspSpgemmReuseKernel<block_size, sege_size>), dim3(n_block), dim3(block_size), 0,
                               stream, (idxType *)mat_A->rows, (idxType *)mat_A->cols, (double *)mat_A->vals,
                               mat_A->mat_descr->base, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                               (double *)mat_B->vals, mat_B->mat_descr->base, (idxType *)mat_C->rows,
                               (idxType *)mat_C->cols, (double *)spgemm_descr->vals, mat_C->mat_descr->base);
            break;
        }
        case MACA_C_32F: {
            MACA_ASSERT(mcMemsetAsync(spgemm_descr->vals, 0, spgemm_descr->nnz * sizeof(mcspComplexFloat), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
            mcLaunchKernelGGL((mcspSpgemmReuseKernel<block_size, sege_size>), dim3(n_block), dim3(block_size), 0,
                               stream, (idxType *)mat_A->rows, (idxType *)mat_A->cols, (mcspComplexFloat *)mat_A->vals,
                               mat_A->mat_descr->base, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                               (mcspComplexFloat *)mat_B->vals, mat_B->mat_descr->base, (idxType *)mat_C->rows,
                               (idxType *)mat_C->cols, (mcspComplexFloat *)spgemm_descr->vals, mat_C->mat_descr->base);
            break;
        }
        case MACA_C_64F: {
            MACA_ASSERT(mcMemsetAsync(spgemm_descr->vals, 0, spgemm_descr->nnz * sizeof(mcspComplexDouble), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
            mcLaunchKernelGGL((mcspSpgemmReuseKernel<block_size, sege_size>), dim3(n_block), dim3(block_size), 0,
                               stream, (idxType *)mat_A->rows, (idxType *)mat_A->cols, (mcspComplexDouble *)mat_A->vals,
                               mat_A->mat_descr->base, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                               (mcspComplexDouble *)mat_B->vals, mat_B->mat_descr->base, (idxType *)mat_C->rows,
                               (idxType *)mat_C->cols, (mcspComplexDouble *)spgemm_descr->vals, mat_C->mat_descr->base);
            break;
        }
#ifdef __MACA__
        case MACA_R_16F: {
            MACA_ASSERT(mcMemsetAsync(spgemm_descr->vals, 0, spgemm_descr->nnz * sizeof(__half), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
            mcLaunchKernelGGL((mcspSpgemmReuseKernel<block_size, sege_size>), dim3(n_block), dim3(block_size), 0,
                               stream, (idxType *)mat_A->rows, (idxType *)mat_A->cols, (__half *)mat_A->vals,
                               mat_A->mat_descr->base, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                               (__half *)mat_B->vals, mat_B->mat_descr->base, (idxType *)mat_C->rows,
                               (idxType *)mat_C->cols, (__half *)spgemm_descr->vals, mat_C->mat_descr->base);
            break;
        }
        case MACA_R_16BF: {
            MACA_ASSERT(mcMemsetAsync(spgemm_descr->vals, 0, spgemm_descr->nnz * sizeof(mcsp_bfloat16), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
            mcLaunchKernelGGL((mcspSpgemmReuseKernel<block_size, sege_size>), dim3(n_block), dim3(block_size), 0,
                               stream, (idxType *)mat_A->rows, (idxType *)mat_A->cols, (mcsp_bfloat16 *)mat_A->vals,
                               mat_A->mat_descr->base, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                               (mcsp_bfloat16 *)mat_B->vals, mat_B->mat_descr->base, (idxType *)mat_C->rows,
                               (idxType *)mat_C->cols, (mcsp_bfloat16 *)spgemm_descr->vals, mat_C->mat_descr->base);
            break;
        }
        case MACA_C_16F: {
            MACA_ASSERT(mcMemsetAsync(spgemm_descr->vals, 0, spgemm_descr->nnz * sizeof(__half2), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
            mcLaunchKernelGGL((mcspSpgemmReuseKernel<block_size, sege_size>), dim3(n_block), dim3(block_size), 0,
                               stream, (idxType *)mat_A->rows, (idxType *)mat_A->cols, (__half2 *)mat_A->vals,
                               mat_A->mat_descr->base, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                               (__half2 *)mat_B->vals, mat_B->mat_descr->base, (idxType *)mat_C->rows,
                               (idxType *)mat_C->cols, (__half2 *)spgemm_descr->vals, mat_C->mat_descr->base);
            break;
        }
        case MACA_C_16BF: {
            MACA_ASSERT(mcMemsetAsync(spgemm_descr->vals, 0, spgemm_descr->nnz * sizeof(mcsp_bfloat162), stream));
            MACA_ASSERT(mcStreamSynchronize(stream));
            mcLaunchKernelGGL((mcspSpgemmReuseKernel<block_size, sege_size>), dim3(n_block), dim3(block_size), 0,
                               stream, (idxType *)mat_A->rows, (idxType *)mat_A->cols, (mcsp_bfloat162 *)mat_A->vals,
                               mat_A->mat_descr->base, (idxType *)mat_B->rows, (idxType *)mat_B->cols,
                               (mcsp_bfloat162 *)mat_B->vals, mat_B->mat_descr->base, (idxType *)mat_C->rows,
                               (idxType *)mat_C->cols, (mcsp_bfloat162 *)spgemm_descr->vals, mat_C->mat_descr->base);
            break;
        }
#endif
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    n_block = (spgemm_descr->nnz + block_size - 1) / block_size;
    switch (compute_type) {
        case MACA_R_32F: {
            switch (data_type) {
                case MACA_R_32F: {
                    float h_beta = getScalarToHost((float *)beta, handle->ptr_mode);
                    float h_alpha = getScalarToHost((float *)alpha, handle->ptr_mode);
                    denseAxpby<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha, (float *)spgemm_descr->vals,
                                           h_beta, (float *)mat_C->vals);
                    break;
                }
#ifdef __MACA__
                case MACA_R_16F: {
                    float h_beta = getScalarToHost((float *)beta, handle->ptr_mode);
                    float h_alpha = getScalarToHost((float *)alpha, handle->ptr_mode);
                    denseAxpby<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha, (__half *)spgemm_descr->vals,
                                           h_beta, (__half *)mat_C->vals);
                    break;
                }
                case MACA_R_16BF: {
                    float h_beta = getScalarToHost((float *)beta, handle->ptr_mode);
                    float h_alpha = getScalarToHost((float *)alpha, handle->ptr_mode);
                    denseAxpby<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                           (mcsp_bfloat16 *)spgemm_descr->vals, h_beta, (mcsp_bfloat16 *)mat_C->vals);
                    break;
                }
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
            break;
        }
        case MACA_R_64F: {
            double h_beta = getScalarToHost((double *)beta, handle->ptr_mode);
            double h_alpha = getScalarToHost((double *)alpha, handle->ptr_mode);
            denseAxpby<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha, (double *)spgemm_descr->vals, h_beta,
                                   (double *)mat_C->vals);
            break;
        }
        case MACA_C_32F: {
            mcspComplexFloat h_beta = getScalarToHost((mcspComplexFloat *)beta, handle->ptr_mode);
            mcspComplexFloat h_alpha = getScalarToHost((mcspComplexFloat *)alpha, handle->ptr_mode);
            denseAxpby<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha, (mcspComplexFloat *)spgemm_descr->vals,
                                   h_beta, (mcspComplexFloat *)mat_C->vals);
            break;
        }
        case MACA_C_64F: {
            mcspComplexDouble h_beta = getScalarToHost((mcspComplexDouble *)beta, handle->ptr_mode);
            mcspComplexDouble h_alpha = getScalarToHost((mcspComplexDouble *)alpha, handle->ptr_mode);
            denseAxpby<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha, (mcspComplexDouble *)spgemm_descr->vals,
                                   h_beta, (mcspComplexDouble *)mat_C->vals);
            break;
        }
#ifdef __MACA__
        case MACA_R_16F: {
            __half h_beta = getScalarToHost((__half *)beta, handle->ptr_mode);
            __half h_alpha = getScalarToHost((__half *)alpha, handle->ptr_mode);
            denseAxpby<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha, (__half *)spgemm_descr->vals, h_beta,
                                   (__half *)mat_C->vals);
            break;
        }
        case MACA_R_16BF: {
            mcsp_bfloat16 h_beta = getScalarToHost((mcsp_bfloat16 *)beta, handle->ptr_mode);
            mcsp_bfloat16 h_alpha = getScalarToHost((mcsp_bfloat16 *)alpha, handle->ptr_mode);
            denseAxpby<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha, (mcsp_bfloat16 *)spgemm_descr->vals,
                                   h_beta, (mcsp_bfloat16 *)mat_C->vals);
            break;
        }
        case MACA_C_16F: {
            __half2 h_beta = getScalarToHost((__half2 *)beta, handle->ptr_mode);
            __half2 h_alpha = getScalarToHost((__half2 *)alpha, handle->ptr_mode);
            denseAxpbyLowPrecisionComplex<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                                      (__half2 *)spgemm_descr->vals, h_beta, (__half2 *)mat_C->vals);
            break;
        }
        case MACA_C_16BF: {
            mcsp_bfloat162 h_beta = getScalarToHost((mcsp_bfloat162 *)beta, handle->ptr_mode);
            mcsp_bfloat162 h_alpha = getScalarToHost((mcsp_bfloat162 *)alpha, handle->ptr_mode);
            denseAxpbyLowPrecisionComplex<block_size>(stream, n_block, spgemm_descr->nnz, h_alpha,
                                                      (mcsp_bfloat162 *)spgemm_descr->vals, h_beta,
                                                      (mcsp_bfloat162 *)mat_C->vals);
            break;
        }
#endif
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspSpGEMMreuse_workEstimation(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                            mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, mcspSpMatDescr_t mat_C,
                                            mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr,
                                            size_t *buffer_size1, void *external_buffer1) {
    return mcspSpGEMMreuse_workEstimationImpl<mcspInt>(handle, op_A, op_B, mat_A, mat_B, mat_C, alg, spgemm_descr,
                                                       buffer_size1, external_buffer1);
}

mcspStatus_t mcspSpGEMMreuse_nnz(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                 mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, mcspSpMatDescr_t mat_C,
                                 mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr, size_t *buffer_size2,
                                 void *external_buffer2, size_t *nnz_buffer_size, void *nnz_external_buffer,
                                 size_t *row_buffer_size, void *row_external_buffer) {
    return mcspSpGEMMreuse_nnzImpl<mcspInt>(handle, op_A, op_B, mat_A, mat_B, mat_C, alg, spgemm_descr, buffer_size2,
                                            external_buffer2, nnz_buffer_size, nnz_external_buffer, row_buffer_size,
                                            row_external_buffer);
}

mcspStatus_t mcspSpGEMMreuse_copy(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                  mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B, mcspSpMatDescr_t mat_C,
                                  mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr, size_t *col_val_buffer_size,
                                  void *col_val_external_buffer) {
    return mcspSpGEMMreuse_copyImpl<mcspInt>(handle, op_A, op_B, mat_A, mat_B, mat_C, alg, spgemm_descr,
                                             col_val_buffer_size, col_val_external_buffer);
}

mcspStatus_t mcspSpGEMMreuse_compute(mcspHandle_t handle, mcsparseOperation_t op_A, mcsparseOperation_t op_B,
                                     const void *alpha, mcspSpMatDescr_t mat_A, mcspSpMatDescr_t mat_B,
                                     const void *beta, mcspSpMatDescr_t mat_C, macaDataType compute_type,
                                     mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemm_descr) {
    return mcspSpGEMMreuse_computeImpl<mcspInt>(handle, op_A, op_B, alpha, mat_A, mat_B, beta, mat_C, compute_type, alg,
                                                spgemm_descr);
}

#ifdef __cplusplus
}
#endif