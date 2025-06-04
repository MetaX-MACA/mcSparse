#ifndef COMMON_MCSP_INTERNAL_TRANSPOSE_SPARSE_HPP_
#define COMMON_MCSP_INTERNAL_TRANSPOSE_SPARSE_HPP_

#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "internal_interface/mcsp_internal_conversion.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_internal_host_utils.hpp"

static mcspStatus_t mcspCallCsr2CscBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                              const mcspInt *csr_rows, const mcspInt *csr_cols,
                                              mcsparseAction_t csc_action, size_t *buffer_size) {
    return mcspCsr2CscBufferSize(handle, m, n, nnz, csr_rows, csr_cols, csc_action, buffer_size);
}

static mcspStatus_t mcspCallCsr2CscBufferSize(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz,
                                              const int64_t *csr_rows, const int64_t *csr_cols,
                                              mcsparseAction_t csc_action, size_t *buffer_size) {
    return mcspCsr2CscBufferSize64(handle, m, n, nnz, csr_rows, csr_cols, csc_action, buffer_size);
}

static mcspStatus_t mcspCallCoo2Csr(mcspHandle_t handle, const mcspInt *coo_rows, mcspInt nnz, mcspInt m,
                                    mcspInt *csr_rows, mcsparseIndexBase_t idx_base) {
    return mcspCoo2Csr(handle, coo_rows, nnz, m, csr_rows, idx_base);
}

static mcspStatus_t mcspCallCoo2Csr(mcspHandle_t handle, const int64_t *coo_rows, int64_t nnz, int64_t m,
                                    int64_t *csr_rows, mcsparseIndexBase_t idx_base) {
    return mcspCoo2Csr64(handle, coo_rows, nnz, m, csr_rows, idx_base);
}

static mcspStatus_t mcspCallScsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const float *csr_val,
                                     const mcspInt *csr_rows, const mcspInt *csr_cols, float *csc_val,
                                     mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                                     mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspScsr2Csc(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                        idx_base, temp_buffer);
}

static mcspStatus_t mcspCallScsr2Csc(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const float *csr_val,
                                     const int64_t *csr_rows, const int64_t *csr_cols, float *csc_val,
                                     int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                                     mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspScsr2Csc64(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                          idx_base, temp_buffer);
}

static mcspStatus_t mcspCallDcsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const double *csr_val,
                                     const mcspInt *csr_rows, const mcspInt *csr_cols, double *csc_val,
                                     mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                                     mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspDcsr2Csc(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                        idx_base, temp_buffer);
}

static mcspStatus_t mcspCallDcsr2Csc(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const double *csr_val,
                                     const int64_t *csr_rows, const int64_t *csr_cols, double *csc_val,
                                     int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                                     mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspDcsr2Csc64(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                          idx_base, temp_buffer);
}

static mcspStatus_t mcspCallCcsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                     const mcspComplexFloat *csr_val, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                     mcspComplexFloat *csc_val, mcspInt *csc_rows, mcspInt *csc_cols,
                                     mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCcsr2Csc(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                        idx_base, temp_buffer);
}

static mcspStatus_t mcspCallCcsr2Csc(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz,
                                     const mcspComplexFloat *csr_val, const int64_t *csr_rows, const int64_t *csr_cols,
                                     mcspComplexFloat *csc_val, int64_t *csc_rows, int64_t *csc_cols,
                                     mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCcsr2Csc64(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                          idx_base, temp_buffer);
}

static mcspStatus_t mcspCallZcsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                     const mcspComplexDouble *csr_val, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                     mcspComplexDouble *csc_val, mcspInt *csc_rows, mcspInt *csc_cols,
                                     mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspZcsr2Csc(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                        idx_base, temp_buffer);
}

static mcspStatus_t mcspCallZcsr2Csc(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz,
                                     const mcspComplexDouble *csr_val, const int64_t *csr_rows, const int64_t *csr_cols,
                                     mcspComplexDouble *csc_val, int64_t *csc_rows, int64_t *csc_cols,
                                     mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspZcsr2Csc64(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                          idx_base, temp_buffer);
}

#if defined(__MACA__)
static mcspStatus_t mcspCallR16fCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const __half *csr_val,
                                        const mcspInt *csr_rows, const mcspInt *csr_cols, __half *csc_val,
                                        mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                                        mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspR16fCsr2Csc(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                           idx_base, temp_buffer);
}

static mcspStatus_t mcspCallR16fCsr2Csc(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const __half *csr_val,
                                        const int64_t *csr_rows, const int64_t *csr_cols, __half *csc_val,
                                        int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                                        mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspR16fCsr2Csc64(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                             idx_base, temp_buffer);
}

static mcspStatus_t mcspCallC16fCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const __half2 *csr_val,
                                        const mcspInt *csr_rows, const mcspInt *csr_cols, __half2 *csc_val,
                                        mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                                        mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspC16fCsr2Csc(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                           idx_base, temp_buffer);
}

static mcspStatus_t mcspCallC16fCsr2Csc(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const __half2 *csr_val,
                                        const int64_t *csr_rows, const int64_t *csr_cols, __half2 *csc_val,
                                        int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                                        mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspC16fCsr2Csc64(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                             idx_base, temp_buffer);
}

static mcspStatus_t mcspCallR16bfCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                         const mcsp_bfloat16 *csr_val, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                         mcsp_bfloat16 *csc_val, mcspInt *csc_rows, mcspInt *csc_cols,
                                         mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspR16bfCsr2Csc(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                            idx_base, temp_buffer);
}

static mcspStatus_t mcspCallR16bfCsr2Csc(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz,
                                         const mcsp_bfloat16 *csr_val, const int64_t *csr_rows, const int64_t *csr_cols,
                                         mcsp_bfloat16 *csc_val, int64_t *csc_rows, int64_t *csc_cols,
                                         mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspR16bfCsr2Csc64(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                              idx_base, temp_buffer);
}

static mcspStatus_t mcspCallC16bfCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                         const mcsp_bfloat162 *csr_val, const mcspInt *csr_rows,
                                         const mcspInt *csr_cols, mcsp_bfloat162 *csc_val, mcspInt *csc_rows,
                                         mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                                         void *temp_buffer) {
    return mcspC16bfCsr2Csc(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                            idx_base, temp_buffer);
}

static mcspStatus_t mcspCallC16bfCsr2Csc(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz,
                                         const mcsp_bfloat162 *csr_val, const int64_t *csr_rows,
                                         const int64_t *csr_cols, mcsp_bfloat162 *csc_val, int64_t *csc_rows,
                                         int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                                         void *temp_buffer) {
    return mcspC16bfCsr2Csc64(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                              idx_base, temp_buffer);
}

static mcspStatus_t mcspCallR8iCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const int8_t *csr_val,
                                       const mcspInt *csr_rows, const mcspInt *csr_cols, int8_t *csc_val,
                                       mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                                       mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspR8iCsr2Csc(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                          idx_base, temp_buffer);
}

static mcspStatus_t mcspCallR8iCsr2Csc(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const int8_t *csr_val,
                                       const int64_t *csr_rows, const int64_t *csr_cols, int8_t *csc_val,
                                       int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                                       mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspR8iCsr2Csc64(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                            idx_base, temp_buffer);
}

#endif

template <typename valType>
static mcspStatus_t mcspGetCsrTransExBuffersize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                                const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info,
                                                size_t &trans_buffer_size) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcspStatus_t stat =
        mcspCsr2CscBufferSize(handle, m, n, nnz, csr_rows, csr_cols, MCSPARSE_ACTION_NUMERIC, &trans_buffer_size);
    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }

    trans_buffer_size = ALIGN(trans_buffer_size, ALIGNED_SIZE);                             // csr2csc buffer
    info->assist_index_buffer_size = ALIGN((n + 1 + nnz) * sizeof(mcspInt), ALIGNED_SIZE);  // assistant row, col buffer
    info->fixed_length_buffer_size =
        info->assist_index_buffer_size + ALIGN(nnz * sizeof(valType), ALIGNED_SIZE);  // assistant values buffer
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
static mcspStatus_t mcspGetGenericTransExBuffersize(mcspHandle_t handle, mcsparseOperation_t opA,
                                                    macaDataType computeType, const idxType *csr_rows,
                                                    const idxType *csr_cols, mcspSpMatDescr_t matA,
                                                    size_t &trans_buffer_size) {
    if (matA == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (matA->format == MCSPARSE_FORMAT_COO) {
        matA->assist_index_buffer_size = ALIGN((matA->row_num + 1) * sizeof(idxType), ALIGNED_SIZE);  // coo2csr buffer
    }

    size_t vals_size = 0;
    if (opA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        stat = mcspCsr2CscBufferSize(handle, (idxType)matA->row_num, (idxType)matA->col_num, (idxType)matA->nnz,
                                     (idxType *)matA->rows, (idxType *)matA->cols, MCSPARSE_ACTION_NUMERIC,
                                     &trans_buffer_size);
        if (stat != MCSP_STATUS_SUCCESS) {
            return stat;
        }

        trans_buffer_size = ALIGN(trans_buffer_size, ALIGNED_SIZE);  // csr2csc buffer
        matA->assist_index_buffer_size +=
            ALIGN((matA->col_num + 1 + matA->nnz) * sizeof(idxType), ALIGNED_SIZE);     // assistant row, col buffer
        vals_size = ALIGN(matA->nnz * GetMacaDataTypeSize(computeType), ALIGNED_SIZE);  // assistant values buffer
    }
    matA->fixed_length_buffer_size = matA->assist_index_buffer_size + vals_size;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
static mcspStatus_t mcspTransposeSparseByCsr2Csc(mcspHandle_t handle, mcsparseOperation_t op_a, idxType m, idxType n,
                                                 idxType nnz, const void *csr_vals, const idxType *csr_rows,
                                                 const idxType *csr_cols, void *csc_vals, idxType *csc_rows,
                                                 idxType *csc_cols, mcsparseIndexBase_t idx_base, macaDataType a_type,
                                                 void *external_buffer) {
    const int block_size = 512;
    const int n_blocks = CEIL(nnz, block_size);
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    mcStream_t stream = mcspGetStreamInternal(handle);

    switch (a_type) {
        case MACA_R_32F: {
            stat = mcspCallScsr2Csc(handle, (idxType)m, (idxType)n, (idxType)nnz, (float *)csr_vals,
                                    (idxType *)csr_rows, (idxType *)csr_cols, (float *)csc_vals, (idxType *)csc_rows,
                                    (idxType *)csc_cols, MCSPARSE_ACTION_NUMERIC, idx_base, external_buffer);
            break;
        }
        case MACA_R_64F: {
            stat = mcspCallDcsr2Csc(handle, (idxType)m, (idxType)n, (idxType)nnz, (double *)csr_vals,
                                    (idxType *)csr_rows, (idxType *)csr_cols, (double *)csc_vals, (idxType *)csc_rows,
                                    (idxType *)csc_cols, MCSPARSE_ACTION_NUMERIC, idx_base, external_buffer);
            break;
        }
        case MACA_C_32F: {
            stat = mcspCallCcsr2Csc(handle, (idxType)m, (idxType)n, (idxType)nnz, (mcspComplexFloat *)csr_vals,
                                    (idxType *)csr_rows, (idxType *)csr_cols, (mcspComplexFloat *)csc_vals,
                                    (idxType *)csc_rows, (idxType *)csc_cols, MCSPARSE_ACTION_NUMERIC, idx_base,
                                    external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }

            if (op_a == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
                mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream, nnz,
                                   (mcspComplexFloat *)csc_vals);
            }
            break;
        }
        case MACA_C_64F: {
            stat = mcspCallZcsr2Csc(handle, (idxType)m, (idxType)n, (idxType)nnz, (mcspComplexDouble *)csr_vals,
                                    (idxType *)csr_rows, (idxType *)csr_cols, (mcspComplexDouble *)csc_vals,
                                    (idxType *)csc_rows, (idxType *)csc_cols, MCSPARSE_ACTION_NUMERIC, idx_base,
                                    external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }

            if (op_a == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
                mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream, nnz,
                                   (mcspComplexDouble *)csc_vals);
            }
            break;
        }
#if defined(__MACA__)
        case MACA_R_16F: {
            stat =
                mcspCallR16fCsr2Csc(handle, (idxType)m, (idxType)n, (idxType)nnz, (__half *)csr_vals,
                                    (idxType *)csr_rows, (idxType *)csr_cols, (__half *)csc_vals, (idxType *)csc_rows,
                                    (idxType *)csc_cols, MCSPARSE_ACTION_NUMERIC, idx_base, external_buffer);
            break;
        }
        case MACA_C_16F: {
            stat =
                mcspCallC16fCsr2Csc(handle, (idxType)m, (idxType)n, (idxType)nnz, (__half2 *)csr_vals,
                                    (idxType *)csr_rows, (idxType *)csr_cols, (__half2 *)csc_vals, (idxType *)csc_rows,
                                    (idxType *)csc_cols, MCSPARSE_ACTION_NUMERIC, idx_base, external_buffer);

            if (op_a == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
                mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream, nnz,
                                   (__half2 *)csc_vals);
            }
            break;
        }
        case MACA_R_16BF: {
            stat = mcspCallR16bfCsr2Csc(handle, (idxType)m, (idxType)n, (idxType)nnz, (mcsp_bfloat16 *)csr_vals,
                                        (idxType *)csr_rows, (idxType *)csr_cols, (mcsp_bfloat16 *)csc_vals,
                                        (idxType *)csc_rows, (idxType *)csc_cols, MCSPARSE_ACTION_NUMERIC, idx_base,
                                        external_buffer);
            break;
        }
        case MACA_C_16BF: {
            stat = mcspCallC16bfCsr2Csc(handle, (idxType)m, (idxType)n, (idxType)nnz, (mcsp_bfloat162 *)csr_vals,
                                        (idxType *)csr_rows, (idxType *)csr_cols, (mcsp_bfloat162 *)csc_vals,
                                        (idxType *)csc_rows, (idxType *)csc_cols, MCSPARSE_ACTION_NUMERIC, idx_base,
                                        external_buffer);

            if (op_a == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
                mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream, nnz,
                                   (mcsp_bfloat162 *)csc_vals);
            }
            break;
        }
        case MACA_R_8I: {
            stat = mcspCallR8iCsr2Csc(handle, (idxType)m, (idxType)n, (idxType)nnz, (int8_t *)csr_vals,
                                      (idxType *)csr_rows, (idxType *)csr_cols, (int8_t *)csc_vals, (idxType *)csc_rows,
                                      (idxType *)csc_cols, MCSPARSE_ACTION_NUMERIC, idx_base, external_buffer);
            break;
        }
#endif
        default:
            stat = MCSP_STATUS_NOT_IMPLEMENTED;
    }
    return stat;
}

template <typename valType>
static mcspStatus_t mcspTransposeSpMatForRawAPI(mcspHandle_t handle, mcsparseOperation_t transA, mcspInt m, mcspInt n,
                                                mcspInt nnz, const valType *csr_vals, const mcspInt *csr_rows,
                                                const mcspInt *csr_cols, mcsparseIndexBase_t idx_base,
                                                mcspMatInfo_t info, void *pBuffer) {
    if (info == nullptr || pBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcspInt *buffer_head = reinterpret_cast<mcspInt *>(pBuffer);
    info->to_csc_cols = buffer_head;
    buffer_head += m + 1;
    info->to_csc_rows = buffer_head;
    info->to_csc_vals = reinterpret_cast<valType *>(reinterpret_cast<char *>(pBuffer) + info->assist_index_buffer_size);
    void *csr2csc_buffer = (void *)(reinterpret_cast<char *>(pBuffer) + info->fixed_length_buffer_size);

    macaDataType compute_type = GetMacaDataTypeFromTypename<valType>();
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    stat = mcspTransposeSparseByCsr2Csc(handle, transA, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (valType *)csr_vals,
                                        (mcspInt *)csr_rows, (mcspInt *)csr_cols, (valType *)info->to_csc_vals,
                                        (mcspInt *)info->to_csc_rows, (mcspInt *)info->to_csc_cols, idx_base,
                                        compute_type, csr2csc_buffer);
    return stat;
}

template <typename idxType>
static mcspStatus_t mcspTransposeSpMatForGenericAPI(mcspHandle_t handle, mcsparseOperation_t opA,
                                                    macaDataType computeType, mcspSpMatDescr_t matA,
                                                    void *externalBuffer) {
    if (matA == nullptr || externalBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType *buffer_head = reinterpret_cast<idxType *>(externalBuffer);
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (matA->format == MCSPARSE_FORMAT_COO) {
        matA->to_csr_rows = (void *)buffer_head;
        buffer_head += matA->row_num + 1;
        stat = mcspCallCoo2Csr(handle, (idxType *)matA->rows, (idxType)matA->nnz, (idxType)matA->row_num,
                               (idxType *)matA->to_csr_rows, matA->idxBase);
        if (stat != MCSP_STATUS_SUCCESS) {
            return stat;
        }
    }

    if (opA != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        matA->to_csc_cols = buffer_head;
        buffer_head += matA->row_num + 1;
        matA->to_csc_rows = buffer_head;
        matA->to_csc_vals = (void *)(reinterpret_cast<char *>(externalBuffer) + matA->assist_index_buffer_size);
        void *csr2csc_buffer = (void *)(reinterpret_cast<char *>(externalBuffer) + matA->fixed_length_buffer_size);
        stat = mcspTransposeSparseByCsr2Csc(handle, opA, (idxType)matA->row_num, (idxType)matA->col_num,
                                            (idxType)matA->nnz, matA->vals, (idxType *)matA->to_csr_rows,
                                            (idxType *)matA->cols, matA->to_csc_vals, (idxType *)matA->to_csc_rows,
                                            (idxType *)matA->to_csc_cols, matA->idxBase, computeType, csr2csc_buffer);
    }
    return stat;
}

template <typename valType>
static mcspStatus_t mcspGetBsrTransExBuffersize(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                                mcspInt row_block_dim, mcspInt col_block_dim, const mcspInt *bsr_rows,
                                                const mcspInt *bsr_cols, const valType *bsr_vals, mcspMatInfo_t info,
                                                size_t &trans_buffer_size) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    mcspStatus_t stat = mcspSgebsr2gebsc_bufferSizeExt(handle, mb, nb, nnzb, (float *)bsr_vals, bsr_rows, bsr_cols,
                                                       row_block_dim, col_block_dim, &trans_buffer_size);
    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }
    trans_buffer_size = ALIGN(trans_buffer_size, ALIGNED_SIZE);  // gebsr2gebsc buffer
    info->assist_index_buffer_size =
        ALIGN((nb + 1 + nnzb) * sizeof(mcspInt), ALIGNED_SIZE);  // assistant row, col buffer
    info->fixed_length_buffer_size =
        info->assist_index_buffer_size +
        ALIGN(nnzb * row_block_dim * col_block_dim * sizeof(valType), ALIGNED_SIZE);  // assistant values buffer
    return stat;
}

static mcspStatus_t mcspTransposeBsr(mcspHandle_t handle, mcsparseOperation_t transA, mcspInt mb, mcspInt nb,
                                     mcspInt nnzb, mcspInt row_block_dim, mcspInt col_block_dim,
                                     const mcspInt *bsr_rows, const mcspInt *bsr_cols, const void *bsr_vals,
                                     mcspInt *bsrt_rows, mcspInt *bsrt_cols, void *bsrt_vals,
                                     mcsparseIndexBase_t idx_base, mcsparseDirection_t bsrt_dir,
                                     macaDataType compute_type, void *external_buffer) {
    constexpr int block_size = 512;
    const int n_blocks = CEIL(nnzb * row_block_dim * col_block_dim, block_size);
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    mcStream_t stream = mcspGetStreamInternal(handle);
    switch (compute_type) {
        case MACA_R_32F: {
            stat = mcspSgebsr2gebsc(handle, mb, nb, nnzb, (float *)bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                    col_block_dim, (float *)bsrt_vals, bsrt_cols, bsrt_rows, MCSPARSE_ACTION_NUMERIC,
                                    idx_base, external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }
            mcLaunchKernelGGL((transposeBsrBlockKernel<block_size>), dim3(n_blocks), dim3(block_size), 0, stream,
                               nnzb * row_block_dim * col_block_dim, row_block_dim * col_block_dim, row_block_dim,
                               col_block_dim, (float *)bsrt_vals, bsrt_dir);
            break;
        }
        case MACA_R_64F: {
            stat = mcspDgebsr2gebsc(handle, mb, nb, nnzb, (double *)bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                    col_block_dim, (double *)bsrt_vals, bsrt_cols, bsrt_rows, MCSPARSE_ACTION_NUMERIC,
                                    idx_base, external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }
            mcLaunchKernelGGL((transposeBsrBlockKernel<block_size>), dim3(n_blocks), dim3(block_size), 0, stream,
                               nnzb * row_block_dim * col_block_dim, row_block_dim * col_block_dim, row_block_dim,
                               col_block_dim, (double *)bsrt_vals, bsrt_dir);
            break;
        }
        case MACA_C_32F: {
            stat = mcspCgebsr2gebsc(handle, mb, nb, nnzb, (mcFloatComplex *)bsr_vals, bsr_rows, bsr_cols, row_block_dim,
                                    col_block_dim, (mcFloatComplex *)bsrt_vals, bsrt_cols, bsrt_rows,
                                    MCSPARSE_ACTION_NUMERIC, idx_base, external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }
            mcLaunchKernelGGL((transposeBsrBlockKernel<block_size>), dim3(n_blocks), dim3(block_size), 0, stream,
                               nnzb * row_block_dim * col_block_dim, row_block_dim * col_block_dim, row_block_dim,
                               col_block_dim, (mcspComplexFloat *)bsrt_vals, bsrt_dir);
            if (transA == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
                mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                   nnzb * row_block_dim * col_block_dim, (mcspComplexFloat *)bsrt_vals);
            }
            break;
        }
        case MACA_C_64F: {
            stat = mcspZgebsr2gebsc(handle, mb, nb, nnzb, (mcspComplexDouble *)bsr_vals, bsr_rows, bsr_cols,
                                    row_block_dim, col_block_dim, (mcspComplexDouble *)bsrt_vals, bsrt_cols, bsrt_rows,
                                    MCSPARSE_ACTION_NUMERIC, idx_base, external_buffer);
            if (stat != MCSP_STATUS_SUCCESS) {
                break;
            }
            mcLaunchKernelGGL((transposeBsrBlockKernel<block_size>), dim3(n_blocks), dim3(block_size), 0, stream,
                               nnzb * row_block_dim * col_block_dim, row_block_dim * col_block_dim, row_block_dim,
                               col_block_dim, (mcspComplexDouble *)bsrt_vals, bsrt_dir);
            if (transA == MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
                mcLaunchKernelGGL((oppositeImagePartInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream,
                                   nnzb * row_block_dim * col_block_dim, (mcspComplexDouble *)bsrt_vals);
            }
            break;
        }
        default:
            stat = MCSP_STATUS_NOT_IMPLEMENTED;
    }
    return stat;
}

template <typename valType>
static mcspStatus_t mcspRawTransposeBsr(mcspHandle_t handle, mcsparseOperation_t transA, mcspInt mb, mcspInt nb,
                                        mcspInt nnzb, mcspInt row_block_dim, mcspInt col_block_dim,
                                        const mcspInt *bsr_rows, const mcspInt *bsr_cols, const valType *bsr_vals,
                                        mcsparseIndexBase_t idx_base, mcspMatInfo_t info, void *pBuffer) {
    if (info == nullptr || pBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    mcspInt *buffer_head = reinterpret_cast<mcspInt *>(pBuffer);
    info->bsrt_rows = buffer_head;
    buffer_head += nb + 1;
    info->bsrt_cols = buffer_head;
    info->bsrt_vals = reinterpret_cast<valType *>(reinterpret_cast<char *>(pBuffer) + info->assist_index_buffer_size);
    void *bsr2bsc_buffer = (void *)(reinterpret_cast<char *>(pBuffer) + info->fixed_length_buffer_size);
    macaDataType compute_type = GetMacaDataTypeFromTypename<valType>();
    return mcspTransposeBsr(handle, transA, mb, nb, nnzb, row_block_dim, col_block_dim, bsr_rows, bsr_cols, bsr_vals,
                            (mcspInt *)info->bsrt_rows, (mcspInt *)info->bsrt_cols, info->bsrt_vals, idx_base,
                            info->bsrt_dir, compute_type, bsr2bsc_buffer);
}

template <typename idxType>
mcspStatus_t CalculateAssistBufferSizeForTranspose(mcspHandle_t handle, mcspSpMatDescr_t matA, macaDataType a_type,
                                                   size_t *buffer_size,
                                                   mcsparseFormat_t source_format = MCSPARSE_FORMAT_CSR) {
    size_t trans_buffer_size = 0;
    mcspStatus_t stat;

    idxType working_row_num = (idxType)matA->row_num;
    idxType working_col_num = (idxType)matA->col_num;
    idxType *working_rows = (idxType *)matA->rows;
    idxType *working_cols = (idxType *)matA->cols;
    if (source_format == MCSPARSE_FORMAT_CSC) {
        std::swap(working_row_num, working_col_num);
        std::swap(working_rows, working_cols);
    }

    stat = mcspCallCsr2CscBufferSize(handle, working_row_num, working_col_num, (idxType)matA->nnz, working_rows,
                                     working_cols, MCSPARSE_ACTION_NUMERIC, &trans_buffer_size);

    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }
    trans_buffer_size = ALIGN(trans_buffer_size, ALIGNED_SIZE);  // csr2csc buffersize

    size_t vals_size = GetMacaDataTypeSize(a_type);
    matA->assist_index_buffer_size = ALIGN((matA->row_num + matA->col_num + 2 + matA->nnz) * sizeof(idxType),
                                           ALIGNED_SIZE);  // ->csr_rows ->csc_cols ->csc_rows
    matA->fixed_length_buffer_size =
        matA->assist_index_buffer_size + ALIGN(matA->nnz * vals_size, ALIGNED_SIZE);  // ->csc_vals
    *buffer_size = matA->fixed_length_buffer_size + trans_buffer_size;                // max_of(trans_buffer, 0)
    matA->is_buffersize_called = 1;
    return MCSP_STATUS_SUCCESS;
}
#endif
