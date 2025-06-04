#include "common/mcsp_types.h"
#include "internal_interface/mcsp_internal_generic.h"
#include "internal_interface/mcsp_internal_level1.h"
#include "mcsp_config.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

mcspStatus_t mcspSpVV_bufferSize(mcspHandle_t handle, mcsparseOperation_t op_x, mcspSpVecDescr_t vec_x,
                                 mcspDnVecDescr_t vec_y, void *result, macaDataType compute_type, size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (vec_y == nullptr || vec_x == nullptr || result == nullptr || buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    macaDataType xType = vec_x->valueType;
    macaDataType yType = vec_y->valueType;
    if (xType != yType) {
        return MCSP_STATUS_TYPE_MISMATCH;
    }
    *buffer_size = MIN_BUFFER_SIZE;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpVV(mcspHandle_t handle, mcsparseOperation_t op_x, mcspSpVecDescr_t vec_x, mcspDnVecDescr_t vec_y,
                      void *result, macaDataType compute_type, void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (vec_y == nullptr || vec_x == nullptr || result == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    macaDataType xType = vec_x->valueType;
    macaDataType yType = vec_y->valueType;
    if (xType != yType) {
        return MCSP_STATUS_TYPE_MISMATCH;
    }
    switch (compute_type) {
        case MACA_R_32F: {
            switch (xType) {
                case MACA_R_32F: {
                    return mcspSdoti(handle, vec_x->nnz, (float *)vec_x->values, (mcspInt *)vec_x->indices,
                                     (float *)vec_y->values, (float *)result, vec_x->idxBase);
                }
#if defined(__MACA__)
                case MACA_R_16F: {
                    return mcspR16fR32fDoti(handle, vec_x->nnz, (__half *)vec_x->values, (mcspInt *)vec_x->indices,
                                            (__half *)vec_y->values, (float *)result, vec_x->idxBase);
                }
                case MACA_R_16BF: {
                    return mcspR16bfR32fDoti(handle, vec_x->nnz, (mcsp_bfloat16 *)vec_x->values,
                                             (mcspInt *)vec_x->indices, (mcsp_bfloat16 *)vec_y->values, (float *)result,
                                             vec_x->idxBase);
                }
                case MACA_R_8I: {
                    return mcspR8iR32fDoti(handle, vec_x->nnz, (int8_t *)vec_x->values, (mcspInt *)vec_x->indices,
                                           (int8_t *)vec_y->values, (float *)result, vec_x->idxBase);
                }
                default: {
                    return MCSP_STATUS_NOT_IMPLEMENTED;
                }
#endif
            }
        }
        case MACA_R_64F: {
            return mcspDdoti(handle, vec_x->nnz, (double *)vec_x->values, (mcspInt *)vec_x->indices,
                             (double *)vec_y->values, (double *)result, vec_x->idxBase);
        }
        case MACA_C_32F: {
            switch (xType) {
                case MACA_C_32F: {
                    switch (op_x) {
                        case MCSPARSE_OPERATION_NON_TRANSPOSE: {
                            return mcspCdoti(handle, vec_x->nnz, (mcspComplexFloat *)vec_x->values,
                                             (mcspInt *)vec_x->indices, (mcspComplexFloat *)vec_y->values,
                                             (mcspComplexFloat *)result, vec_x->idxBase);
                        }
                        case MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE: {
                            return mcspCdotci(handle, vec_x->nnz, (mcspComplexFloat *)vec_x->values,
                                              (mcspInt *)vec_x->indices, (mcspComplexFloat *)vec_y->values,
                                              (mcspComplexFloat *)result, vec_x->idxBase);
                        }
                        default: {
                            return MCSP_STATUS_NOT_IMPLEMENTED;
                        }
                    }
                }
#if defined(__MACA__)
                case MACA_C_16F: {
                    switch (op_x) {
                        case MCSPARSE_OPERATION_NON_TRANSPOSE: {
                            return mcspC16fC32fDoti(handle, vec_x->nnz, (__half2 *)vec_x->values,
                                                    (mcspInt *)vec_x->indices, (__half2 *)vec_y->values,
                                                    (mcspComplexFloat *)result, vec_x->idxBase);
                        }
                        case MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE: {
                            return mcspC16fC32fDotci(handle, vec_x->nnz, (__half2 *)vec_x->values,
                                                     (mcspInt *)vec_x->indices, (__half2 *)vec_y->values,
                                                     (mcspComplexFloat *)result, vec_x->idxBase);
                        }
                        default: {
                            return MCSP_STATUS_NOT_IMPLEMENTED;
                        }
                    }
                }
                case MACA_C_16BF: {
                    switch (op_x) {
                        case MCSPARSE_OPERATION_NON_TRANSPOSE: {
                            return mcspC16bfC32fDoti(handle, vec_x->nnz, (mcsp_bfloat162 *)vec_x->values,
                                                     (mcspInt *)vec_x->indices, (mcsp_bfloat162 *)vec_y->values,
                                                     (mcspComplexFloat *)result, vec_x->idxBase);
                        }
                        case MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE: {
                            return mcspC16bfC32fDotci(handle, vec_x->nnz, (mcsp_bfloat162 *)vec_x->values,
                                                      (mcspInt *)vec_x->indices, (mcsp_bfloat162 *)vec_y->values,
                                                      (mcspComplexFloat *)result, vec_x->idxBase);
                        }
                        default: {
                            return MCSP_STATUS_NOT_IMPLEMENTED;
                        }
                    }
                }
#endif
                default: {
                    return MCSP_STATUS_NOT_IMPLEMENTED;
                }
            }
        }
        case MACA_C_64F: {
            switch (op_x) {
                case MCSPARSE_OPERATION_NON_TRANSPOSE: {
                    return mcspZdoti(handle, vec_x->nnz, (mcspComplexDouble *)vec_x->values, (mcspInt *)vec_x->indices,
                                     (mcspComplexDouble *)vec_y->values, (mcspComplexDouble *)result, vec_x->idxBase);
                }
                case MCSPARSE_OPERATION_CONJUGATE_TRANSPOSE: {
                    return mcspZdotci(handle, vec_x->nnz, (mcspComplexDouble *)vec_x->values, (mcspInt *)vec_x->indices,
                                      (mcspComplexDouble *)vec_y->values, (mcspComplexDouble *)result, vec_x->idxBase);
                }
                default: {
                    return MCSP_STATUS_NOT_IMPLEMENTED;
                }
            }
        }
#if defined(__MACA__)
        case MACA_R_32I: {
            return mcspR8iR32iDoti(handle, vec_x->nnz, (int8_t *)vec_x->values, (mcspInt *)vec_x->indices,
                                   (int8_t *)vec_y->values, (int32_t *)result, vec_x->idxBase);
        }
#endif
        default: {
            return MCSP_STATUS_NOT_IMPLEMENTED;
        }
    }
}