#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "internal_interface/mcsp_internal_conversion.h"
#include "internal_interface/mcsp_internal_level1.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

//------------------------------------------------------------------------------
// ### Generic API functions ###

template <typename idxType>
mcspStatus_t mcspGatherImpl(mcspHandle_t handle, mcspDnVecDescr_t vecY, mcspSpVecDescr_t vecX) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (vecY == nullptr || vecX == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (vecY->valueType != vecX->valueType) {
        return MCSP_STATUS_TYPE_MISMATCH;
    }

    idxType nnz = vecX->nnz;
    macaDataType type = vecX->valueType;
    switch (type) {
        case MACA_R_32F:
            return mcspSgthr(handle, (idxType)nnz, (float *)vecY->values, (float *)vecX->values,
                             (idxType *)vecX->indices, vecX->idxBase);
        case MACA_R_64F:
            return mcspDgthr(handle, (idxType)nnz, (double *)vecY->values, (double *)vecX->values,
                             (idxType *)vecX->indices, vecX->idxBase);
        case MACA_C_32F:
            return mcspCgthr(handle, (idxType)nnz, (mcspComplexFloat *)vecY->values, (mcspComplexFloat *)vecX->values,
                             (idxType *)vecX->indices, vecX->idxBase);
        case MACA_C_64F:
            return mcspZgthr(handle, (idxType)nnz, (mcspComplexDouble *)vecY->values, (mcspComplexDouble *)vecX->values,
                             (idxType *)vecX->indices, vecX->idxBase);

#if defined(__MACA__)
        case MACA_R_16F: {
            return mcspR16Fgthr(handle, (idxType)nnz, (__half *)vecY->values, (__half *)vecX->values,
                                (idxType *)vecX->indices, vecX->idxBase);
        }
#endif

#ifdef __MACA__
        case MACA_C_16F: {
            return mcspC16Fgthr(handle, (idxType)nnz, (__half2 *)vecY->values, (__half2 *)vecX->values,
                                (idxType *)vecX->indices, vecX->idxBase);
        }
        case MACA_R_16BF: {
            return mcspR16BFgthr(handle, (idxType)nnz, (mcsp_bfloat16 *)vecY->values, (mcsp_bfloat16 *)vecX->values,
                                 (idxType *)vecX->indices, vecX->idxBase);
        }
        case MACA_C_16BF: {
            return mcspC16BFgthr(handle, (idxType)nnz, (mcsp_bfloat162 *)vecY->values, (mcsp_bfloat162 *)vecX->values,
                                 (idxType *)vecX->indices, vecX->idxBase);
        }
#endif

        default:
            return MCSP_STATUS_TYPE_MISMATCH;
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspScatterImpl(mcspHandle_t handle, mcspSpVecDescr_t vecX, mcspDnVecDescr_t vecY) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (vecY == nullptr || vecX == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (vecY->valueType != vecX->valueType) {
        return MCSP_STATUS_TYPE_MISMATCH;
    }

    idxType nnz = vecX->nnz;
    macaDataType type = vecX->valueType;
    switch (type) {
        case MACA_R_32F: {
            return mcspSsctr(handle, (idxType)nnz, (float *)vecX->values, (idxType *)vecX->indices,
                             (float *)vecY->values, vecX->idxBase);
        }
        case MACA_R_64F: {
            return mcspDsctr(handle, (idxType)nnz, (double *)vecX->values, (idxType *)vecX->indices,
                             (double *)vecY->values, vecX->idxBase);
        }
        case MACA_C_32F: {
            return mcspCsctr(handle, (idxType)nnz, (mcspComplexFloat *)vecX->values, (idxType *)vecX->indices,
                             (mcspComplexFloat *)vecY->values, vecX->idxBase);
        }
        case MACA_C_64F: {
            return mcspZsctr(handle, (idxType)nnz, (mcspComplexDouble *)vecX->values, (idxType *)vecX->indices,
                             (mcspComplexDouble *)vecY->values, vecX->idxBase);
        }

#if defined(__MACA__)
        case MACA_R_16F: {
            return mcspR16Fsctr(handle, (idxType)nnz, (__half *)vecX->values, (idxType *)vecX->indices,
                                (__half *)vecY->values, vecX->idxBase);
        }
#endif

#ifdef __MACA__
        case MACA_C_16F: {
            return mcspC16Fsctr(handle, (idxType)nnz, (__half2 *)vecX->values, (idxType *)vecX->indices,
                                (__half2 *)vecY->values, vecX->idxBase);
        }
        case MACA_R_16BF: {
            return mcspR16BFsctr(handle, (idxType)nnz, (mcsp_bfloat16 *)vecX->values, (idxType *)vecX->indices,
                                 (mcsp_bfloat16 *)vecY->values, vecX->idxBase);
        }
        case MACA_C_16BF: {
            return mcspC16BFsctr(handle, (idxType)nnz, (mcsp_bfloat162 *)vecX->values, (idxType *)vecX->indices,
                                 (mcsp_bfloat162 *)vecY->values, vecX->idxBase);
        }
#endif
        default:
            return MCSP_STATUS_TYPE_MISMATCH;
    }
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspRotImpl(mcspHandle_t handle, const void *c_coeff, const void *s_coeff, mcspSpVecDescr_t vecX,
                         mcspDnVecDescr_t vecY) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (vecY == nullptr || vecX == nullptr || c_coeff == nullptr || s_coeff == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (vecY->valueType != vecX->valueType) {
        return MCSP_STATUS_TYPE_MISMATCH;
    }

    idxType nnz = vecX->nnz;
    macaDataType type = vecX->valueType;
    switch (type) {
        case MACA_R_32F: {
            return mcspSroti(handle, (idxType)nnz, (float *)vecX->values, (idxType *)vecX->indices,
                             (float *)vecY->values, (float *)c_coeff, (float *)s_coeff, vecX->idxBase);
        }
        case MACA_R_64F: {
            return mcspDroti(handle, (idxType)nnz, (double *)vecX->values, (idxType *)vecX->indices,
                             (double *)vecY->values, (double *)c_coeff, (double *)s_coeff, vecX->idxBase);
        }
        case MACA_C_32F: {
            return mcspCroti(handle, (idxType)nnz, (mcspComplexFloat *)vecX->values, (idxType *)vecX->indices,
                             (mcspComplexFloat *)vecY->values, (mcspComplexFloat *)c_coeff, (mcspComplexFloat *)s_coeff,
                             vecX->idxBase);
        }
        case MACA_C_64F: {
            return mcspZroti(handle, (idxType)nnz, (mcspComplexDouble *)vecX->values, (idxType *)vecX->indices,
                             (mcspComplexDouble *)vecY->values, (mcspComplexDouble *)c_coeff,
                             (mcspComplexDouble *)s_coeff, vecX->idxBase);
        }
#if defined(__MACA__)
        case MACA_R_16F: {
            return mcspR16fRoti(handle, (idxType)nnz, (__half *)vecX->values, (idxType *)vecX->indices,
                                (__half *)vecY->values, (float *)c_coeff, (float *)s_coeff, vecX->idxBase);
        }
#endif
#ifdef __MACA__
        case MACA_R_16BF: {
            return mcspR16bfRoti(handle, (idxType)nnz, (mcsp_bfloat16 *)vecX->values, (idxType *)vecX->indices,
                                 (mcsp_bfloat16 *)vecY->values, (float *)c_coeff, (float *)s_coeff, vecX->idxBase);
        }
        case MACA_C_16F: {
            return mcspC16fRoti(handle, (idxType)nnz, (__half2 *)vecX->values, (idxType *)vecX->indices,
                                (__half2 *)vecY->values, (mcspComplexFloat *)c_coeff, (mcspComplexFloat *)s_coeff,
                                vecX->idxBase);
        }
        case MACA_C_16BF: {
            return mcspC16bfRoti(handle, (idxType)nnz, (mcsp_bfloat162 *)vecX->values, (idxType *)vecX->indices,
                                 (mcsp_bfloat162 *)vecY->values, (mcspComplexFloat *)c_coeff,
                                 (mcspComplexFloat *)s_coeff, vecX->idxBase);
        }
#endif
        default:
            return MCSP_STATUS_TYPE_MISMATCH;
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSparseToDense_bufferSizeImpl(mcspHandle_t handle, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB,
                                              mcsparseSparseToDenseAlg_t alg, size_t *bufferSize) {
    *bufferSize = MIN_BUFFER_SIZE;
    return MCSP_STATUS_SUCCESS;
}

// The function converts the sparse matrix matA in CSR, CSC, or COO format into its dense representation matB
template <typename idxType>
mcspStatus_t mcspSparseToDenseImpl(mcspHandle_t handle, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB,
                                   mcsparseSparseToDenseAlg_t alg, void *externalBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (matA == nullptr || matB == nullptr || matA->mat_descr == nullptr || externalBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (matA->valueType != matB->valueType ||
        !(matA->format == MCSPARSE_FORMAT_COO || matA->format == MCSPARSE_FORMAT_CSR ||
          matA->format == MCSPARSE_FORMAT_CSC)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    switch (matA->format) {
        case MCSPARSE_FORMAT_CSR: {
            switch (matA->valueType) {
                case MACA_R_32F: {
                    return mcspSgenericCsr2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (float *)matA->vals, (idxType *)matA->rows,
                                                 (idxType *)matA->cols, (float *)matB->values, (idxType)matB->ld,
                                                 matB->order);
                }
                case MACA_R_64F: {
                    return mcspDgenericCsr2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (double *)matA->vals, (idxType *)matA->rows,
                                                 (idxType *)matA->cols, (double *)matB->values, (idxType)matB->ld,
                                                 matB->order);
                }
                case MACA_C_32F: {
                    return mcspCgenericCsr2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexFloat *)matA->vals, (idxType *)matA->rows,
                                                 (idxType *)matA->cols, (mcspComplexFloat *)matB->values,
                                                 (idxType)matB->ld, matB->order);
                }
                case MACA_C_64F: {
                    return mcspZgenericCsr2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexDouble *)matA->vals,
                                                 (idxType *)matA->rows, (idxType *)matA->cols,
                                                 (mcspComplexDouble *)matB->values, (idxType)matB->ld, matB->order);
                }
#if defined(__MACA__)
                case MACA_R_16F: {
                    return mcspR16FgenericCsr2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half *)matA->vals, (idxType *)matA->rows,
                                                    (idxType *)matA->cols, (__half *)matB->values, (idxType)matB->ld,
                                                    matB->order);
                }
#endif
#ifdef __MACA__
                case MACA_C_16F: {
                    return mcspC16FgenericCsr2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half2 *)matA->vals, (idxType *)matA->rows,
                                                    (idxType *)matA->cols, (__half2 *)matB->values, (idxType)matB->ld,
                                                    matB->order);
                }
                case MACA_R_16BF: {
                    return mcspR16BgenericCsr2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (mcsp_bfloat16 *)matA->vals, (idxType *)matA->rows,
                                                    (idxType *)matA->cols, (mcsp_bfloat16 *)matB->values,
                                                    (idxType)matB->ld, matB->order);
                }
                case MACA_C_16BF: {
                    return mcspC16BgenericCsr2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (mcsp_bfloat162 *)matA->vals,
                                                    (idxType *)matA->rows, (idxType *)matA->cols,
                                                    (mcsp_bfloat162 *)matB->values, (idxType)matB->ld, matB->order);
                }
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_COO: {
            switch (matA->valueType) {
                case MACA_R_32F: {
                    return mcspSgenericCoo2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 (idxType)matA->nnz, matA->mat_descr, (float *)matA->vals,
                                                 (idxType *)matA->rows, (idxType *)matA->cols, (float *)matB->values,
                                                 (idxType)matB->ld, matB->order);
                }
                case MACA_R_64F: {
                    return mcspDgenericCoo2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 (idxType)matA->nnz, matA->mat_descr, (double *)matA->vals,
                                                 (idxType *)matA->rows, (idxType *)matA->cols, (double *)matB->values,
                                                 (idxType)matB->ld, matB->order);
                }
                case MACA_C_32F: {
                    return mcspCgenericCoo2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 (idxType)matA->nnz, matA->mat_descr, (mcspComplexFloat *)matA->vals,
                                                 (idxType *)matA->rows, (idxType *)matA->cols,
                                                 (mcspComplexFloat *)matB->values, (idxType)matB->ld, matB->order);
                }
                case MACA_C_64F: {
                    return mcspZgenericCoo2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 (idxType)matA->nnz, matA->mat_descr, (mcspComplexDouble *)matA->vals,
                                                 (idxType *)matA->rows, (idxType *)matA->cols,
                                                 (mcspComplexDouble *)matB->values, (idxType)matB->ld, matB->order);
                }
#if defined(__MACA__)
                case MACA_R_16F: {
                    return mcspR16FgenericCoo2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    (idxType)matA->nnz, matA->mat_descr, (__half *)matA->vals,
                                                    (idxType *)matA->rows, (idxType *)matA->cols,
                                                    (__half *)matB->values, (idxType)matB->ld, matB->order);
                }
#endif
#ifdef __MACA__
                case MACA_C_16F: {
                    return mcspC16FgenericCoo2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    (idxType)matA->nnz, matA->mat_descr, (__half2 *)matA->vals,
                                                    (idxType *)matA->rows, (idxType *)matA->cols,
                                                    (__half2 *)matB->values, (idxType)matB->ld, matB->order);
                }
                case MACA_R_16BF: {
                    return mcspR16BgenericCoo2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    (idxType)matA->nnz, matA->mat_descr, (mcsp_bfloat16 *)matA->vals,
                                                    (idxType *)matA->rows, (idxType *)matA->cols,
                                                    (mcsp_bfloat16 *)matB->values, (idxType)matB->ld, matB->order);
                }
                case MACA_C_16BF: {
                    return mcspC16BgenericCoo2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    (idxType)matA->nnz, matA->mat_descr, (mcsp_bfloat162 *)matA->vals,
                                                    (idxType *)matA->rows, (idxType *)matA->cols,
                                                    (mcsp_bfloat162 *)matB->values, (idxType)matB->ld, matB->order);
                }
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_CSC: {
            switch (matA->valueType) {
                case MACA_R_32F: {
                    return mcspSgenericCsc2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (float *)matA->vals, (idxType *)matA->rows,
                                                 (idxType *)matA->cols, (float *)matB->values, (idxType)matB->ld,
                                                 matB->order);
                }
                case MACA_R_64F: {
                    return mcspDgenericCsc2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (double *)matA->vals, (idxType *)matA->rows,
                                                 (idxType *)matA->cols, (double *)matB->values, (idxType)matB->ld,
                                                 matB->order);
                }
                case MACA_C_32F: {
                    return mcspCgenericCsc2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexFloat *)matA->vals, (idxType *)matA->rows,
                                                 (idxType *)matA->cols, (mcspComplexFloat *)matB->values,
                                                 (idxType)matB->ld, matB->order);
                }
                case MACA_C_64F: {
                    return mcspZgenericCsc2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexDouble *)matA->vals,
                                                 (idxType *)matA->rows, (idxType *)matA->cols,
                                                 (mcspComplexDouble *)matB->values, (idxType)matB->ld, matB->order);
                }
#if defined(__MACA__)
                case MACA_R_16F: {
                    return mcspR16FgenericCsc2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half *)matA->vals, (idxType *)matA->rows,
                                                    (idxType *)matA->cols, (__half *)matB->values, (idxType)matB->ld,
                                                    matB->order);
                }
#endif
#ifdef __MACA__
                case MACA_C_16F: {
                    return mcspC16FgenericCsc2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half2 *)matA->vals, (idxType *)matA->rows,
                                                    (idxType *)matA->cols, (__half2 *)matB->values, (idxType)matB->ld,
                                                    matB->order);
                }
                case MACA_R_16BF: {
                    return mcspR16BgenericCsc2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (mcsp_bfloat16 *)matA->vals, (idxType *)matA->rows,
                                                    (idxType *)matA->cols, (mcsp_bfloat16 *)matB->values,
                                                    (idxType)matB->ld, matB->order);
                }
                case MACA_C_16BF: {
                    return mcspC16BgenericCsc2Dense(handle, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (mcsp_bfloat162 *)matA->vals,
                                                    (idxType *)matA->rows, (idxType *)matA->cols,
                                                    (mcsp_bfloat162 *)matB->values, (idxType)matB->ld, matB->order);
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

mcspStatus_t mcspDenseToSparse_bufferSizeImpl(mcspHandle_t handle, mcspDnMatDescr_t matA, mcspSpMatDescr_t matB,
                                              mcsparseDenseToSparseAlg_t alg, size_t *bufferSize) {
    *bufferSize = MIN_BUFFER_SIZE;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspDenseToSparse_analysisImpl(mcspHandle_t handle, mcspDnMatDescr_t matA, mcspSpMatDescr_t matB,
                                            mcsparseDenseToSparseAlg_t alg, void *externalBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (matA == nullptr || matA->values == nullptr || externalBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (matA->valueType != matB->valueType ||
        !(matB->format == MCSPARSE_FORMAT_COO || matB->format == MCSPARSE_FORMAT_CSR ||
          matB->format == MCSPARSE_FORMAT_CSC)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    mcsparseDirection_t dir;
    size_t nnz_array_size = 0;
    if (matB->format == MCSPARSE_FORMAT_CSR) {
        nnz_array_size = matA->row_num;
        dir = MCSPARSE_DIRECTION_ROW;
    } else if (matB->format == MCSPARSE_FORMAT_CSC) {
        nnz_array_size = matA->col_num;
        dir = MCSPARSE_DIRECTION_COLUMN;
    } else if (matB->format == MCSPARSE_FORMAT_COO) {
        nnz_array_size = matA->row_num;
        dir = MCSPARSE_DIRECTION_ROW;
    } else {
        return MCSP_STATUS_INVALID_VALUE;
    }
    MACA_ASSERT(mcMalloc(&(matB->nnz_array), nnz_array_size * sizeof(idxType)));
    switch (matA->valueType) {
        case MACA_R_32F: {
            return mcspSgenericNnz(handle, matA->order, dir, (idxType)matA->row_num, (idxType)matA->col_num,
                                   matA->mat_descr, (float *)matA->values, (idxType)matA->ld,
                                   (idxType *)matB->nnz_array, (idxType *)&(matB->nnz));
        }
        case MACA_R_64F: {
            return mcspDgenericNnz(handle, matA->order, dir, (idxType)matA->row_num, (idxType)matA->col_num,
                                   matA->mat_descr, (double *)matA->values, (idxType)matA->ld,
                                   (idxType *)matB->nnz_array, (idxType *)&(matB->nnz));
        }
        case MACA_C_32F: {
            return mcspCgenericNnz(handle, matA->order, dir, (idxType)matA->row_num, (idxType)matA->col_num,
                                   matA->mat_descr, (mcspComplexFloat *)matA->values, (idxType)matA->ld,
                                   (idxType *)matB->nnz_array, (idxType *)&(matB->nnz));
        }
        case MACA_C_64F: {
            return mcspZgenericNnz(handle, matA->order, dir, (idxType)matA->row_num, (idxType)matA->col_num,
                                   matA->mat_descr, (mcspComplexDouble *)matA->values, (idxType)matA->ld,
                                   (idxType *)matB->nnz_array, (idxType *)&(matB->nnz));
        }
#if defined(__MACA__)
        case MACA_R_16F: {
            return mcspR16FgenericNnz(handle, matA->order, dir, (idxType)matA->row_num, (idxType)matA->col_num,
                                      matA->mat_descr, (__half *)matA->values, (idxType)matA->ld,
                                      (idxType *)matB->nnz_array, (idxType *)&(matB->nnz));
        }
#endif

#ifdef __MACA__
        case MACA_C_16F: {
            return mcspC16FgenericNnz(handle, matA->order, dir, (idxType)matA->row_num, (idxType)matA->col_num,
                                      matA->mat_descr, (__half2 *)matA->values, (idxType)matA->ld,
                                      (idxType *)matB->nnz_array, (idxType *)&(matB->nnz));
        }
        case MACA_R_16BF: {
            return mcspR16BFgenericNnz(handle, matA->order, dir, (idxType)matA->row_num, (idxType)matA->col_num,
                                       matA->mat_descr, (mcsp_bfloat16 *)matA->values, (idxType)matA->ld,
                                       (idxType *)matB->nnz_array, (idxType *)&(matB->nnz));
        }
        case MACA_C_16BF: {
            return mcspC16BFgenericNnz(handle, matA->order, dir, (idxType)matA->row_num, (idxType)matA->col_num,
                                       matA->mat_descr, (mcsp_bfloat162 *)matA->values, (idxType)matA->ld,
                                       (idxType *)matB->nnz_array, (idxType *)&(matB->nnz));
        }
#endif
        default:
            return MCSP_STATUS_NOT_IMPLEMENTED;
    }
}

template <typename idxType>
mcspStatus_t mcspDenseToSparse_convertImpl(mcspHandle_t handle, mcspDnMatDescr_t matA, mcspSpMatDescr_t matB,
                                           mcsparseDenseToSparseAlg_t alg, void *externalBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (matA == nullptr || matA->mat_descr == nullptr || matA->values == nullptr || matB->rows == nullptr ||
        matB->cols == nullptr || matB->vals == nullptr || matB->nnz_array == nullptr || externalBuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (matA->valueType != matB->valueType ||
        !(matB->format == MCSPARSE_FORMAT_COO || matB->format == MCSPARSE_FORMAT_CSR ||
          matB->format == MCSPARSE_FORMAT_CSC)) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    switch (matB->format) {
        case MCSPARSE_FORMAT_CSR: {
            switch (matA->valueType) {
                case MACA_R_32F: {
                    return mcspSgenericDense2Csr(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (float *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (float *)matB->vals, (idxType *)matB->rows,
                                                 (idxType *)matB->cols);
                }
                case MACA_R_64F: {
                    return mcspDgenericDense2Csr(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (double *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (double *)matB->vals,
                                                 (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_C_32F: {
                    return mcspCgenericDense2Csr(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexFloat *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (mcspComplexFloat *)matB->vals,
                                                 (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_C_64F: {
                    return mcspZgenericDense2Csr(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexDouble *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (mcspComplexDouble *)matB->vals,
                                                 (idxType *)matB->rows, (idxType *)matB->cols);
                }
#if defined(__MACA__)
                case MACA_R_16F: {
                    return mcspR16FgenericDense2Csr(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half *)matA->values, (idxType)matA->ld,
                                                    (idxType *)matB->nnz_array, (__half *)matB->vals,
                                                    (idxType *)matB->rows, (idxType *)matB->cols);
                }
#endif

#ifdef __MACA__
                case MACA_C_16F: {
                    return mcspC16FgenericDense2Csr(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half2 *)matA->values, (idxType)matA->ld,
                                                    (idxType *)matB->nnz_array, (__half2 *)matB->vals,
                                                    (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_R_16BF: {
                    return mcspR16BFgenericDense2Csr(
                        handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num, matA->mat_descr,
                        (mcsp_bfloat16 *)matA->values, (idxType)matA->ld, (idxType *)matB->nnz_array,
                        (mcsp_bfloat16 *)matB->vals, (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_C_16BF: {
                    return mcspC16BFgenericDense2Csr(
                        handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num, matA->mat_descr,
                        (mcsp_bfloat162 *)matA->values, (idxType)matA->ld, (idxType *)matB->nnz_array,
                        (mcsp_bfloat162 *)matB->vals, (idxType *)matB->rows, (idxType *)matB->cols);
                }
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_COO: {
            switch (matA->valueType) {
                case MACA_R_32F: {
                    return mcspSgenericDense2Coo(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (float *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (float *)matB->vals, (idxType *)matB->rows,
                                                 (idxType *)matB->cols);
                }
                case MACA_R_64F: {
                    return mcspDgenericDense2Coo(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (double *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (double *)matB->vals,
                                                 (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_C_32F: {
                    return mcspCgenericDense2Coo(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexFloat *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (mcspComplexFloat *)matB->vals,
                                                 (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_C_64F: {
                    return mcspZgenericDense2Coo(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexDouble *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (mcspComplexDouble *)matB->vals,
                                                 (idxType *)matB->rows, (idxType *)matB->cols);
                }
#if defined(__MACA__)
                case MACA_R_16F: {
                    return mcspR16FgenericDense2Coo(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half *)matA->values, (idxType)matA->ld,
                                                    (idxType *)matB->nnz_array, (__half *)matB->vals,
                                                    (idxType *)matB->rows, (idxType *)matB->cols);
                }
#endif

#ifdef __MACA__
                case MACA_C_16F: {
                    return mcspC16FgenericDense2Coo(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half2 *)matA->values, (idxType)matA->ld,
                                                    (idxType *)matB->nnz_array, (__half2 *)matB->vals,
                                                    (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_R_16BF: {
                    return mcspR16BFgenericDense2Coo(
                        handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num, matA->mat_descr,
                        (mcsp_bfloat16 *)matA->values, (idxType)matA->ld, (idxType *)matB->nnz_array,
                        (mcsp_bfloat16 *)matB->vals, (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_C_16BF: {
                    return mcspC16BFgenericDense2Coo(
                        handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num, matA->mat_descr,
                        (mcsp_bfloat162 *)matA->values, (idxType)matA->ld, (idxType *)matB->nnz_array,
                        (mcsp_bfloat162 *)matB->vals, (idxType *)matB->rows, (idxType *)matB->cols);
                }
#endif
                default:
                    return MCSP_STATUS_NOT_IMPLEMENTED;
            }
        }
        case MCSPARSE_FORMAT_CSC: {
            switch (matA->valueType) {
                case MACA_R_32F: {
                    return mcspSgenericDense2Csc(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (float *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (float *)matB->vals, (idxType *)matB->rows,
                                                 (idxType *)matB->cols);
                }
                case MACA_R_64F: {
                    return mcspDgenericDense2Csc(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (double *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (double *)matB->vals,
                                                 (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_C_32F: {
                    return mcspCgenericDense2Csc(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexFloat *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (mcspComplexFloat *)matB->vals,
                                                 (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_C_64F: {
                    return mcspZgenericDense2Csc(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                 matA->mat_descr, (mcspComplexDouble *)matA->values, (idxType)matA->ld,
                                                 (idxType *)matB->nnz_array, (mcspComplexDouble *)matB->vals,
                                                 (idxType *)matB->rows, (idxType *)matB->cols);
                }
#if defined(__MACA__)
                case MACA_R_16F: {
                    return mcspR16FgenericDense2Csc(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half *)matA->values, (idxType)matA->ld,
                                                    (idxType *)matB->nnz_array, (__half *)matB->vals,
                                                    (idxType *)matB->rows, (idxType *)matB->cols);
                }
#endif

#ifdef __MACA__
                case MACA_C_16F: {
                    return mcspC16FgenericDense2Csc(handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num,
                                                    matA->mat_descr, (__half2 *)matA->values, (idxType)matA->ld,
                                                    (idxType *)matB->nnz_array, (__half2 *)matB->vals,
                                                    (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_R_16BF: {
                    return mcspR16BFgenericDense2Csc(
                        handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num, matA->mat_descr,
                        (mcsp_bfloat16 *)matA->values, (idxType)matA->ld, (idxType *)matB->nnz_array,
                        (mcsp_bfloat16 *)matB->vals, (idxType *)matB->rows, (idxType *)matB->cols);
                }
                case MACA_C_16BF: {
                    return mcspC16BFgenericDense2Csc(
                        handle, matA->order, (idxType)matA->row_num, (idxType)matA->col_num, matA->mat_descr,
                        (mcsp_bfloat162 *)matA->values, (idxType)matA->ld, (idxType *)matB->nnz_array,
                        (mcsp_bfloat162 *)matB->vals, (idxType *)matB->rows, (idxType *)matB->cols);
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
mcspStatus_t mcspCooSetStridedBatchImpl(mcspSpMatDescr_t spMatDescr, idxType batchCount, int64_t batchStride) {
    if (spMatDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (batchCount <= 0 || batchStride < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    spMatDescr->batchCount = batchCount;
    spMatDescr->batchStride = batchStride;

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCsrSetStridedBatchImpl(mcspSpMatDescr_t spMatDescr, idxType batchCount, int64_t offsetsBatchStride,
                                        int64_t columnsValuesBatchStride) {
    if (spMatDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (batchCount <= 0 || offsetsBatchStride < 0 || columnsValuesBatchStride < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    spMatDescr->batchCount = batchCount;
    spMatDescr->offsetsBatchStride = offsetsBatchStride;
    spMatDescr->batchStride = columnsValuesBatchStride;

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspDnMatSetStridedBatchImpl(mcspDnMatDescr_t dnMatDescr, idxType batchCount, int64_t batchStride) {
    if (dnMatDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (batchCount <= 0 || batchStride < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    dnMatDescr->batchCount = batchCount;
    dnMatDescr->batchStride = batchStride;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspDnMatGetStridedBatchImpl(mcspDnMatDescr_t dnMatDescr, idxType *batchCount, int64_t *batchStride) {
    if (dnMatDescr == nullptr || batchCount == nullptr || batchStride == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *batchCount = dnMatDescr->batchCount;
    *batchStride = dnMatDescr->batchStride;
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspGather(mcspHandle_t handle, mcspDnVecDescr_t vecY, mcspSpVecDescr_t vecX) {
    return mcspGatherImpl<mcspInt>(handle, vecY, vecX);
}

mcspStatus_t mcspSparseToDense_bufferSize(mcspHandle_t handle, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB,
                                          mcsparseSparseToDenseAlg_t alg, size_t *bufferSize) {
    return mcspSparseToDense_bufferSizeImpl(handle, matA, matB, alg, bufferSize);
}

mcspStatus_t mcspSparseToDense(mcspHandle_t handle, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB,
                               mcsparseSparseToDenseAlg_t alg, void *externalBuffer) {
    return mcspSparseToDenseImpl<mcspInt>(handle, matA, matB, alg, externalBuffer);
}

mcspStatus_t mcspDenseToSparse_bufferSize(mcspHandle_t handle, mcspDnMatDescr_t matA, mcspSpMatDescr_t matB,
                                          mcsparseDenseToSparseAlg_t alg, size_t *bufferSize) {
    return mcspDenseToSparse_bufferSizeImpl(handle, matA, matB, alg, bufferSize);
}

mcspStatus_t mcspDenseToSparse_analysis(mcspHandle_t handle, mcspDnMatDescr_t matA, mcspSpMatDescr_t matB,
                                        mcsparseDenseToSparseAlg_t alg, void *externalBuffer) {
    return mcspDenseToSparse_analysisImpl<mcspInt>(handle, matA, matB, alg, externalBuffer);
}

mcspStatus_t mcspDenseToSparse_convert(mcspHandle_t handle, mcspDnMatDescr_t matA, mcspSpMatDescr_t matB,
                                       mcsparseDenseToSparseAlg_t alg, void *externalBuffer) {
    return mcspDenseToSparse_convertImpl<mcspInt>(handle, matA, matB, alg, externalBuffer);
}

mcspStatus_t mcspScatter(mcspHandle_t handle, mcspSpVecDescr_t vecX, mcspDnVecDescr_t vecY) {
    return mcspScatterImpl<mcspInt>(handle, vecX, vecY);
}

mcspStatus_t mcspRot(mcspHandle_t handle, const void *c_coeff, const void *s_coeff, mcspSpVecDescr_t vecX,
                     mcspDnVecDescr_t vecY) {
    return mcspRotImpl<mcspInt>(handle, c_coeff, s_coeff, vecX, vecY);
}

mcspStatus_t mcspCooSetStridedBatch(mcspSpMatDescr_t spMatDescr, mcspInt batchCount, int64_t batchStride) {
    return mcspCooSetStridedBatchImpl(spMatDescr, batchCount, batchStride);
}

mcspStatus_t mcspCsrSetStridedBatch(mcspSpMatDescr_t spMatDescr, mcspInt batchCount, int64_t offsetsBatchStride,
                                    int64_t columnsValuesBatchStride) {
    return mcspCsrSetStridedBatchImpl(spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride);
}

mcspStatus_t mcspDnMatSetStridedBatch(mcspDnMatDescr_t dnMatDescr, mcspInt batchCount, int64_t batchStride) {
    return mcspDnMatSetStridedBatchImpl(dnMatDescr, batchCount, batchStride);
}

mcspStatus_t mcspDnMatGetStridedBatch(mcspDnMatDescr_t dnMatDescr, mcspInt *batchCount, int64_t *batchStride) {
    return mcspDnMatGetStridedBatchImpl(dnMatDescr, batchCount, batchStride);
}

#ifdef __cplusplus
}
#endif
