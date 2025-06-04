#include "common/mcsp_types.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "utils/mcsp_logger.h"

#define MCSP_VER_MAJOR 1
#define MCSP_VER_MINOR 0
#define MCSP_PATCH 0

// #############################################################################
// # Not export functions
// #############################################################################

mcspStatus_t mcspCreateTrmInfo(mcspTrmInfo_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *info = new mcspTrmInfo();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDestroyTrmInfo(mcspTrmInfo_t info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info->row_map != nullptr) {
        if (mcFree(info->row_map) != mcSuccess) {
            stat = MCSP_STATUS_INTERNAL_ERROR;
        } else {
            info->row_map = nullptr;
        }
    }
    if (info->trm_diag_ind != nullptr) {
        if (mcFree(info->trm_diag_ind) != mcSuccess) {
            stat = MCSP_STATUS_INTERNAL_ERROR;
        } else {
            info->trm_diag_ind = nullptr;
        }
    }
    if (info->zero_pivot_array != nullptr) {
        if (mcFree(info->zero_pivot_array) != mcSuccess) {
            stat = MCSP_STATUS_INTERNAL_ERROR;
        } else {
            info->zero_pivot_array = nullptr;
        }
    }
    if (info->zero_pivot_lead != nullptr) {
        if (mcFree(info->zero_pivot_lead) != mcSuccess) {
            stat = MCSP_STATUS_INTERNAL_ERROR;
        } else {
            info->zero_pivot_lead = nullptr;
        }
    }
    delete info;
    info = nullptr;
    return stat;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspCreateHandle(mcspHandle_t *handle) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *handle = new mcspHandle();
    if (*handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    mcspStatus_t ret = (*handle)->mcspMallocPoolBuffer();
    return ret;
}

mcspStatus_t mcspDestroyHandle(mcspHandle_t handle) {
    if (handle == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    delete handle;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCreateMatDescr(mcspMatDescr_t *descr) {
    if (descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *descr = new mcspMatDescr();
    if (*descr == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDestroyMatDescr(mcspMatDescr_t descr) {
    if (descr == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    delete descr;
    descr = nullptr;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCreatePruneInfo(mcspPruneInfo_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *info = new mcspPruneInfo();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDestroyPruneInfo(mcspPruneInfo_t info) {
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    delete info;
    info = nullptr;
    return MCSP_STATUS_SUCCESS;
}
// TODO
mcspStatus_t mcspCopyMatDescr(mcspMatDescr_t dest, const mcspMatDescr_t src) {
    return MCSP_STATUS_NOT_IMPLEMENTED;
}

mcspStatus_t mcspCreateMatInfo(mcspMatInfo_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *info = new mcspMatInfo();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDestroyMatInfo(mcspMatInfo_t info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    delete info;
    info = nullptr;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSetStream(mcspHandle_t handle, mcStream_t sid) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    handle->stream = sid;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspGetStream(const mcspHandle_t handle, mcStream_t *sid) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (sid == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *sid = handle->stream;
    return MCSP_STATUS_SUCCESS;
}

mcStream_t mcspGetStreamInternal(const mcspHandle_t handle) {
    mcStream_t stream = nullptr;
    if (mcspGetStream(handle, &stream) == MCSP_STATUS_SUCCESS) {
        return stream;
    }
    return nullptr;
}

mcspStatus_t mcspSetPointerMode(mcspHandle_t handle, mcsparsePointerMode_t pointer_mode) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    handle->ptr_mode = pointer_mode;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspGetPointerMode(const mcspHandle_t handle, mcsparsePointerMode_t *pointer_mode) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (pointer_mode == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *pointer_mode = handle->ptr_mode;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSetMatIndexBase(mcspMatDescr_t descr, mcsparseIndexBase_t index_base) {
    if (descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if ((index_base != MCSPARSE_INDEX_BASE_ZERO) && (index_base != MCSPARSE_INDEX_BASE_ONE)) {
        return MCSP_STATUS_INVALID_VALUE;
    }
    descr->base = index_base;
    return MCSP_STATUS_SUCCESS;
}

mcsparseIndexBase_t mcspGetMatIndexBase(const mcspMatDescr_t descr) {
    if (descr == nullptr) {
        return (mcsparseIndexBase_t)-1;
    }

    return descr->base;
}

mcspStatus_t mcspSetMatType(mcspMatDescr_t descr, mcsparseMatrixType_t matrix_type) {
    if (descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if ((matrix_type != MCSPARSE_MATRIX_TYPE_GENERAL) && (matrix_type != MCSPARSE_MATRIX_TYPE_HERMITIAN) &&
        (matrix_type != MCSPARSE_MATRIX_TYPE_SYMMETRIC) && (matrix_type != MCSPARSE_MATRIX_TYPE_TRIANGULAR)) {
        return MCSP_STATUS_INVALID_VALUE;
    }
    descr->type = matrix_type;
    return MCSP_STATUS_SUCCESS;
}

mcsparseMatrixType_t mcspGetMatType(const mcspMatDescr_t descr) {
    if (descr == nullptr) {
        return (mcsparseMatrixType_t)-1;
    }

    return descr->type;
}

mcspStatus_t mcspSetMatFillMode(mcspMatDescr_t descr, mcsparseFillMode_t mode) {
    if (descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if ((mode != MCSPARSE_FILL_MODE_FULL) && (mode != MCSPARSE_FILL_MODE_LOWER) && (mode != MCSPARSE_FILL_MODE_UPPER)) {
        return MCSP_STATUS_INVALID_VALUE;
    }
    descr->fill_mode = mode;
    return MCSP_STATUS_SUCCESS;
}

mcsparseFillMode_t mcspGetMatFillMode(const mcspMatDescr_t descr) {
    if (descr == nullptr) {
        return (mcsparseFillMode_t)-1;
    }
    return descr->fill_mode;
}

mcspStatus_t mcspSetStorageMode(mcspMatDescr_t descr, mcsparseStorageMode_t storage_mode) {
    if (descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if ((storage_mode != MCSPARSE_STORAGE_MODE_SORTED) && (storage_mode != MCSPARSE_STORAGE_MODE_UNSORTED)) {
        return MCSP_STATUS_INVALID_VALUE;
    }
    descr->storage_mode = storage_mode;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspGetStorageMode(const mcspMatDescr_t descr, mcsparseStorageMode_t *storage_mode) {
    if ((descr == nullptr) || (storage_mode == nullptr)) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *storage_mode = descr->storage_mode;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSetMatDiagType(mcspMatDescr_t descr, mcsparseDiagType_t diag_type) {
    if (descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if ((diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) && (diag_type != MCSPARSE_DIAG_TYPE_UNIT)) {
        return MCSP_STATUS_INVALID_VALUE;
    }
    descr->diag_type = diag_type;
    return MCSP_STATUS_SUCCESS;
}

mcsparseDiagType_t mcspGetMatDiagType(const mcspMatDescr_t descr) {
    if (descr == nullptr) {
        return (mcsparseDiagType_t)-1;
    }
    return descr->diag_type;
}

mcspStatus_t mcspGetVersion(mcspHandle_t handle, int *version) {
    if (version == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    *version = MCSP_VER_MAJOR * 1000000 + MCSP_VER_MINOR * 1000 + MCSP_PATCH;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspGetProperty(libraryPropertyType type, int *value) {
    switch (type) {
        case MAJOR_VERSION:
            *value = MCSP_VER_MAJOR;
            break;
        case MINOR_VERSION:
            *value = MCSP_VER_MINOR;
            break;
        case PATCH_LEVEL:
            *value = MCSP_PATCH;
            break;
        default:
            return MCSP_STATUS_TYPE_MISMATCH;
    }

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCreateDnVec(mcspDnVecDescr_t *dnVecDescr, int64_t size, void *values, macaDataType valueType) {
    if (size < 0) {
        LOG_FS_ERR("Wrong input size\n");
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (dnVecDescr == nullptr || values == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (!(valueType == MACA_R_32F || valueType == MACA_R_64F || valueType == MACA_C_32F || valueType == MACA_C_64F ||
          valueType == MACA_R_16F || valueType == MACA_R_16BF || valueType == MACA_C_16F || valueType == MACA_C_16BF ||
          valueType == MACA_R_8I || valueType == MACA_R_32I)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    *dnVecDescr = new mcspDnVecDescr();
    if (*dnVecDescr == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    (*dnVecDescr)->valueType = valueType;
    (*dnVecDescr)->size = size;
    (*dnVecDescr)->values = values;

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDestroyDnVec(mcspDnVecDescr_t dnVecDescr) {
    if (dnVecDescr == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    delete dnVecDescr;
    dnVecDescr = nullptr;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDnVecGet(mcspDnVecDescr_t dnVecDescr, int64_t *size, void **values, macaDataType *valueType) {
    if (dnVecDescr == nullptr || size == nullptr || values == nullptr || valueType == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *size = dnVecDescr->size;
    *values = dnVecDescr->values;
    *valueType = dnVecDescr->valueType;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDnVecGetValues(mcspDnVecDescr_t dnVecDescr, void **values) {
    if (dnVecDescr == nullptr || values == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *values = dnVecDescr->values;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDnVecSetValues(mcspDnVecDescr_t dnVecDescr, void *values) {
    if (dnVecDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    dnVecDescr->values = values;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCreateSpMat(mcspSpMatDescr_t *spMatDescr, int64_t row_num, int64_t col_num, int64_t nnz, void *rows,
                             void *cols, void *vals, mcsparseIndexType_t rowIndexType, mcsparseIndexType_t colIndexType,
                             mcsparseIndexBase_t idxBase, macaDataType valueType, mcsparseFormat_t format) {
    if (row_num < 0 || col_num < 0 || nnz < 0) {
        LOG_FS_ERR("Wrong input size\n");
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (!((valueType == MACA_R_32F || valueType == MACA_R_64F || valueType == MACA_C_32F || valueType == MACA_C_64F ||
           valueType == MACA_R_16F || valueType == MACA_R_16BF || valueType == MACA_C_16F || valueType == MACA_C_16BF ||
           valueType == MACA_R_8I) &&
          (format == MCSPARSE_FORMAT_CSR || format == MCSPARSE_FORMAT_COO || format == MCSPARSE_FORMAT_CSC ||
           format == MCSPARSE_FORMAT_BLOCKED_ELL || format == MCSPARSE_FORMAT_COO_AOS))) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (spMatDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *spMatDescr = new mcspSpMatDescr();
    if (*spMatDescr == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    (*spMatDescr)->format = format;
    (*spMatDescr)->row_num = row_num;
    (*spMatDescr)->col_num = col_num;
    (*spMatDescr)->nnz = nnz;
    (*spMatDescr)->rows = rows;
    (*spMatDescr)->cols = cols;
    (*spMatDescr)->vals = vals;
    (*spMatDescr)->rowIdxType = rowIndexType;
    (*spMatDescr)->colIdxType = colIndexType;
    (*spMatDescr)->idxBase = idxBase;
    (*spMatDescr)->valueType = valueType;
    mcspStatus_t stat = mcspCreateMatDescr(&((*spMatDescr)->mat_descr));
    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }
    stat = mcspSetMatIndexBase((*spMatDescr)->mat_descr, idxBase);
    return stat;
}

mcspStatus_t mcspCreateCoo(mcspSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void *cooRowInd,
                           void *cooColInd, void *cooValues, mcsparseIndexType_t cooIdxType,
                           mcsparseIndexBase_t idxBase, macaDataType valueType) {
    return mcspCreateSpMat(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, cooIdxType,
                           idxBase, valueType, MCSPARSE_FORMAT_COO);
}

mcspStatus_t mcspCreateCsr(mcspSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void *csrRowOffsets,
                           void *csrColInd, void *csrValues, mcsparseIndexType_t csrRowOffsetsType,
                           mcsparseIndexType_t csrColIndType, mcsparseIndexBase_t idxBase, macaDataType valueType) {
    return mcspCreateSpMat(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType,
                           csrColIndType, idxBase, valueType, MCSPARSE_FORMAT_CSR);
}

mcspStatus_t mcspCreateCsc(mcspSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void *cscColOffsets,
                           void *cscRowInd, void *cscValues, mcsparseIndexType_t cscColOffsetsType,
                           mcsparseIndexType_t cscRowIndType, mcsparseIndexBase_t idxBase, macaDataType valueType) {
    return mcspCreateSpMat(spMatDescr, rows, cols, nnz, cscRowInd, cscColOffsets, cscValues, cscRowIndType,
                           cscColOffsetsType, idxBase, valueType, MCSPARSE_FORMAT_CSC);
}

mcspStatus_t mcspCreateBlockedEll(mcspSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize,
                                  int64_t ellCols, void *ellColInd, void *ellValue, mcsparseIndexType_t ellIdxType,
                                  mcsparseIndexBase_t idxBase, macaDataType valueType) {
    mcspStatus_t stat = mcspCreateSpMat(spMatDescr, rows, cols, 0, nullptr, nullptr, nullptr, ellIdxType, ellIdxType,
                                        idxBase, valueType, MCSPARSE_FORMAT_BLOCKED_ELL);
    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }

    if (rows % ellBlockSize != 0) {
        LOG_FS_ERR("number of rows (%d) is not a multiple of block size (%d)\n", rows, ellBlockSize);
        return MCSP_STATUS_INVALID_SIZE;
    }
    if (cols % ellBlockSize != 0) {
        LOG_FS_ERR("number of columns (%d) is not a multiple of block size (%d)\n", cols, ellBlockSize);
        return MCSP_STATUS_INVALID_SIZE;
    }
    if (ellColInd == nullptr || ellValue == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (ellCols < 0 || ellBlockSize < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    (*spMatDescr)->ellBlockSize = ellBlockSize;
    (*spMatDescr)->ellCols = ellCols;
    (*spMatDescr)->ellColInd = ellColInd;
    (*spMatDescr)->ellValue = ellValue;
    return stat;
}

mcspStatus_t mcspCreateCooAos(mcspSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void *cooInd,
                              void *cooValues, mcsparseIndexType_t cooIdxType, mcsparseIndexBase_t idxBase,
                              macaDataType valueType) {
    if (cooInd == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    mcspStatus_t stat = mcspCreateSpMat(spMatDescr, rows, cols, nnz, nullptr, nullptr, cooValues, cooIdxType,
                                        cooIdxType, idxBase, valueType, MCSPARSE_FORMAT_COO_AOS);
    if (stat != MCSP_STATUS_SUCCESS) {
        return stat;
    }

    (*spMatDescr)->coo_aos_ind = cooInd;
    return stat;
}

mcspStatus_t mcspCsrGet(mcspSpMatDescr_t spMatDescr, int64_t *rowNum, int64_t *colNum, int64_t *nnz,
                        void **csrRowOffsets, void **csrColInd, void **csrValues,
                        mcsparseIndexType_t *csrRowOffsetsType, mcsparseIndexType_t *csrColIndType,
                        mcsparseIndexBase_t *idxBase, macaDataType *valueType) {
    if (spMatDescr == nullptr || rowNum == nullptr || colNum == nullptr || nnz == nullptr || csrRowOffsets == nullptr ||
        csrColInd == nullptr || csrValues == nullptr || csrRowOffsetsType == nullptr || csrColIndType == nullptr ||
        idxBase == nullptr || valueType == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *rowNum = spMatDescr->row_num;
    *colNum = spMatDescr->col_num;
    *nnz = spMatDescr->nnz;
    *csrRowOffsets = spMatDescr->rows;
    *csrColInd = spMatDescr->cols;
    *csrValues = spMatDescr->vals;
    *csrRowOffsetsType = spMatDescr->rowIdxType;
    *csrColIndType = spMatDescr->colIdxType;
    *idxBase = spMatDescr->idxBase;
    *valueType = spMatDescr->valueType;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCooGet(mcspSpMatDescr_t spMatDescr, int64_t *rowNum, int64_t *colNum, int64_t *nnz, void **cooRowInd,
                        void **cooColInd, void **cooValues, mcsparseIndexType_t *indexType,
                        mcsparseIndexBase_t *idxBase, macaDataType *valueType) {
    if (spMatDescr == nullptr || rowNum == nullptr || colNum == nullptr || nnz == nullptr || cooRowInd == nullptr ||
        cooColInd == nullptr || cooValues == nullptr || indexType == nullptr || idxBase == nullptr ||
        valueType == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *rowNum = spMatDescr->row_num;
    *colNum = spMatDescr->col_num;
    *nnz = spMatDescr->nnz;
    *cooRowInd = spMatDescr->rows;
    *cooColInd = spMatDescr->cols;
    *cooValues = spMatDescr->vals;
    *indexType = spMatDescr->rowIdxType;
    *idxBase = spMatDescr->idxBase;
    *valueType = spMatDescr->valueType;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspBlockedEllGet(mcsparseSpMatDescr_t spMatDescr, int64_t *rowNum, int64_t *colNum, int64_t *ellBlockSize,
                               int64_t *ellCols, void **ellColInd, void **ellValue, mcsparseIndexType_t *ellIdxType,
                               mcsparseIndexBase_t *idxBase, macaDataType *valueType) {
    if (spMatDescr == nullptr || rowNum == nullptr || colNum == nullptr || ellBlockSize == nullptr ||
        ellCols == nullptr || ellColInd == nullptr || ellValue == nullptr || ellIdxType == nullptr ||
        idxBase == nullptr || valueType == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *rowNum = spMatDescr->row_num;
    *colNum = spMatDescr->col_num;
    *ellBlockSize = spMatDescr->ellBlockSize;
    *ellCols = spMatDescr->ellCols;
    *ellColInd = spMatDescr->ellColInd;
    *ellValue = spMatDescr->ellValue;
    *ellIdxType = spMatDescr->rowIdxType;
    *idxBase = spMatDescr->idxBase;
    *valueType = spMatDescr->valueType;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCooAosGet(mcspSpMatDescr_t spMatDescr, int64_t *rowNum, int64_t *colNum, int64_t *nnz, void **cooInd,
                           void **cooValues, mcsparseIndexType_t *indexType, mcsparseIndexBase_t *idxBase,
                           macaDataType *valueType) {
    if (spMatDescr == nullptr || rowNum == nullptr || colNum == nullptr || nnz == nullptr || cooInd == nullptr ||
        cooValues == nullptr || indexType == nullptr || idxBase == nullptr || valueType == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *rowNum = spMatDescr->row_num;
    *colNum = spMatDescr->col_num;
    *nnz = spMatDescr->nnz;
    *cooInd = spMatDescr->coo_aos_ind;
    *cooValues = spMatDescr->vals;
    *indexType = spMatDescr->rowIdxType;
    *idxBase = spMatDescr->idxBase;
    *valueType = spMatDescr->valueType;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCsrSetPointers(mcspSpMatDescr_t spMatDescr, void *csrRowOffsets, void *csrColInd, void *csrValues) {
    if (spMatDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    spMatDescr->rows = csrRowOffsets;
    spMatDescr->cols = csrColInd;
    spMatDescr->vals = csrValues;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCooSetPointers(mcspSpMatDescr_t spMatDescr, void *cooRows, void *cooColumns, void *cooValues) {
    if (spMatDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    spMatDescr->rows = cooRows;
    spMatDescr->cols = cooColumns;
    spMatDescr->vals = cooValues;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCscSetPointers(mcspSpMatDescr_t spMatDescr, void *cscColOffsets, void *cscRowInd, void *cscValues) {
    if (spMatDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    spMatDescr->rows = cscRowInd;
    spMatDescr->cols = cscColOffsets;
    spMatDescr->vals = cscValues;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpMatGetSize(mcspSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz) {
    if (spMatDescr == nullptr || rows == nullptr || cols == nullptr || nnz == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *rows = spMatDescr->row_num;
    *cols = spMatDescr->col_num;
    *nnz = spMatDescr->nnz;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDestroySpMat(mcspSpMatDescr_t spMatDescr) {
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (spMatDescr == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    if (spMatDescr->mat_descr != nullptr) {
        stat = mcspDestroyMatDescr(spMatDescr->mat_descr);
        spMatDescr->mat_descr = nullptr;
    }

    if (spMatDescr->nnz_array != nullptr) {
        if (mcFree(spMatDescr->nnz_array) != mcSuccess) {
            stat = MCSP_STATUS_INTERNAL_ERROR;
        }
    }

    delete spMatDescr;
    spMatDescr = nullptr;
    return stat;
}

mcspStatus_t mcspSpMatGetIndexBase(mcspSpMatDescr_t spMatDescr, mcsparseIndexBase_t *idxBase) {
    if (spMatDescr == nullptr || idxBase == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *idxBase = spMatDescr->idxBase;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpMatGetValues(mcspSpMatDescr_t spMatDescr, void **values) {
    if (spMatDescr == nullptr || values == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *values = spMatDescr->vals;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpMatSetValues(mcspSpMatDescr_t spMatDescr, void *values) {
    if (spMatDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    spMatDescr->vals = values;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpMatGetFormat(mcspSpMatDescr_t spMatDescr, mcsparseFormat_t *format) {
    if (spMatDescr == nullptr || format == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *format = spMatDescr->format;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpMatGetAttribute(mcspSpMatDescr_t spMatDescr, mcsparseSpMatAttribute_t attribute, void *data,
                                   size_t dataSize) {
    if (data == nullptr || spMatDescr == nullptr || spMatDescr->mat_descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (attribute == MCSPARSE_SPMAT_FILL_MODE) {
        MACA_ASSERT(mcMemcpy(data, &(spMatDescr->mat_descr->fill_mode), sizeof(spMatDescr->mat_descr->fill_mode),
                             mcMemcpyHostToHost));
    } else if (attribute == MCSPARSE_SPMAT_DIAG_TYPE) {
        MACA_ASSERT(mcMemcpy(data, &(spMatDescr->mat_descr->diag_type), sizeof(spMatDescr->mat_descr->diag_type),
                             mcMemcpyHostToHost));
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpMatSetAttribute(mcspSpMatDescr_t spMatDescr, mcsparseSpMatAttribute_t attribute, const void *data,
                                   size_t dataSize) {
    if (data == nullptr || spMatDescr == nullptr || spMatDescr->mat_descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (attribute == MCSPARSE_SPMAT_FILL_MODE) {
        MACA_ASSERT(mcMemcpy(&(spMatDescr->mat_descr->fill_mode), data, sizeof(spMatDescr->mat_descr->fill_mode),
                             mcMemcpyHostToHost));
    } else if (attribute == MCSPARSE_SPMAT_DIAG_TYPE) {
        MACA_ASSERT(mcMemcpy(&(spMatDescr->mat_descr->diag_type), data, sizeof(spMatDescr->mat_descr->diag_type),
                             mcMemcpyHostToHost));
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCreateDnMat(mcspDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void *values,
                             macaDataType valueType, mcsparseOrder_t order) {
    if (rows < 0 || cols < 0 || ld < 0) {
        LOG_FS_ERR("Wrong input size\n");
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (!(valueType == MACA_R_32F || valueType == MACA_R_64F || valueType == MACA_C_32F || valueType == MACA_C_64F ||
          valueType == MACA_R_16F || valueType == MACA_R_16BF || valueType == MACA_C_16F || valueType == MACA_C_16BF ||
          valueType == MACA_R_32I || valueType == MACA_R_8I)) {
        LOG_FS_ERR("Dense matrix data type not supported\n");
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (dnMatDescr == nullptr || values == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *dnMatDescr = new mcspDnMatDescr();
    if (*dnMatDescr == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    (*dnMatDescr)->valueType = valueType;
    (*dnMatDescr)->order = order;
    (*dnMatDescr)->row_num = rows;
    (*dnMatDescr)->col_num = cols;
    (*dnMatDescr)->ld = ld;
    (*dnMatDescr)->values = values;

    return mcspCreateMatDescr(&((*dnMatDescr)->mat_descr));
}

mcspStatus_t mcspDestroyDnMat(mcspDnMatDescr_t dnMatDescr) {
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (dnMatDescr == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }

    if (dnMatDescr->mat_descr != nullptr) {
        stat = mcspDestroyMatDescr(dnMatDescr->mat_descr);
        dnMatDescr->mat_descr = nullptr;
    }
    delete dnMatDescr;
    dnMatDescr = nullptr;
    return stat;
}

mcspStatus_t mcspDnMatGet(mcspDnMatDescr_t dnMatDescr, int64_t *rows, int64_t *cols, int64_t *ld, void **values,
                          macaDataType *type, mcsparseOrder_t *order) {
    if (dnMatDescr == nullptr || rows == nullptr || cols == nullptr || ld == nullptr || values == nullptr ||
        type == nullptr || order == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *rows = dnMatDescr->row_num;
    *cols = dnMatDescr->col_num;
    *ld = dnMatDescr->ld;
    *values = dnMatDescr->values;
    *type = dnMatDescr->valueType;
    *order = dnMatDescr->order;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDnMatGetValues(mcspDnMatDescr_t dnMatDescr, void **values) {
    if (dnMatDescr == nullptr || values == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *values = dnMatDescr->values;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDnMatSetValues(mcspDnMatDescr_t dnMatDescr, void *values) {
    if (dnMatDescr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    dnMatDescr->values = values;
    return MCSP_STATUS_SUCCESS;
}

// #############################################################################
// # SPARSE VECTOR API
// #############################################################################

mcspStatus_t mcspCreateSpVec(mcspSpVecDescr_t *spVecDescr, int64_t size, int64_t nnz, void *indices, void *values,
                             mcsparseIndexType_t idxType, mcsparseIndexBase_t idxBase, macaDataType valueType) {
    if (size < 0 || nnz < 0 || size < nnz) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (!(valueType == MACA_R_32F || valueType == MACA_R_64F || valueType == MACA_C_32F || valueType == MACA_C_64F ||
          valueType == MACA_R_16F || valueType == MACA_R_16BF || valueType == MACA_C_16F || valueType == MACA_C_16BF ||
          valueType == MACA_R_8I)) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (nnz > 0 && (spVecDescr == nullptr || indices == nullptr || values == nullptr)) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *spVecDescr = new mcspSpVecDescr();
    if (*spVecDescr == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }
    (*spVecDescr)->size = size;
    (*spVecDescr)->nnz = nnz;
    (*spVecDescr)->indices = indices;
    (*spVecDescr)->values = values;
    (*spVecDescr)->idxType = idxType;
    (*spVecDescr)->idxBase = idxBase;
    (*spVecDescr)->valueType = valueType;

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDestroySpVec(mcspSpVecDescr_t spVecDescr) {
    if (spVecDescr == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    delete spVecDescr;
    spVecDescr = nullptr;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpVecGet(mcspSpVecDescr_t spVecDescr, int64_t *size, int64_t *nnz, void **indices, void **values,
                          mcsparseIndexType_t *idxType, mcsparseIndexBase_t *idxBase, macaDataType *valueType) {
    if (spVecDescr == nullptr || size == nullptr || nnz == nullptr || indices == nullptr || values == nullptr ||
        values == nullptr || idxType == nullptr || idxBase == nullptr || valueType == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *size = spVecDescr->size;
    *nnz = spVecDescr->nnz;
    *indices = spVecDescr->indices;
    *values = spVecDescr->values;
    *idxType = spVecDescr->idxType;
    *idxBase = spVecDescr->idxBase;
    *valueType = spVecDescr->valueType;

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpVecGetIndexBase(mcspSpVecDescr_t spVecDescr, mcsparseIndexBase_t *idxBase) {
    if (spVecDescr == nullptr || idxBase == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *idxBase = spVecDescr->idxBase;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpVecGetValues(mcspSpVecDescr_t spVecDescr, void **values) {
    if (spVecDescr == nullptr || values == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *values = spVecDescr->values;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspSpVecSetValues(mcspSpVecDescr_t spVecDescr, void *values) {
    if (spVecDescr == nullptr || values == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    spVecDescr->values = values;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCreateCsrilu02Info(mcspCsrilu02Info_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspCsrilu02Info();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*info)->csrilu0_mat));
}

mcspStatus_t mcspDestroyCsrilu02Info(mcspCsrilu02Info_t info) {
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info->csrilu0_mat != nullptr) {
        if (info->csrilu0_mat->csrilu0_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->csrilu0_mat->csrilu0_info);
            info->csrilu0_mat->csrilu0_info = nullptr;
        }
        stat = mcspDestroyMatInfo(info->csrilu0_mat);
        info->csrilu0_mat = nullptr;
    }
    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspCreateCsric02Info(mcspCsric02Info_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspCsric02Info();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*info)->csric0_mat));
}

mcspStatus_t mcspDestroyCsric02Info(mcspCsric02Info_t info) {
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info->csric0_mat != nullptr) {
        if (info->csric0_mat->csric0_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->csric0_mat->csric0_info);
            info->csric0_mat->csric0_info = nullptr;
        }
        stat = mcspDestroyMatInfo(info->csric0_mat);
        info->csric0_mat = nullptr;
    }
    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspCreateColorInfo(mcspColorInfo_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspColorInfo();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*info)->mat_info));
}

mcspStatus_t mcspDestroyColorInfo(mcspColorInfo_t info) {
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info->mat_info != nullptr) {
        stat = mcspDestroyMatInfo(info->mat_info);
    }
    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspSetColorAlgs(mcspColorInfo_t info, mcsparseColorAlg_t alg) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    info->algo = alg;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspGetColorAlgs(mcspColorInfo_t info, mcsparseColorAlg_t *alg) {
    if (info == nullptr || alg == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *alg = info->algo;
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspCreateCsrgemm2Info(mcspCsrgemm2Info_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    *info = new mcspCsrgemm2Info();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }
    return mcspCreateMatInfo(&((*info)->mat_info));
}

mcspStatus_t mcspDestroyCsrgemm2Info(mcspCsrgemm2Info_t info) {
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info->mat_info != nullptr) {
        stat = mcspDestroyMatInfo(info->mat_info);
    }
    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspCreateCsrsv2Info(mcspCsrsv2Info_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspCsrsv2Info();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*info)->mat_info));
}

mcspStatus_t mcspDestroyCsrsv2Info(mcspCsrsv2Info_t info) {
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    if (info->mat_info != nullptr) {
        if (info->mat_info->csr_spsv_lower_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->mat_info->csr_spsv_lower_info);
            info->mat_info->csr_spsv_lower_info = nullptr;
        } else if (info->mat_info->csr_spsv_upper_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->mat_info->csr_spsv_upper_info);
            info->mat_info->csr_spsv_upper_info = nullptr;
        }
        stat = mcspDestroyMatInfo(info->mat_info);
        info->mat_info = nullptr;
    }

    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspSpSV_createDescr(mcspSpSVDescr_t *descr) {
    if (descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *descr = new mcspSpSVDescr();
    if (*descr == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*descr)->mat_info));
}

mcspStatus_t mcspSpSV_destroyDescr(mcspSpSVDescr_t descr) {
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (descr == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    if (descr->mat_info != nullptr) {
        if (descr->mat_info->csr_spsv_lower_info != nullptr) {
            stat = mcspDestroyTrmInfo(descr->mat_info->csr_spsv_lower_info);
            descr->mat_info->csr_spsv_lower_info = nullptr;
        } else if (descr->mat_info->csr_spsv_upper_info != nullptr) {
            stat = mcspDestroyTrmInfo(descr->mat_info->csr_spsv_upper_info);
            descr->mat_info->csr_spsv_upper_info = nullptr;
        }
        stat = mcspDestroyMatInfo(descr->mat_info);
    }

    delete descr;
    descr = nullptr;
    return stat;
}

mcspStatus_t mcspCreateCsrsm2Info(mcspCsrsm2Info_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspCsrsm2Info();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*info)->mat_info));
}

mcspStatus_t mcspDestroyCsrsm2Info(mcspCsrsm2Info_t info) {
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    if (info->mat_info != nullptr) {
        if (info->mat_info->csr_spsm_lower_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->mat_info->csr_spsm_lower_info);
            info->mat_info->csr_spsm_lower_info = nullptr;
        } else if (info->mat_info->csr_spsm_upper_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->mat_info->csr_spsm_upper_info);
            info->mat_info->csr_spsm_upper_info = nullptr;
        }
        stat = mcspDestroyMatInfo(info->mat_info);
        info->mat_info = nullptr;
    }

    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspSpSM_createDescr(mcspSpSMDescr_t *descr) {
    if (descr == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *descr = new mcspSpSMDescr();
    if (*descr == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*descr)->mat_info));
}

mcspStatus_t mcspSpSM_destroyDescr(mcspSpSMDescr_t descr) {
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (descr == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    if (descr->mat_info != nullptr) {
        if (descr->mat_info->csr_spsm_lower_info != nullptr) {
            stat = mcspDestroyTrmInfo(descr->mat_info->csr_spsm_lower_info);
            descr->mat_info->csr_spsm_lower_info = nullptr;
        } else if (descr->mat_info->csr_spsm_upper_info != nullptr) {
            stat = mcspDestroyTrmInfo(descr->mat_info->csr_spsm_upper_info);
            descr->mat_info->csr_spsm_upper_info = nullptr;
        }
        stat = mcspDestroyMatInfo(descr->mat_info);
    }

    delete descr;
    descr = nullptr;
    return stat;
}

mcspStatus_t mcspCreateCsru2csrInfo(mcspCsru2csrInfo_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspCsru2csrInfo();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }
    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspDestroyCsru2csrInfo(mcspCsru2csrInfo_t info) {
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    if (info->perm != nullptr) {
        if (mcFree(info->perm) != mcSuccess) {
            stat = MCSP_STATUS_INTERNAL_ERROR;
        } else {
            info->perm = nullptr;
        }
    }
    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspCreateBsrsv2Info(mcspBsrsv2Info_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspBsrsv2Info();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*info)->mat_info));
}

mcspStatus_t mcspDestroyBsrsv2Info(mcspBsrsv2Info_t info) {
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    if (info->mat_info != nullptr) {
        if (info->mat_info->bsrsv_lower_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->mat_info->bsrsv_lower_info);
            info->mat_info->bsrsv_lower_info = nullptr;
        } else if (info->mat_info->bsrsv_upper_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->mat_info->bsrsv_upper_info);
            info->mat_info->bsrsv_upper_info = nullptr;
        }
        stat = mcspDestroyMatInfo(info->mat_info);
        info->mat_info = nullptr;
    }

    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspCreateBsrsm2Info(mcspBsrsm2Info_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspBsrsm2Info();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*info)->mat_info));
}

mcspStatus_t mcspDestroyBsrsm2Info(mcspBsrsm2Info_t info) {
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    if (info->mat_info != nullptr) {
        if (info->mat_info->bsrsm_lower_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->mat_info->bsrsm_lower_info);
            info->mat_info->bsrsm_lower_info = nullptr;
        } else if (info->mat_info->bsrsm_upper_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->mat_info->bsrsm_upper_info);
            info->mat_info->bsrsm_upper_info = nullptr;
        }
        stat = mcspDestroyMatInfo(info->mat_info);
        info->mat_info = nullptr;
    }

    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspCreateBsrilu02Info(mcspBsrilu02Info_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspBsrilu02Info();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*info)->bsrilu0_mat));
}

mcspStatus_t mcspDestroyBsrilu02Info(mcspBsrilu02Info_t info) {
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info->bsrilu0_mat != nullptr) {
        if (info->bsrilu0_mat->bsrilu0_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->bsrilu0_mat->bsrilu0_info);
            info->bsrilu0_mat->bsrilu0_info = nullptr;
        }
        stat = mcspDestroyMatInfo(info->bsrilu0_mat);
        info->bsrilu0_mat = nullptr;
    }
    delete info;
    info = nullptr;
    return stat;
}

mcspStatus_t mcspCreateBsric02Info(mcspBsric02Info_t *info) {
    if (info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    *info = new mcspBsric02Info();
    if (*info == nullptr) {
        return MCSP_STATUS_INTERNAL_ERROR;
    }

    return mcspCreateMatInfo(&((*info)->bsric0_mat));
}

mcspStatus_t mcspDestroyBsric02Info(mcspBsric02Info_t info) {
    if (info == nullptr) {
        return MCSP_STATUS_SUCCESS;
    }
    mcspStatus_t stat = MCSP_STATUS_SUCCESS;
    if (info->bsric0_mat != nullptr) {
        if (info->bsric0_mat->bsric0_info != nullptr) {
            stat = mcspDestroyTrmInfo(info->bsric0_mat->bsric0_info);
            info->bsric0_mat->bsric0_info = nullptr;
        }
        stat = mcspDestroyMatInfo(info->bsric0_mat);
        info->bsric0_mat = nullptr;
    }
    delete info;
    info = nullptr;
    return stat;
}
#ifdef __cplusplus
}
#endif
