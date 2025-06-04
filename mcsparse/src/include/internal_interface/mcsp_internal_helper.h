#ifndef COMMON_MCSP_INTERNAL_HELPER_H_
#define COMMON_MCSP_INTERNAL_HELPER_H_

#include "mcr/mc_runtime.h"

#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspCreateHandle(mcspHandle_t* handle);
mcspStatus_t mcspDestroyHandle(mcspHandle_t handle);

mcspStatus_t mcspGetVersion(mcspHandle_t handle, int* version);
mcspStatus_t mcspGetProperty(libraryPropertyType type, int* value);

mcspStatus_t mcspSetStream(mcspHandle_t handle, mcStream_t sid);
mcspStatus_t mcspGetStream(const mcspHandle_t handle, mcStream_t* sid);

mcStream_t mcspGetStreamInternal(const mcspHandle_t handle);

mcspStatus_t mcspCreateMatDescr(mcspMatDescr_t* descr);
mcspStatus_t mcspDestroyMatDescr(mcspMatDescr_t descr);
mcspStatus_t mcspCopyMatDescr(mcspMatDescr_t dest, const mcspMatDescr_t src);

mcspStatus_t mcspCreateMatInfo(mcspMatInfo_t* info);
mcspStatus_t mcspDestroyMatInfo(mcspMatInfo_t info);

mcspStatus_t mcspSetPointerMode(mcspHandle_t handle, mcsparsePointerMode_t pointer_mode);
mcspStatus_t mcspGetPointerMode(const mcspHandle_t handle, mcsparsePointerMode_t* pointer_mode);

mcspStatus_t mcspSetMatIndexBase(mcspMatDescr_t descr, mcsparseIndexBase_t index_base);
mcsparseIndexBase_t mcspGetMatIndexBase(const mcspMatDescr_t descr);

mcspStatus_t mcspSetMatType(mcspMatDescr_t descr, mcsparseMatrixType_t matrix_type);
mcsparseMatrixType_t mcspGetMatType(const mcspMatDescr_t descr);

mcspStatus_t mcspSetMatFillMode(mcspMatDescr_t descr, mcsparseFillMode_t mode);
mcsparseFillMode_t mcspGetMatFillMode(const mcspMatDescr_t descr);

mcspStatus_t mcspSetMatDiagType(mcspMatDescr_t descr, mcsparseDiagType_t diag_type);
mcsparseDiagType_t mcspGetMatDiagType(const mcspMatDescr_t descr);

mcspStatus_t mcspSetStorageMode(mcspMatDescr_t descr, mcsparseStorageMode_t storage_mode);
mcspStatus_t mcspGetStorageMode(const mcspMatDescr_t descr, mcsparseStorageMode_t* storage_mode);

mcspStatus_t mcspCreateDnVec(mcspDnVecDescr_t* dnVecDescr, int64_t size, void* values, macaDataType valueType);
mcspStatus_t mcspDestroyDnVec(mcspDnVecDescr_t dnVecDescr);
mcspStatus_t mcspDnVecGet(mcspDnVecDescr_t dnVecDescr, int64_t* size, void** values, macaDataType* valueType);
mcspStatus_t mcspDnVecGetValues(mcspDnVecDescr_t dnVecDescr, void** values);
mcspStatus_t mcspDnVecSetValues(mcspDnVecDescr_t dnVecDescr, void* values);

mcspStatus_t mcspCreateCoo(mcspSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooRowInd,
                           void* cooColInd, void* cooValues, mcsparseIndexType_t cooIdxType,
                           mcsparseIndexBase_t idxBase, macaDataType valueType);
mcspStatus_t mcspCooGet(mcspSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd,
                        void** cooColInd, void** cooValues, mcsparseIndexType_t* idxType, mcsparseIndexBase_t* idxBase,
                        macaDataType* valueType);
mcspStatus_t mcspCooSetPointers(mcspSpMatDescr_t spMatDescr, void* cooRows, void* cooColumns, void* cooValues);

mcspStatus_t mcspCreateCooAos(mcspSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooInd,
                              void* cooValues, mcsparseIndexType_t cooIdxType, mcsparseIndexBase_t idxBase,
                              macaDataType valueType);
mcspStatus_t mcspCooAosGet(mcspSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooInd,
                           void** cooValues, mcsparseIndexType_t* idxType, mcsparseIndexBase_t* idxBase,
                           macaDataType* valueType);

mcspStatus_t mcspCreateCsr(mcspSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets,
                           void* csrColInd, void* csrValues, mcsparseIndexType_t csrRowOffsetsType,
                           mcsparseIndexType_t csrColIndType, mcsparseIndexBase_t idxBase, macaDataType valueType);
mcspStatus_t mcspCsrGet(mcspSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csrRowOffsets,
                        void** csrColInd, void** csrValues, mcsparseIndexType_t* csrRowOffsetsType,
                        mcsparseIndexType_t* csrColIndType, mcsparseIndexBase_t* idxBase, macaDataType* valueType);
mcspStatus_t mcspCsrSetPointers(mcspSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd, void* csrValues);

mcspStatus_t mcspCreateCsc(mcspSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cscColOffsets,
                           void* cscRowInd, void* cscValues, mcsparseIndexType_t cscColOffsetsType,
                           mcsparseIndexType_t cscRowIndType, mcsparseIndexBase_t idxBase, macaDataType valueType);
mcspStatus_t mcspCscSetPointers(mcspSpMatDescr_t spMatDescr, void* cscColOffsets, void* cscRowInd, void* cscValues);

mcspStatus_t mcspCreateBlockedEll(mcspSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize,
                                  int64_t ellCols, void* ellColInd, void* ellValue, mcsparseIndexType_t ellIdxType,
                                  mcsparseIndexBase_t idxBase, macaDataType valueType);
mcspStatus_t mcspBlockedEllGet(mcsparseSpMatDescr_t spMatDescr, int64_t* rowNum, int64_t* colNum, int64_t* ellBlockSize,
                               int64_t* ellCols, void** ellColInd, void** ellValue, mcsparseIndexType_t* ellIdxType,
                               mcsparseIndexBase_t* idxBase, macaDataType* valueType);

mcspStatus_t mcspDestroySpMat(mcspSpMatDescr_t spMatDescr);
mcspStatus_t mcspSpMatGetSize(mcspSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz);
mcspStatus_t mcspSpMatGetFormat(mcspSpMatDescr_t spMatDescr, mcsparseFormat_t* format);
mcspStatus_t mcspSpMatGetIndexBase(mcspSpMatDescr_t spMatDescr, mcsparseIndexBase_t* idxBase);
mcspStatus_t mcspSpMatGetValues(mcspSpMatDescr_t spMatDescr, void** values);
mcspStatus_t mcspSpMatSetValues(mcspSpMatDescr_t spMatDescr, void* values);
mcspStatus_t mcspSpMatGetAttribute(mcspSpMatDescr_t spMatDescr, mcsparseSpMatAttribute_t attribute, void* data,
                                   size_t dataSize);
mcspStatus_t mcspSpMatSetAttribute(mcspSpMatDescr_t spMatDescr, mcsparseSpMatAttribute_t attribute, const void* data,
                                   size_t dataSize);

mcspStatus_t mcspCreateDnMat(mcspDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values,
                             macaDataType valueType, mcsparseOrder_t order);
mcspStatus_t mcspDestroyDnMat(mcspDnMatDescr_t dnMatDescr);
mcspStatus_t mcspDnMatGet(mcspDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values,
                          macaDataType* type, mcsparseOrder_t* order);
mcspStatus_t mcspDnMatGetValues(mcspDnMatDescr_t dnMatDescr, void** values);
mcspStatus_t mcspDnMatSetValues(mcspDnMatDescr_t dnMatDescr, void* values);

mcspStatus_t mcspCreateCsrsv2Info(mcspCsrsv2Info_t* info);
mcspStatus_t mcspDestroyCsrsv2Info(mcspCsrsv2Info_t info);

mcspStatus_t mcspCreateCsrsm2Info(mcspCsrsm2Info_t* info);
mcspStatus_t mcspDestroyCsrsm2Info(mcspCsrsm2Info_t info);

mcspStatus_t mcspCreateSpVec(mcspSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values,
                             mcsparseIndexType_t idxType, mcsparseIndexBase_t idxBase, macaDataType valueType);
mcspStatus_t mcspDestroySpVec(mcspSpVecDescr_t spVecDescr);
mcspStatus_t mcspSpVecGet(mcspSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values,
                          mcsparseIndexType_t* idxType, mcsparseIndexBase_t* idxBase, macaDataType* valueType);
mcspStatus_t mcspSpVecGetIndexBase(mcspSpVecDescr_t spVecDescr, mcsparseIndexBase_t* idxBase);
mcspStatus_t mcspSpVecGetValues(mcspSpVecDescr_t spVecDescr, void** values);
mcspStatus_t mcspSpVecSetValues(mcspSpVecDescr_t spVecDescr, void* values);

mcspStatus_t mcspCreateCsrilu02Info(mcspCsrilu02Info_t* info);
mcspStatus_t mcspDestroyCsrilu02Info(mcspCsrilu02Info_t info);

mcspStatus_t mcspCreateCsric02Info(mcspCsric02Info_t* info);
mcspStatus_t mcspDestroyCsric02Info(mcspCsric02Info_t info);

mcspStatus_t mcspCreateColorInfo(mcspColorInfo_t* info);
mcspStatus_t mcspDestroyColorInfo(mcspColorInfo_t info);
mcspStatus_t mcspSetColorAlgs(mcspColorInfo_t info, mcsparseColorAlg_t alg);
mcspStatus_t mcspGetColorAlgs(mcspColorInfo_t info, mcsparseColorAlg_t* alg);

mcspStatus_t mcspCreateCsrgemm2Info(mcspCsrgemm2Info_t* info);
mcspStatus_t mcspDestroyCsrgemm2Info(mcspCsrgemm2Info_t info);

mcspStatus_t mcspCreatePruneInfo(mcspPruneInfo_t* info);
mcspStatus_t mcspDestroyPruneInfo(mcspPruneInfo_t info);

mcspStatus_t mcspCreateCsru2csrInfo(mcspCsru2csrInfo_t* info);
mcspStatus_t mcspDestroyCsru2csrInfo(mcspCsru2csrInfo_t info);

mcspStatus_t mcspCreateBsrsv2Info(mcspBsrsv2Info_t* info);
mcspStatus_t mcspDestroyBsrsv2Info(mcspBsrsv2Info_t info);

mcspStatus_t mcspCreateBsrsm2Info(mcspBsrsm2Info_t* info);
mcspStatus_t mcspDestroyBsrsm2Info(mcspBsrsm2Info_t info);

mcspStatus_t mcspCreateBsrilu02Info(mcspBsrilu02Info_t* info);
mcspStatus_t mcspDestroyBsrilu02Info(mcspBsrilu02Info_t info);

mcspStatus_t mcspCreateBsric02Info(mcspBsric02Info_t* info);
mcspStatus_t mcspDestroyBsric02Info(mcspBsric02Info_t info);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSP_HELPER_H_
