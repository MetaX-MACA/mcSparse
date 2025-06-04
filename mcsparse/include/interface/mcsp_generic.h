#ifndef INTERFACE_MCSPARSE_GENERIC_H_
#define INTERFACE_MCSPARSE_GENERIC_H_

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Axpby
mcsparseStatus_t mcsparseAxpby(mcsparseHandle_t handle, const void* alpha, mcsparseSpVecDescr_t vecX, const void* beta,
                               mcsparseDnVecDescr_t vecY);

// Gather
mcsparseStatus_t mcsparseGather(mcsparseHandle_t handle, mcsparseDnVecDescr_t vecY, mcsparseSpVecDescr_t vecX);

// Scatter
mcsparseStatus_t mcsparseScatter(mcsparseHandle_t handle, mcsparseSpVecDescr_t vecX, mcsparseDnVecDescr_t vecY);

// Rot
mcsparseStatus_t mcsparseRot(mcsparseHandle_t handle, const void* c_coeff, const void* s_coeff,
                             mcsparseSpVecDescr_t vecX, mcsparseDnVecDescr_t vecY);

// SparseToDense
mcsparseStatus_t mcsparseSparseToDense_bufferSize(mcsparseHandle_t handle, mcsparseSpMatDescr_t matA,
                                                  mcsparseDnMatDescr_t matB, mcsparseSparseToDenseAlg_t alg,
                                                  size_t* bufferSize);
mcsparseStatus_t mcsparseSparseToDense(mcsparseHandle_t handle, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                       mcsparseSparseToDenseAlg_t alg, void* externalBuffer);

// DenseToSparse
mcsparseStatus_t mcsparseDenseToSparse_bufferSize(mcsparseHandle_t handle, mcsparseDnMatDescr_t matA,
                                                  mcsparseSpMatDescr_t matB, mcsparseDenseToSparseAlg_t alg,
                                                  size_t* bufferSize);
mcsparseStatus_t mcsparseDenseToSparse_analysis(mcsparseHandle_t handle, mcsparseDnMatDescr_t matA,
                                                mcsparseSpMatDescr_t matB, mcsparseDenseToSparseAlg_t alg,
                                                void* externalBuffer);
mcsparseStatus_t mcsparseDenseToSparse_convert(mcsparseHandle_t handle, mcsparseDnMatDescr_t matA,
                                               mcsparseSpMatDescr_t matB, mcsparseDenseToSparseAlg_t alg,
                                               void* externalBuffer);

// SpMV
mcsparseStatus_t mcsparseSpMV(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                              mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX, const void* beta,
                              mcsparseDnVecDescr_t vecY, macaDataType computeType, mcsparseSpMVAlg_t alg,
                              void* externalBuffer);

mcsparseStatus_t mcsparseSpMV_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                                         mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX, const void* beta,
                                         mcsparseDnVecDescr_t vecY, macaDataType computeType, mcsparseSpMVAlg_t alg,
                                         size_t* bufferSize);

// SpMM
mcsparseStatus_t mcsparseSpMM_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                         const void* beta, mcsparseDnMatDescr_t matC, macaDataType computeType,
                                         mcsparseSpMMAlg_t alg, size_t* bufferSize);

mcsparseStatus_t mcsparseSpMM_preprocess(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                         const void* beta, mcsparseDnMatDescr_t matC, macaDataType computeType,
                                         mcsparseSpMMAlg_t alg, void* externalBuffer);

mcsparseStatus_t mcsparseSpMM(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                              const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB, const void* beta,
                              mcsparseDnMatDescr_t matC, macaDataType computeType, mcsparseSpMMAlg_t alg,
                              void* externalBuffer);

// SpGEMM
mcsparseStatus_t mcsparseSpGEMM_createDescr(mcsparseSpGEMMDescr_t* spgemm_descr);
mcsparseStatus_t mcsparseSpGEMM_destroyDescr(mcsparseSpGEMMDescr_t spgemm_descr);
mcsparseStatus_t mcsparseSpGEMM_workEstimation(mcsparseHandle_t handle, mcsparseOperation_t opA,
                                               mcsparseOperation_t opB, const void* alpha, mcsparseSpMatDescr_t matA,
                                               mcsparseSpMatDescr_t matB, const void* beta, mcsparseSpMatDescr_t matC,
                                               macaDataType computeType, mcsparseSpGEMMAlg_t alg,
                                               mcsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1,
                                               void* externalBuffer1);

mcsparseStatus_t mcsparseSpGEMM_compute(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                        const void* alpha, mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                        const void* beta, mcsparseSpMatDescr_t matC, macaDataType computeType,
                                        mcsparseSpGEMMAlg_t alg, mcsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2,
                                        void* externalBuffer2);

mcsparseStatus_t mcsparseSpGEMM_copy(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                     const void* alpha, mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                     const void* beta, mcsparseSpMatDescr_t matC, macaDataType computeType,
                                     mcsparseSpGEMMAlg_t alg, mcsparseSpGEMMDescr_t spgemmDescr);

// SpGEMMreuse
mcsparseStatus_t mcsparseSpGEMMreuse_workEstimation(mcsparseHandle_t handle, mcsparseOperation_t opA,
                                                    mcsparseOperation_t opB, mcsparseSpMatDescr_t matA,
                                                    mcsparseSpMatDescr_t matB, mcsparseSpMatDescr_t matC,
                                                    mcsparseSpGEMMAlg_t alg, mcsparseSpGEMMDescr_t spgemmDescr,
                                                    size_t* bufferSize1, void* externalBuffer1);

mcsparseStatus_t mcsparseSpGEMMreuse_nnz(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                         mcsparseSpMatDescr_t matC, mcsparseSpGEMMAlg_t alg,
                                         mcsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2,
                                         size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize4,
                                         void* externalBuffer4);

mcsparseStatus_t mcsparseSpGEMMreuse_copy(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                          mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                          mcsparseSpMatDescr_t matC, mcsparseSpGEMMAlg_t alg,
                                          mcsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize5,
                                          void* externalBuffer5);

mcsparseStatus_t mcsparseSpGEMMreuse_compute(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                             const void* alpha, mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                             const void* beta, mcsparseSpMatDescr_t matC, macaDataType computeType,
                                             mcsparseSpGEMMAlg_t alg, mcsparseSpGEMMDescr_t spgemmDescr);

// SDDMM
mcsparseStatus_t mcsparseSDDMM_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                          const void* alpha, const mcsparseDnMatDescr_t A, const mcsparseDnMatDescr_t B,
                                          const void* beta, mcsparseSpMatDescr_t C, macaDataType compute_type,
                                          mcsparseSDDMMAlg_t alg, size_t* buffer_size);

mcsparseStatus_t mcsparseSDDMM_preprocess(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                          const void* alpha, const mcsparseDnMatDescr_t A, const mcsparseDnMatDescr_t B,
                                          const void* beta, mcsparseSpMatDescr_t C, macaDataType compute_type,
                                          mcsparseSDDMMAlg_t alg, void* temp_buffer);

mcsparseStatus_t mcsparseSDDMM(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                               const void* alpha, const mcsparseDnMatDescr_t A, const mcsparseDnMatDescr_t B,
                               const void* beta, mcsparseSpMatDescr_t C, macaDataType compute_type,
                               mcsparseSDDMMAlg_t alg, void* temp_buffer);

// SpVV
mcsparseStatus_t mcsparseSpVV_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t op_x, mcsparseSpVecDescr_t vec_x,
                                         mcsparseDnVecDescr_t vec_y, void* result, macaDataType compute_type,
                                         size_t* buffer_size);

mcsparseStatus_t mcsparseSpVV(mcsparseHandle_t handle, mcsparseOperation_t op_x, mcsparseSpVecDescr_t vec_x,
                              mcsparseDnVecDescr_t vec_y, void* result, macaDataType compute_type, void* temp_buffer);

// SpSV
mcsparseStatus_t mcsparseSpSV_createDescr(mcsparseSpSVDescr_t* descr);

mcsparseStatus_t mcsparseSpSV_destroyDescr(mcsparseSpSVDescr_t descr);

mcsparseStatus_t mcsparseSpSV_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                                         mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX,
                                         mcsparseDnVecDescr_t vecY, macaDataType computeType, mcsparseSpSVAlg_t alg,
                                         mcsparseSpSVDescr_t spsvDescr, size_t* bufferSize);

mcsparseStatus_t mcsparseSpSV_analysis(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                                       mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX, mcsparseDnVecDescr_t vecY,
                                       macaDataType computeType, mcsparseSpSVAlg_t alg, mcsparseSpSVDescr_t spsvDescr,
                                       void* externalBuffer);

mcsparseStatus_t mcsparseSpSV_solve(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                                    mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX, mcsparseDnVecDescr_t vecY,
                                    macaDataType computeType, mcsparseSpSVAlg_t alg, mcsparseSpSVDescr_t spsvDescr);

// SpSM
mcsparseStatus_t mcsparseSpSM_createDescr(mcsparseSpSMDescr_t* descr);

mcsparseStatus_t mcsparseSpSM_destroyDescr(mcsparseSpSMDescr_t descr);

mcsparseStatus_t mcsparseSpSM_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                         mcsparseDnMatDescr_t matC, macaDataType computeType, mcsparseSpSMAlg_t alg,
                                         mcsparseSpSMDescr_t spsmDescr, size_t* bufferSize);

mcsparseStatus_t mcsparseSpSM_analysis(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                       const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                       mcsparseDnMatDescr_t matC, macaDataType computeType, mcsparseSpSMAlg_t alg,
                                       mcsparseSpSMDescr_t spsmDescr, void* externalBuffer);

mcsparseStatus_t mcsparseSpSM_solve(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                    const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                    mcsparseDnMatDescr_t matC, macaDataType computeType, mcsparseSpSMAlg_t alg,
                                    mcsparseSpSMDescr_t spsmDescr);

mcsparseStatus_t mcsparseCooSetStridedBatch(mcsparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride);

mcsparseStatus_t mcsparseCsrSetStridedBatch(mcsparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride,
                                            int64_t columnsValuesBatchStride);

mcsparseStatus_t mcsparseDnMatSetStridedBatch(mcsparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride);

mcsparseStatus_t mcsparseDnMatGetStridedBatch(mcsparseDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride);

// AOS COO
mcsparseStatus_t mcsparseCreateCooAoS(mcsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                      void* cooInd, void* cooValues, mcsparseIndexType_t cooIdxType,
                                      mcsparseIndexBase_t idxBase, macaDataType valueType);

mcsparseStatus_t mcsparseCooAoSGet(mcsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz,
                                   void** cooInd,     // COO indices
                                   void** cooValues,  // COO values
                                   mcsparseIndexType_t* idxType, mcsparseIndexBase_t* idxBase, macaDataType* valueType);

#ifdef __cplusplus
}
#endif

#endif  // end of INTERFACE_MCSPARSE_GENERIC_H_