#ifndef MCSPARSE_TBD_GENERIC_H
#define MCSPARSE_TBD_GENERIC_H

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

mcsparseStatus_t mcsparseSpMatSetStridedBatch(mcsparseSpMatDescr_t spMatDescr, int batchCount);

mcsparseStatus_t mcsparseSpMatGetStridedBatch(mcsparseSpMatDescr_t spMatDescr, int* batchCount);

//------------------------------------------------------------------------------
// ### COO ###

// #############################################################################
// # SAMPLED DENSE-DENSE MATRIX MULTIPLICATION
// #############################################################################

mcsparseStatus_t mcsparseConstrainedGeMM(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         const void* alpha, mcsparseDnMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                         const void* beta, mcsparseSpMatDescr_t matC, macaDataType computeType,
                                         void* externalBuffer);

mcsparseStatus_t mcsparseConstrainedGeMM_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA,
                                                    mcsparseOperation_t opB, const void* alpha,
                                                    mcsparseDnMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                                    const void* beta, mcsparseSpMatDescr_t matC,
                                                    macaDataType computeType, size_t* bufferSize);

// #############################################################################
// # GENERIC APIs WITH CUSTOM OPERATORS (PREVIEW)
// #############################################################################

mcsparseStatus_t mcsparseSpMMOp_createPlan(mcsparseHandle_t handle, mcsparseSpMMOpPlan_t* plan, mcsparseOperation_t opA,
                                           mcsparseOperation_t opB, mcsparseSpMatDescr_t matA,
                                           mcsparseDnMatDescr_t matB, mcsparseDnMatDescr_t matC,
                                           macaDataType computeType, mcsparseSpMMOpAlg_t alg,
                                           const void* addOperationNvvmBuffer, size_t addOperationBufferSize,
                                           const void* mulOperationNvvmBuffer, size_t mulOperationBufferSize,
                                           const void* epilogueNvvmBuffer, size_t epilogueBufferSize,
                                           size_t* SpMMWorkspaceSize);

mcsparseStatus_t mcsparseSpMMOp(mcsparseSpMMOpPlan_t plan, void* externalBuffer);

mcsparseStatus_t mcsparseSpMMOp_destroyPlan(mcsparseSpMMOpPlan_t plan);

#ifdef __cplusplus
}
#endif

#endif