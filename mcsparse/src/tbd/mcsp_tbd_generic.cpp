#include "interface/tbd/mcsp_tbd_generic.h"

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// #############################################################################
// # SPARSE MATRIX API
// #############################################################################

mcsparseStatus_t mcsparseSpMatSetStridedBatch(mcsparseSpMatDescr_t spMatDescr, int batchCount) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcsparseSpMatGetStridedBatch(mcsparseSpMatDescr_t spMatDescr, int* batchCount) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

//------------------------------------------------------------------------------
// ### COO ###

//------------------------------------------------------------------------------
// ### Dense matrix APIs ###

// #############################################################################
// # SAMPLED DENSE-DENSE MATRIX MULTIPLICATION
// #############################################################################

mcsparseStatus_t mcsparseConstrainedGeMM(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         const void* alpha, mcsparseDnMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                         const void* beta, mcsparseSpMatDescr_t matC, macaDataType computeType,
                                         void* externalBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcsparseConstrainedGeMM_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA,
                                                    mcsparseOperation_t opB, const void* alpha,
                                                    mcsparseDnMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                                    const void* beta, mcsparseSpMatDescr_t matC,
                                                    macaDataType computeType, size_t* bufferSize) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

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
                                           size_t* SpMMWorkspaceSize) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcsparseSpMMOp(mcsparseSpMMOpPlan_t plan, void* externalBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcsparseSpMMOp_destroyPlan(mcsparseSpMMOpPlan_t plan) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif
