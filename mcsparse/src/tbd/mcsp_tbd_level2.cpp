#include "interface/tbd/mcsp_tbd_level2.h"

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

mcsparseStatus_t mcspXcsrsv2_zeroPivot(mcsparseHandle_t handle, mcsparseCsrsv2Info_t info, int* position) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspScsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                        const mcsparseMatDescr_t descrA, float* csrSortedValA,
                                        const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                        mcsparseCsrsv2Info_t info, int* pBufferSizeInBytes) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspDcsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                        const mcsparseMatDescr_t descrA, double* csrSortedValA,
                                        const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                        mcsparseCsrsv2Info_t info, int* pBufferSizeInBytes) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspCcsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                        const mcsparseMatDescr_t descrA, mcFloatComplex* csrSortedValA,
                                        const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                        mcsparseCsrsv2Info_t info, int* pBufferSizeInBytes) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspZcsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                        const mcsparseMatDescr_t descrA, mcDoubleComplex* csrSortedValA,
                                        const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                        mcsparseCsrsv2Info_t info, int* pBufferSizeInBytes) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspScsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                           const mcsparseMatDescr_t descrA, float* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                           mcsparseCsrsv2Info_t info, size_t* pBufferSize) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspDcsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                           const mcsparseMatDescr_t descrA, double* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                           mcsparseCsrsv2Info_t info, size_t* pBufferSize) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspCcsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                           const mcsparseMatDescr_t descrA, mcFloatComplex* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                           mcsparseCsrsv2Info_t info, size_t* pBufferSize) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspZcsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                           const mcsparseMatDescr_t descrA, mcDoubleComplex* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                           mcsparseCsrsv2Info_t info, size_t* pBufferSize) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspScsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcsparseMatDescr_t descrA, const float* csrSortedValA,
                                      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspDcsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcsparseMatDescr_t descrA, const double* csrSortedValA,
                                      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspCcsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcsparseMatDescr_t descrA, const mcFloatComplex* csrSortedValA,
                                      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspZcsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcsparseMatDescr_t descrA, const mcDoubleComplex* csrSortedValA,
                                      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspScsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                   const float* alpha, const mcsparseMatDescr_t descrA, const float* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, mcsparseCsrsv2Info_t info,
                                   const float* f, float* x, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspDcsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                   const double* alpha, const mcsparseMatDescr_t descrA, const double* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, mcsparseCsrsv2Info_t info,
                                   const double* f, double* x, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspCcsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                   const mcFloatComplex* alpha, const mcsparseMatDescr_t descrA,
                                   const mcFloatComplex* csrSortedValA, const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA, mcsparseCsrsv2Info_t info, const mcFloatComplex* f,
                                   mcFloatComplex* x, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

mcsparseStatus_t mcspZcsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                   const mcDoubleComplex* alpha, const mcsparseMatDescr_t descrA,
                                   const mcDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA, mcsparseCsrsv2Info_t info, const mcDoubleComplex* f,
                                   mcDoubleComplex* x, mcsparseSolvePolicy_t policy, void* pBuffer) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif
