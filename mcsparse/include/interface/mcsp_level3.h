#ifndef INTERFACE_MCSPARSE_LEVEL3_H_
#define INTERFACE_MCSPARSE_LEVEL3_H_

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// gemmi
mcsparseStatus_t mcsparseSgemmi(mcsparseHandle_t handle, int m, int n, int k, int nnz, const float *alpha,
                                const float *A, int lda, const float *csc_vals, const int *csc_cols,
                                const int *csc_rows, const float *beta, float *C, int ldc);
mcsparseStatus_t mcsparseDgemmi(mcsparseHandle_t handle, int m, int n, int k, int nnz, const double *alpha,
                                const double *A, int lda, const double *csc_vals, const int *csc_cols,
                                const int *csc_rows, const double *beta, double *C, int ldc);
mcsparseStatus_t mcsparseCgemmi(mcsparseHandle_t handle, int m, int n, int k, int nnz, const mcComplex *alpha,
                                const mcComplex *A, int lda, const mcComplex *csc_vals, const int *csc_cols,
                                const int *csc_rows, const mcComplex *beta, mcComplex *C, int ldc);
mcsparseStatus_t mcsparseZgemmi(mcsparseHandle_t handle, int m, int n, int k, int nnz, const mcDoubleComplex *alpha,
                                const mcDoubleComplex *A, int lda, const mcDoubleComplex *csc_vals, const int *csc_cols,
                                const int *csc_rows, const mcDoubleComplex *beta, mcDoubleComplex *C, int ldc);

// csrsm2
mcsparseStatus_t mcsparseXcsrsm2_zeroPivot(mcsparseHandle_t handle, mcsparseCsrsm2Info_t info, int *position);

mcsparseStatus_t mcsparseScsrsm2_bufferSizeExt(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                               mcsparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
                                               const mcsparseMatDescr_t descrA, const float *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B,
                                               int ldb, mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                               size_t *pBufferSize);

mcsparseStatus_t mcsparseDcsrsm2_bufferSizeExt(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                               mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                               const double *alpha, const mcsparseMatDescr_t descrA,
                                               const double *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, const double *B, int ldb,
                                               mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                               size_t *pBufferSize);

mcsparseStatus_t mcsparseCcsrsm2_bufferSizeExt(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                               mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                               const mcComplex *alpha, const mcsparseMatDescr_t descrA,
                                               const mcComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, const mcComplex *B, int ldb,
                                               mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                               size_t *pBufferSize);

mcsparseStatus_t mcsparseZcsrsm2_bufferSizeExt(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                               mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                               const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                               const mcDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA, const mcDoubleComplex *B, int ldb,
                                               mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                               size_t *pBufferSize);

mcsparseStatus_t mcsparseScsrsm2_analysis(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                          mcsparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
                                          const mcsparseMatDescr_t descrA, const float *csrSortedValA,
                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B,
                                          int ldb, mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                          void *pBuffer);

mcsparseStatus_t mcsparseDcsrsm2_analysis(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                          mcsparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
                                          const mcsparseMatDescr_t descrA, const double *csrSortedValA,
                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *B,
                                          int ldb, mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                          void *pBuffer);

mcsparseStatus_t mcsparseCcsrsm2_analysis(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                          mcsparseOperation_t transB, int m, int nrhs, int nnz, const mcComplex *alpha,
                                          const mcsparseMatDescr_t descrA, const mcComplex *csrSortedValA,
                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA, const mcComplex *B,
                                          int ldb, mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                          void *pBuffer);

mcsparseStatus_t mcsparseZcsrsm2_analysis(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                          mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                          const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                          const mcDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                          const int *csrSortedColIndA, const mcDoubleComplex *B, int ldb,
                                          mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseScsrsm2_solve(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
                                       const mcsparseMatDescr_t descrA, const float *csrSortedValA,
                                       const int *csrSortedRowPtrA, const int *csrSortedColIndA, float *B, int ldb,
                                       mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseDcsrsm2_solve(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
                                       const mcsparseMatDescr_t descrA, const double *csrSortedValA,
                                       const int *csrSortedRowPtrA, const int *csrSortedColIndA, double *B, int ldb,
                                       mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseCcsrsm2_solve(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, int m, int nrhs, int nnz, const mcComplex *alpha,
                                       const mcsparseMatDescr_t descrA, const mcComplex *csrSortedValA,
                                       const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcComplex *B, int ldb,
                                       mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseZcsrsm2_solve(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                       const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                       const mcDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                       const int *csrSortedColIndA, mcDoubleComplex *B, int ldb,
                                       mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

// bsrmm
mcsparseStatus_t mcsparseSbsrmm(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const float *alpha,
                                const mcsparseMatDescr_t descrA, const float *bsrSortedValA,
                                const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize,
                                const float *B, const int ldb, const float *beta, float *C, int ldc);

mcsparseStatus_t mcsparseDbsrmm(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const double *alpha,
                                const mcsparseMatDescr_t descrA, const double *bsrSortedValA,
                                const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize,
                                const double *B, const int ldb, const double *beta, double *C, int ldc);

mcsparseStatus_t mcsparseCbsrmm(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                mcsparseOperation_t transB, int mb, int n, int kb, int nnzb,
                                const mcFloatComplex *alpha, const mcsparseMatDescr_t descrA,
                                const mcFloatComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                const int *bsrSortedColIndA, const int blockSize, const mcFloatComplex *B,
                                const int ldb, const mcFloatComplex *beta, mcFloatComplex *C, int ldc);

mcsparseStatus_t mcsparseZbsrmm(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                mcsparseOperation_t transB, int mb, int n, int kb, int nnzb,
                                const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                const mcDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                const int *bsrSortedColIndA, const int blockSize, const mcDoubleComplex *B,
                                const int ldb, const mcDoubleComplex *beta, mcDoubleComplex *C, int ldc);

// csrmm2
mcsparseStatus_t mcsparseScsrmm2(mcsparseHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m,
                                 int n, int k, int nnz, const float *alpha, const mcsparseMatDescr_t descrA,
                                 const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *B,
                                 int ldb, const float *beta, float *C, int ldc);

mcsparseStatus_t mcsparseDcsrmm2(mcsparseHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m,
                                 int n, int k, int nnz, const double *alpha, const mcsparseMatDescr_t descrA,
                                 const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *B,
                                 int ldb, const double *beta, double *C, int ldc);

mcsparseStatus_t mcsparseCcsrmm2(mcsparseHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m,
                                 int n, int k, int nnz, const mcComplex *alpha, const mcsparseMatDescr_t descrA,
                                 const mcComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                 const mcComplex *B, int ldb, const mcComplex *beta, mcComplex *C, int ldc);

mcsparseStatus_t mcsparseZcsrmm2(mcsparseHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m,
                                 int n, int k, int nnz, const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                 const mcDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                 const mcDoubleComplex *B, int ldb, const mcDoubleComplex *beta, mcDoubleComplex *C,
                                 int ldc);

// csrmm
mcsparseStatus_t mcsparseScsrmm(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int k, int nnz,
                                const float *alpha, const mcsparseMatDescr_t descrA, const float *csrValA,
                                const int *csrRowPtrA, const int *csrColIndA, const float *B, int ldb,
                                const float *beta, float *C, int ldc);

mcsparseStatus_t mcsparseDcsrmm(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int k, int nnz,
                                const double *alpha, const mcsparseMatDescr_t descrA, const double *csrValA,
                                const int *csrRowPtrA, const int *csrColIndA, const double *B, int ldb,
                                const double *beta, double *C, int ldc);

mcsparseStatus_t mcsparseCcsrmm(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int k, int nnz,
                                const mcComplex *alpha, const mcsparseMatDescr_t descrA, const mcComplex *csrValA,
                                const int *csrRowPtrA, const int *csrColIndA, const mcComplex *B, int ldb,
                                const mcComplex *beta, mcComplex *C, int ldc);

mcsparseStatus_t mcsparseZcsrmm(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int k, int nnz,
                                const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                const mcDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                const mcDoubleComplex *B, int ldb, const mcDoubleComplex *beta, mcDoubleComplex *C,
                                int ldc);

// bsrsm
mcsparseStatus_t mcsparseXbsrsm2_zeroPivot(mcsparseHandle_t handle, mcsparseBsrsm2Info_t info, int *position);

mcsparseStatus_t mcsparseSbsrsm2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, mcsparseOperation_t transXY, int mb, int n,
                                            int nnzb, const mcsparseMatDescr_t descrA, float *bsrSortedVal,
                                            const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                            mcsparseBsrsm2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseDbsrsm2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, mcsparseOperation_t transXY, int mb, int n,
                                            int nnzb, const mcsparseMatDescr_t descrA, double *bsrSortedVal,
                                            const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                            mcsparseBsrsm2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseCbsrsm2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, mcsparseOperation_t transXY, int mb, int n,
                                            int nnzb, const mcsparseMatDescr_t descrA, mcComplex *bsrSortedVal,
                                            const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                            mcsparseBsrsm2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseZbsrsm2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, mcsparseOperation_t transXY, int mb, int n,
                                            int nnzb, const mcsparseMatDescr_t descrA, mcDoubleComplex *bsrSortedVal,
                                            const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                            mcsparseBsrsm2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseSbsrsm2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, mcsparseOperation_t transB, int mb, int n,
                                               int nnzb, const mcsparseMatDescr_t descrA, float *bsrSortedVal,
                                               const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                               mcsparseBsrsm2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseDbsrsm2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, mcsparseOperation_t transB, int mb, int n,
                                               int nnzb, const mcsparseMatDescr_t descrA, double *bsrSortedVal,
                                               const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                               mcsparseBsrsm2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseCbsrsm2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, mcsparseOperation_t transB, int mb, int n,
                                               int nnzb, const mcsparseMatDescr_t descrA, mcComplex *bsrSortedVal,
                                               const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                               mcsparseBsrsm2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseZbsrsm2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, mcsparseOperation_t transB, int mb, int n,
                                               int nnzb, const mcsparseMatDescr_t descrA, mcDoubleComplex *bsrSortedVal,
                                               const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                               mcsparseBsrsm2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseSbsrsm2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                          const mcsparseMatDescr_t descrA, const float *bsrSortedVal,
                                          const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                          mcsparseBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseDbsrsm2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                          const mcsparseMatDescr_t descrA, const double *bsrSortedVal,
                                          const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                          mcsparseBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseCbsrsm2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                          const mcsparseMatDescr_t descrA, const mcComplex *bsrSortedVal,
                                          const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                          mcsparseBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseZbsrsm2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                          const mcsparseMatDescr_t descrA, const mcDoubleComplex *bsrSortedVal,
                                          const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                          mcsparseBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseSbsrsm2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transXY, int mb, int n, int nnzb, const float *alpha,
                                       const mcsparseMatDescr_t descrA, const float *bsrSortedVal,
                                       const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                       mcsparseBsrsm2Info_t info, const float *B, int ldb, float *X, int ldx,
                                       mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseDbsrsm2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transXY, int mb, int n, int nnzb, const double *alpha,
                                       const mcsparseMatDescr_t descrA, const double *bsrSortedVal,
                                       const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                       mcsparseBsrsm2Info_t info, const double *B, int ldb, double *X, int ldx,
                                       mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseCbsrsm2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transXY, int mb, int n, int nnzb, const mcComplex *alpha,
                                       const mcsparseMatDescr_t descrA, const mcComplex *bsrSortedVal,
                                       const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
                                       mcsparseBsrsm2Info_t info, const mcComplex *B, int ldb, mcComplex *X, int ldx,
                                       mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseZbsrsm2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                       const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                       const mcDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                       const int *bsrSortedColInd, int blockSize, mcsparseBsrsm2Info_t info,
                                       const mcDoubleComplex *B, int ldb, mcDoubleComplex *X, int ldx,
                                       mcsparseSolvePolicy_t policy, void *pBuffer);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSPARSE_LEVEL3_H_
