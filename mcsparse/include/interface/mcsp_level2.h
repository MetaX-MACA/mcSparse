#ifndef INTERFACE_MCSPARSE_LEVEL2_H_
#define INTERFACE_MCSPARSE_LEVEL2_H_

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// CsrmvEx
mcsparseStatus_t mcsparseCsrmvEx_bufferSize(mcsparseHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA,
                                            int m, int n, int nnz, const void *alpha, macaDataType alphatype,
                                            const mcsparseMatDescr_t descrA, const void *csrValA,
                                            macaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA,
                                            const void *x, macaDataType xtype, const void *beta, macaDataType betatype,
                                            void *y, macaDataType ytype, macaDataType executiontype,
                                            size_t *bufferSizeInBytes);

mcsparseStatus_t mcsparseCsrmvEx(mcsparseHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, int m,
                                 int n, int nnz, const void *alpha, macaDataType alphatype,
                                 const mcsparseMatDescr_t descrA, const void *csrValA, macaDataType csrValAtype,
                                 const int *csrRowPtrA, const int *csrColIndA, const void *x, macaDataType xtype,
                                 const void *beta, macaDataType betatype, void *y, macaDataType ytype,
                                 macaDataType executiontype, void *buffer);
// csrmv
mcsparseStatus_t mcsparseScsrmv(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                                const float *alpha, const mcsparseMatDescr_t descrA, const float *csrValA,
                                const int *csrRowPtrA, const int *csrColIndA, const float *x, const float *beta,
                                float *y);

mcsparseStatus_t mcsparseDcsrmv(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                                const double *alpha, const mcsparseMatDescr_t descrA, const double *csrValA,
                                const int *csrRowPtrA, const int *csrColIndA, const double *x, const double *beta,
                                double *y);

mcsparseStatus_t mcsparseCcsrmv(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                                const mcComplex *alpha, const mcsparseMatDescr_t descrA, const mcComplex *csrValA,
                                const int *csrRowPtrA, const int *csrColIndA, const mcComplex *x, const mcComplex *beta,
                                mcComplex *y);

mcsparseStatus_t mcsparseZcsrmv(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                                const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                const mcDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                const mcDoubleComplex *x, const mcDoubleComplex *beta, mcDoubleComplex *y);

// csrsv2
mcsparseStatus_t mcsparseXcsrsv2_zeroPivot(mcsparseHandle_t handle, mcsparseCsrsv2Info_t info, int *position);

mcsparseStatus_t mcsparseScsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                            const mcsparseMatDescr_t descrA, float *csrSortedValA,
                                            const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                            mcsparseCsrsv2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseDcsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                            const mcsparseMatDescr_t descrA, double *csrSortedValA,
                                            const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                            mcsparseCsrsv2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseCcsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                            const mcsparseMatDescr_t descrA, mcComplex *csrSortedValA,
                                            const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                            mcsparseCsrsv2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseZcsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                            const mcsparseMatDescr_t descrA, mcDoubleComplex *csrSortedValA,
                                            const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                            mcsparseCsrsv2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseScsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                               const mcsparseMatDescr_t descrA, float *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                               mcsparseCsrsv2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseDcsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                               const mcsparseMatDescr_t descrA, double *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                               mcsparseCsrsv2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseCcsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                               const mcsparseMatDescr_t descrA, mcComplex *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                               mcsparseCsrsv2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseZcsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                               const mcsparseMatDescr_t descrA, mcDoubleComplex *csrSortedValA,
                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                               mcsparseCsrsv2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseScsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                          const mcsparseMatDescr_t descrA, const float *csrSortedValA,
                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                          mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseDcsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                          const mcsparseMatDescr_t descrA, const double *csrSortedValA,
                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                          mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseCcsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                          const mcsparseMatDescr_t descrA, const mcComplex *csrSortedValA,
                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                          mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseZcsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                          const mcsparseMatDescr_t descrA, const mcDoubleComplex *csrSortedValA,
                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                          mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseScsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                       const float *alpha, const mcsparseMatDescr_t descrA, const float *csrSortedValA,
                                       const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                       mcsparseCsrsv2Info_t info, const float *f, float *x,
                                       mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseDcsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                       const double *alpha, const mcsparseMatDescr_t descrA,
                                       const double *csrSortedValA, const int *csrSortedRowPtrA,
                                       const int *csrSortedColIndA, mcsparseCsrsv2Info_t info, const double *f,
                                       double *x, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseCcsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                       const mcComplex *alpha, const mcsparseMatDescr_t descrA,
                                       const mcComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                       const int *csrSortedColIndA, mcsparseCsrsv2Info_t info, const mcComplex *f,
                                       mcComplex *x, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseZcsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                       const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                       const mcDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                                       const int *csrSortedColIndA, mcsparseCsrsv2Info_t info, const mcDoubleComplex *f,
                                       mcDoubleComplex *x, mcsparseSolvePolicy_t policy, void *pBuffer);

// gemvi
mcsparseStatus_t mcsparseSgemvi_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                           int *buffer_size);
mcsparseStatus_t mcsparseDgemvi_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                           int *buffer_size);
mcsparseStatus_t mcsparseCgemvi_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                           int *buffer_size);
mcsparseStatus_t mcsparseZgemvi_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                           int *buffer_size);

mcsparseStatus_t mcsparseSgemvi(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, const float *alpha,
                                const float *A, int lda, int nnz, const float *x_val, const int *x_ind,
                                const float *beta, float *y, mcsparseIndexBase_t idx_base, void *temp_buffer);

mcsparseStatus_t mcsparseDgemvi(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, const double *alpha,
                                const double *A, int lda, int nnz, const double *x_val, const int *x_ind,
                                const double *beta, double *y, mcsparseIndexBase_t idx_base, void *temp_buffer);

mcsparseStatus_t mcsparseCgemvi(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n,
                                const mcComplex *alpha, const mcComplex *A, int lda, int nnz, const mcComplex *x_val,
                                const int *x_ind, const mcComplex *beta, mcComplex *y, mcsparseIndexBase_t idx_base,
                                void *temp_buffer);

mcsparseStatus_t mcsparseZgemvi(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n,
                                const mcDoubleComplex *alpha, const mcDoubleComplex *A, int lda, int nnz,
                                const mcDoubleComplex *x_val, const int *x_ind, const mcDoubleComplex *beta,
                                mcDoubleComplex *y, mcsparseIndexBase_t idx_base, void *temp_buffer);

// bsrmv
mcsparseStatus_t mcsparseSbsrmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                int nb, int nnzb, const float *alpha, const mcsparseMatDescr_t descrA,
                                const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                int blockDim, const float *x, const float *beta, float *y);

mcsparseStatus_t mcsparseDbsrmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                int nb, int nnzb, const double *alpha, const mcsparseMatDescr_t descrA,
                                const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                int blockDim, const double *x, const double *beta, double *y);

mcsparseStatus_t mcsparseCbsrmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                int nb, int nnzb, const mcComplex *alpha, const mcsparseMatDescr_t descrA,
                                const mcComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                const int *bsrSortedColIndA, int blockDim, const mcComplex *x, const mcComplex *beta,
                                mcComplex *y);

mcsparseStatus_t mcsparseZbsrmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                int nb, int nnzb, const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                const mcDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                const int *bsrSortedColIndA, int blockDim, const mcDoubleComplex *x,
                                const mcDoubleComplex *beta, mcDoubleComplex *y);

// bsrxmv
mcsparseStatus_t mcsparseSbsrxmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                 int sizeOfMask, int mb, int nb, int nnzb, const float *alpha,
                                 const mcsparseMatDescr_t descrA, const float *bsrSortedValA,
                                 const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                                 const int *bsrSortedColIndA, int blockDim, const float *x, const float *beta,
                                 float *y);

mcsparseStatus_t mcsparseDbsrxmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                 int sizeOfMask, int mb, int nb, int nnzb, const double *alpha,
                                 const mcsparseMatDescr_t descrA, const double *bsrSortedValA,
                                 const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                                 const int *bsrSortedColIndA, int blockDim, const double *x, const double *beta,
                                 double *y);

mcsparseStatus_t mcsparseCbsrxmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                 int sizeOfMask, int mb, int nb, int nnzb, const mcFloatComplex *alpha,
                                 const mcsparseMatDescr_t descrA, const mcFloatComplex *bsrSortedValA,
                                 const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                                 const int *bsrSortedColIndA, int blockDim, const mcFloatComplex *x,
                                 const mcFloatComplex *beta, mcFloatComplex *y);

mcsparseStatus_t mcsparseZbsrxmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                 int sizeOfMask, int mb, int nb, int nnzb, const mcDoubleComplex *alpha,
                                 const mcsparseMatDescr_t descrA, const mcDoubleComplex *bsrSortedValA,
                                 const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA,
                                 const int *bsrSortedColIndA, int blockDim, const mcDoubleComplex *x,
                                 const mcDoubleComplex *beta, mcDoubleComplex *y);
// bsrsv2
mcsparseStatus_t mcsparseSbsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, float *bsrSortedValA,
                                            const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                            mcsparseBsrsv2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseDbsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, double *bsrSortedValA,
                                            const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                            mcsparseBsrsv2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseCbsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, mcFloatComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                            mcsparseBsrsv2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseZbsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, mcDoubleComplex *bsrSortedValA,
                                            const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                            mcsparseBsrsv2Info_t info, int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseSbsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, int mb, int nnzb,
                                               const mcsparseMatDescr_t descrA, float *bsrSortedValA,
                                               const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
                                               mcsparseBsrsv2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseDbsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, int mb, int nnzb,
                                               const mcsparseMatDescr_t descrA, double *bsrSortedValA,
                                               const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
                                               mcsparseBsrsv2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseCbsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, int mb, int nnzb,
                                               const mcsparseMatDescr_t descrA, mcFloatComplex *bsrSortedValA,
                                               const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
                                               mcsparseBsrsv2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseZbsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, int mb, int nnzb,
                                               const mcsparseMatDescr_t descrA, mcDoubleComplex *bsrSortedValA,
                                               const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
                                               mcsparseBsrsv2Info_t info, size_t *pBufferSize);

mcsparseStatus_t mcsparseSbsrsv2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          int mb, int nnzb, const mcsparseMatDescr_t descrA, const float *bsrSortedValA,
                                          const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                          mcsparseBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseDbsrsv2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          int mb, int nnzb, const mcsparseMatDescr_t descrA,
                                          const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                                          const int *bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                          mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseCbsrsv2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          int mb, int nnzb, const mcsparseMatDescr_t descrA,
                                          const mcFloatComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                          const int *bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                          mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseZbsrsv2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          int mb, int nnzb, const mcsparseMatDescr_t descrA,
                                          const mcDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                          const int *bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                          mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseSbsrsv2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       int mb, int nnzb, const float *alpha, const mcsparseMatDescr_t descrA,
                                       const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                                       const int *bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                       const float *f, float *x, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseDbsrsv2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       int mb, int nnzb, const double *alpha, const mcsparseMatDescr_t descrA,
                                       const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                                       const int *bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                       const double *f, double *x, mcsparseSolvePolicy_t policy, void *pBuffer);

mcsparseStatus_t mcsparseCbsrsv2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       int mb, int nnzb, const mcFloatComplex *alpha, const mcsparseMatDescr_t descrA,
                                       const mcFloatComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                       const int *bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                       const mcFloatComplex *f, mcFloatComplex *x, mcsparseSolvePolicy_t policy,
                                       void *pBuffer);

mcsparseStatus_t mcsparseZbsrsv2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       int mb, int nnzb, const mcDoubleComplex *alpha, const mcsparseMatDescr_t descrA,
                                       const mcDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                       const int *bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                       const mcDoubleComplex *f, mcDoubleComplex *x, mcsparseSolvePolicy_t policy,
                                       void *pBuffer);

mcsparseStatus_t mcsparseXbsrsv2_zeroPivot(mcsparseHandle_t handle, mcsparseBsrsv2Info_t info, int *position);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSPARSE_LEVEL2_H_
