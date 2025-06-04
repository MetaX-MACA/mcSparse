#ifndef INTERFACE_MCSPARSE_CONVERSION_H_
#define INTERFACE_MCSPARSE_CONVERSION_H_

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Xcoo2csr
mcsparseStatus_t mcsparseXcoo2csr(mcsparseHandle_t handle, const int *coo_rows, int nnz, int m, int *csr_rows,
                                  mcsparseIndexBase_t idx_base);

// Xcsr2coo
mcsparseStatus_t mcsparseXcsr2coo(mcsparseHandle_t handle, const int *csr_rows, int nnz, int m, int *coo_rows,
                                  mcsparseIndexBase_t idx_base);

// Csr2cscEx2
mcsparseStatus_t mcsparseCsr2cscEx2(mcsparseHandle_t handle, int m, int n, int nnz, const void *csr_val,
                                    const int *csr_rows, const int *csr_cols, void *csc_val, int *csc_cols,
                                    int *csc_rows, macaDataType val_type, mcsparseAction_t csc_action,
                                    mcsparseIndexBase_t idx_base, mcsparseCsr2CscAlg_t alg, void *temp_buffer);

mcsparseStatus_t mcsparseCsr2cscEx2_bufferSize(mcsparseHandle_t handle, int m, int n, int nnz, const void *csr_val,
                                               const int *csr_rows, const int *csr_cols, void *csc_val, int *csc_cols,
                                               int *csc_rows, macaDataType val_type, mcsparseAction_t csc_action,
                                               mcsparseIndexBase_t idx_base, mcsparseCsr2CscAlg_t alg,
                                               size_t *buffer_size);
// nnz_compress
mcsparseStatus_t mcsparseSnnz_compress(mcsparseHandle_t handle, int m, const mcsparseMatDescr_t mcsp_descr_A,
                                       const float *csr_val_A, const int *csr_row_A, int *nnz_per_row, int *nnz_C,
                                       float tol);

mcsparseStatus_t mcsparseDnnz_compress(mcsparseHandle_t handle, int m, const mcsparseMatDescr_t mcsp_descr_A,
                                       const double *csr_val_A, const int *csr_row_A, int *nnz_per_row, int *nnz_C,
                                       double tol);

mcsparseStatus_t mcsparseCnnz_compress(mcsparseHandle_t handle, int m, const mcsparseMatDescr_t mcsp_descr_A,
                                       const mcComplex *csr_val_A, const int *csr_row_A, int *nnz_per_row, int *nnz_C,
                                       mcComplex tol);

mcsparseStatus_t mcsparseZnnz_compress(mcsparseHandle_t handle, int m, const mcsparseMatDescr_t mcsp_descr_A,
                                       const mcDoubleComplex *csr_val_A, const int *csr_row_A, int *nnz_per_row,
                                       int *nnz_C, mcDoubleComplex tol);

// csr2csr_compress
mcsparseStatus_t mcsparseScsr2csr_compress(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                           const float *csr_val_A, const int *csr_col_A, const int *csr_row_A,
                                           int nnz_A, const int *nnz_per_row, float *csr_val_C, int *csr_col_C,
                                           int *csr_row_C, float tol);

mcsparseStatus_t mcsparseDcsr2csr_compress(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                           const double *csr_val_A, const int *csr_col_A, const int *csr_row_A,
                                           int nnz_A, const int *nnz_per_row, double *csr_val_C, int *csr_col_C,
                                           int *csr_row_C, double tol);

mcsparseStatus_t mcsparseCcsr2csr_compress(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                           const mcComplex *csr_val_A, const int *csr_col_A, const int *csr_row_A,
                                           int nnz_A, const int *nnz_per_row, mcComplex *csr_val_C, int *csr_col_C,
                                           int *csr_row_C, mcComplex tol);
mcsparseStatus_t mcsparseZcsr2csr_compress(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                           const mcDoubleComplex *csr_val_A, const int *csr_col_A, const int *csr_row_A,
                                           int nnz_A, const int *nnz_per_row, mcDoubleComplex *csr_val_C,
                                           int *csr_col_C, int *csr_row_C, mcDoubleComplex tol);
// Xcsrsort
mcsparseStatus_t mcsparseXcsrsort_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, const int *csr_rows,
                                                const int *csr_cols, size_t *buffer_size);

mcsparseStatus_t mcsparseXcsrsort(mcsparseHandle_t handle, int m, int n, int nnz, mcsparseMatDescr_t mcsp_descr_A,
                                  const int *csr_rows, int *csr_cols, int *perm, void *temp_buffer);

// csr2csru
mcsparseStatus_t mcsparseScsr2csru(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, float *csr_vals, const int *csr_rows,
                                   int *csr_cols, mcsparseCsru2csrInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseDcsr2csru(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, double *csr_vals, const int *csr_rows,
                                   int *csr_cols, mcsparseCsru2csrInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseCcsr2csru(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, mcComplex *csr_vals, const int *csr_rows,
                                   int *csr_cols, mcsparseCsru2csrInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseZcsr2csru(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, mcDoubleComplex *csr_vals,
                                   const int *csr_rows, int *csr_cols, mcsparseCsru2csrInfo_t info, void *temp_buffer);

// Xcscsort
mcsparseStatus_t mcsparseXcscsort_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, const int *csc_cols,
                                                const int *csc_rows, size_t *buffer_size);

mcsparseStatus_t mcsparseXcscsort(mcsparseHandle_t handle, int m, int n, int nnz, mcsparseMatDescr_t mcsp_descr_A,
                                  const int *csc_cols, int *csc_rows, int *perm, void *temp_buffer);

// coosort
mcsparseStatus_t mcsparseXcoosort_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, const int *coo_rows,
                                                const int *coo_cols, size_t *buffer_size);

mcsparseStatus_t mcsparseXcoosortByRow(mcsparseHandle_t handle, int m, int n, int nnz, int *coo_rows, int *coo_cols,
                                       int *perm, void *temp_buffer);

mcsparseStatus_t mcsparseXcoosortByColumn(mcsparseHandle_t handle, int m, int n, int nnz, int *coo_rows, int *coo_cols,
                                          int *perm, void *temp_buffer);

// CreateIdentityPermutation
mcsparseStatus_t mcsparseCreateIdentityPermutation(mcsparseHandle_t handle, int n, int *p);

// nnz
mcsparseStatus_t mcsparseSnnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcsparseMatDescr_t mcsp_descr_A, const float *dense_matrix, int lda,
                              int *nnz_per_row_or_column, int *nnz);

mcsparseStatus_t mcsparseDnnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcsparseMatDescr_t mcsp_descr_A, const double *dense_matrix, int lda,
                              int *nnz_per_row_or_column, int *nnz);

mcsparseStatus_t mcsparseCnnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcsparseMatDescr_t mcsp_descr_A, const mcComplex *dense_matrix, int lda,
                              int *nnz_per_row_or_column, int *nnz);

mcsparseStatus_t mcsparseZnnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcsparseMatDescr_t mcsp_descr_A, const mcDoubleComplex *dense_matrix, int lda,
                              int *nnz_per_row_or_column, int *nnz);

// dense2csr
mcsparseStatus_t mcsparseSdense2csr(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const float *dense_matrix, int lda, int *nnz_per_row_or_column, float *csr_vals,
                                    int *csr_rows, int *csr_cols);

mcsparseStatus_t mcsparseDdense2csr(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const double *dense_matrix, int lda, int *nnz_per_row_or_column, double *csr_vals,
                                    int *csr_rows, int *csr_cols);

mcsparseStatus_t mcsparseCdense2csr(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcComplex *dense_matrix, int lda, int *nnz_per_row_or_column,
                                    mcComplex *csr_vals, int *csr_rows, int *csr_cols);

mcsparseStatus_t mcsparseZdense2csr(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcDoubleComplex *dense_matrix, int lda, int *nnz_per_row_or_column,
                                    mcDoubleComplex *csr_vals, int *csr_rows, int *csr_cols);

// dense2csc
mcsparseStatus_t mcsparseSdense2csc(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const float *dense_matrix, int lda, int *nnz_per_row_or_column, float *csc_vals,
                                    int *csc_rows, int *csc_cols);

mcsparseStatus_t mcsparseDdense2csc(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const double *dense_matrix, int lda, int *nnz_per_row_or_column, double *csc_vals,
                                    int *csc_rows, int *csc_cols);

mcsparseStatus_t mcsparseCdense2csc(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcComplex *dense_matrix, int lda, int *nnz_per_row_or_column,
                                    mcComplex *csc_vals, int *csc_rows, int *csc_cols);

mcsparseStatus_t mcsparseZdense2csc(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcDoubleComplex *dense_matrix, int lda, int *nnz_per_row_or_column,
                                    mcDoubleComplex *csc_vals, int *csc_rows, int *csc_cols);

// csr2dense
mcsparseStatus_t mcsparseScsr2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const float *csr_vals, const int *csr_rows, const int *csr_cols,
                                    float *dense_matrix, int lda);

mcsparseStatus_t mcsparseDcsr2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const double *csr_vals, const int *csr_rows, const int *csr_cols,
                                    double *dense_matrix, int lda);

mcsparseStatus_t mcsparseCcsr2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcComplex *csr_vals, const int *csr_rows, const int *csr_cols,
                                    mcComplex *dense_matrix, int lda);

mcsparseStatus_t mcsparseZcsr2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcDoubleComplex *csr_vals, const int *csr_rows, const int *csr_cols,
                                    mcDoubleComplex *dense_matrix, int lda);

// csc2dense
mcsparseStatus_t mcsparseScsc2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const float *csc_vals, const int *csc_rows, const int *csc_cols,
                                    float *dense_matrix, int lda);

mcsparseStatus_t mcsparseDcsc2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const double *csc_vals, const int *csc_rows, const int *csc_cols,
                                    double *dense_matrix, int lda);

mcsparseStatus_t mcsparseCcsc2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcComplex *csc_vals, const int *csc_rows, const int *csc_cols,
                                    mcComplex *dense_matrix, int lda);

mcsparseStatus_t mcsparseZcsc2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcDoubleComplex *csc_vals, const int *csc_rows, const int *csc_cols,
                                    mcDoubleComplex *dense_matrix, int lda);

// pruneDense2csr
mcsparseStatus_t mcsparseSpruneDense2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const float *dense_matrix,
                                                       int lda, const float *threshold,
                                                       const mcsparseMatDescr_t mcsp_descr_A, const float *csr_vals,
                                                       const int *csr_rows, const int *csr_cols, size_t *buffer_size);

mcsparseStatus_t mcsparseDpruneDense2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n,
                                                       const double *dense_matrix, int lda, const double *threshold,
                                                       const mcsparseMatDescr_t mcsp_descr_A, const double *csr_vals,
                                                       const int *csr_rows, const int *csr_cols, size_t *buffer_size);

mcsparseStatus_t mcsparseSpruneDense2csrNnz(mcsparseHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                            const float *threshold, const mcsparseMatDescr_t mcsp_descr_A,
                                            int *csr_rows, int *nnz, void *temp_buffer);

mcsparseStatus_t mcsparseDpruneDense2csrNnz(mcsparseHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                            const double *threshold, const mcsparseMatDescr_t mcsp_descr_A,
                                            int *csr_rows, int *nnz, void *temp_buffer);

mcsparseStatus_t mcsparseSpruneDense2csr(mcsparseHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                         const float *threshold, const mcsparseMatDescr_t mcsp_descr_A, float *csr_vals,
                                         const int *csr_rows, int *csr_cols, void *temp_buffer);

mcsparseStatus_t mcsparseDpruneDense2csr(mcsparseHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                         const double *threshold, const mcsparseMatDescr_t mcsp_descr_A,
                                         double *csr_vals, const int *csr_rows, int *csr_cols, void *temp_buffer);

#if defined(__cplusplus)
mcsparseStatus_t mcsparseHpruneDense2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const __half *A, int lda,
                                                       const __half *threshold, const mcsparseMatDescr_t descrC,
                                                       const __half *csrSortedValC, const int *csrSortedRowPtrC,
                                                       const int *csrSortedColIndC, size_t *pBufferSizeInBytes);

mcsparseStatus_t mcsparseHpruneDense2csrNnz(mcsparseHandle_t handle, int m, int n, const __half *A, int lda,
                                            const __half *threshold, const mcsparseMatDescr_t descrC, int *csrRowPtrC,
                                            int *nnzTotalDevHostPtr, void *pBuffer);

mcsparseStatus_t mcsparseHpruneDense2csr(mcsparseHandle_t handle, int m, int n, const __half *A, int lda,
                                         const __half *threshold, const mcsparseMatDescr_t descrC,
                                         __half *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
                                         void *pBuffer);
#endif

// pruneDense2csrByPercentage
mcsparseStatus_t mcsparseSpruneDense2csrByPercentage_bufferSizeExt(mcsparseHandle_t handle, int m, int n,
                                                                   const float *dense_matrix, int lda, float percentage,
                                                                   const mcsparseMatDescr_t mcsp_descr_A,
                                                                   const float *csr_vals, const int *csr_rows,
                                                                   const int *csr_cols, const mcsparsePruneInfo_t info,
                                                                   size_t *buffer_size);

mcsparseStatus_t mcsparseDpruneDense2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, const double *dense_matrix, int lda, double percentage,
    const mcsparseMatDescr_t mcsp_descr_A, const double *csr_vals, const int *csr_rows, const int *csr_cols,
    const mcsparsePruneInfo_t info, size_t *buffer_size);

mcsparseStatus_t mcsparseSpruneDense2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n,
                                                        const float *dense_matrix, int lda, float percentage,
                                                        const mcsparseMatDescr_t mcsp_descr_A, int *csr_rows, int *nnz,
                                                        const mcsparsePruneInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseDpruneDense2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n,
                                                        const double *dense_matrix, int lda, double percentage,
                                                        const mcsparseMatDescr_t mcsp_descr_A, int *csr_rows, int *nnz,
                                                        const mcsparsePruneInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseSpruneDense2csrByPercentage(mcsparseHandle_t handle, int m, int n, const float *dense_matrix,
                                                     int lda, float percentage, const mcsparseMatDescr_t mcsp_descr_A,
                                                     float *csr_vals, const int *csr_rows, int *csr_cols,
                                                     const mcsparsePruneInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseDpruneDense2csrByPercentage(mcsparseHandle_t handle, int m, int n, const double *dense_matrix,
                                                     int lda, double percentage, const mcsparseMatDescr_t mcsp_descr_A,
                                                     double *csr_vals, const int *csr_rows, int *csr_cols,
                                                     const mcsparsePruneInfo_t info, void *temp_buffer);

#if defined(__cplusplus)
mcsparseStatus_t mcsparseHpruneDense2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, const __half *A, int lda, float percentage, const mcsparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, mcsparsePruneInfo_t info,
    size_t *pBufferSizeInBytes);

mcsparseStatus_t mcsparseHpruneDense2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n, const __half *A, int lda,
                                                        float percentage, const mcsparseMatDescr_t descrC,
                                                        int *csrRowPtrC, int *nnzTotalDevHostPtr,
                                                        mcsparsePruneInfo_t info, void *pBuffer);

mcsparseStatus_t mcsparseHpruneDense2csrByPercentage(mcsparseHandle_t handle, int m, int n, const __half *A, int lda,
                                                     float percentage, const mcsparseMatDescr_t descrC,
                                                     __half *csrSortedValC, const int *csrSortedRowPtrC,
                                                     int *csrSortedColIndC, mcsparsePruneInfo_t info, void *pBuffer);
#endif

// pruneCsr2csr
mcsparseStatus_t mcsparseSpruneCsr2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                     const mcsparseMatDescr_t mcsp_descr_A, const float *csr_val_A,
                                                     const int *csr_row_A, const int *csr_col_A, const float *tol,
                                                     const mcsparseMatDescr_t mcsp_descr_C, const float *csr_val_C,
                                                     const int *csr_row_C, const int *csr_col_C, size_t *buffer_size);

mcsparseStatus_t mcsparseDpruneCsr2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                     const mcsparseMatDescr_t mcsp_descr_A, const double *csr_val_A,
                                                     const int *csr_row_A, const int *csr_col_A, const double *tol,
                                                     const mcsparseMatDescr_t mcsp_descr_C, const double *csr_val_C,
                                                     const int *csr_row_C, const int *csr_col_C, size_t *buffer_size);

mcsparseStatus_t mcsparseSpruneCsr2csrNnz(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                          const mcsparseMatDescr_t mcsp_descr_A, const float *csr_val_A,
                                          const int *csr_row_A, const int *csr_col_A, const float *tol,
                                          const mcsparseMatDescr_t mcsp_descr_C, int *csr_row_C, int *nnz_C,
                                          void *temp_buffer);

mcsparseStatus_t mcsparseDpruneCsr2csrNnz(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                          const mcsparseMatDescr_t mcsp_descr_A, const double *csr_val_A,
                                          const int *csr_row_A, const int *csr_col_A, const double *tol,
                                          const mcsparseMatDescr_t mcsp_descr_C, int *csr_row_C, int *nnz_C,
                                          void *temp_buffer);

mcsparseStatus_t mcsparseSpruneCsr2csr(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                       const mcsparseMatDescr_t mcsp_descr_A, const float *csr_val_A,
                                       const int *csr_row_A, const int *csr_col_A, const float *tol,
                                       const mcsparseMatDescr_t mcsp_descr_C, float *csr_val_C, const int *csr_row_C,
                                       int *csr_col_C, void *temp_buffer);

mcsparseStatus_t mcsparseDpruneCsr2csr(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                       const mcsparseMatDescr_t mcsp_descr_A, const double *csr_val_A,
                                       const int *csr_row_A, const int *csr_col_A, const double *tol,
                                       const mcsparseMatDescr_t mcsp_descr_C, double *csr_val_C, const int *csr_row_C,
                                       int *csr_col_C, void *temp_buffer);

#if defined(__cplusplus)
mcsparseStatus_t mcsparseHpruneCsr2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnzA,
                                                     const mcsparseMatDescr_t descrA, const __half *csrSortedValA,
                                                     const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                     const __half *threshold, const mcsparseMatDescr_t descrC,
                                                     const __half *csrSortedValC, const int *csrSortedRowPtrC,
                                                     const int *csrSortedColIndC, size_t *pBufferSizeInBytes);

mcsparseStatus_t mcsparseHpruneCsr2csrNnz(mcsparseHandle_t handle, int m, int n, int nnzA,
                                          const mcsparseMatDescr_t descrA, const __half *csrSortedValA,
                                          const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                          const __half *threshold, const mcsparseMatDescr_t descrC,
                                          int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer);

mcsparseStatus_t mcsparseHpruneCsr2csr(mcsparseHandle_t handle, int m, int n, int nnzA, const mcsparseMatDescr_t descrA,
                                       const __half *csrSortedValA, const int *csrSortedRowPtrA,
                                       const int *csrSortedColIndA, const __half *threshold,
                                       const mcsparseMatDescr_t descrC, __half *csrSortedValC,
                                       const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer);
#endif

// pruneCsr2csrByPercentage
mcsparseStatus_t mcsparseSpruneCsr2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, int nnz_A, const mcsparseMatDescr_t mcsp_descr_A, const float *csr_vals_A,
    const int *csr_rows_A, const int *csr_cols_A, float percentage, const mcsparseMatDescr_t mcsp_descr_C,
    const float *csr_vals_C, const int *csr_rows_C, const int *csr_cols_C, const mcsparsePruneInfo_t info,
    size_t *buffer_size);

mcsparseStatus_t mcsparseDpruneCsr2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, int nnz_A, const mcsparseMatDescr_t mcsp_descr_A, const double *csr_vals_A,
    const int *csr_rows_A, const int *csr_cols_A, double percentage, const mcsparseMatDescr_t mcsp_descr_C,
    const double *csr_vals_C, const int *csr_rows_C, const int *csr_cols_C, const mcsparsePruneInfo_t info,
    size_t *buffer_size);

mcsparseStatus_t mcsparseSpruneCsr2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                      const mcsparseMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                                      const int *csr_rows_A, const int *csr_cols_A, float percentage,
                                                      const mcsparseMatDescr_t mcsp_descr_C, int *csr_rows_C,
                                                      int *nnz_C, const mcsparsePruneInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseDpruneCsr2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                      const mcsparseMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                                      const int *csr_rows_A, const int *csr_cols_A, double percentage,
                                                      const mcsparseMatDescr_t mcsp_descr_C, int *csr_rows_C,
                                                      int *nnz_C, const mcsparsePruneInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseSpruneCsr2csrByPercentage(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                   const mcsparseMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                                   const int *csr_rows_A, const int *csr_cols_A, float percentage,
                                                   const mcsparseMatDescr_t mcsp_descr_C, float *csr_vals_C,
                                                   const int *csr_rows_C, int *csr_cols_C,
                                                   const mcsparsePruneInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseDpruneCsr2csrByPercentage(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                   const mcsparseMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                                   const int *csr_rows_A, const int *csr_cols_A, double percentage,
                                                   const mcsparseMatDescr_t mcsp_descr_C, double *csr_vals_C,
                                                   const int *csr_rows_C, int *csr_cols_C,
                                                   const mcsparsePruneInfo_t info, void *temp_buffer);

#if defined(__cplusplus)
mcsparseStatus_t mcsparseHpruneCsr2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, int nnzA, const mcsparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const mcsparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, mcsparsePruneInfo_t info,
    size_t *pBufferSizeInBytes);

mcsparseStatus_t mcsparseHpruneCsr2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n, int nnzA,
                                                      const mcsparseMatDescr_t descrA, const __half *csrSortedValA,
                                                      const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                      float percentage, const mcsparseMatDescr_t descrC,
                                                      int *csrSortedRowPtrC, int *nnzTotalDevHostPtr,
                                                      mcsparsePruneInfo_t info, void *pBuffer);

mcsparseStatus_t mcsparseHpruneCsr2csrByPercentage(mcsparseHandle_t handle, int m, int n, int nnzA,
                                                   const mcsparseMatDescr_t descrA, const __half *csrSortedValA,
                                                   const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                   float percentage, /* between 0 to 100 */
                                                   const mcsparseMatDescr_t descrC, __half *csrSortedValC,
                                                   const int *csrSortedRowPtrC, int *csrSortedColIndC,
                                                   mcsparsePruneInfo_t info, void *pBuffer);
#endif

// csru2csr
mcsparseStatus_t mcsparseScsru2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, float *csr_vals,
                                                 const int *csr_rows, int *csr_cols, mcsparseCsru2csrInfo_t info,
                                                 size_t *buffer_size);

mcsparseStatus_t mcsparseDcsru2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, double *csr_vals,
                                                 const int *csr_rows, int *csr_cols, mcsparseCsru2csrInfo_t info,
                                                 size_t *buffer_size);

mcsparseStatus_t mcsparseCcsru2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, mcComplex *csr_vals,
                                                 const int *csr_rows, int *csr_cols, mcsparseCsru2csrInfo_t info,
                                                 size_t *buffer_size);

mcsparseStatus_t mcsparseZcsru2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz,
                                                 mcDoubleComplex *csr_vals, const int *csr_rows, int *csr_cols,
                                                 mcsparseCsru2csrInfo_t info, size_t *buffer_size);

mcsparseStatus_t mcsparseScsru2csr(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, float *csr_vals, const int *csr_rows,
                                   int *csr_cols, mcsparseCsru2csrInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseDcsru2csr(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, double *csr_vals, const int *csr_rows,
                                   int *csr_cols, mcsparseCsru2csrInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseCcsru2csr(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, mcComplex *csr_vals, const int *csr_rows,
                                   int *csr_cols, mcsparseCsru2csrInfo_t info, void *temp_buffer);

mcsparseStatus_t mcsparseZcsru2csr(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, mcDoubleComplex *csr_vals,
                                   const int *csr_rows, int *csr_cols, mcsparseCsru2csrInfo_t info, void *temp_buffer);

// csr2gebsr
mcsparseStatus_t mcsparseScsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                               const mcsparseMatDescr_t csr_descr, const float *csr_val,
                                               const int *csr_row, const int *csr_col, int row_block_dim,
                                               int col_block_dim, int *buffer_size);

mcsparseStatus_t mcsparseDcsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                               const mcsparseMatDescr_t csr_descr, const double *csr_val,
                                               const int *csr_row, const int *csr_col, int row_block_dim,
                                               int col_block_dim, int *buffer_size);

mcsparseStatus_t mcsparseCcsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                               const mcsparseMatDescr_t csr_descr, const mcComplex *csr_val,
                                               const int *csr_row, const int *csr_col, int row_block_dim,
                                               int col_block_dim, int *buffer_size);

mcsparseStatus_t mcsparseZcsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                               const mcsparseMatDescr_t csr_descr, const mcDoubleComplex *csr_val,
                                               const int *csr_row, const int *csr_col, int row_block_dim,
                                               int col_block_dim, int *buffer_size);

mcsparseStatus_t mcsparseScsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                                  const mcsparseMatDescr_t csr_descr, const float *csr_val,
                                                  const int *csr_row, const int *csr_col, int row_block_dim,
                                                  int col_block_dim, size_t *buffer_size);

mcsparseStatus_t mcsparseDcsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                                  const mcsparseMatDescr_t csr_descr, const double *csr_val,
                                                  const int *csr_row, const int *csr_col, int row_block_dim,
                                                  int col_block_dim, size_t *buffer_size);

mcsparseStatus_t mcsparseCcsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                                  const mcsparseMatDescr_t csr_descr, const mcComplex *csr_val,
                                                  const int *csr_row, const int *csr_col, int row_block_dim,
                                                  int col_block_dim, size_t *buffer_size);

mcsparseStatus_t mcsparseZcsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                                  const mcsparseMatDescr_t csr_descr, const mcDoubleComplex *csr_val,
                                                  const int *csr_row, const int *csr_col, int row_block_dim,
                                                  int col_block_dim, size_t *buffer_size);

mcsparseStatus_t mcsparseXcsr2gebsrNnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                       const mcsparseMatDescr_t csr_descr, const int *csr_row, const int *csr_col,
                                       const mcsparseMatDescr_t bsr_descr, int *bsr_row, int row_block_dim,
                                       int col_block_dim, int *nnzb, void *temp_buffer);

mcsparseStatus_t mcsparseScsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                    const mcsparseMatDescr_t csr_descr, const float *csr_val, const int *csr_row,
                                    const int *csr_col, const mcsparseMatDescr_t bsr_descr, float *bsr_val,
                                    int *bsr_row, int *bsr_col, int row_block_dim, int col_block_dim,
                                    void *temp_buffer);

mcsparseStatus_t mcsparseDcsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                    const mcsparseMatDescr_t csr_descr, const double *csr_val, const int *csr_row,
                                    const int *csr_col, const mcsparseMatDescr_t bsr_descr, double *bsr_val,
                                    int *bsr_row, int *bsr_col, int row_block_dim, int col_block_dim,
                                    void *temp_buffer);

mcsparseStatus_t mcsparseCcsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                    const mcsparseMatDescr_t csr_descr, const mcComplex *csr_val, const int *csr_row,
                                    const int *csr_col, const mcsparseMatDescr_t bsr_descr, mcComplex *bsr_val,
                                    int *bsr_row, int *bsr_col, int row_block_dim, int col_block_dim,
                                    void *temp_buffer);

mcsparseStatus_t mcsparseZcsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                    const mcsparseMatDescr_t csr_descr, const mcDoubleComplex *csr_val,
                                    const int *csr_row, const int *csr_col, const mcsparseMatDescr_t bsr_descr,
                                    mcDoubleComplex *bsr_val, int *bsr_row, int *bsr_col, int row_block_dim,
                                    int col_block_dim, void *temp_buffer);

// csr2bsr
mcsparseStatus_t mcsparseXcsr2bsrNnz(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                     const mcsparseMatDescr_t descrA, const int *csrSortedRowPtrA,
                                     const int *csrSortedColIndA, int blockDim, const mcsparseMatDescr_t descrC,
                                     int *bsrSortedRowPtrC, int *nnzTotalDevHostPtr);

mcsparseStatus_t mcsparseScsr2bsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                  const mcsparseMatDescr_t descrA, const float *csrSortedValA,
                                  const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, float *bsrSortedValC, int *bsrSortedRowPtrC,
                                  int *bsrSortedColIndC);

mcsparseStatus_t mcsparseDcsr2bsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                  const mcsparseMatDescr_t descrA, const double *csrSortedValA,
                                  const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, double *bsrSortedValC, int *bsrSortedRowPtrC,
                                  int *bsrSortedColIndC);

mcsparseStatus_t mcsparseCcsr2bsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                  const mcsparseMatDescr_t descrA, const mcFloatComplex *csrSortedValA,
                                  const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, mcFloatComplex *bsrSortedValC, int *bsrSortedRowPtrC,
                                  int *bsrSortedColIndC);

mcsparseStatus_t mcsparseZcsr2bsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                  const mcsparseMatDescr_t descrA, const mcDoubleComplex *csrSortedValA,
                                  const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, mcDoubleComplex *bsrSortedValC,
                                  int *bsrSortedRowPtrC, int *bsrSortedColIndC);

// gebsr2gebsc
mcsparseStatus_t mcsparseSgebsr2gebsc_bufferSize(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                 const float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                 const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                 int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseDgebsr2gebsc_bufferSize(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                 const double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                 const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                 int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseCgebsr2gebsc_bufferSize(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                 const mcComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                 const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                 int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseZgebsr2gebsc_bufferSize(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                 const mcDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                 const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                 int *pBufferSizeInBytes);

mcsparseStatus_t mcsparseSgebsr2gebsc_bufferSizeExt(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                    const float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                    size_t *pBufferSize);

mcsparseStatus_t mcsparseDgebsr2gebsc_bufferSizeExt(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                    const double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                    size_t *pBufferSize);

mcsparseStatus_t mcsparseCgebsr2gebsc_bufferSizeExt(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                    const mcComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                    size_t *pBufferSize);

mcsparseStatus_t mcsparseZgebsr2gebsc_bufferSizeExt(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                    const mcDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                    size_t *pBufferSize);

mcsparseStatus_t mcsparseSgebsr2gebsc(mcsparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrSortedVal,
                                      const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim,
                                      int colBlockDim, float *bscVal, int *bscRowInd, int *bscColPtr,
                                      mcsparseAction_t copyValues, mcsparseIndexBase_t idxBase, void *pBuffer);

mcsparseStatus_t mcsparseDgebsr2gebsc(mcsparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrSortedVal,
                                      const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim,
                                      int colBlockDim, double *bscVal, int *bscRowInd, int *bscColPtr,
                                      mcsparseAction_t copyValues, mcsparseIndexBase_t idxBase, void *pBuffer);

mcsparseStatus_t mcsparseCgebsr2gebsc(mcsparseHandle_t handle, int mb, int nb, int nnzb, const mcComplex *bsrSortedVal,
                                      const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim,
                                      int colBlockDim, mcComplex *bscVal, int *bscRowInd, int *bscColPtr,
                                      mcsparseAction_t copyValues, mcsparseIndexBase_t idxBase, void *pBuffer);

mcsparseStatus_t mcsparseZgebsr2gebsc(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                      const mcDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                      const int *bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                      mcDoubleComplex *bscVal, int *bscRowInd, int *bscColPtr,
                                      mcsparseAction_t copyValues, mcsparseIndexBase_t idxBase, void *pBuffer);
#ifdef __cplusplus
}
#endif

#endif
