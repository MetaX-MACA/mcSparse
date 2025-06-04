#ifndef INTERFACE_MCSP_INTERNAL_LEVEL3_H_
#define INTERFACE_MCSP_INTERNAL_LEVEL3_H_

#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief   Compute CSR-based SpMM in single and double precision in GPU, csr_scalar algorithm
 *          C = alpha * A * B + beta * C
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans_A       [in]        matrix operation type of sparse matrix A
 * @param trans_B       [in]        matrix operation type of dense matrix B
 * @param m             [in]        number of rows of op(A) and C
 * @param n             [in]        number of columns of op(B) and C
 * @param k             [in]        number of columns of op(A) and rows of op(B)
 * @param nnz           [in]        number of nonzeros of A
 * @param alpha         [in]        alpha
 * @param descr         [in]        descriptor of the sparse matrix A
 * @param csr_rows      [in]        pointer to the row offset in CSR
 * @param csr_cols      [in]        pointer to the column indexes of nonzeros in CSR
 * @param csr_vals      [in]        pointer to the values of nonzeros in CSR
 * @param mtx_B         [in]        pointer to the dense matrix B
 * @param ldb           [in]        leading dimension of B
 * @param beta          [in]        beta
 * @param mtx_C         [in/out]    pointer to the dense matrix C
 * @param ldc           [in]        leading dimension of C
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrSpmm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, mcspInt nnz, const float *alpha, const mcspMatDescr_t descr,
                          const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols, const float *mtx_B,
                          mcspInt ldb, const float *beta, float *mtx_C, mcspInt ldc);
mcspStatus_t mcspDcsrSpmm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, mcspInt nnz, const double *alpha, const mcspMatDescr_t descr,
                          const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols, const double *mtx_B,
                          mcspInt ldb, const double *beta, double *mtx_C, mcspInt ldc);
mcspStatus_t mcspCcsrSpmm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, mcspInt nnz, const mcspComplexFloat *alpha, const mcspMatDescr_t descr,
                          const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                          const mcspComplexFloat *mtx_B, mcspInt ldb, const mcspComplexFloat *beta,
                          mcspComplexFloat *mtx_C, mcspInt ldc);
mcspStatus_t mcspZcsrSpmm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, mcspInt nnz, const mcspComplexDouble *alpha, const mcspMatDescr_t descr,
                          const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                          const mcspComplexDouble *mtx_B, mcspInt ldb, const mcspComplexDouble *beta,
                          mcspComplexDouble *mtx_C, mcspInt ldc);

/**
 * @brief   Dense matrix sparse matrix multiplication. B is a sparse matrix in CSR format
 *          and C is a dense matrix.
 *          C = alpha * op(A) * op(B) + beta * C
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans         [in]        matrix operation type
 * @param m             [in]        number of rows of dense matrix A
 * @param n             [in]        number of columns of dense matrix C and sparse CSR matrix op(B)
 * @param k             [in]        number of columns of dense matrix A
 * @param nnz           [in]        number of non-zeros in the sparse CSR matrix B
 * @param alpha         [in]        scalar alpha
 * @param A             [in]        pointer to the dense matrix A
 * @param lda           [in]        leading dimension of the dense matrix A
 * @param descr         [in]        descriptor of the sparse matrix B
 * @param csr_vals      [in]        array of nnz elements of the sparse matrix B
 * @param csr_rows      [in]        row pointer of the sparse CSR matrix B
 * @param csr_cols      [in]        column index of the sparse CSR matrix B
 * @param beta          [in]        scalar beta
 * @param C             [in/out]    array of ldcxn that holds the values of dense matrix C
 * @param ldc           [in]        leading dimension of dense matrix C
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgemmi(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                        mcspInt n, mcspInt k, mcspInt nnz, const float *alpha, const float *A, mcspInt lda,
                        const mcspMatDescr_t descr, const float *csr_vals, const mcspInt *csr_rows,
                        const mcspInt *csr_cols, const float *beta, float *C, mcspInt ldc);
mcspStatus_t mcspDgemmi(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                        mcspInt n, mcspInt k, mcspInt nnz, const double *alpha, const double *A, mcspInt lda,
                        const mcspMatDescr_t descr, const double *csr_vals, const mcspInt *csr_rows,
                        const mcspInt *csr_cols, const double *beta, double *C, mcspInt ldc);
mcspStatus_t mcspCgemmi(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                        mcspInt n, mcspInt k, mcspInt nnz, const mcspComplexFloat *alpha, const mcspComplexFloat *A,
                        mcspInt lda, const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals,
                        const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspComplexFloat *beta,
                        mcspComplexFloat *C, mcspInt ldc);
mcspStatus_t mcspZgemmi(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                        mcspInt n, mcspInt k, mcspInt nnz, const mcspComplexDouble *alpha, const mcspComplexDouble *A,
                        mcspInt lda, const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                        const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspComplexDouble *beta,
                        mcspComplexDouble *C, mcspInt ldc);

mcspStatus_t mcspCuinSgemmi(mcspHandle_t handle, int m, int n, int k, int nnz, const float *alpha, const float *A,
                          int lda, const float *csc_vals, const int *csc_cols, const int *csc_rows, const float *beta,
                          float *C, int ldc);
mcspStatus_t mcspCuinDgemmi(mcspHandle_t handle, int m, int n, int k, int nnz, const double *alpha, const double *A,
                          int lda, const double *csc_vals, const int *csc_cols, const int *csc_rows, const double *beta,
                          double *C, int ldc);
mcspStatus_t mcspCuinCgemmi(mcspHandle_t handle, int m, int n, int k, int nnz, const mcspComplexFloat *alpha,
                          const mcspComplexFloat *A, int lda, const mcspComplexFloat *csc_vals, const int *csc_cols,
                          const int *csc_rows, const mcspComplexFloat *beta, mcspComplexFloat *C, int ldc);
mcspStatus_t mcspCuinZgemmi(mcspHandle_t handle, int m, int n, int k, int nnz, const mcspComplexDouble *alpha,
                          const mcspComplexDouble *A, int lda, const mcspComplexDouble *csc_vals, const int *csc_cols,
                          const int *csc_rows, const mcspComplexDouble *beta, mcspComplexDouble *C, int ldc);

/**
 * @brief Sparse triangular system solve using CSR storage format
 *        return zero pivot and its position if structual or numerical zero exist in A
 *
 * @param handle        [in]        handle of mcsp library
 * @param info          [in]        meta data for CSR matrix
 * @param position      [out]       pointer to the position of zero pivot
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrSpsmZeroPivot(mcspHandle_t handle, mcspMatInfo_t info, int *position);

/**
 * @brief Sparse triangular system solve using CSR storage format
 *        op(A) * C = alpha * op(B)
 *        1st step: query assistant buffer size
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans_A       [in]        matrix A operation type
 * @param trans_B       [in]        matrix B operation type
 * @param m             [in]        number of rows of the sparse CSR matrix A
 * @param nrhs          [in]        number of columns of the dense matrix op(B)
 * @param nnz           [in]        number of nonzeros of CSR matrix A
 * @param alpha         [in]        scalar variable
 * @param descr         [in]        descriptor of the sparse matrix
 * @param csr_val       [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_row_ptr   [in]        pointer to the row offset in CSR matrix A
 * @param csr_col_ind   [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param B             [in]        array of mxnrhs elements of the rhs matrix B
 * @param ldb           [in]        leading dimension of rhs matrix B
 * @param info          [in]        meta data for CSR matrix A
 * @param policy        [in]        spsm solve policy
 * @param buffer_size   [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrSpsmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt nrhs, mcspInt nnz, const float *alpha,
                                    const mcspMatDescr_t descr, const float *csr_val, const mcspInt *csr_row_ptr,
                                    const mcspInt *csr_col_ind, const float *B, mcspInt ldb, mcspMatInfo_t info,
                                    mcsparseSolvePolicy_t policy, size_t *buffer_size);
mcspStatus_t mcspDcsrSpsmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt nrhs, mcspInt nnz, const double *alpha,
                                    const mcspMatDescr_t descr, const double *csr_val, const mcspInt *csr_row_ptr,
                                    const mcspInt *csr_col_ind, const double *B, mcspInt ldb, mcspMatInfo_t info,
                                    mcsparseSolvePolicy_t policy, size_t *buffer_size);
mcspStatus_t mcspCcsrSpsmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt nrhs, mcspInt nnz, const mcspComplexFloat *alpha,
                                    const mcspMatDescr_t descr, const mcspComplexFloat *csr_val,
                                    const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind, const mcspComplexFloat *B,
                                    mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy, size_t *buffer_size);
mcspStatus_t mcspZcsrSpsmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt nrhs, mcspInt nnz, const mcspComplexDouble *alpha,
                                    const mcspMatDescr_t descr, const mcspComplexDouble *csr_val,
                                    const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind, const mcspComplexDouble *B,
                                    mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy, size_t *buffer_size);

/**
 * @brief Sparse triangular system solve using CSR storage format
 *        op(A) * C = alpha * op(B)
 *        2nd step: analyze the square sparse matrix A
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans_A       [in]        matrix A operation type
 * @param trans_B       [in]        matrix B operation type
 * @param m             [in]        number of rows of the sparse CSR matrix A
 * @param nrhs          [in]        number of columns of the dense matrix op(B)
 * @param nnz           [in]        number of nonzeros of CSR matrix A
 * @param alpha         [in]        scalar variable
 * @param descr         [in]        descriptor of the sparse matrix
 * @param csr_val       [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_row_ptr   [in]        pointer to the row offset in CSR matrix A
 * @param csr_col_ind   [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param B             [in]        array of mxnrhs elements of the rhs matrix B
 * @param ldb           [in]        leading dimension of rhs matrix B
 * @param info          [out]       meta data for CSR matrix A
 * @param analysis      [in]        analysis policy
 * @param solve         [in]        solve policy
 * @param temp_buffer   [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrSpsmAnalysis(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                  mcspInt m, mcspInt nrhs, mcspInt nnz, const float *alpha, const mcspMatDescr_t descr,
                                  const float *csr_val, const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind,
                                  const float *B, mcspInt ldb, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis,
                                  mcsparseSolvePolicy_t solve, void *temp_buffer);
mcspStatus_t mcspDcsrSpsmAnalysis(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                  mcspInt m, mcspInt nrhs, mcspInt nnz, const double *alpha, const mcspMatDescr_t descr,
                                  const double *csr_val, const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind,
                                  const double *B, mcspInt ldb, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis,
                                  mcsparseSolvePolicy_t solve, void *temp_buffer);
mcspStatus_t mcspCcsrSpsmAnalysis(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                  mcspInt m, mcspInt nrhs, mcspInt nnz, const mcspComplexFloat *alpha,
                                  const mcspMatDescr_t descr, const mcspComplexFloat *csr_val,
                                  const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind, const mcspComplexFloat *B,
                                  mcspInt ldb, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis,
                                  mcsparseSolvePolicy_t solve, void *temp_buffer);
mcspStatus_t mcspZcsrSpsmAnalysis(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                  mcspInt m, mcspInt nrhs, mcspInt nnz, const mcspComplexDouble *alpha,
                                  const mcspMatDescr_t descr, const mcspComplexDouble *csr_val,
                                  const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind, const mcspComplexDouble *B,
                                  mcspInt ldb, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis,
                                  mcsparseSolvePolicy_t solve, void *temp_buffer);

/**
 * @brief Sparse triangular system solve using CSR storage format
 *        op(A) * C = alpha * op(B)
 *        3rd step: compute triangular solve
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans_A       [in]        matrix A operation type
 * @param trans_B       [in]        matrix B operation type
 * @param m             [in]        number of rows of the sparse CSR matrix A
 * @param nrhs          [in]        number of columns of the dense matrix op(B)
 * @param nnz           [in]        number of nonzeros of CSR matrix A
 * @param alpha         [in]        scalar variable
 * @param descr         [in]        descriptor of the sparse matrix
 * @param csr_val       [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_row_ptr   [in]        pointer to the row offset in CSR matrix A
 * @param csr_col_ind   [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param B             [in/out]    array of mxnrhs elements of the rhs matrix B
 * @param ldb           [in]        leading dimension of rhs matrix B
 * @param info          [out]       meta data for CSR matrix A
 * @param policy        [in]        solve policy
 * @param temp_buffer   [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrSpsmSolve(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                               mcspInt nrhs, mcspInt nnz, const float *alpha, const mcspMatDescr_t descr,
                               const float *csr_val, const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind, float *B,
                               mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy, void *temp_buffer);
mcspStatus_t mcspDcsrSpsmSolve(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                               mcspInt nrhs, mcspInt nnz, const double *alpha, const mcspMatDescr_t descr,
                               const double *csr_val, const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind, double *B,
                               mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy, void *temp_buffer);
mcspStatus_t mcspCcsrSpsmSolve(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                               mcspInt nrhs, mcspInt nnz, const mcspComplexFloat *alpha, const mcspMatDescr_t descr,
                               const mcspComplexFloat *csr_val, const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind,
                               mcspComplexFloat *B, mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy,
                               void *temp_buffer);
mcspStatus_t mcspZcsrSpsmSolve(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                               mcspInt nrhs, mcspInt nnz, const mcspComplexDouble *alpha, const mcspMatDescr_t descr,
                               const mcspComplexDouble *csr_val, const mcspInt *csr_row_ptr, const mcspInt *csr_col_ind,
                               mcspComplexDouble *B, mcspInt ldb, mcspMatInfo_t info, mcsparseSolvePolicy_t policy,
                               void *temp_buffer);

/**
 * @brief Sparse triangular system solve using CSR storage format
 *        op(A) * C = alpha * op(B)
 *        4th step: deallocate the memory allocated
 *
 * @param handle        [in]        handle of mcsp library
 * @param info          [in/out]    meta data for CSR matrix
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrSpsmClear(mcspHandle_t handle, mcspMatInfo_t info);

mcspStatus_t mcspCuinXcsrsm2_zeroPivot(mcspHandle_t handle, mcspCsrsm2Info_t info, int *position);

mcspStatus_t mcspCuinScsrsm2_bufferSizeExt(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                         mcsparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
                                         const mcspMatDescr_t descrA, const float *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B,
                                         int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                         size_t *pBufferSize);

mcspStatus_t mcspCuinDcsrsm2_bufferSizeExt(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                         mcsparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
                                         const mcspMatDescr_t descrA, const double *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *B,
                                         int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                         size_t *pBufferSize);

mcspStatus_t mcspCuinCcsrsm2_bufferSizeExt(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                         mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                         const mcspComplexFloat *alpha, const mcspMatDescr_t descrA,
                                         const mcspComplexFloat *csrSortedValA, const int *csrSortedRowPtrA,
                                         const int *csrSortedColIndA, const mcspComplexFloat *B, int ldb,
                                         mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, size_t *pBufferSize);

mcspStatus_t mcspCuinZcsrsm2_bufferSizeExt(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                         mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                         const mcspComplexDouble *alpha, const mcspMatDescr_t descrA,
                                         const mcspComplexDouble *csrSortedValA, const int *csrSortedRowPtrA,
                                         const int *csrSortedColIndA, const mcspComplexDouble *B, int ldb,
                                         mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, size_t *pBufferSize);

mcspStatus_t mcspCuinScsrsm2_analysis(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                    mcsparseOperation_t transB, int m, int nrhs, int nnz, const float *alpha,
                                    const mcspMatDescr_t descrA, const float *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *B, int ldb,
                                    mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinDcsrsm2_analysis(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                    mcsparseOperation_t transB, int m, int nrhs, int nnz, const double *alpha,
                                    const mcspMatDescr_t descrA, const double *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *B, int ldb,
                                    mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinCcsrsm2_analysis(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                    mcsparseOperation_t transB, int m, int nrhs, int nnz, const mcspComplexFloat *alpha,
                                    const mcspMatDescr_t descrA, const mcspComplexFloat *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const mcspComplexFloat *B,
                                    int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinZcsrsm2_analysis(mcspHandle_t handle, int algo, mcsparseOperation_t transA,
                                    mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                    const mcspComplexDouble *alpha, const mcspMatDescr_t descrA,
                                    const mcspComplexDouble *csrSortedValA, const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA, const mcspComplexDouble *B, int ldb,
                                    mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinScsrsm2_solve(mcspHandle_t handle, int algo, mcsparseOperation_t transA, mcsparseOperation_t transB,
                                 int m, int nrhs, int nnz, const float *alpha, const mcspMatDescr_t descrA,
                                 const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                 float *B, int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinDcsrsm2_solve(mcspHandle_t handle, int algo, mcsparseOperation_t transA, mcsparseOperation_t transB,
                                 int m, int nrhs, int nnz, const double *alpha, const mcspMatDescr_t descrA,
                                 const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                 double *B, int ldb, mcspCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                 void *pBuffer);

mcspStatus_t mcspCuinCcsrsm2_solve(mcspHandle_t handle, int algo, mcsparseOperation_t transA, mcsparseOperation_t transB,
                                 int m, int nrhs, int nnz, const mcspComplexFloat *alpha, const mcspMatDescr_t descrA,
                                 const mcspComplexFloat *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspComplexFloat *B, int ldb, mcspCsrsm2Info_t info,
                                 mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinZcsrsm2_solve(mcspHandle_t handle, int algo, mcsparseOperation_t transA, mcsparseOperation_t transB,
                                 int m, int nrhs, int nnz, const mcspComplexDouble *alpha, const mcspMatDescr_t descrA,
                                 const mcspComplexDouble *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspComplexDouble *B, int ldb, mcspCsrsm2Info_t info,
                                 mcsparseSolvePolicy_t policy, void *pBuffer);

/**
 * @brief   Compute BSR-based SpMV in GPU.
 *          C = alpha * op(A) * op(B) + beta * C
 *
 * @param handle             [in]        handle of mcsp library
 * @param dir                [in]        storage format of blocks
 * @param trans_A            [in]        matrix A operation type
 * @param trans_B            [in]        matrix B operation type
 * @param mb                 [in]        number of rows of bsr matrix
 * @param n                  [in]        number of columns of dense matrix op(B) and A
 * @param kb                 [in]        number of block columns of sparse matrix A
 * @param nnzb               [in]        total number of nonzero blocks of BSR matrix
 * @param alpha              [in]        scalar used for multiplication
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_vals           [in]        pointer to the val offset in BSR matrix
 * @param bsr_rows_ind       [in]        pointer to the row offset in BSR matrix
 * @param bsr_cols_ind       [in]        pointer to the col offset in BSR matrix
 * @param block_dim          [in]        number of rows within a block of BSR matrix
 * @param d_B                [in]        array of dimensions (ldb, n) if op(B)=B and (ldb, k) otherwise
 * @param ldb                [in]        leading dimension of B
 * @param beta               [in]        scalar beta
 * @param d_C                [in/out]    array of dimensions (ldc, n)
 * @param ldc                [in]        leading dimension of C
 * @return mcspStatus_t
 */
mcspStatus_t mcspSbsrmm(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans_A,
                        mcsparseOperation_t trans_B, mcspInt mb, mcspInt n, mcspInt kb, mcspInt nnzb,
                        const float *alpha, const mcspMatDescr_t bsr_descr, const float *bsr_vals,
                        const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim, const float *d_B,
                        mcspInt ldb, const float *beta, float *d_C, mcspInt ldc);

mcspStatus_t mcspDbsrmm(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans_A,
                        mcsparseOperation_t trans_B, mcspInt mb, mcspInt n, mcspInt kb, mcspInt nnzb,
                        const double *alpha, const mcspMatDescr_t bsr_descr, const double *bsr_vals,
                        const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim, const double *d_B,
                        mcspInt ldb, const double *beta, double *d_C, mcspInt ldc);

mcspStatus_t mcspCbsrmm(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans_A,
                        mcsparseOperation_t trans_B, mcspInt mb, mcspInt n, mcspInt kb, mcspInt nnzb,
                        const mcspComplexFloat *alpha, const mcspMatDescr_t bsr_descr, const mcspComplexFloat *bsr_vals,
                        const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                        const mcspComplexFloat *d_B, mcspInt ldb, const mcspComplexFloat *beta, mcspComplexFloat *d_C,
                        mcspInt ldc);

mcspStatus_t mcspZbsrmm(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans_A,
                        mcsparseOperation_t trans_B, mcspInt mb, mcspInt n, mcspInt kb, mcspInt nnzb,
                        const mcspComplexDouble *alpha, const mcspMatDescr_t bsr_descr,
                        const mcspComplexDouble *bsr_vals, const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind,
                        mcspInt block_dim, const mcspComplexDouble *d_B, mcspInt ldb, const mcspComplexDouble *beta,
                        mcspComplexDouble *d_C, mcspInt ldc);

mcspStatus_t mcspCuinSbsrmm(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                          mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const float *alpha,
                          const mcspMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                          const int *bsrSortedColIndA, const int blockSize, const float *B, const int ldb,
                          const float *beta, float *C, int ldc);

mcspStatus_t mcspCuinDbsrmm(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                          mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const double *alpha,
                          const mcspMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                          const int *bsrSortedColIndA, const int blockSize, const double *B, const int ldb,
                          const double *beta, double *C, int ldc);

mcspStatus_t mcspCuinCbsrmm(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                          mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const mcFloatComplex *alpha,
                          const mcspMatDescr_t descrA, const mcFloatComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                          const int *bsrSortedColIndA, const int blockSize, const mcFloatComplex *B, const int ldb,
                          const mcFloatComplex *beta, mcFloatComplex *C, int ldc);

mcspStatus_t mcspCuinZbsrmm(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                          mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const mcDoubleComplex *alpha,
                          const mcspMatDescr_t descrA, const mcDoubleComplex *bsrSortedValA,
                          const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize,
                          const mcDoubleComplex *B, const int ldb, const mcDoubleComplex *beta, mcDoubleComplex *C,
                          int ldc);

// bsrsm
mcspStatus_t mcspXbsrsm2_zeroPivot(mcspHandle_t handle, mcspBsrsm2Info_t info, int *position);

mcspStatus_t mcspSbsrsm2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                    mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, float *bsrSortedVal, const mcspInt *bsrSortedRowPtr,
                                    const mcspInt *bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                                    int *pBufferSizeInBytes);

mcspStatus_t mcspDbsrsm2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                    mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, double *bsrSortedVal, const mcspInt *bsrSortedRowPtr,
                                    const mcspInt *bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                                    int *pBufferSizeInBytes);

mcspStatus_t mcspCbsrsm2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                    mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, mcFloatComplex *bsrSortedVal,
                                    const mcspInt *bsrSortedRowPtr, const mcspInt *bsrSortedColInd, mcspInt blockSize,
                                    mcspBsrsm2Info_t info, int *pBufferSizeInBytes);

mcspStatus_t mcspZbsrsm2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                    mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, mcDoubleComplex *bsrSortedVal,
                                    const mcspInt *bsrSortedRowPtr, const mcspInt *bsrSortedColInd, mcspInt blockSize,
                                    mcspBsrsm2Info_t info, int *pBufferSizeInBytes);

mcspStatus_t mcspSbsrsm2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, mcspInt mb, mcspInt n, mcspInt nnzb,
                                       const mcspMatDescr_t descrA, float *bsrSortedVal, const mcspInt *bsrSortedRowPtr,
                                       const mcspInt *bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                                       size_t *pBufferSize);

mcspStatus_t mcspDbsrsm2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, mcspInt mb, mcspInt n, mcspInt nnzb,
                                       const mcspMatDescr_t descrA, double *bsrSortedVal,
                                       const mcspInt *bsrSortedRowPtr, const mcspInt *bsrSortedColInd,
                                       mcspInt blockSize, mcspBsrsm2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspCbsrsm2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, mcspInt mb, mcspInt n, mcspInt nnzb,
                                       const mcspMatDescr_t descrA, mcFloatComplex *bsrSortedVal,
                                       const mcspInt *bsrSortedRowPtr, const mcspInt *bsrSortedColInd,
                                       mcspInt blockSize, mcspBsrsm2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspZbsrsm2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, mcspInt mb, mcspInt n, mcspInt nnzb,
                                       const mcspMatDescr_t descrA, mcDoubleComplex *bsrSortedVal,
                                       const mcspInt *bsrSortedRowPtr, const mcspInt *bsrSortedColInd,
                                       mcspInt blockSize, mcspBsrsm2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspSbsrsm2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                  mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                  const mcspMatDescr_t descrA, const float *bsrSortedVal,
                                  const mcspInt *bsrSortedRowPtr, const mcspInt *bsrSortedColInd, mcspInt blockSize,
                                  mcspBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspDbsrsm2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                  mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                  const mcspMatDescr_t descrA, const double *bsrSortedVal,
                                  const mcspInt *bsrSortedRowPtr, const mcspInt *bsrSortedColInd, mcspInt blockSize,
                                  mcspBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCbsrsm2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                  mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                  const mcspMatDescr_t descrA, const mcFloatComplex *bsrSortedVal,
                                  const mcspInt *bsrSortedRowPtr, const mcspInt *bsrSortedColInd, mcspInt blockSize,
                                  mcspBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspZbsrsm2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                  mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                                  const mcspMatDescr_t descrA, const mcDoubleComplex *bsrSortedVal,
                                  const mcspInt *bsrSortedRowPtr, const mcspInt *bsrSortedColInd, mcspInt blockSize,
                                  mcspBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspSbsrsm2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                               mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb, const float *alpha,
                               const mcspMatDescr_t descrA, const float *bsrSortedVal, const mcspInt *bsrSortedRowPtr,
                               const mcspInt *bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info, const float *B,
                               mcspInt ldb, float *X, mcspInt ldx, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspDbsrsm2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                               mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb, const double *alpha,
                               const mcspMatDescr_t descrA, const double *bsrSortedVal, const mcspInt *bsrSortedRowPtr,
                               const mcspInt *bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                               const double *B, mcspInt ldb, double *X, mcspInt ldx, mcsparseSolvePolicy_t policy,
                               void *pBuffer);

mcspStatus_t mcspCbsrsm2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                               mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                               const mcFloatComplex *alpha, const mcspMatDescr_t descrA,
                               const mcFloatComplex *bsrSortedVal, const mcspInt *bsrSortedRowPtr,
                               const mcspInt *bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                               const mcFloatComplex *B, mcspInt ldb, mcFloatComplex *X, mcspInt ldx,
                               mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspZbsrsm2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                               mcsparseOperation_t transXY, mcspInt mb, mcspInt n, mcspInt nnzb,
                               const mcDoubleComplex *alpha, const mcspMatDescr_t descrA,
                               const mcDoubleComplex *bsrSortedVal, const mcspInt *bsrSortedRowPtr,
                               const mcspInt *bsrSortedColInd, mcspInt blockSize, mcspBsrsm2Info_t info,
                               const mcDoubleComplex *B, mcspInt ldb, mcDoubleComplex *X, mcspInt ldx,
                               mcsparseSolvePolicy_t policy, void *pBuffer);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSP_LEVEL3_H_
