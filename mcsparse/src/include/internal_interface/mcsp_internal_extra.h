#ifndef INTERFACE_MCSP_INTERNAL_EXTRA_H_
#define INTERFACE_MCSP_INTERNAL_EXTRA_H_

#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspCuinScsrgeam2_bufferSizeExt(mcspHandle_t handle, int m, int n, const float* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta,
                                           const mcspMatDescr_t descrB, int nnzB, const float* csrSortedValB,
                                           const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                           const mcspMatDescr_t descrC, const float* csrSortedValC,
                                           const int* csrSortedRowPtrC, const int* csrSortedColIndC,
                                           size_t* pBufferSizeInBytes);

mcspStatus_t mcspCuinDcsrgeam2_bufferSizeExt(mcspHandle_t handle, int m, int n, const double* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta,
                                           const mcspMatDescr_t descrB, int nnzB, const double* csrSortedValB,
                                           const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                           const mcspMatDescr_t descrC, const double* csrSortedValC,
                                           const int* csrSortedRowPtrC, const int* csrSortedColIndC,
                                           size_t* pBufferSizeInBytes);

mcspStatus_t mcspCuinCcsrgeam2_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcspComplexFloat* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const mcspComplexFloat* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                           const mcspComplexFloat* beta, const mcspMatDescr_t descrB, int nnzB,
                                           const mcspComplexFloat* csrSortedValB, const int* csrSortedRowPtrB,
                                           const int* csrSortedColIndB, const mcspMatDescr_t descrC,
                                           const mcspComplexFloat* csrSortedValC, const int* csrSortedRowPtrC,
                                           const int* csrSortedColIndC, size_t* pBufferSizeInBytes);

mcspStatus_t mcspCuinZcsrgeam2_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcspComplexDouble* alpha,
                                           const mcspMatDescr_t descrA, int nnzA,
                                           const mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                           const int* csrSortedColIndA, const mcspComplexDouble* beta,
                                           const mcspMatDescr_t descrB, int nnzB,
                                           const mcspComplexDouble* csrSortedValB, const int* csrSortedRowPtrB,
                                           const int* csrSortedColIndB, const mcspMatDescr_t descrC,
                                           const mcspComplexDouble* csrSortedValC, const int* csrSortedRowPtrC,
                                           const int* csrSortedColIndC, size_t* pBufferSizeInBytes);

/**
 * @brief   Compute CSR-based Geam in single or double precision in GPU
 *          1st step: calculate nnz and row array of output matrix C
 *          C = alpha * A + beta * B
 *
 * @param handle        [in]        handle of mcsp library
 * @param m             [in]        number of rows of A, B and C
 * @param n             [in]        number of columns of A, B and C
 * @param descr_A       [in]        descriptor of the sparse matrix A
 * @param nnz_A         [in]        number of nonzeros of A
 * @param csr_rows_A    [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols_A    [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param descr_B       [in]        descriptor of the sparse matrix B
 * @param nnz_B         [in]        number of nonzeros of B
 * @param csr_rows_B    [in]        pointer to the row offset in CSR matrix B
 * @param csr_cols_B    [in]        pointer to the column indexes of nonzeros in CSR matrix B
 * @param descr_C       [in]        descriptor of the sparse matrix C
 * @param nnz_C         [out]        number of nonzeros of C
 * @param csr_rows_C    [out]        pointer to the row offset in CSR matrix C
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrGeamNnz(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t descr_A, mcspInt nnz_A,
                            const mcspInt* csr_rows_A, const mcspInt* csr_cols_A, const mcspMatDescr_t descr_B,
                            mcspInt nnz_B, const mcspInt* csr_rows_B, const mcspInt* csr_cols_B,
                            const mcspMatDescr_t descr_C, mcspInt* csr_rows_C, mcspInt* nnz_C);

mcspStatus_t mcspCuinXcsrgeam2Nnz(mcspHandle_t handle, int m, int n, const mcspMatDescr_t descrA, int nnzA,
                                const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcspMatDescr_t descrB,
                                int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                const mcspMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr,
                                void* workspace);

/**
 * @brief   Compute CSR-based Geam in single or double precision in GPU
 *          2nd step: calculate colomn and value array of output matrix C
 *          C = alpha * A + beta * B
 *
 * @param handle        [in]        handle of mcsp library
 * @param m             [in]        number of rows of A, B and C
 * @param n             [in]        number of columns of A, B and C
 * @param alpha         [in]        alpha
 * @param descr_A       [in]        descriptor of the sparse matrix A
 * @param nnz_A         [in]        number of nonzeros of A
 * @param csr_vals_A    [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows_A    [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols_A    [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param beta          [in]        beta
 * @param descr_B       [in]        descriptor of the sparse matrix B
 * @param nnz_B         [in]        number of nonzeros of B
 * @param csr_vals_B    [in]        pointer to the values of nonzeros in CSR matrix B
 * @param csr_rows_B    [in]        pointer to the row offset in CSR matrix B
 * @param csr_cols_B    [in]        pointer to the column indexes of nonzeros in CSR matrix B
 * @param descr_C       [in]        descriptor of the sparse matrix C
 * @param csr_vals_C    [out]        pointer to the values of nonzeros in CSR matrix C
 * @param csr_rows_C    [in]        pointer to the row offset in CSR matrix C
 * @param csr_cols_C    [out]        pointer to the column indexes of nonzeros in CSR matrix C
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrGeam(mcspHandle_t handle, mcspInt m, mcspInt n, const float* alpha, const mcspMatDescr_t descr_A,
                          mcspInt nnz_A, const float* csr_vals_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                          const float* beta, const mcspMatDescr_t descr_B, mcspInt nnz_B, const float* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspMatDescr_t descr_C,
                          float* csr_vals_C, const mcspInt* csr_rows_C, mcspInt* csr_cols_C);

mcspStatus_t mcspDcsrGeam(mcspHandle_t handle, mcspInt m, mcspInt n, const double* alpha, const mcspMatDescr_t descr_A,
                          mcspInt nnz_A, const double* csr_vals_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                          const double* beta, const mcspMatDescr_t descr_B, mcspInt nnz_B, const double* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspMatDescr_t descr_C,
                          double* csr_vals_C, const mcspInt* csr_rows_C, mcspInt* csr_cols_C);

mcspStatus_t mcspCcsrGeam(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* alpha,
                          const mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspComplexFloat* csr_vals_A,
                          const mcspInt* csr_rows_A, const mcspInt* csr_cols_A, const mcspComplexFloat* beta,
                          const mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspComplexFloat* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspMatDescr_t descr_C,
                          mcspComplexFloat* csr_vals_C, const mcspInt* csr_rows_C, mcspInt* csr_cols_C);

mcspStatus_t mcspZcsrGeam(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* alpha,
                          const mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspComplexDouble* csr_vals_A,
                          const mcspInt* csr_rows_A, const mcspInt* csr_cols_A, const mcspComplexDouble* beta,
                          const mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspComplexDouble* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspMatDescr_t descr_C,
                          mcspComplexDouble* csr_vals_C, const mcspInt* csr_rows_C, mcspInt* csr_cols_C);

mcspStatus_t mcspCuinScsrgeam2(mcspHandle_t handle, int m, int n, const float* alpha, const mcspMatDescr_t descrA,
                             int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA, const float* beta, const mcspMatDescr_t descrB, int nnzB,
                             const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                             const mcspMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC,
                             int* csrSortedColIndC, void* pBuffer);

mcspStatus_t mcspCuinDcsrgeam2(mcspHandle_t handle, int m, int n, const double* alpha, const mcspMatDescr_t descrA,
                             int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA, const double* beta, const mcspMatDescr_t descrB, int nnzB,
                             const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                             const mcspMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC,
                             int* csrSortedColIndC, void* pBuffer);

mcspStatus_t mcspCuinCcsrgeam2(mcspHandle_t handle, int m, int n, const mcspComplexFloat* alpha,
                             const mcspMatDescr_t descrA, int nnzA, const mcspComplexFloat* csrSortedValA,
                             const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcspComplexFloat* beta,
                             const mcspMatDescr_t descrB, int nnzB, const mcspComplexFloat* csrSortedValB,
                             const int* csrSortedRowPtrB, const int* csrSortedColIndB, const mcspMatDescr_t descrC,
                             mcspComplexFloat* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,
                             void* pBuffer);

mcspStatus_t mcspCuinZcsrgeam2(mcspHandle_t handle, int m, int n, const mcspComplexDouble* alpha,
                             const mcspMatDescr_t descrA, int nnzA, const mcspComplexDouble* csrSortedValA,
                             const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcspComplexDouble* beta,
                             const mcspMatDescr_t descrB, int nnzB, const mcspComplexDouble* csrSortedValB,
                             const int* csrSortedRowPtrB, const int* csrSortedColIndB, const mcspMatDescr_t descrC,
                             mcspComplexDouble* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,
                             void* pBuffer);

/**
 * @brief   Compute CSR-based SpGemm in single or double precision in GPU, csr_scalar algorithm
 *          1st step: determine buffer size needed for the calculation
 *          D = alpha * A * B + beta * C
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans_A       [in]        matrix operation type of sparse matrix A
 * @param trans_B       [in]        matrix operation type of dense matrix B
 * @param m             [in]        number of rows of op(A) and C
 * @param n             [in]        number of columns of op(B) and C
 * @param k             [in]        number of columns of op(A) and rows of op(B)
 * @param alpha         [in]        alpha
 * @param descr_A       [in]        descriptor of the sparse matrix A
 * @param nnz_A         [in]        number of nonzeros of A
 * @param csr_rows_A    [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols_A    [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param descr_B       [in]        descriptor of the sparse matrix B
 * @param nnz_B         [in]        number of nonzeros of B
 * @param csr_rows_B    [in]        pointer to the row offset in CSR matrix B
 * @param csr_cols_B    [in]        pointer to the column indexes of nonzeros in CSR matrix B
 * @param beta          [in]        beta
 * @param descr_C       [in]        descriptor of the sparse matrix C
 * @param nnz_C         [in]        number of nonzeros of C
 * @param csr_rows_C    [in]        pointer to the row offset in CSR matrix C
 * @param csr_cols_C    [in]        pointer to the column indexes of nonzeros in CSR matrix C
 * @param info_D        [in/out]    meta data for CSR matrix D
 * @param buffer_size   [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrgemmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt n, mcspInt k, const float* alpha, mcspMatDescr_t descr_A,
                                    mcspInt nnz_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                                    mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspInt* csr_rows_B,
                                    const mcspInt* csr_cols_B, const float* beta, mcspMatDescr_t descr_C, mcspInt nnz_C,
                                    const mcspInt* csr_rows_C, const mcspInt* csr_cols_C, mcspMatInfo_t info_D,
                                    size_t* buffer_size);
mcspStatus_t mcspDcsrgemmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt n, mcspInt k, const double* alpha, mcspMatDescr_t descr_A,
                                    mcspInt nnz_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                                    mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspInt* csr_rows_B,
                                    const mcspInt* csr_cols_B, const double* beta, mcspMatDescr_t descr_C,
                                    mcspInt nnz_C, const mcspInt* csr_rows_C, const mcspInt* csr_cols_C,
                                    mcspMatInfo_t info_D, size_t* buffer_size);
mcspStatus_t mcspCcsrgemmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt n, mcspInt k, const mcspComplexFloat* alpha,
                                    mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspInt* csr_rows_A,
                                    const mcspInt* csr_cols_A, mcspMatDescr_t descr_B, mcspInt nnz_B,
                                    const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspComplexFloat* beta,
                                    mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspInt* csr_rows_C,
                                    const mcspInt* csr_cols_C, mcspMatInfo_t info_D, size_t* buffer_size);
mcspStatus_t mcspZcsrgemmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt n, mcspInt k, const mcspComplexDouble* alpha,
                                    mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspInt* csr_rows_A,
                                    const mcspInt* csr_cols_A, mcspMatDescr_t descr_B, mcspInt nnz_B,
                                    const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspComplexDouble* beta,
                                    mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspInt* csr_rows_C,
                                    const mcspInt* csr_cols_C, mcspMatInfo_t info_D, size_t* buffer_size);

mcspStatus_t mcspCuinScsrgemm2_bufferSizeExt(mcspHandle_t handle, int m, int n, int k, const float* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
                                           const int* csrSortedColIndA, const mcspMatDescr_t descrB, int nnzB,
                                           const int* csrSortedRowPtrB, const int* csrSortedColIndB, const float* beta,
                                           const mcspMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD,
                                           const int* csrSortedColIndD, mcspCsrgemm2Info_t info,
                                           size_t* pBufferSizeInBytes);

mcspStatus_t mcspCuinDcsrgemm2_bufferSizeExt(mcspHandle_t handle, int m, int n, int k, const double* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
                                           const int* csrSortedColIndA, const mcspMatDescr_t descrB, int nnzB,
                                           const int* csrSortedRowPtrB, const int* csrSortedColIndB, const double* beta,
                                           const mcspMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD,
                                           const int* csrSortedColIndD, mcspCsrgemm2Info_t info,
                                           size_t* pBufferSizeInBytes);

mcspStatus_t mcspCuinCcsrgemm2_bufferSizeExt(mcspHandle_t handle, int m, int n, int k, const mcspComplexFloat* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
                                           const int* csrSortedColIndA, const mcspMatDescr_t descrB, int nnzB,
                                           const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                           const mcspComplexFloat* beta, const mcspMatDescr_t descrD, int nnzD,
                                           const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                           mcspCsrgemm2Info_t info, size_t* pBufferSizeInBytes);

mcspStatus_t mcspCuinZcsrgemm2_bufferSizeExt(mcspHandle_t handle, int m, int n, int k, const mcspComplexDouble* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
                                           const int* csrSortedColIndA, const mcspMatDescr_t descrB, int nnzB,
                                           const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                           const mcspComplexDouble* beta, const mcspMatDescr_t descrD, int nnzD,
                                           const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                           mcspCsrgemm2Info_t info, size_t* pBufferSizeInBytes);

/**
 * @brief   Compute CSR-based SpGemm in single or double precision in GPU, csr_scalar algorithm
 *          2nd step: calculate nnz and row array of output matrix D
 *          D = alpha * A * B + beta * C
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans_A       [in]        matrix operation type of sparse matrix A
 * @param trans_B       [in]        matrix operation type of dense matrix B
 * @param m             [in]        number of rows of op(A) and C
 * @param n             [in]        number of columns of op(B) and C
 * @param k             [in]        number of columns of op(A) and rows of op(B)
 * @param alpha         [in]        alpha
 * @param descr_A       [in]        descriptor of the sparse matrix A
 * @param nnz_A         [in]        number of nonzeros of A
 * @param csr_rows_A    [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols_A    [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param descr_B       [in]        descriptor of the sparse matrix B
 * @param nnz_B         [in]        number of nonzeros of B
 * @param csr_rows_B    [in]        pointer to the row offset in CSR matrix B
 * @param csr_cols_B    [in]        pointer to the column indexes of nonzeros in CSR matrix B
 * @param beta          [in]        beta
 * @param descr_C       [in]        descriptor of the sparse matrix C
 * @param nnz_C         [in]        number of nonzeros of C
 * @param csr_rows_C    [in]        pointer to the row offset in CSR matrix C
 * @param csr_cols_C    [in]        pointer to the column indexes of nonzeros in CSR matrix C
 * @param descr_D       [in]        descriptor of the sparse matrix D
 * @param csr_rows_D    [out]       pointer to the row offset in CSR matrix D
 * @param nnz_D         [out]       number of nonzeros of D
 * @param info_D        [in]        meta data for CSR matrix D
 * @param buffer        [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrgemmNnz(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                            mcspInt n, mcspInt k, mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspInt* csr_rows_A,
                            const mcspInt* csr_cols_A, mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspInt* csr_rows_B,
                            const mcspInt* csr_cols_B, mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspInt* csr_rows_C,
                            const mcspInt* csr_cols_C, mcspMatDescr_t descr_D, mcspInt* csr_rows_D, mcspInt* nnz_D,
                            mcspMatInfo_t info_D, void* buffer);

mcspStatus_t mcspCuinXcsrgemm2Nnz(mcspHandle_t handle, int m, int n, int k, const mcspMatDescr_t descrA, int nnzA,
                                const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcspMatDescr_t descrB,
                                int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                const mcspMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD,
                                const int* csrSortedColIndD, const mcspMatDescr_t descrC, int* csrSortedRowPtrC,
                                int* nnzTotalDevHostPtr, const mcspCsrgemm2Info_t info, void* pBuffer);

/**
 * @brief   Compute CSR-based SpGemm in single or double precision in GPU, csr_scalar algorithm
 *          3rd step: calculate colomn and value array of output matrix D
 *          D = alpha * A * B + beta * C
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans_A       [in]        matrix operation type of sparse matrix A
 * @param trans_B       [in]        matrix operation type of dense matrix B
 * @param m             [in]        number of rows of op(A) and C
 * @param n             [in]        number of columns of op(B) and C
 * @param k             [in]        number of columns of op(A) and rows of op(B)
 * @param alpha         [in]        alpha
 * @param descr_A       [in]        descriptor of the sparse matrix A
 * @param nnz_A         [in]        number of nonzeros of A
 * @param csr_vals_A    [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows_A    [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols_A    [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param descr_B       [in]        descriptor of the sparse matrix B
 * @param nnz_B         [in]        number of nonzeros of B
 * @param csr_vals_B    [in]        pointer to the values of nonzeros in CSR matrix B
 * @param csr_rows_B    [in]        pointer to the row offset in CSR matrix B
 * @param csr_cols_B    [in]        pointer to the column indexes of nonzeros in CSR matrix B
 * @param beta          [in]        beta
 * @param descr_C       [in]        descriptor of the sparse matrix C
 * @param nnz_C         [in]        number of nonzeros of C
 * @param csr_vals_C    [in]        pointer to the values of nonzeros in CSR matrix C
 * @param csr_rows_C    [in]        pointer to the row offset in CSR matrix C
 * @param csr_cols_C    [in]        pointer to the column indexes of nonzeros in CSR matrix C
 * @param descr_D       [in]        descriptor of the sparse matrix D
 * @param csr_vals_D    [out]       pointer to the values of nonzeros in CSR matrix D
 * @param csr_rows_D    [in]        pointer to the row offset in CSR matrix D
 * @param csr_cols_D    [out]       pointer to the column indexes of nonzeros in CSR matrix D
 * @param info_D        [in]        meta data for CSR matrix D
 * @param buffer        [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrgemm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, const float* alpha, mcspMatDescr_t descr_A, mcspInt nnz_A,
                          const float* csr_vals_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                          mcspMatDescr_t descr_B, mcspInt nnz_B, const float* csr_vals_B, const mcspInt* csr_rows_B,
                          const mcspInt* csr_cols_B, const float* beta, mcspMatDescr_t descr_C, mcspInt nnz_C,
                          const float* csr_vals_C, const mcspInt* csr_rows_C, const mcspInt* csr_cols_C,
                          mcspMatDescr_t descr_D, float* csr_vals_D, const mcspInt* csr_rows_D, mcspInt* csr_cols_D,
                          mcspMatInfo_t info_D, void* buffer);
mcspStatus_t mcspDcsrgemm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, const double* alpha, mcspMatDescr_t descr_A, mcspInt nnz_A,
                          const double* csr_vals_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                          mcspMatDescr_t descr_B, mcspInt nnz_B, const double* csr_vals_B, const mcspInt* csr_rows_B,
                          const mcspInt* csr_cols_B, const double* beta, mcspMatDescr_t descr_C, mcspInt nnz_C,
                          const double* csr_vals_C, const mcspInt* csr_rows_C, const mcspInt* csr_cols_C,
                          mcspMatDescr_t descr_D, double* csr_vals_D, const mcspInt* csr_rows_D, mcspInt* csr_cols_D,
                          mcspMatInfo_t info_D, void* buffer);
mcspStatus_t mcspCcsrgemm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, const mcspComplexFloat* alpha, mcspMatDescr_t descr_A, mcspInt nnz_A,
                          const mcspComplexFloat* csr_vals_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                          mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspComplexFloat* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspComplexFloat* beta,
                          mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspComplexFloat* csr_vals_C,
                          const mcspInt* csr_rows_C, const mcspInt* csr_cols_C, mcspMatDescr_t descr_D,
                          mcspComplexFloat* csr_vals_D, const mcspInt* csr_rows_D, mcspInt* csr_cols_D,
                          mcspMatInfo_t info_D, void* buffer);
mcspStatus_t mcspZcsrgemm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, const mcspComplexDouble* alpha, mcspMatDescr_t descr_A, mcspInt nnz_A,
                          const mcspComplexDouble* csr_vals_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                          mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspComplexDouble* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspComplexDouble* beta,
                          mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspComplexDouble* csr_vals_C,
                          const mcspInt* csr_rows_C, const mcspInt* csr_cols_C, mcspMatDescr_t descr_D,
                          mcspComplexDouble* csr_vals_D, const mcspInt* csr_rows_D, mcspInt* csr_cols_D,
                          mcspMatInfo_t info_D, void* buffer);

mcspStatus_t mcspCuinScsrgemm2(mcspHandle_t handle, int m, int n, int k, const float* alpha, const mcspMatDescr_t descrA,
                             int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA, const mcspMatDescr_t descrB, int nnzB,
                             const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                             const float* beta, const mcspMatDescr_t descrD, int nnzD, const float* csrSortedValD,
                             const int* csrSortedRowPtrD, const int* csrSortedColIndD, const mcspMatDescr_t descrC,
                             float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC,
                             const mcspCsrgemm2Info_t info, void* pBuffer);

mcspStatus_t mcspCuinDcsrgemm2(mcspHandle_t handle, int m, int n, int k, const double* alpha, const mcspMatDescr_t descrA,
                             int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA, const mcspMatDescr_t descrB, int nnzB,
                             const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                             const double* beta, const mcspMatDescr_t descrD, int nnzD, const double* csrSortedValD,
                             const int* csrSortedRowPtrD, const int* csrSortedColIndD, const mcspMatDescr_t descrC,
                             double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC,
                             const mcspCsrgemm2Info_t info, void* pBuffer);

mcspStatus_t mcspCuinCcsrgemm2(mcspHandle_t handle, int m, int n, int k, const mcspComplexFloat* alpha,
                             const mcspMatDescr_t descrA, int nnzA, const mcspComplexFloat* csrSortedValA,
                             const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcspMatDescr_t descrB,
                             int nnzB, const mcspComplexFloat* csrSortedValB, const int* csrSortedRowPtrB,
                             const int* csrSortedColIndB, const mcspComplexFloat* beta, const mcspMatDescr_t descrD,
                             int nnzD, const mcspComplexFloat* csrSortedValD, const int* csrSortedRowPtrD,
                             const int* csrSortedColIndD, const mcspMatDescr_t descrC, mcspComplexFloat* csrSortedValC,
                             const int* csrSortedRowPtrC, int* csrSortedColIndC, const mcspCsrgemm2Info_t info,
                             void* pBuffer);

mcspStatus_t mcspCuinZcsrgemm2(mcspHandle_t handle, int m, int n, int k, const mcspComplexDouble* alpha,
                             const mcspMatDescr_t descrA, int nnzA, const mcspComplexDouble* csrSortedValA,
                             const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcspMatDescr_t descrB,
                             int nnzB, const mcspComplexDouble* csrSortedValB, const int* csrSortedRowPtrB,
                             const int* csrSortedColIndB, const mcspComplexDouble* beta, const mcspMatDescr_t descrD,
                             int nnzD, const mcspComplexDouble* csrSortedValD, const int* csrSortedRowPtrD,
                             const int* csrSortedColIndD, const mcspMatDescr_t descrC, mcspComplexDouble* csrSortedValC,
                             const int* csrSortedRowPtrC, int* csrSortedColIndC, const mcspCsrgemm2Info_t info,
                             void* pBuffer);

mcspStatus_t mcspCuinScsrmv(mcspHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz, const float* alpha,
                          const mcspMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA,
                          const int* csrColIndA, const float* x, const float* beta, float* y);

mcspStatus_t mcspCuinDcsrmv(mcspHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz, const double* alpha,
                          const mcspMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA,
                          const int* csrColIndA, const double* x, const double* beta, double* y);

mcspStatus_t mcspCuinCcsrmv(mcspHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                          const mcspComplexFloat* alpha, const mcspMatDescr_t descrA, const mcspComplexFloat* csrValA,
                          const int* csrRowPtrA, const int* csrColIndA, const mcspComplexFloat* x,
                          const mcspComplexFloat* beta, mcspComplexFloat* y);

mcspStatus_t mcspCuinZcsrmv(mcspHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                          const mcspComplexDouble* alpha, const mcspMatDescr_t descrA, const mcspComplexDouble* csrValA,
                          const int* csrRowPtrA, const int* csrColIndA, const mcspComplexDouble* x,
                          const mcspComplexDouble* beta, mcspComplexDouble* y);

mcspStatus_t mcspCuinScsrmm2(mcspHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m, int n,
                           int k, int nnz, const float* alpha, const mcspMatDescr_t descrA, const float* csrValA,
                           const int* csrRowPtrA, const int* csrColIndA, const float* B, int ldb, const float* beta,
                           float* C, int ldc);

mcspStatus_t mcspCuinDcsrmm2(mcspHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m, int n,
                           int k, int nnz, const double* alpha, const mcspMatDescr_t descrA, const double* csrValA,
                           const int* csrRowPtrA, const int* csrColIndA, const double* B, int ldb, const double* beta,
                           double* C, int ldc);

mcspStatus_t mcspCuinCcsrmm2(mcspHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m, int n,
                           int k, int nnz, const mcspComplexFloat* alpha, const mcspMatDescr_t descrA,
                           const mcspComplexFloat* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                           const mcspComplexFloat* B, int ldb, const mcspComplexFloat* beta, mcspComplexFloat* C,
                           int ldc);

mcspStatus_t mcspCuinZcsrmm2(mcspHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m, int n,
                           int k, int nnz, const mcspComplexDouble* alpha, const mcspMatDescr_t descrA,
                           const mcspComplexDouble* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                           const mcspComplexDouble* B, int ldb, const mcspComplexDouble* beta, mcspComplexDouble* C,
                           int ldc);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSP_EXTRA_H_
