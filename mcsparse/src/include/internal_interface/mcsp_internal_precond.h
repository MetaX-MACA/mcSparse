#ifndef INTERFACE_MCSP_INTERNAL_PRECOND_H_
#define INTERFACE_MCSP_INTERNAL_PRECOND_H_

#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief   Compute incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage format
 *          1st step: determine buffer size needed for the calculation
 *          A '=. L * U
 *
 * @param handle        [in]        handle of mcsp library
 * @param m             [in]        number of rows of A
 * @param nnz           [in]        number of nonzeros of A
 * @param descr         [in]        descriptor of the sparse matrix A
 * @param csr_vals      [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows      [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols      [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param info          [in]        meta data for CSR matrix A
 * @param buffer_size   [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrilu0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                    const float* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                    mcspMatInfo_t info, size_t* buffersize);
mcspStatus_t mcspDcsrilu0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                    const double* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                    mcspMatInfo_t info, size_t* buffersize);
mcspStatus_t mcspCcsrilu0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                    const mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                    mcspMatInfo_t info, size_t* buffersize);
mcspStatus_t mcspZcsrilu0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                    const mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                    mcspMatInfo_t info, size_t* buffersize);

/**
 * @brief   Compute incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage format
 *          2nd step: analyze sparse matrix A for the calculation
 *          A '=. L * U
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of A
 * @param nnz             [in]        number of nonzeros of A
 * @param descr           [in]        descriptor of the sparse matrix A
 * @param csr_vals        [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows        [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols        [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param info            [in/out]    meta data for CSR matrix A
 * @param analysis_policy [in]        analysis policy
 * @param solve_policy    [in]        solve policy
 * @param temp_buffer     [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrilu0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                  const float* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                  mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer);
mcspStatus_t mcspDcsrilu0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                  const double* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                  mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer);
mcspStatus_t mcspCcsrilu0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                  const mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                  mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer);
mcspStatus_t mcspZcsrilu0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                  const mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                  mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

/**
 * @brief   Compute incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage format
 *          3rd step: compute incomplete LU factorization
 *          A '=. L * U
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of A
 * @param nnz             [in]        number of nonzeros of A
 * @param descr           [in]        descriptor of the sparse matrix A
 * @param csr_vals        [in/out]    pointer to the values of nonzeros in CSR matrix A and L, U
 * @param csr_rows        [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols        [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param info            [in]        meta data for CSR matrix A
 * @param solve_policy    [in]        solve policy
 * @param temp_buffer     [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrilu0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, float* csr_vals,
                          const mcspInt* csr_rows, const mcspInt* csr_cols, mcspMatInfo_t info,
                          mcsparseSolvePolicy_t solve_policy, void* temp_buffer);
mcspStatus_t mcspDcsrilu0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, double* csr_vals,
                          const mcspInt* csr_rows, const mcspInt* csr_cols, mcspMatInfo_t info,
                          mcsparseSolvePolicy_t solve_policy, void* temp_buffer);
mcspStatus_t mcspCcsrilu0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                          mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                          mcspMatInfo_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);
mcspStatus_t mcspZcsrilu0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                          mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                          mcspMatInfo_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

/**
 * @brief   Check whether the sparse matrix have structual or numerical zero pivot
 *
 * @param handle          [in]        handle of mcsp library
 * @param info            [in]        meta data for CSR matrix A
 * @param position        [out]       position of first zero pivot if exists
 * @return mcspStatus_t
 */
mcspStatus_t mcspXcsrilu0ZeroPivot(mcspHandle_t handle, mcspMatInfo_t info, int* position);

/**
 * @brief   Clear memory allocation in "2nd step: analyze sparse matrix A for the calculation"
 *
 * @param handle          [in]        handle of mcsp library
 * @param info            [in/out]    meta data for CSR matrix A
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrilu0Clear(mcspHandle_t handle, mcspMatInfo_t info);

mcspStatus_t mcspCuinXcsrilu02_zeroPivot(mcspHandle_t handle, mcspCsrilu02Info_t info, int* position);

mcspStatus_t mcspCuinScsrilu02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                        float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                        mcspCsrilu02Info_t info, int* pBufferSizeInBytes);

mcspStatus_t mcspCuinDcsrilu02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                        double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                        mcspCsrilu02Info_t info, int* pBufferSizeInBytes);

mcspStatus_t mcspCuinCcsrilu02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                        mcspComplexFloat* csrSortedValA, const int* csrSortedRowPtrA,
                                        const int* csrSortedColIndA, mcspCsrilu02Info_t info, int* pBufferSizeInBytes);

mcspStatus_t mcspCuinZcsrilu02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                        mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                        const int* csrSortedColIndA, mcspCsrilu02Info_t info, int* pBufferSizeInBytes);

mcspStatus_t mcspCuinScsrilu02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                           float* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd,
                                           mcspCsrilu02Info_t info, size_t* pBufferSize);

mcspStatus_t mcspCuinDcsrilu02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                           double* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd,
                                           mcspCsrilu02Info_t info, size_t* pBufferSize);

mcspStatus_t mcspCuinCcsrilu02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                           mcspComplexFloat* csrSortedVal, const int* csrSortedRowPtr,
                                           const int* csrSortedColInd, mcspCsrilu02Info_t info, size_t* pBufferSize);

mcspStatus_t mcspCuinZcsrilu02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                           mcspComplexDouble* csrSortedVal, const int* csrSortedRowPtr,
                                           const int* csrSortedColInd, mcspCsrilu02Info_t info, size_t* pBufferSize);

mcspStatus_t mcspCuinScsrilu02_numericBoost(mcspHandle_t handle, mcspCsrilu02Info_t info, int enable_boost, double* tol,
                                          float* boost_val);

mcspStatus_t mcspCuinDcsrilu02_numericBoost(mcspHandle_t handle, mcspCsrilu02Info_t info, int enable_boost, double* tol,
                                          double* boost_val);

mcspStatus_t mcspCuinCcsrilu02_numericBoost(mcspHandle_t handle, mcspCsrilu02Info_t info, int enable_boost, double* tol,
                                          mcspComplexFloat* boost_val);

mcspStatus_t mcspCuinZcsrilu02_numericBoost(mcspHandle_t handle, mcspCsrilu02Info_t info, int enable_boost, double* tol,
                                          mcspComplexDouble* boost_val);

mcspStatus_t mcspCuinScsrilu02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                      const float* csrSortedValA, const int* csrSortedRowPtrA,
                                      const int* csrSortedColIndA, mcspCsrilu02Info_t info,
                                      mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspCuinDcsrilu02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                      const double* csrSortedValA, const int* csrSortedRowPtrA,
                                      const int* csrSortedColIndA, mcspCsrilu02Info_t info,
                                      mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspCuinCcsrilu02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                      const mcspComplexFloat* csrSortedValA, const int* csrSortedRowPtrA,
                                      const int* csrSortedColIndA, mcspCsrilu02Info_t info,
                                      mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspCuinZcsrilu02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                                      const mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                      const int* csrSortedColIndA, mcspCsrilu02Info_t info,
                                      mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspCuinScsrilu02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                             float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                             mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspCuinDcsrilu02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                             double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                             mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspCuinCcsrilu02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                             mcspComplexFloat* csrSortedValA_valM, const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA, mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy,
                             void* pBuffer);

mcspStatus_t mcspCuinZcsrilu02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descrA,
                             mcspComplexDouble* csrSortedValA_valM, const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA, mcspCsrilu02Info_t info, mcsparseSolvePolicy_t policy,
                             void* pBuffer);

/**
 * @brief   Compute incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR storage format
 *          1st step: determine buffer size needed for the calculation
 *          A '=. L * L^H
 *
 * @param handle        [in]        handle of mcsp library
 * @param m             [in]        number of rows of A
 * @param nnz           [in]        number of nonzeros of A
 * @param descr         [in]        descriptor of the sparse matrix A
 * @param csr_vals      [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows      [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols      [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param info          [in]        meta data for CSR matrix A
 * @param buffer_size   [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsric0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                   const float* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                   mcspMatInfo_t info, size_t* buffersize);

mcspStatus_t mcspDcsric0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                   const double* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                   mcspMatInfo_t info, size_t* buffersize);

mcspStatus_t mcspCcsric0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                   const mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                   mcspMatInfo_t info, size_t* buffersize);

mcspStatus_t mcspZcsric0Buffersize(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                   const mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                   mcspMatInfo_t info, size_t* buffersize);

/**
 * @brief   Compute incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR storage format
 *          2nd step: analyze sparse matrix A for the calculation
 *          A '=. L * L^H
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of A
 * @param nnz             [in]        number of nonzeros of A
 * @param descr           [in]        descriptor of the sparse matrix A
 * @param csr_vals        [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows        [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols        [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param info            [in/out]    meta data for CSR matrix A
 * @param analysis_policy [in]        analysis policy
 * @param solve_policy    [in]        solve policy
 * @param temp_buffer     [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsric0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                 const float* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                 mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                 mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspDcsric0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                 const double* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                 mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                 mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspCcsric0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                 const mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                 mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                 mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspZcsric0Analysis(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                                 const mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                                 mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                 mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

/**
 * @brief   Compute incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR storage format
 *          3rd step: compute incomplete Cholesky factorization
 *          A '=. L * L^H
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of A
 * @param nnz             [in]        number of nonzeros of A
 * @param descr           [in]        descriptor of the sparse matrix A
 * @param csr_vals        [in/out]    pointer to the values of nonzeros in CSR matrix A and L
 * @param csr_rows        [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols        [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param info            [in]        meta data for CSR matrix A
 * @param solve_policy    [in]        solve policy
 * @param temp_buffer     [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsric0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, float* csr_vals,
                         const mcspInt* csr_rows, const mcspInt* csr_cols, mcspMatInfo_t info,
                         mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspDcsric0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, double* csr_vals,
                         const mcspInt* csr_rows, const mcspInt* csr_cols, mcspMatInfo_t info,
                         mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspCcsric0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                         mcspComplexFloat* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                         mcspMatInfo_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspZcsric0(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                         mcspComplexDouble* csr_vals, const mcspInt* csr_rows, const mcspInt* csr_cols,
                         mcspMatInfo_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

/**
 * @brief   Check whether the sparse matrix have structual or numerical zero pivot
 *
 * @param handle          [in]        handle of mcsp library
 * @param info            [in]        meta data for CSR matrix A
 * @param position        [out]       position of first zero pivot if exists
 * @return mcspStatus_t
 */
mcspStatus_t mcspXcsric0ZeroPivot(mcspHandle_t handle, mcspMatInfo_t info, int* position);

/**
 * @brief   Clear memory allocation in "2nd step: analyze sparse matrix A for the calculation"
 *
 * @param handle          [in]        handle of mcsp library
 * @param info            [in/out]    meta data for CSR matrix A
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsric0Clear(mcspHandle_t handle, mcspMatInfo_t info);

mcspStatus_t mcspCuinScsric02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                       const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                       mcspCsric02Info_t info, int* buffersize);

mcspStatus_t mcspCuinDcsric02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                       const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                       mcspCsric02Info_t info, int* buffersize);

mcspStatus_t mcspCuinCcsric02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                       const mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                       mcspCsric02Info_t info, int* buffersize);

mcspStatus_t mcspCuinZcsric02_bufferSize(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                       const mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                       mcspCsric02Info_t info, int* buffersize);

mcspStatus_t mcspCuinScsric02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                          const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                          mcspCsric02Info_t info, size_t* buffersize);

mcspStatus_t mcspCuinDcsric02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                          const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                          mcspCsric02Info_t info, size_t* buffersize);

mcspStatus_t mcspCuinCcsric02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                          const mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                          mcspCsric02Info_t info, size_t* buffersize);

mcspStatus_t mcspCuinZcsric02_bufferSizeExt(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                          const mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                          mcspCsric02Info_t info, size_t* buffersize);

mcspStatus_t mcspCuinScsric02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                     const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                     mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspCuinDcsric02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                     const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                     mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspCuinCcsric02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                     const mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                     mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspCuinZcsric02_analysis(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                                     const mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                     mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspCuinXcsric02_zeroPivot(mcspHandle_t handle, mcspCsric02Info_t info, int* position);

mcspStatus_t mcspCuinScsric02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, float* csr_vals,
                            const int* csr_rows, const int* csr_cols, mcspCsric02Info_t info,
                            mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspCuinDcsric02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, double* csr_vals,
                            const int* csr_rows, const int* csr_cols, mcspCsric02Info_t info,
                            mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspCuinCcsric02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, mcspComplexFloat* csr_vals,
                            const int* csr_rows, const int* csr_cols, mcspCsric02Info_t info,
                            mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcspStatus_t mcspCuinZcsric02(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                            mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                            mcspCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

/**
 * @brief   Calculates the required buffer size for gtsv API.
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of tridiagonal system
 * @param n               [in]        number of cols of tridiagonal system
 * @param dl              [in]        pointer to the lower diagonal of tridiagonal system
 * @param d               [in]        pointer to the main diagonal of tridiagonal system
 * @param du              [in]        pointer to the upper diagonal of tridiagonal system
 * @param B               [in]    pointer to the dense matrix
 * @param ldb             [in]        leading dimension of B
 * @param buffer_size     [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgtsvBuffersize(mcspHandle_t handle, mcspInt m, mcspInt n, const float* dl, const float* d,
                                 const float* du, const float* B, mcspInt ldb, size_t* buffer_size);

mcspStatus_t mcspDgtsvBuffersize(mcspHandle_t handle, mcspInt m, mcspInt n, const double* dl, const double* d,
                                 const double* du, const double* B, mcspInt ldb, size_t* buffer_size);

mcspStatus_t mcspCgtsvBuffersize(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* dl,
                                 const mcspComplexFloat* d, const mcspComplexFloat* du, const mcspComplexFloat* B,
                                 mcspInt ldb, size_t* buffer_size);

mcspStatus_t mcspZgtsvBuffersize(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* dl,
                                 const mcspComplexDouble* d, const mcspComplexDouble* du, const mcspComplexDouble* B,
                                 mcspInt ldb, size_t* buffer_size);

/**
 * @brief   Solve a tridiagonal system for multiple right hand sides with pivoting.
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of tridiagonal system
 * @param n               [in]        number of cols of tridiagonal system
 * @param dl              [in]        pointer to the lower diagonal of tridiagonal system
 * @param d               [in]        pointer to the main diagonal of tridiagonal system
 * @param du              [in]        pointer to the upper diagonal of tridiagonal system
 * @param B               [in/out]    pointer to the dense matrix
 * @param ldb             [in]        leading dimension of B
 * @param buffer          [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgtsv(mcspHandle_t handle, mcspInt m, mcspInt n, const float* dl, const float* d, const float* du,
                       float* B, mcspInt ldb, void* buffer);

mcspStatus_t mcspDgtsv(mcspHandle_t handle, mcspInt m, mcspInt n, const double* dl, const double* d, const double* du,
                       double* B, mcspInt ldb, void* buffer);

mcspStatus_t mcspCgtsv(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* dl, const mcspComplexFloat* d,
                       const mcspComplexFloat* du, mcspComplexFloat* B, mcspInt ldb, void* buffer);

mcspStatus_t mcspZgtsv(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* dl,
                       const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B, mcspInt ldb,
                       void* buffer);

mcspStatus_t mcspCuinSgtsv2_bufferSizeExt(mcspHandle_t handle, int m, int n, const float* dl, const float* d,
                                        const float* du, const float* B, int ldb, size_t* bufferSizeInBytes);

mcspStatus_t mcspCuinDgtsv2_bufferSizeExt(mcspHandle_t handle, int m, int n, const double* dl, const double* d,
                                        const double* du, const double* B, int ldb, size_t* bufferSizeInBytes);

mcspStatus_t mcspCuinCgtsv2_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcspComplexFloat* dl,
                                        const mcspComplexFloat* d, const mcspComplexFloat* du,
                                        const mcspComplexFloat* B, int ldb, size_t* bufferSizeInBytes);

mcspStatus_t mcspCuinZgtsv2_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcspComplexDouble* dl,
                                        const mcspComplexDouble* d, const mcspComplexDouble* du,
                                        const mcspComplexDouble* B, int ldb, size_t* bufferSizeInBytes);

mcspStatus_t mcspCuinSgtsv2(mcspHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B,
                          int ldb, void* pBuffer);

mcspStatus_t mcspCuinDgtsv2(mcspHandle_t handle, int m, int n, const double* dl, const double* d, const double* du,
                          double* B, int ldb, void* pBuffer);

mcspStatus_t mcspCuinCgtsv2(mcspHandle_t handle, int m, int n, const mcspComplexFloat* dl, const mcspComplexFloat* d,
                          const mcspComplexFloat* du, mcspComplexFloat* B, int ldb, void* pBuffer);

mcspStatus_t mcspCuinZgtsv2(mcspHandle_t handle, int m, int n, const mcspComplexDouble* dl, const mcspComplexDouble* d,
                          const mcspComplexDouble* du, mcspComplexDouble* B, int ldb, void* pBuffer);

mcspStatus_t mcspSgtsv10x(mcspHandle_t handle, mcspInt m, mcspInt n, const float* dl, const float* d, const float* du,
                          float* B, mcspInt ldb);

mcspStatus_t mcspDgtsv10x(mcspHandle_t handle, mcspInt m, mcspInt n, const double* dl, const double* d,
                          const double* du, double* B, mcspInt ldb);

mcspStatus_t mcspCgtsv10x(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* dl,
                          const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* B, mcspInt ldb);

mcspStatus_t mcspZgtsv10x(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* dl,
                          const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B, mcspInt ldb);

/**
 * @brief   Calculates the required buffer size for gtsv no pivot API.
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of tridiagonal system
 * @param n               [in]        number of cols of tridiagonal system
 * @param dl              [in]        pointer to the lower diagonal of tridiagonal system.
 * @param d               [in]        pointer to the main diagonal of tridiagonal system.
 * @param du              [in]        pointer to the upper diagonal of tridiagonal system.
 * @param B               [in]        pointer to the dense matrix
 * @param ldb             [in]        leading dimension of B
 * @param buffer_size     [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgtsvNopivotBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const float* dl, const float* d,
                                        const float* du, const float* B, mcspInt ldb, size_t* buffer_size);

mcspStatus_t mcspDgtsvNopivotBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const double* dl, const double* d,
                                        const double* du, const double* B, mcspInt ldb, size_t* buffer_size);

mcspStatus_t mcspCgtsvNopivotBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* dl,
                                        const mcspComplexFloat* d, const mcspComplexFloat* du,
                                        const mcspComplexFloat* B, mcspInt ldb, size_t* buffer_size);

mcspStatus_t mcspZgtsvNopivotBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* dl,
                                        const mcspComplexDouble* d, const mcspComplexDouble* du,
                                        const mcspComplexDouble* B, mcspInt ldb, size_t* buffer_size);

/**
 * @brief   Solve a tridiagonal system for multiple right hand sides with no pivot.
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of tridiagonal system
 * @param n               [in]        number of cols of tridiagonal system
 * @param dl              [in]        pointer to the lower diagonal of tridiagonal system.
 * @param d               [in]        pointer to the main diagonal of tridiagonal system.
 * @param du              [in]        pointer to the upper diagonal of tridiagonal system.
 * @param B               [in/out]    pointer to the dense matrix
 * @param ldb             [in]        leading dimension of B
 * @param temp_buffer     [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgtsvNopivot(mcspHandle_t handle, mcspInt m, mcspInt n, const float* dl, const float* d,
                              const float* du, float* B, mcspInt ldb, void* temp_buffer);

mcspStatus_t mcspDgtsvNopivot(mcspHandle_t handle, mcspInt m, mcspInt n, const double* dl, const double* d,
                              const double* du, double* B, mcspInt ldb, void* temp_buffer);

mcspStatus_t mcspCgtsvNopivot(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* dl,
                              const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* B, mcspInt ldb,
                              void* temp_buffer);

mcspStatus_t mcspZgtsvNopivot(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* dl,
                              const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B,
                              mcspInt ldb, void* temp_buffer);
mcspStatus_t mcspCuinSgtsv2_nopivot_bufferSizeExt(mcspHandle_t handle, int m, int n, const float* dl, const float* d,
                                                const float* du, const float* B, int ldb, size_t* buffer_size);

mcspStatus_t mcspCuinDgtsv2_nopivot_bufferSizeExt(mcspHandle_t handle, int m, int n, const double* dl, const double* d,
                                                const double* du, const double* B, int ldb, size_t* buffer_size);

mcspStatus_t mcspCuinCgtsv2_nopivot_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcspComplexFloat* dl,
                                                const mcspComplexFloat* d, const mcspComplexFloat* du,
                                                const mcspComplexFloat* B, int ldb, size_t* buffer_size);

mcspStatus_t mcspCuinZgtsv2_nopivot_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcspComplexDouble* dl,
                                                const mcspComplexDouble* d, const mcspComplexDouble* du,
                                                const mcspComplexDouble* B, int ldb, size_t* buffer_size);

mcspStatus_t mcspCuinSgtsv2_nopivot(mcspHandle_t handle, int m, int n, const float* dl, const float* d, const float* du,
                                  float* B, int ldb, void* temp_buffer);

mcspStatus_t mcspCuinDgtsv2_nopivot(mcspHandle_t handle, int m, int n, const double* dl, const double* d,
                                  const double* du, double* B, int ldb, void* temp_buffer);

mcspStatus_t mcspCuinCgtsv2_nopivot(mcspHandle_t handle, int m, int n, const mcspComplexFloat* dl,
                                  const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* B, int ldb,
                                  void* temp_buffer);

mcspStatus_t mcspCuinZgtsv2_nopivot(mcspHandle_t handle, int m, int n, const mcspComplexDouble* dl,
                                  const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B,
                                  int ldb, void* temp_buffer);

mcspStatus_t mcspSgtsv_nopivot10x(mcspHandle_t handle, mcspInt m, mcspInt n, const float* dl, const float* d,
                                  const float* du, float* B, mcspInt ldb);

mcspStatus_t mcspDgtsv_nopivot10x(mcspHandle_t handle, mcspInt m, mcspInt n, const double* dl, const double* d,
                                  const double* du, double* B, mcspInt ldb);

mcspStatus_t mcspCgtsv_nopivot10x(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* dl,
                                  const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* B,
                                  mcspInt ldb);

mcspStatus_t mcspZgtsv_nopivot10x(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* dl,
                                  const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B,
                                  mcspInt ldb);

/**
 * @brief   Calculates the required buffer size for gtsv strided batch API.
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of tridiagonal systems
 * @param dl              [in]        pointer to the lower diagonals of tridiagonal systems
 * @param d               [in]        pointer to the main diagonals of tridiagonal systems
 * @param du              [in]        pointer to the upper diagonals of tridiagonal systems
 * @param x               [in]    pointer to the dense matrices
 * @param batch_count     [in]        number of systems to solve
 * @param batch_stride    [in]        stride that separates the vectors of each system
 * @param buffer_size     [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgtsvStridedBatchBuffersize(mcspHandle_t handle, mcspInt m, const float* dl, const float* d,
                                             const float* du, const float* x, mcspInt batch_count, mcspInt batch_stride,
                                             size_t* buffer_size);

mcspStatus_t mcspDgtsvStridedBatchBuffersize(mcspHandle_t handle, mcspInt m, const double* dl, const double* d,
                                             const double* du, const double* x, mcspInt batch_count,
                                             mcspInt batch_stride, size_t* buffer_size);

mcspStatus_t mcspCgtsvStridedBatchBuffersize(mcspHandle_t handle, mcspInt m, const mcspComplexFloat* dl,
                                             const mcspComplexFloat* d, const mcspComplexFloat* du,
                                             const mcspComplexFloat* x, mcspInt batch_count, mcspInt batch_stride,
                                             size_t* buffer_size);

mcspStatus_t mcspZgtsvStridedBatchBuffersize(mcspHandle_t handle, mcspInt m, const mcspComplexDouble* dl,
                                             const mcspComplexDouble* d, const mcspComplexDouble* du,
                                             const mcspComplexDouble* x, mcspInt batch_count, mcspInt batch_stride,
                                             size_t* buffer_size);

/**
 * @brief   Solve multiple tridiagonal systems Ai * yi = xi with pivoting.
 *
 * @param handle          [in]        handle of mcsp library
 * @param m               [in]        number of rows of tridiagonal systems
 * @param dl              [in]        pointer to the lower diagonals of tridiagonal systems
 * @param d               [in]        pointer to the main diagonals of tridiagonal systems
 * @param du              [in]        pointer to the upper diagonals of tridiagonal systems
 * @param x               [in/out]    pointer to the dense matrices
 * @param batch_count     [in]        number of systems to solve
 * @param batch_stride    [in]        stride that separates the vectors of each system
 * @param buffer          [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgtsvStridedBatch(mcspHandle_t handle, mcspInt m, const float* dl, const float* d, const float* du,
                                   float* x, mcspInt batch_count, mcspInt batch_stride, void* buffer);

mcspStatus_t mcspDgtsvStridedBatch(mcspHandle_t handle, mcspInt m, const double* dl, const double* d, const double* du,
                                   double* x, mcspInt batch_count, mcspInt batch_stride, void* buffer);

mcspStatus_t mcspCgtsvStridedBatch(mcspHandle_t handle, mcspInt m, const mcspComplexFloat* dl,
                                   const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* x,
                                   mcspInt batch_count, mcspInt batch_stride, void* buffer);

mcspStatus_t mcspZgtsvStridedBatch(mcspHandle_t handle, mcspInt m, const mcspComplexDouble* dl,
                                   const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* x,
                                   mcspInt batch_count, mcspInt batch_stride, void* buffer);

mcspStatus_t mcspCuinSgtsv2StridedBatch_bufferSizeExt(mcspHandle_t handle, int m, const float* dl, const float* d,
                                                    const float* du, const float* x, int batchCount, int batchStride,
                                                    size_t* bufferSizeInBytes);

mcspStatus_t mcspCuinDgtsv2StridedBatch_bufferSizeExt(mcspHandle_t handle, int m, const double* dl, const double* d,
                                                    const double* du, const double* x, int batchCount, int batchStride,
                                                    size_t* bufferSizeInBytes);

mcspStatus_t mcspCuinCgtsv2StridedBatch_bufferSizeExt(mcspHandle_t handle, int m, const mcspComplexFloat* dl,
                                                    const mcspComplexFloat* d, const mcspComplexFloat* du,
                                                    const mcspComplexFloat* x, int batchCount, int batchStride,
                                                    size_t* bufferSizeInBytes);

mcspStatus_t mcspCuinZgtsv2StridedBatch_bufferSizeExt(mcspHandle_t handle, int m, const mcspComplexDouble* dl,
                                                    const mcspComplexDouble* d, const mcspComplexDouble* du,
                                                    const mcspComplexDouble* x, int batchCount, int batchStride,
                                                    size_t* bufferSizeInBytes);

mcspStatus_t mcspCuinSgtsv2StridedBatch(mcspHandle_t handle, int m, const float* dl, const float* d, const float* du,
                                      float* x, int batchCount, int batchStride, void* pBuffer);

mcspStatus_t mcspCuinDgtsv2StridedBatch(mcspHandle_t handle, int m, const double* dl, const double* d, const double* du,
                                      double* x, int batchCount, int batchStride, void* pBuffer);

mcspStatus_t mcspCuinCgtsv2StridedBatch(mcspHandle_t handle, int m, const mcspComplexFloat* dl, const mcspComplexFloat* d,
                                      const mcspComplexFloat* du, mcspComplexFloat* x, int batchCount, int batchStride,
                                      void* pBuffer);

mcspStatus_t mcspCuinZgtsv2StridedBatch(mcspHandle_t handle, int m, const mcspComplexDouble* dl,
                                      const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* x,
                                      int batchCount, int batchStride, void* pBuffer);

mcspStatus_t mcspSgtsvStridedBatch10x(mcspHandle_t handle, mcspInt m, const float* dl, const float* d, const float* du,
                                      float* x, mcspInt batch_count, mcspInt batch_stride);

mcspStatus_t mcspDgtsvStridedBatch10x(mcspHandle_t handle, mcspInt m, const double* dl, const double* d,
                                      const double* du, double* x, mcspInt batch_count, mcspInt batch_stride);

mcspStatus_t mcspCgtsvStridedBatch10x(mcspHandle_t handle, mcspInt m, const mcspComplexFloat* dl,
                                      const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* x,
                                      mcspInt batch_count, mcspInt batch_stride);

mcspStatus_t mcspZgtsvStridedBatch10x(mcspHandle_t handle, mcspInt m, const mcspComplexDouble* dl,
                                      const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* x,
                                      mcspInt batch_count, mcspInt batch_stride);

/**
 * @brief   Solves a batch of tridiagonal linear systems.
 *
 * @param handle            [in]        handle of mcsp library
 * @param alg               [in]        algorithm to solve the linear system
 * @param row_num           [in]        size of the pentadiagonal linear system
 * @param dl                [in]        lower diagonal of pentadiagonal system
 * @param d                 [in]        main diagonal of pentadiagonal system
 * @param du                [in]        upper diagonal of pentadiagonal system
 * @param x                 [in/out]    dense array of right-hand-sides with dimension
 * @param batch_count       [in]        number of systems to solve
 * @param batch_stride      [in]        number of elements that separate consecutive elements in a system
 * @param buffer_size       [out]       number of bytes of the temporary storage buffer required
 * @param temp_buffer       [in]        number of bytes of the temporary storage buffer required
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgtsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                                 const float* dl, const float* d, const float* du, const float* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size);

mcspStatus_t mcspDgtsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                                 const double* dl, const double* d, const double* du, const double* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size);

mcspStatus_t mcspCgtsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                                 const mcspComplexFloat* dl, const mcspComplexFloat* d,
                                                 const mcspComplexFloat* du, const mcspComplexFloat* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size);

mcspStatus_t mcspZgtsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                                 const mcspComplexDouble* dl, const mcspComplexDouble* d,
                                                 const mcspComplexDouble* du, const mcspComplexDouble* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size);

mcspStatus_t mcspSgtsvInterleavedBatch(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                       float* dl, float* d, float* du, float* x, mcspInt batch_count,
                                       mcspInt batch_stride, void* temp_buffer);

mcspStatus_t mcspDgtsvInterleavedBatch(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                       double* dl, double* d, double* du, double* x, mcspInt batch_count,
                                       mcspInt batch_stride, void* temp_buffer);

mcspStatus_t mcspCgtsvInterleavedBatch(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                       mcspComplexFloat* dl, mcspComplexFloat* d, mcspComplexFloat* du,
                                       mcspComplexFloat* x, mcspInt batch_count, mcspInt batch_stride,
                                       void* temp_buffer);

mcspStatus_t mcspZgtsvInterleavedBatch(mcspHandle_t handle, mcsparseGtsvInterleavedAlg_t alg, mcspInt row_num,
                                       mcspComplexDouble* dl, mcspComplexDouble* d, mcspComplexDouble* du,
                                       mcspComplexDouble* x, mcspInt batch_count, mcspInt batch_stride,
                                       void* temp_buffer);

mcspStatus_t mcspCuinSgtsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, const float* dl,
                                                       const float* d, const float* du, const float* x, int batch_count,
                                                       size_t* buffer_size);

mcspStatus_t mcspCuinDgtsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, const double* dl,
                                                       const double* d, const double* du, const double* x,
                                                       int batch_count, size_t* buffer_size);

mcspStatus_t mcspCuinCgtsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num,
                                                       const mcspComplexFloat* dl, const mcspComplexFloat* d,
                                                       const mcspComplexFloat* du, const mcspComplexFloat* x,
                                                       int batch_count, size_t* buffer_size);

mcspStatus_t mcspCuinZgtsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num,
                                                       const mcspComplexDouble* dl, const mcspComplexDouble* d,
                                                       const mcspComplexDouble* du, const mcspComplexDouble* x,
                                                       int batch_count, size_t* buffer_size);

mcspStatus_t mcspCuinSgtsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, float* dl, float* d, float* du,
                                         float* x, int batch_count, void* temp_buffer);

mcspStatus_t mcspCuinDgtsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, double* dl, double* d, double* du,
                                         double* x, int batch_count, void* temp_buffer);

mcspStatus_t mcspCuinCgtsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, mcspComplexFloat* dl,
                                         mcspComplexFloat* d, mcspComplexFloat* du, mcspComplexFloat* x,
                                         int batch_count, void* temp_buffer);

mcspStatus_t mcspCuinZgtsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, mcspComplexDouble* dl,
                                         mcspComplexDouble* d, mcspComplexDouble* du, mcspComplexDouble* x,
                                         int batch_count, void* temp_buffer);

/**
 * @brief   Calculates the required buffer size for gpsv interleaved batch API.
 *
 * @param handle            [in]        handle of mcsp library
 * @param alg               [in]        algorithm to solve the linear system
 * @param row_num           [in]        size of the pentadiagonal linear system
 * @param ds                [in]        lower diagonal (distance 2) of pentadiagonal system
 * @param dl                [in]        lower diagonal of pentadiagonal system
 * @param d                 [in]        main diagonal of pentadiagonal system
 * @param du                [in]        upper diagonal of pentadiagonal system
 * @param dw                [in]        upper diagonal (distance 2) of pentadiagonal system
 * @param x                 [in]        dense array of right-hand-sides with dimension
 * @param batch_count       [in]        number of systems to solve
 * @param batch_stride      [in]        number of elements that separate consecutive elements in a system
 * @param buffer_size       [out]       number of bytes of the temporary storage buffer required
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgpsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                                 float* ds, float* dl, float* d, float* du, float* dw, float* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size);

mcspStatus_t mcspDgpsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                                 double* ds, double* dl, double* d, double* du, double* dw, double* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size);

mcspStatus_t mcspCgpsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                                 mcspComplexFloat* ds, mcspComplexFloat* dl, mcspComplexFloat* d,
                                                 mcspComplexFloat* du, mcspComplexFloat* dw, mcspComplexFloat* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size);

mcspStatus_t mcspZgpsvInterleavedBatchBuffersize(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                                 mcspComplexDouble* ds, mcspComplexDouble* dl, mcspComplexDouble* d,
                                                 mcspComplexDouble* du, mcspComplexDouble* dw, mcspComplexDouble* x,
                                                 mcspInt batch_count, mcspInt batch_stride, size_t* buffer_size);

/**
 * @brief   Solves a batch of pentadiagonal linear systems.
 *
 * @param handle            [in]        handle of mcsp library
 * @param alg               [in]        algorithm to solve the linear system
 * @param row_num           [in]        size of the pentadiagonal linear system
 * @param ds                [in]        lower diagonal (distance 2) of pentadiagonal system
 * @param dl                [in]        lower diagonal of pentadiagonal system
 * @param d                 [in]        main diagonal of pentadiagonal system
 * @param du                [in]        upper diagonal of pentadiagonal system
 * @param dw                [in]        upper diagonal (distance 2) of pentadiagonal system
 * @param x                 [in/out]    dense array of right-hand-sides with dimension
 * @param batch_count       [in]        number of systems to solve
 * @param batch_stride      [in]        number of elements that separate consecutive elements in a system
 * @param temp_buffer       [in]        number of bytes of the temporary storage buffer required
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgpsvInterleavedBatch(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                       float* ds, float* dl, float* d, float* du, float* dw, float* x,
                                       mcspInt batch_count, mcspInt batch_stride, void* temp_buffer);

mcspStatus_t mcspDgpsvInterleavedBatch(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                       double* ds, double* dl, double* d, double* du, double* dw, double* x,
                                       mcspInt batch_count, mcspInt batch_stride, void* temp_buffer);

mcspStatus_t mcspCgpsvInterleavedBatch(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                       mcspComplexFloat* ds, mcspComplexFloat* dl, mcspComplexFloat* d,
                                       mcspComplexFloat* du, mcspComplexFloat* dw, mcspComplexFloat* x,
                                       mcspInt batch_count, mcspInt batch_stride, void* temp_buffer);

mcspStatus_t mcspZgpsvInterleavedBatch(mcspHandle_t handle, mcsparseGpsvInterleavedAlg_t alg, mcspInt row_num,
                                       mcspComplexDouble* ds, mcspComplexDouble* dl, mcspComplexDouble* d,
                                       mcspComplexDouble* du, mcspComplexDouble* dw, mcspComplexDouble* x,
                                       mcspInt batch_count, mcspInt batch_stride, void* temp_buffer);

mcspStatus_t mcspCuinSgpsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, float* ds, float* dl,
                                                       float* d, float* du, float* dw, float* x, int batch_count,
                                                       size_t* buffer_size);

mcspStatus_t mcspCuinDgpsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, double* ds,
                                                       double* dl, double* d, double* du, double* dw, double* x,
                                                       int batch_count, size_t* buffer_size);

mcspStatus_t mcspCuinCgpsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, mcspComplexFloat* ds,
                                                       mcspComplexFloat* dl, mcspComplexFloat* d, mcspComplexFloat* du,
                                                       mcspComplexFloat* dw, mcspComplexFloat* x, int batch_count,
                                                       size_t* buffer_size);

mcspStatus_t mcspCuinZgpsvInterleavedBatch_bufferSizeExt(mcspHandle_t handle, int alg, int row_num, mcspComplexDouble* ds,
                                                       mcspComplexDouble* dl, mcspComplexDouble* d,
                                                       mcspComplexDouble* du, mcspComplexDouble* dw,
                                                       mcspComplexDouble* x, int batch_count, size_t* buffer_size);

mcspStatus_t mcspCuinSgpsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, float* ds, float* dl, float* d,
                                         float* du, float* dw, float* x, int batch_count, void* temp_buffer);

mcspStatus_t mcspCuinDgpsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, double* ds, double* dl, double* d,
                                         double* du, double* dw, double* x, int batch_count, void* temp_buffer);

mcspStatus_t mcspCuinCgpsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, mcspComplexFloat* ds,
                                         mcspComplexFloat* dl, mcspComplexFloat* d, mcspComplexFloat* du,
                                         mcspComplexFloat* dw, mcspComplexFloat* x, int batch_count, void* temp_buffer);

mcspStatus_t mcspCuinZgpsvInterleavedBatch(mcspHandle_t handle, int alg, int row_num, mcspComplexDouble* ds,
                                         mcspComplexDouble* dl, mcspComplexDouble* d, mcspComplexDouble* du,
                                         mcspComplexDouble* dw, mcspComplexDouble* x, int batch_count,
                                         void* temp_buffer);

// bsrilu0
mcspStatus_t mcspSbsrilu02_numericBoost(mcspHandle_t handle, mcspBsrilu02Info_t info, mcspInt enable_boost, double* tol,
                                        float* boost_val);

mcspStatus_t mcspDbsrilu02_numericBoost(mcspHandle_t handle, mcspBsrilu02Info_t info, mcspInt enable_boost, double* tol,
                                        double* boost_val);

mcspStatus_t mcspCbsrilu02_numericBoost(mcspHandle_t handle, mcspBsrilu02Info_t info, mcspInt enable_boost, double* tol,
                                        mcFloatComplex* boost_val);

mcspStatus_t mcspZbsrilu02_numericBoost(mcspHandle_t handle, mcspBsrilu02Info_t info, mcspInt enable_boost, double* tol,
                                        mcDoubleComplex* boost_val);

mcspStatus_t mcspXbsrilu02_zeroPivot(mcspHandle_t handle, mcspBsrilu02Info_t info, int* position);

mcspStatus_t mcspSbsrilu02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                      const mcspMatDescr_t descrA, float* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                      const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                                      int* pBufferSizeInBytes);

mcspStatus_t mcspDbsrilu02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                      const mcspMatDescr_t descrA, double* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                      const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                                      int* pBufferSizeInBytes);

mcspStatus_t mcspCbsrilu02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                      const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                      const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockDim,
                                      mcspBsrilu02Info_t info, int* pBufferSizeInBytes);

mcspStatus_t mcspZbsrilu02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                      const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                      const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockDim,
                                      mcspBsrilu02Info_t info, int* pBufferSizeInBytes);

mcspStatus_t mcspSbsrilu02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                         const mcspMatDescr_t descrA, float* bsrSortedVal,
                                         const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                         mcspInt blockSize, mcspBsrilu02Info_t info, size_t* pBufferSize);

mcspStatus_t mcspDbsrilu02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                         const mcspMatDescr_t descrA, double* bsrSortedVal,
                                         const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                         mcspInt blockSize, mcspBsrilu02Info_t info, size_t* pBufferSize);

mcspStatus_t mcspCbsrilu02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                         const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                         const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                         mcspInt blockSize, mcspBsrilu02Info_t info, size_t* pBufferSize);

mcspStatus_t mcspZbsrilu02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                         const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                         const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd,
                                         mcspInt blockSize, mcspBsrilu02Info_t info, size_t* pBufferSize);

mcspStatus_t mcspSbsrilu02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, float* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                    const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                                    mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspDbsrilu02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, double* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                                    const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                                    mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspCbsrilu02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                    const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockDim,
                                    mcspBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspZbsrilu02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                    const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                    const mcspInt* bsrSortedRowPtr, const mcspInt* bsrSortedColInd, mcspInt blockDim,
                                    mcspBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspSbsrilu02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                           const mcspMatDescr_t descrA, float* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                           const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                           mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspDbsrilu02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                           const mcspMatDescr_t descrA, double* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                           const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                           mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspCbsrilu02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                           const mcspMatDescr_t descrA, mcFloatComplex* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                           const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                           mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspZbsrilu02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                           const mcspMatDescr_t descrA, mcDoubleComplex* bsrSortedVal, const mcspInt* bsrSortedRowPtr,
                           const mcspInt* bsrSortedColInd, mcspInt blockDim, mcspBsrilu02Info_t info,
                           mcsparseSolvePolicy_t policy, void* pBuffer);

// bsric0
mcspStatus_t mcspXbsric02_zeroPivot(mcspHandle_t handle, mcspBsric02Info_t info, int* position);

mcspStatus_t mcspSbsric02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                     const mcspMatDescr_t descrA, float* bsr_vals, const mcspInt* bsr_rows,
                                     const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                     int* buffer_size);

mcspStatus_t mcspDbsric02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                     const mcspMatDescr_t descrA, double* bsr_vals, const mcspInt* bsr_rows,
                                     const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                     int* buffer_size);

mcspStatus_t mcspCbsric02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                     const mcspMatDescr_t descrA, mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                                     const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                     int* buffer_size);

mcspStatus_t mcspZbsric02_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                     const mcspMatDescr_t descrA, mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows,
                                     const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                     int* buffer_size);

mcspStatus_t mcspSbsric02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                        const mcspMatDescr_t descrA, float* bsr_vals, const mcspInt* bsr_rows,
                                        const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                        size_t* buffer_size);

mcspStatus_t mcspDbsric02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                        const mcspMatDescr_t descrA, double* bsr_vals, const mcspInt* bsr_rows,
                                        const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                        size_t* buffer_size);

mcspStatus_t mcspCbsric02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                        const mcspMatDescr_t descrA, mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                                        const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                        size_t* buffer_size);

mcspStatus_t mcspZbsric02_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                        const mcspMatDescr_t descrA, mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows,
                                        const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                        size_t* buffer_size);

mcspStatus_t mcspSbsric02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                   const mcspMatDescr_t descrA, const float* bsr_vals, const mcspInt* bsr_rows,
                                   const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* buffer);

mcspStatus_t mcspDbsric02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                   const mcspMatDescr_t descrA, const double* bsr_vals, const mcspInt* bsr_rows,
                                   const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* buffer);

mcspStatus_t mcspCbsric02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                   const mcspMatDescr_t descrA, const mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                                   const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* buffer);

mcspStatus_t mcspZbsric02_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                                   const mcspMatDescr_t descrA, const mcDoubleComplex* bsr_vals,
                                   const mcspInt* bsr_rows, const mcspInt* bsr_cols, mcspInt block_dim,
                                   mcspBsric02Info_t info, mcsparseSolvePolicy_t policy, void* buffer);

mcspStatus_t mcspSbsric02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                          const mcspMatDescr_t descrA, float* bsr_vals, const mcspInt* bsr_rows,
                          const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                          mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspDbsric02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                          const mcspMatDescr_t descrA, double* bsr_vals, const mcspInt* bsr_rows,
                          const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                          mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspCbsric02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                          const mcspMatDescr_t descrA, mcFloatComplex* bsr_vals, const mcspInt* bsr_rows,
                          const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                          mcsparseSolvePolicy_t policy, void* pBuffer);

mcspStatus_t mcspZbsric02(mcspHandle_t handle, mcsparseDirection_t dirA, mcspInt mb, mcspInt nnzb,
                          const mcspMatDescr_t descrA, mcDoubleComplex* bsr_vals, const mcspInt* bsr_rows,
                          const mcspInt* bsr_cols, mcspInt block_dim, mcspBsric02Info_t info,
                          mcsparseSolvePolicy_t policy, void* pBuffer);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSP_PRECOND_H_
