#ifndef INTERFACE_MCSP_INTERNAL_LEVEL2_H_
#define INTERFACE_MCSP_INTERNAL_LEVEL2_H_

#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief   compute COO-based SpMV in GPU
 *          GPU COO spmv needs A in row-major order
 *          y = alpha * A * x + beta * y
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans         [in]        matrix operation type
 * @param row_num       [in]        number of rows
 * @param col_num       [in]        number of columns
 * @param nnz           [in]        number of nonzeros
 * @param alpha         [in]        alpha
 * @param descr         [in]        descriptor of the sparse matrix
 * @param coo_vals      [in]        pointer to the values of nonzeros in COO
 * @param coo_rows      [in]        pointer to the row indexes of nonzeros in COO
 * @param coo_cols      [in]        pointer to the column indexes of nonzeros in COO
 * @param vec_x         [in]        pointer to the vector x
 * @param beta          [in]        beta
 * @param vec_y         [in/out]    pointer to the vector b
 * @return mcspStatus_t
 */
mcspStatus_t mcspScooSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const float *alpha, const mcspMatDescr_t descr, const float *coo_vals,
                          const mcspInt *coo_rows, const mcspInt *coo_cols, const float *vec_x, const float *beta,
                          float *vec_y);
mcspStatus_t mcspDcooSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const double *alpha, const mcspMatDescr_t descr, const double *coo_vals,
                          const mcspInt *coo_rows, const mcspInt *coo_cols, const double *vec_x, const double *beta,
                          double *vec_y);
mcspStatus_t mcspCcooSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const mcspComplexFloat *alpha, const mcspMatDescr_t descr, const mcspComplexFloat *coo_vals,
                          const mcspInt *coo_rows, const mcspInt *coo_cols, const mcspComplexFloat *vec_x,
                          const mcspComplexFloat *beta, mcspComplexFloat *vec_y);
mcspStatus_t mcspZcooSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const mcspComplexDouble *alpha, const mcspMatDescr_t descr, const mcspComplexDouble *coo_vals,
                          const mcspInt *coo_rows, const mcspInt *coo_cols, const mcspComplexDouble *vec_x,
                          const mcspComplexDouble *beta, mcspComplexDouble *vec_y);
/**
 * @brief performs the analysis step for csrmv. If the matrix sparsity pattern changes, the gathered
 *        information will become invalid.
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans         [in]        matrix operation type
 * @param row_num       [in]        number of rows
 * @param col_num       [in]        number of columns
 * @param nnz           [in]        number of nonzeros
 * @param descr         [in]        descriptor of the sparse matrix
 * @param csr_vals      [in]        pointer to the values of nonzeros in CSR
 * @param csr_rows      [in]        pointer to the row offset in CSR
 * @param csr_cols      [in]        pointer to the column indexes of nonzeros in CSR
 * @param mat_info      [out]       meta data for CSR matrix
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrSpmvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                                  mcspInt nnz, const mcspMatDescr_t descr, const float *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t mat_info);
mcspStatus_t mcspDcsrSpmvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                                  mcspInt nnz, const mcspMatDescr_t descr, const double *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t mat_info);
mcspStatus_t mcspCcsrSpmvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                                  mcspInt nnz, const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t mat_info);
mcspStatus_t mcspZcsrSpmvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                                  mcspInt nnz, const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t mat_info);

/**
 * @brief clear the analysis information for csrmv. Clearing is optional. All allocated resources
 *        will be cleared when mcspMatInfo struct is destroyed using mcspDestroyMatInfo.
 *
 * @param handle            [in]        handle of mcsp library
 * @param mat_info          [out]       meta data for CSR matrix D
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrSpmvAnalysisClear(mcspHandle_t handle, mcspMatInfo_t mat_info);

/**
 * @brief   compute CSR-based SpMV in GPU
 *          y = alpha * A * x + beta * y
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans         [in]        matrix operation type
 * @param row_num       [in]        number of rows
 * @param col_num       [in]        number of columns
 * @param nnz           [in]        number of nonzeros
 * @param alpha         [in]        alpha
 * @param descr         [in]        descriptor of the sparse matrix
 * @param csr_vals      [in]        pointer to the values of nonzeros in CSR
 * @param csr_rows      [in]        pointer to the row offset in CSR
 * @param csr_cols      [in]        pointer to the column indexes of nonzeros in CSR
 * @param mat_info      [in]        meta data for CSR matrix
 * @param vec_x         [in]        pointer to the vector x
 * @param beta          [in]        beta
 * @param vec_y         [in/out]    pointer to the vector
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const float *alpha, const mcspMatDescr_t descr, const float *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t mat_info, const float *vec_x,
                          const float *beta, float *vec_y);
mcspStatus_t mcspDcsrSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const double *alpha, const mcspMatDescr_t descr, const double *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t mat_info, const double *vec_x,
                          const double *beta, double *vec_y);
mcspStatus_t mcspCcsrSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const mcspComplexFloat *alpha, const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t mat_info,
                          const mcspComplexFloat *vec_x, const mcspComplexFloat *beta, mcspComplexFloat *vec_y);
mcspStatus_t mcspZcsrSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num, mcspInt nnz,
                          const mcspComplexDouble *alpha, const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t mat_info,
                          const mcspComplexDouble *vec_x, const mcspComplexDouble *beta, mcspComplexDouble *vec_y);

/**
 * @brief   compute ELL-based SpMV in GPU
 *          y = alpha * A * x + beta * y
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans         [in]        matrix operation type
 * @param row_num       [in]        number of rows
 * @param col_num       [in]        number of columns
 * @param alpha         [in]        alpha
 * @param descr         [in]        descriptor of the sparse matrix
 * @param ell_vals      [in]        pointer to the values of element in ELL
 * @param ell_cols      [in]        pointer to the column indexes of element in ELL
 * @param ell_k         [in]        threshold value of ELL matrix
 * @param vec_x         [in]        pointer to the vector x
 * @param beta          [in]        beta
 * @param vec_y         [in/out]    pointer to the vector y
 * @return mcspStatus_t
 */
mcspStatus_t mcspSellSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                          const float *alpha, const mcspMatDescr_t descr, const float *ell_vals,
                          const mcspInt *ell_cols, mcspInt ell_k, const float *vec_x, const float *beta, float *vec_y);
mcspStatus_t mcspDellSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                          const double *alpha, const mcspMatDescr_t descr, const double *ell_vals,
                          const mcspInt *ell_cols, mcspInt ell_k, const double *vec_x, const double *beta,
                          double *vec_y);
mcspStatus_t mcspCellSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                          const mcspComplexFloat *alpha, const mcspMatDescr_t descr, const mcspComplexFloat *ell_vals,
                          const mcspInt *ell_cols, mcspInt ell_k, const mcspComplexFloat *vec_x,
                          const mcspComplexFloat *beta, mcspComplexFloat *vec_y);
mcspStatus_t mcspZellSpmv(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt col_num,
                          const mcspComplexDouble *alpha, const mcspMatDescr_t descr, const mcspComplexDouble *ell_vals,
                          const mcspInt *ell_cols, mcspInt ell_k, const mcspComplexDouble *vec_x,
                          const mcspComplexDouble *beta, mcspComplexDouble *vec_y);

mcspStatus_t mcspCsrmvEx_bufferSize(mcspHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, mcspInt m,
                                    mcspInt n, mcspInt nnz, const void *alpha, macaDataType alphatype,
                                    const mcspMatDescr_t descrA, const void *csrValA, macaDataType csrValAtype,
                                    const mcspInt *csrRowPtrA, const mcspInt *csrColIndA, const void *x,
                                    macaDataType xtype, const void *beta, macaDataType betatype, void *y,
                                    macaDataType ytype, macaDataType executiontype, size_t *bufferSizeInBytes);

mcspStatus_t mcspCsrmvEx(mcspHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, mcspInt m, mcspInt n,
                         mcspInt nnz, const void *alpha, macaDataType alphatype, const mcspMatDescr_t descrA,
                         const void *csrValA, macaDataType csrValAtype, const mcspInt *csrRowPtrA,
                         const mcspInt *csrColIndA, const void *x, macaDataType xtype, const void *beta,
                         macaDataType betatype, void *y, macaDataType ytype, macaDataType executiontype, void *buffer);

mcspStatus_t mcspCuinCsrmvEx_bufferSize(mcspHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, int m,
                                      int n, int nnz, const void *alpha, macaDataType alphatype,
                                      const mcspMatDescr_t descrA, const void *csrValA, macaDataType csrValAtype,
                                      const int *csrRowPtrA, const int *csrColIndA, const void *x, macaDataType xtype,
                                      const void *beta, macaDataType betatype, void *y, macaDataType ytype,
                                      macaDataType executiontype, size_t *bufferSizeInBytes);

mcspStatus_t mcspCuinCsrmvEx(mcspHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, int m, int n,
                           int nnz, const void *alpha, macaDataType alphatype, const mcspMatDescr_t descrA,
                           const void *csrValA, macaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA,
                           const void *x, macaDataType xtype, const void *beta, macaDataType betatype, void *y,
                           macaDataType ytype, macaDataType executiontype, void *buffer);

/**
 * @brief   compute CSR-based Sparse triangular solve in GPU
 *          op(A) * y = alpha * x
 *          1st step: determine buffer size needed for the calculation
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans         [in]        matrix operation type
 * @param row_num       [in]        number of rows
 * @param nnz           [in]        number of nonzeros
 * @param descr         [in]        descriptor of the sparse matrix
 * @param csr_vals      [in]        pointer to the values of nonzeros in CSR
 * @param csr_rows      [in]        pointer to the row offset in CSR
 * @param csr_cols      [in]        pointer to the column indexes of nonzeros in CSR
 * @param info          [in]        meta data for CSR matrix
 * @param buffer_size   [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrSpsvBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                    const mcspMatDescr_t descr, const float *csr_vals, const mcspInt *csr_rows,
                                    const mcspInt *csr_cols, mcspMatInfo_t info, size_t *buffer_size);
mcspStatus_t mcspDcsrSpsvBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                    const mcspMatDescr_t descr, const double *csr_vals, const mcspInt *csr_rows,
                                    const mcspInt *csr_cols, mcspMatInfo_t info, size_t *buffer_size);
mcspStatus_t mcspCcsrSpsvBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                    const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals,
                                    const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info,
                                    size_t *buffer_size);
mcspStatus_t mcspZcsrSpsvBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                    const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                                    const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info,
                                    size_t *buffer_size);

/**
 * @brief   compute CSR-based Sparse triangular solve in GPU
 *          op(A) * y = alpha * x
 *          2nd step: analyze the square sparse matrix A
 *
 * @param handle           [in]        handle of mcsp library
 * @param trans            [in]        matrix operation type
 * @param row_num          [in]        number of rows
 * @param nnz              [in]        number of nonzeros
 * @param descr            [in]        descriptor of the sparse matrix
 * @param csr_vals         [in]        pointer to the values of nonzeros in CSR
 * @param csr_rows         [in]        pointer to the row offset in CSR
 * @param csr_cols         [in]        pointer to the column indexes of nonzeros in CSR
 * @param info             [out]       meta data for CSR matrix
 * @param analysis_policy  [in]        analysis policy
 * @param solve_policy     [in]        solve policy
 * @param buffer           [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrSpsvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                  const mcspMatDescr_t descr, const float *csr_vals, const mcspInt *csr_rows,
                                  const mcspInt *csr_cols, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void *buffer);
mcspStatus_t mcspDcsrSpsvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                  const mcspMatDescr_t descr, const double *csr_vals, const mcspInt *csr_rows,
                                  const mcspInt *csr_cols, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void *buffer);
mcspStatus_t mcspCcsrSpsvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                  const mcspMatDescr_t descr, const mcspComplexFloat *csr_vals, const mcspInt *csr_rows,
                                  const mcspInt *csr_cols, mcspMatInfo_t info, mcsparseAnalysisPolicy_t analysis_policy,
                                  mcsparseSolvePolicy_t solve_policy, void *buffer);
mcspStatus_t mcspZcsrSpsvAnalysis(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                                  const mcspMatDescr_t descr, const mcspComplexDouble *csr_vals,
                                  const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info,
                                  mcsparseAnalysisPolicy_t analysis_policy, mcsparseSolvePolicy_t solve_policy,
                                  void *buffer);

/**
 * @brief   compute CSR-based Sparse triangular solve in GPU
 *          op(A) * y = alpha * x
 *          3rd step: compute triangular solve
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans         [in]        matrix operation type
 * @param row_num       [in]        number of rows
 * @param nnz           [in]        number of nonzeros
 * @param alpha         [in]        alpha
 * @param descr         [in]        descriptor of the sparse matrix
 * @param csr_vals      [in]        pointer to the values of nonzeros in CSR
 * @param csr_rows      [in]        pointer to the row offset in CSR
 * @param csr_cols      [in]        pointer to the column indexes of nonzeros in CSR
 * @param info          [in]        meta data for CSR matrix
 * @param x             [in]        pointer to the vector x
 * @param y             [out]       pointer to the vector y
 * @param solve_policy  [in]        solve policy
 * @param buffer        [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrSpsvSolve(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                               const float *alpha, const mcspMatDescr_t descr, const float *csr_vals,
                               const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info, const float *x,
                               float *y, mcsparseSolvePolicy_t solve_policy, void *buffer);
mcspStatus_t mcspDcsrSpsvSolve(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                               const double *alpha, const mcspMatDescr_t descr, const double *csr_vals,
                               const mcspInt *csr_rows, const mcspInt *csr_cols, mcspMatInfo_t info, const double *x,
                               double *y, mcsparseSolvePolicy_t solve_policy, void *buffer);
mcspStatus_t mcspCcsrSpsvSolve(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                               const mcspComplexFloat *alpha, const mcspMatDescr_t descr,
                               const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                               mcspMatInfo_t info, const mcspComplexFloat *x, mcspComplexFloat *y,
                               mcsparseSolvePolicy_t solve_policy, void *buffer);
mcspStatus_t mcspZcsrSpsvSolve(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt row_num, mcspInt nnz,
                               const mcspComplexDouble *alpha, const mcspMatDescr_t descr,
                               const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                               mcspMatInfo_t info, const mcspComplexDouble *x, mcspComplexDouble *y,
                               mcsparseSolvePolicy_t solve_policy, void *buffer);

/**
 * @brief   compute CSR-based Sparse triangular solve in GPU
 *          deallocate the memory allocated by CsrSpsvAnalysis()
 *
 * @param handle        [in]        handle of mcsp library
 * @param descr         [in]        descriptor of the sparse matrix
 * @param info          [in/out]    meta data for CSR matrix
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrSpsvClear(mcspHandle_t handle, const mcspMatDescr_t descr, mcspMatInfo_t info,
                              mcsparseOperation_t opA = MCSPARSE_OPERATION_NON_TRANSPOSE);
/**
 * @brief   compute CSR-based Sparse triangular solve in GPU
 *          return zero pivot and its position if structual or numerical zero exist in A
 *
 * @param handle        [in]        handle of mcsp library
 * @param descr         [in]        descriptor of the sparse matrix
 * @param info          [in]        meta data for CSR matrix
 * @param position      [out]       pointer to the position of zero pivot
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrSpsvZeroPivot(mcspHandle_t handle, const mcspMatDescr_t descr, mcspMatInfo_t info,
                                  mcspInt *position);

mcspStatus_t mcspCuinXcsrsv2_zeroPivot(mcspHandle_t handle, mcspCsrsv2Info_t info, int *position);

mcspStatus_t mcspCuinScsrsv2_bufferSize(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcspMatDescr_t descrA, float *csrSortedValA, const int *csrSortedRowPtrA,
                                      const int *csrSortedColIndA, mcspCsrsv2Info_t info, int *pBufferSizeInBytes);

mcspStatus_t mcspCuinDcsrsv2_bufferSize(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcspMatDescr_t descrA, double *csrSortedValA, const int *csrSortedRowPtrA,
                                      const int *csrSortedColIndA, mcspCsrsv2Info_t info, int *pBufferSizeInBytes);

mcspStatus_t mcspCuinCcsrsv2_bufferSize(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcspMatDescr_t descrA, mcspComplexFloat *csrSortedValA,
                                      const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                      int *pBufferSizeInBytes);

mcspStatus_t mcspCuinZcsrsv2_bufferSize(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                      const mcspMatDescr_t descrA, mcspComplexDouble *csrSortedValA,
                                      const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                      int *pBufferSizeInBytes);

mcspStatus_t mcspCuinScsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                         const mcspMatDescr_t descrA, float *csrSortedValA, const int *csrSortedRowPtrA,
                                         const int *csrSortedColIndA, mcspCsrsv2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspCuinDcsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                         const mcspMatDescr_t descrA, double *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                         mcspCsrsv2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspCuinCcsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                         const mcspMatDescr_t descrA, mcspComplexFloat *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                         mcspCsrsv2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspCuinZcsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                         const mcspMatDescr_t descrA, mcspComplexDouble *csrSortedValA,
                                         const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                         mcspCsrsv2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspCuinScsrsv2_analysis(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                    const mcspMatDescr_t descrA, const float *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                    mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinDcsrsv2_analysis(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                    const mcspMatDescr_t descrA, const double *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                    mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinCcsrsv2_analysis(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                    const mcspMatDescr_t descrA, const mcspComplexFloat *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                    mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinZcsrsv2_analysis(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                    const mcspMatDescr_t descrA, const mcspComplexDouble *csrSortedValA,
                                    const int *csrSortedRowPtrA, const int *csrSortedColIndA, mcspCsrsv2Info_t info,
                                    mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinScsrsv2_solve(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz, const float *alpha,
                                 const mcspMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspCsrsv2Info_t info, const float *f, float *x,
                                 mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinDcsrsv2_solve(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz, const double *alpha,
                                 const mcspMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspCsrsv2Info_t info, const double *f, double *x,
                                 mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinCcsrsv2_solve(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                 const mcspComplexFloat *alpha, const mcspMatDescr_t descrA,
                                 const mcspComplexFloat *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspCsrsv2Info_t info, const mcspComplexFloat *f,
                                 mcspComplexFloat *x, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinZcsrsv2_solve(mcspHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                 const mcspComplexDouble *alpha, const mcspMatDescr_t descrA,
                                 const mcspComplexDouble *csrSortedValA, const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA, mcspCsrsv2Info_t info, const mcspComplexDouble *f,
                                 mcspComplexDouble *x, mcsparseSolvePolicy_t policy, void *pBuffer);
/**
 * @brief   Dense matrix sparse vector multiplication.
 *          1st step: determine buffer size needed for the calculation
 *          y = alpha * A * x + beta * y
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans         [in]        matrix operation type
 * @param m             [in]        number of dense matrix rows
 * @param n             [in]        number of dense matrix columns
 * @param nnz           [in]        number of non-zeros in the sparse vector
 * @param buffer_size   [out]       temporary storage buffer size
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgemviBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, mcspInt nnz,
                                  size_t *buffer_size);
mcspStatus_t mcspDgemviBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, mcspInt nnz,
                                  size_t *buffer_size);
mcspStatus_t mcspCgemviBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, mcspInt nnz,
                                  size_t *buffer_size);
mcspStatus_t mcspZgemviBuffersize(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, mcspInt nnz,
                                  size_t *buffer_size);

mcspStatus_t mcspCuinSgemvi_bufferSize(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                     int *buffer_size);
mcspStatus_t mcspCuinDgemvi_bufferSize(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                     int *buffer_size);
mcspStatus_t mcspCuinCgemvi_bufferSize(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                     int *buffer_size);
mcspStatus_t mcspCuinZgemvi_bufferSize(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                     int *buffer_size);

/**
 * @brief   Dense matrix sparse vector multiplication. x is a sparse vector and y is a dense vector.
 *          2nd step: dense matrix sparse vector multiplication
 *          y = alpha * op(A) * x + beta * y
 *
 * @param handle        [in]        handle of mcsp library
 * @param trans         [in]        matrix operation type
 * @param m             [in]        number of rows of dense matrix
 * @param n             [in]        number of columns of dense matrix
 * @param alpha         [in]        scalar alpha
 * @param A             [in]        pointer to the dense matrix
 * @param lda           [in]        leading dimension of the dense matrix
 * @param nnz           [in]        number of non-zeros in the sparse vector
 * @param x_val         [in]        array of nnz elements containing the values of the sparse vector
 * @param x_ind         [in]        array of nnz elements containing the index of the sparse vector
 * @param beta          [in]        scalar beta
 * @param y             [out]       array of m (if op(A)=A) or n(if op(A)=A_T or op(A)=A_H) elements
 * @param idx_base      [in]        MCSPARSE_INDEX_BASE_ZERO or MCSPARSE_INDEX_BASE_ONE
 * @param temp_buffer   [in]        temporary storage buffer
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgemvi(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, const float *alpha,
                        const float *A, mcspInt lda, mcspInt nnz, const float *x_val, const mcspInt *x_ind,
                        const float *beta, float *y, mcsparseIndexBase_t idx_base, void *temp_buffer);
mcspStatus_t mcspDgemvi(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n, const double *alpha,
                        const double *A, mcspInt lda, mcspInt nnz, const double *x_val, const mcspInt *x_ind,
                        const double *beta, double *y, mcsparseIndexBase_t idx_base, void *temp_buffer);
mcspStatus_t mcspCgemvi(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n,
                        const mcspComplexFloat *alpha, const mcspComplexFloat *A, mcspInt lda, mcspInt nnz,
                        const mcspComplexFloat *x_val, const mcspInt *x_ind, const mcspComplexFloat *beta,
                        mcspComplexFloat *y, mcsparseIndexBase_t idx_base, void *temp_buffer);
mcspStatus_t mcspZgemvi(mcspHandle_t handle, mcsparseOperation_t trans, mcspInt m, mcspInt n,
                        const mcspComplexDouble *alpha, const mcspComplexDouble *A, mcspInt lda, mcspInt nnz,
                        const mcspComplexDouble *x_val, const mcspInt *x_ind, const mcspComplexDouble *beta,
                        mcspComplexDouble *y, mcsparseIndexBase_t idx_base, void *temp_buffer);

mcspStatus_t mcspCuinSgemvi(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, const float *alpha,
                          const float *A, int lda, int nnz, const float *x_val, const int *x_ind, const float *beta,
                          float *y, mcsparseIndexBase_t idx_base, void *temp_buffer);

mcspStatus_t mcspCuinDgemvi(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, const double *alpha,
                          const double *A, int lda, int nnz, const double *x_val, const int *x_ind, const double *beta,
                          double *y, mcsparseIndexBase_t idx_base, void *temp_buffer);

mcspStatus_t mcspCuinCgemvi(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, const mcspComplexFloat *alpha,
                          const mcspComplexFloat *A, int lda, int nnz, const mcspComplexFloat *x_val, const int *x_ind,
                          const mcspComplexFloat *beta, mcspComplexFloat *y, mcsparseIndexBase_t idx_base,
                          void *temp_buffer);

mcspStatus_t mcspCuinZgemvi(mcspHandle_t handle, mcsparseOperation_t trans, int m, int n, const mcspComplexDouble *alpha,
                          const mcspComplexDouble *A, int lda, int nnz, const mcspComplexDouble *x_val,
                          const int *x_ind, const mcspComplexDouble *beta, mcspComplexDouble *y,
                          mcsparseIndexBase_t idx_base, void *temp_buffer);
/**
 * @brief   Compute BSR-based SpMV in GPU.
 *          y = alpha * op(A) * x + beta * y
 *
 * @param handle             [in]        handle of mcsp library
 * @param dir                [in]        storage format of blocks
 * @param trans              [in]        matrix operation type
 * @param mb                 [in]        number of rows of bsr matrix
 * @param nb                 [in]        number of columns of bsr matrix
 * @param nnzb               [in]        total number of nonzero blocks of BSR matrix
 * @param alpha              [in]        scalar used for multiplication
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_vals           [in]        pointer to the val offset in BSR matrix
 * @param bsr_rows_ind       [in]        pointer to the row offset in BSR matrix
 * @param bsr_cols_ind       [in]        pointer to the col offset in BSR matrix
 * @param block_dim          [in]        number of rows within a block of BSR matrix
 * @param x                  [in]        vector of nb*block_dim elements
 * @param beta               [in]        scalar beta
 * @param y                  [in/out]    vector of mb*block_dim elements and updated vector
 * @return mcspStatus_t
 */
mcspStatus_t mcspSbsrmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb, mcspInt nb,
                        mcspInt nnzb, const float *alpha, const mcspMatDescr_t bsr_descr, const float *bsr_vals,
                        const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim, const float *x,
                        const float *beta, float *y);

mcspStatus_t mcspDbsrmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb, mcspInt nb,
                        mcspInt nnzb, const double *alpha, const mcspMatDescr_t bsr_descr, const double *bsr_vals,
                        const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim, const double *x,
                        const double *beta, double *y);

mcspStatus_t mcspCbsrmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb, mcspInt nb,
                        mcspInt nnzb, const mcspComplexFloat *alpha, const mcspMatDescr_t bsr_descr,
                        const mcspComplexFloat *bsr_vals, const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind,
                        mcspInt block_dim, const mcspComplexFloat *x, const mcspComplexFloat *beta,
                        mcspComplexFloat *y);

mcspStatus_t mcspZbsrmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb, mcspInt nb,
                        mcspInt nnzb, const mcspComplexDouble *alpha, const mcspMatDescr_t bsr_descr,
                        const mcspComplexDouble *bsr_vals, const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind,
                        mcspInt block_dim, const mcspComplexDouble *x, const mcspComplexDouble *beta,
                        mcspComplexDouble *y);

mcspStatus_t mcspCuinSbsrmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb, int nb,
                          int nnzb, const float *alpha, const mcspMatDescr_t descrA, const float *bsrSortedValA,
                          const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const float *x,
                          const float *beta, float *y);

mcspStatus_t mcspCuinDbsrmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb, int nb,
                          int nnzb, const double *alpha, const mcspMatDescr_t descrA, const double *bsrSortedValA,
                          const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const double *x,
                          const double *beta, double *y);

mcspStatus_t mcspCuinCbsrmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb, int nb,
                          int nnzb, const mcspComplexFloat *alpha, const mcspMatDescr_t descrA,
                          const mcspComplexFloat *bsrSortedValA, const int *bsrSortedRowPtrA,
                          const int *bsrSortedColIndA, int blockDim, const mcspComplexFloat *x,
                          const mcspComplexFloat *beta, mcspComplexFloat *y);

mcspStatus_t mcspCuinZbsrmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb, int nb,
                          int nnzb, const mcspComplexDouble *alpha, const mcspMatDescr_t descrA,
                          const mcspComplexDouble *bsrSortedValA, const int *bsrSortedRowPtrA,
                          const int *bsrSortedColIndA, int blockDim, const mcspComplexDouble *x,
                          const mcspComplexDouble *beta, mcspComplexDouble *y);

/**
 * @brief   Compute masked BSRX-based SpMV in GPU.
 *          y(mask) = (alpha * op(A) * x + beta * y)(mask)
 *
 * @param handle             [in]        handle of mcsp library
 * @param dir                [in]        storage format of blocks
 * @param trans              [in]        matrix operation type
 * @param mask_size          [in]        number of updated block rows of y
 * @param mb                 [in]        number of rows of bsr matrix
 * @param nb                 [in]        number of columns of bsr matrix
 * @param nnzb               [in]        total number of nonzero blocks of BSR matrix
 * @param alpha              [in]        scalar used for multiplication
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_vals           [in]        pointer to the val offset in BSR matrix
 * @param mask_ptr           [in]        array of mask_size elements that contains the indices of updated block rows
 * @param bsr_rows_ind       [in]        array of mb elements that contains the start of every block row
 * @param bsr_ends_ind       [in]        array of mb elements that contains the end of the every block row plus one
 * @param bsr_cols_ind       [in]        pointer to the col offset in BSR matrix
 * @param block_dim          [in]        number of rows within a block of BSR matrix
 * @param x                  [in]        vector of nb*block_dim elements
 * @param beta               [in]        scalar beta
 * @param y                  [in/out]    vector of mb*block_dim elements and updated vector
 * @return mcspStatus_t
 */
mcspStatus_t mcspSbsrxmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mask_size,
                         mcspInt mb, mcspInt nb, mcspInt nnzb, const float *alpha, const mcspMatDescr_t bsr_descr,
                         const float *bsr_vals, const mcspInt *mask_ptr, const mcspInt *bsr_rows_ind,
                         const mcspInt *bsr_ends_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim, const float *x,
                         const float *beta, float *y);

mcspStatus_t mcspDbsrxmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mask_size,
                         mcspInt mb, mcspInt nb, mcspInt nnzb, const double *alpha, const mcspMatDescr_t bsr_descr,
                         const double *bsr_vals, const mcspInt *mask_ptr, const mcspInt *bsr_rows_ind,
                         const mcspInt *bsr_ends_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim, const double *x,
                         const double *beta, double *y);

mcspStatus_t mcspCbsrxmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mask_size,
                         mcspInt mb, mcspInt nb, mcspInt nnzb, const mcspComplexFloat *alpha,
                         const mcspMatDescr_t bsr_descr, const mcspComplexFloat *bsr_vals, const mcspInt *mask_ptr,
                         const mcspInt *bsr_rows_ind, const mcspInt *bsr_ends_ind, const mcspInt *bsr_cols_ind,
                         mcspInt block_dim, const mcspComplexFloat *x, const mcspComplexFloat *beta,
                         mcspComplexFloat *y);

mcspStatus_t mcspZbsrxmv(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mask_size,
                         mcspInt mb, mcspInt nb, mcspInt nnzb, const mcspComplexDouble *alpha,
                         const mcspMatDescr_t bsr_descr, const mcspComplexDouble *bsr_vals, const mcspInt *mask_ptr,
                         const mcspInt *bsr_rows_ind, const mcspInt *bsr_ends_ind, const mcspInt *bsr_cols_ind,
                         mcspInt block_dim, const mcspComplexDouble *x, const mcspComplexDouble *beta,
                         mcspComplexDouble *y);

mcspStatus_t mcspCuinSbsrxmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int sizeOfMask,
                           int mb, int nb, int nnzb, const float *alpha, const mcspMatDescr_t descrA,
                           const float *bsrSortedValA, const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA,
                           const int *bsrSortedEndPtrA, const int *bsrSortedColIndA, int blockDim, const float *x,
                           const float *beta, float *y);

mcspStatus_t mcspCuinDbsrxmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int sizeOfMask,
                           int mb, int nb, int nnzb, const double *alpha, const mcspMatDescr_t descrA,
                           const double *bsrSortedValA, const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA,
                           const int *bsrSortedEndPtrA, const int *bsrSortedColIndA, int blockDim, const double *x,
                           const double *beta, double *y);

mcspStatus_t mcspCuinCbsrxmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int sizeOfMask,
                           int mb, int nb, int nnzb, const mcFloatComplex *alpha, const mcspMatDescr_t descrA,
                           const mcFloatComplex *bsrSortedValA, const int *bsrSortedMaskPtrA,
                           const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA,
                           int blockDim, const mcFloatComplex *x, const mcFloatComplex *beta, mcFloatComplex *y);

mcspStatus_t mcspCuinZbsrxmv(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int sizeOfMask,
                           int mb, int nb, int nnzb, const mcDoubleComplex *alpha, const mcspMatDescr_t descrA,
                           const mcDoubleComplex *bsrSortedValA, const int *bsrSortedMaskPtrA,
                           const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA,
                           int blockDim, const mcDoubleComplex *x, const mcDoubleComplex *beta, mcDoubleComplex *y);

/**
 * @brief   compute BSR-based Sparse triangular solve in GPU
 *          op(A) * y = alpha * x
 *          1st step: determine buffer size needed for the calculation
 *
 * @param handle             [in]        handle of mcsp library
 * @param dir                [in]        storage format of blocks
 * @param trans              [in]        matrix operation type
 * @param mb                 [in]        number of rows of bsr matrix
 * @param nnzb               [in]        total number of nonzero blocks of BSR matrix
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_vals           [in]        pointer to the val offset in BSR matrix
 * @param bsr_rows_ind       [in]        pointer to the row offset in BSR matrix
 * @param bsr_cols_ind       [in]        pointer to the col offset in BSR matrix
 * @param block_dim          [in]        number of rows within a block of BSR matrix
 * @param info               [in]        meta data for BSR matrix
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspSbsrsvBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                  mcspInt nnzb, const mcspMatDescr_t bsr_descr, const float *bsr_vals,
                                  const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                                  mcspBsrsv2Info_t info, size_t *buffer_size);

mcspStatus_t mcspDbsrsvBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                  mcspInt nnzb, const mcspMatDescr_t bsr_descr, const double *bsr_vals,
                                  const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                                  mcspBsrsv2Info_t info, size_t *buffer_size);

mcspStatus_t mcspCbsrsvBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                  mcspInt nnzb, const mcspMatDescr_t bsr_descr, const mcspComplexFloat *bsr_vals,
                                  const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                                  mcspBsrsv2Info_t info, size_t *buffer_size);

mcspStatus_t mcspZbsrsvBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                  mcspInt nnzb, const mcspMatDescr_t bsr_descr, const mcspComplexDouble *bsr_vals,
                                  const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                                  mcspBsrsv2Info_t info, size_t *buffer_size);
/**
 * @brief   compute BSR-based Sparse triangular solve in GPU
 *          op(A) * y = alpha * x
 *          2nd step: analyze the square sparse matrix A
 *
 * @param handle             [in]        handle of mcsp library
 * @param dir                [in]        storage format of blocks
 * @param trans              [in]        matrix operation type
 * @param mb                 [in]        number of rows of bsr matrix
 * @param nnzb               [in]        total number of nonzero blocks of BSR matrix
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_vals           [in]        pointer to the val offset in BSR matrix
 * @param bsr_rows_ind       [in]        pointer to the row offset in BSR matrix
 * @param bsr_cols_ind       [in]        pointer to the col offset in BSR matrix
 * @param block_dim          [in]        number of rows within a block of BSR matrix
 * @param info               [in]        meta data for BSR matrix
 * @param policy             [in]        analysis policy
 * @param buffer             [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSbsrsvAnalysis(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                mcspInt nnzb, const mcspMatDescr_t bsr_descr, const float *bsr_vals,
                                const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                                mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *buffer);

mcspStatus_t mcspDbsrsvAnalysis(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                mcspInt nnzb, const mcspMatDescr_t bsr_descr, const double *bsr_vals,
                                const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                                mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *buffer);

mcspStatus_t mcspCbsrsvAnalysis(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                mcspInt nnzb, const mcspMatDescr_t bsr_descr, const mcspComplexFloat *bsr_vals,
                                const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                                mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *buffer);

mcspStatus_t mcspZbsrsvAnalysis(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                                mcspInt nnzb, const mcspMatDescr_t bsr_descr, const mcspComplexDouble *bsr_vals,
                                const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                                mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *buffer);

/**
 * @brief   compute BSR-based Sparse triangular solve in GPU
 *          op(A) * y = alpha * x
 *          3rd step: compute triangular solve
 *
 * @param handle             [in]        handle of mcsp library
 * @param dir                [in]        storage format of blocks
 * @param trans              [in]        matrix operation type
 * @param mb                 [in]        number of rows of bsr matrix
 * @param nnzb               [in]        total number of nonzero blocks of BSR matrix
 * @param alpha              [in]        alpha
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_vals           [in]        pointer to the val offset in BSR matrix
 * @param bsr_rows_ind       [in]        pointer to the row offset in BSR matrix
 * @param bsr_cols_ind       [in]        pointer to the col offset in BSR matrix
 * @param block_dim          [in]        number of rows within a block of BSR matrix
 * @param info               [in]        meta data for BSR matrix
 * @param x                  [in]        pointer to the vector x
 * @param y                  [out]       pointer to the vector y
 * @param policy             [in]        analysis policy
 * @param buffer             [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSbsrsvSolve(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                             mcspInt nnzb, const float *alpha, const mcspMatDescr_t bsr_descr, const float *bsr_vals,
                             const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                             mcspBsrsv2Info_t info, const float *x, float *y, mcsparseSolvePolicy_t policy,
                             void *buffer);

mcspStatus_t mcspDbsrsvSolve(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                             mcspInt nnzb, const double *alpha, const mcspMatDescr_t bsr_descr, const double *bsr_vals,
                             const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind, mcspInt block_dim,
                             mcspBsrsv2Info_t info, const double *x, double *y, mcsparseSolvePolicy_t policy,
                             void *buffer);

mcspStatus_t mcspCbsrsvSolve(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                             mcspInt nnzb, const mcspComplexFloat *alpha, const mcspMatDescr_t bsr_descr,
                             const mcspComplexFloat *bsr_vals, const mcspInt *bsr_rows_ind, const mcspInt *bsr_cols_ind,
                             mcspInt block_dim, mcspBsrsv2Info_t info, const mcspComplexFloat *x, mcspComplexFloat *y,
                             mcsparseSolvePolicy_t policy, void *buffer);

mcspStatus_t mcspZbsrsvSolve(mcspHandle_t handle, mcsparseDirection_t dir, mcsparseOperation_t trans, mcspInt mb,
                             mcspInt nnzb, const mcspComplexDouble *alpha, const mcspMatDescr_t bsr_descr,
                             const mcspComplexDouble *bsr_vals, const mcspInt *bsr_rows_ind,
                             const mcspInt *bsr_cols_ind, mcspInt block_dim, mcspBsrsv2Info_t info,
                             const mcspComplexDouble *x, mcspComplexDouble *y, mcsparseSolvePolicy_t policy,
                             void *buffer);

/**
 * @brief   compute BSR-based Sparse triangular solve in GPU
 *          return zero pivot and its position if structual or numerical zero exist in A
 *
 * @param handle             [in]        handle of mcsp library
 * @param info               [in]        meta data for BSR matrix
 * @param position           [out]       pointer to the position of zero pivot
 * @return mcspStatus_t
 */
mcspStatus_t mcspBsrsvZeroPivot(mcspHandle_t handle, mcspBsrsv2Info_t info, int *position);

mcspStatus_t mcspCuinXbsrsv2_zeroPivot(mcspHandle_t handle, mcspBsrsv2Info_t info, int *position);

mcspStatus_t mcspCuinSbsrsv2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                      int nnzb, const mcspMatDescr_t descrA, float *bsrSortedValA,
                                      const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                      mcspBsrsv2Info_t info, int *pBufferSizeInBytes);

mcspStatus_t mcspCuinDbsrsv2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                      int nnzb, const mcspMatDescr_t descrA, double *bsrSortedValA,
                                      const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                      mcspBsrsv2Info_t info, int *pBufferSizeInBytes);

mcspStatus_t mcspCuinCbsrsv2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                      int nnzb, const mcspMatDescr_t descrA, mcFloatComplex *bsrSortedValA,
                                      const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                      mcspBsrsv2Info_t info, int *pBufferSizeInBytes);

mcspStatus_t mcspCuinZbsrsv2_bufferSize(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                      int nnzb, const mcspMatDescr_t descrA, mcDoubleComplex *bsrSortedValA,
                                      const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                      mcspBsrsv2Info_t info, int *pBufferSizeInBytes);

mcspStatus_t mcspCuinSbsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                         int mb, int nnzb, const mcspMatDescr_t descrA, float *bsrSortedValA,
                                         const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
                                         mcspBsrsv2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspCuinDbsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                         int mb, int nnzb, const mcspMatDescr_t descrA, double *bsrSortedValA,
                                         const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
                                         mcspBsrsv2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspCuinCbsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                         int mb, int nnzb, const mcspMatDescr_t descrA, mcFloatComplex *bsrSortedValA,
                                         const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
                                         mcspBsrsv2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspCuinZbsrsv2_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                         int mb, int nnzb, const mcspMatDescr_t descrA, mcDoubleComplex *bsrSortedValA,
                                         const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize,
                                         mcspBsrsv2Info_t info, size_t *pBufferSize);

mcspStatus_t mcspCuinSbsrsv2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                    int nnzb, const mcspMatDescr_t descrA, const float *bsrSortedValA,
                                    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                    mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinDbsrsv2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                    int nnzb, const mcspMatDescr_t descrA, const double *bsrSortedValA,
                                    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                    mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinCbsrsv2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                    int nnzb, const mcspMatDescr_t descrA, const mcFloatComplex *bsrSortedValA,
                                    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                    mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinZbsrsv2_analysis(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                    int nnzb, const mcspMatDescr_t descrA, const mcDoubleComplex *bsrSortedValA,
                                    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                    mcspBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinSbsrsv2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                 int nnzb, const float *alpha, const mcspMatDescr_t descrA, const float *bsrSortedValA,
                                 const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
                                 mcspBsrsv2Info_t info, const float *f, float *x, mcsparseSolvePolicy_t policy,
                                 void *pBuffer);

mcspStatus_t mcspCuinDbsrsv2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                 int nnzb, const double *alpha, const mcspMatDescr_t descrA,
                                 const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                 int blockDim, mcspBsrsv2Info_t info, const double *f, double *x,
                                 mcsparseSolvePolicy_t policy, void *pBuffer);

mcspStatus_t mcspCuinCbsrsv2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                 int nnzb, const mcFloatComplex *alpha, const mcspMatDescr_t descrA,
                                 const mcFloatComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                 const int *bsrSortedColIndA, int blockDim, mcspBsrsv2Info_t info,
                                 const mcFloatComplex *f, mcFloatComplex *x, mcsparseSolvePolicy_t policy,
                                 void *pBuffer);

mcspStatus_t mcspCuinZbsrsv2_solve(mcspHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                 int nnzb, const mcDoubleComplex *alpha, const mcspMatDescr_t descrA,
                                 const mcDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                 const int *bsrSortedColIndA, int blockDim, mcspBsrsv2Info_t info,
                                 const mcDoubleComplex *f, mcDoubleComplex *x, mcsparseSolvePolicy_t policy,
                                 void *pBuffer);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSP_LEVEL2_H_
