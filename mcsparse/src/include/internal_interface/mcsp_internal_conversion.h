#ifndef INTERFACE_MCSP_INTERNAL_CONVERSION_H_
#define INTERFACE_MCSP_INTERNAL_CONVERSION_H_

#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief   Convert a sparse COO matrix into a sparse CSR matrix
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param coo_rows           [in]        pointer to the row offset in COO matrix A
 * @param nnz                [in]        number of nonzeros of A
 * @param m                  [in]        number of rows of A
 * @param csr_rows           [out]       pointer to the row offset in CSR matrix A
 * @param idx_base           [in]        index base of matrix A (0 or 1)
 * @return mcspStatus_t
 */
mcspStatus_t mcspCoo2Csr(mcspHandle_t handle, const mcspInt *coo_rows, mcspInt nnz, mcspInt m, mcspInt *csr_rows,
                         mcsparseIndexBase_t idx_base);
mcspStatus_t mcspCoo2Csr64(mcspHandle_t handle, const int64_t *coo_rows, int64_t nnz, int64_t m, int64_t *csr_rows,
                           mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinXcoo2csr(mcspHandle_t handle, const int *coo_rows, int nnz, int m, int *csr_rows,
                            mcsparseIndexBase_t idx_base);

/**
 * @brief   Convert a sparse CSR matrix into a sparse COO matrix
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param nnz                [in]        number of nonzeros of A
 * @param m                  [in]        number of rows of A
 * @param coo_rows           [out]       pointer to the row offset in COO matrix A
 * @param idx_base           [in]        index base of matrix A (0 or 1)
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsr2Coo(mcspHandle_t handle, const mcspInt *csr_rows, mcspInt nnz, mcspInt m, mcspInt *coo_rows,
                         mcsparseIndexBase_t idx_base);
mcspStatus_t mcspCsr2Coo64(mcspHandle_t handle, const int64_t *csr_rows, int64_t nnz, int64_t m, int64_t *coo_rows,
                           mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinXcsr2coo(mcspHandle_t handle, const int *csr_rows, int nnz, int m, int *coo_rows,
                            mcsparseIndexBase_t idx_base);

/**
 * @brief    Convert a sparse CSR matrix into a sparse CSC matrix
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param csr_val            [in]        pointer to the value list in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the col offset in CSR matrix A
 * @param csc_val            [out]       pointer to the value list in CSC matrix A
 * @param csc_rows           [out]       pointer to the row offset in CSC matrix A
 * @param csc_cols           [out]       pointer to the col offset in CSC matrix A
 * @param csc_action         [in]        operate only on indices or data and indices
 * @param idx_base           [in]        index base of matrix A (0 or 1)
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */

mcspStatus_t mcspScsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const float *csr_val,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, float *csc_val, mcspInt *csc_rows,
                          mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                          void *temp_buffer);
mcspStatus_t mcspScsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const float *csr_val,
                            const int64_t *csr_rows, const int64_t *csr_cols, float *csc_val, int64_t *csc_rows,
                            int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                            void *temp_buffer);

mcspStatus_t mcspDcsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const double *csr_val,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, double *csc_val, mcspInt *csc_rows,
                          mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                          void *temp_buffer);
mcspStatus_t mcspDcsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const double *csr_val,
                            const int64_t *csr_rows, const int64_t *csr_cols, double *csc_val, int64_t *csc_rows,
                            int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                            void *temp_buffer);

mcspStatus_t mcspCcsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspComplexFloat *csr_val,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspComplexFloat *csc_val,
                          mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                          mcsparseIndexBase_t idx_base, void *temp_buffer);
mcspStatus_t mcspCcsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const mcspComplexFloat *csr_val,
                            const int64_t *csr_rows, const int64_t *csr_cols, mcspComplexFloat *csc_val,
                            int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                            mcsparseIndexBase_t idx_base, void *temp_buffer);

mcspStatus_t mcspZcsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspComplexDouble *csr_val,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspComplexDouble *csc_val,
                          mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                          mcsparseIndexBase_t idx_base, void *temp_buffer);
mcspStatus_t mcspZcsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const mcspComplexDouble *csr_val,
                            const int64_t *csr_rows, const int64_t *csr_cols, mcspComplexDouble *csc_val,
                            int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                            mcsparseIndexBase_t idx_base, void *temp_buffer);

#if defined(__MACA__)
mcspStatus_t mcspR16fCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const __half *csr_val,
                             const mcspInt *csr_rows, const mcspInt *csr_cols, __half *csc_val, mcspInt *csc_rows,
                             mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                             void *temp_buffer);
mcspStatus_t mcspR16fCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const __half *csr_val,
                               const int64_t *csr_rows, const int64_t *csr_cols, __half *csc_val, int64_t *csc_rows,
                               int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                               void *temp_buffer);
mcspStatus_t mcspC16fCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const __half2 *csr_val,
                             const mcspInt *csr_rows, const mcspInt *csr_cols, __half2 *csc_val, mcspInt *csc_rows,
                             mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                             void *temp_buffer);
mcspStatus_t mcspC16fCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const __half2 *csr_val,
                               const int64_t *csr_rows, const int64_t *csr_cols, __half2 *csc_val, int64_t *csc_rows,
                               int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                               void *temp_buffer);

mcspStatus_t mcspR16bfCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcsp_bfloat16 *csr_val,
                              const mcspInt *csr_rows, const mcspInt *csr_cols, mcsp_bfloat16 *csc_val,
                              mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                              mcsparseIndexBase_t idx_base, void *temp_buffer);
mcspStatus_t mcspR16bfCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const mcsp_bfloat16 *csr_val,
                                const int64_t *csr_rows, const int64_t *csr_cols, mcsp_bfloat16 *csc_val,
                                int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                                mcsparseIndexBase_t idx_base, void *temp_buffer);
mcspStatus_t mcspC16bfCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcsp_bfloat162 *csr_val,
                              const mcspInt *csr_rows, const mcspInt *csr_cols, mcsp_bfloat162 *csc_val,
                              mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                              mcsparseIndexBase_t idx_base, void *temp_buffer);
mcspStatus_t mcspC16bfCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const mcsp_bfloat162 *csr_val,
                                const int64_t *csr_rows, const int64_t *csr_cols, mcsp_bfloat162 *csc_val,
                                int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                                mcsparseIndexBase_t idx_base, void *temp_buffer);
mcspStatus_t mcspR8iCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const int8_t *csr_val,
                            const mcspInt *csr_rows, const mcspInt *csr_cols, int8_t *csc_val, mcspInt *csc_rows,
                            mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                            void *temp_buffer);

mcspStatus_t mcspR8iCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const int8_t *csr_val,
                              const int64_t *csr_rows, const int64_t *csr_cols, int8_t *csc_val, int64_t *csc_rows,
                              int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                              void *temp_buffer);
#endif

mcspStatus_t mcspCuinCsr2cscEx2(mcspHandle_t handle, int m, int n, int nnz, const void *csr_val, const int *csr_rows,
                              const int *csr_cols, void *csc_val, int *csc_cols, int *csc_rows, macaDataType val_type,
                              mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base, mcsparseCsr2CscAlg_t alg,
                              void *temp_buffer);
/**
 * @brief    Get buffer size for csr2csc function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the col offset in CSR matrix A
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsr2CscBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspInt *csr_rows,
                                   const mcspInt *csr_cols, mcsparseAction_t csc_action, size_t *buffer_size);
mcspStatus_t mcspCsr2CscBufferSize64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const int64_t *csr_rows,
                                     const int64_t *csr_cols, mcsparseAction_t csc_action, size_t *buffer_size);

mcspStatus_t mcspCuinCsr2cscEx2_bufferSize(mcspHandle_t handle, int m, int n, int nnz, const void *csr_val,
                                         const int *csr_rows, const int *csr_cols, void *csc_val, int *csc_cols,
                                         int *csc_rows, macaDataType val_type, mcsparseAction_t csc_action,
                                         mcsparseIndexBase_t idx_base, mcsparseCsr2CscAlg_t alg, size_t *buffer_size);
/**
 * @brief        Get the number of elements greater than tol per row
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_val_A          [in]        pointer to the values offset in CSR matrix A
 * @param csr_row_A          [in]        pointer to the row offset in CSR matrix A
 * @param nnz_per_row        [out]       pointer to the number of elements greater than tol per row
 * @param nnz_C              [out]       pointer to the total number of elements grater than tol
 * @param tol                [in]        non-negative tolerence to determine if a number less than or equal to it
 * @return mcspStatus_t
 */
mcspStatus_t mcspSnnzCompress(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A,
                              const mcspInt *csr_row_A, mcspInt *nnz_per_row, mcspInt *nnz_C, float tol);

mcspStatus_t mcspDnnzCompress(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t mcsp_descr_A,
                              const double *csr_val_A, const mcspInt *csr_row_A, mcspInt *nnz_per_row, mcspInt *nnz_C,
                              double tol);

mcspStatus_t mcspCnnzCompress(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat *csr_val_A, const mcspInt *csr_row_A, mcspInt *nnz_per_row,
                              mcspInt *nnz_C, mcspComplexFloat tol);

mcspStatus_t mcspZnnzCompress(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble *csr_val_A, const mcspInt *csr_row_A, mcspInt *nnz_per_row,
                              mcspInt *nnz_C, mcspComplexDouble tol);

mcspStatus_t mcspCuinSnnz_compress(mcspHandle_t handle, int m, const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A,
                                 const int *csr_row_A, int *nnz_per_row, int *nnz_C, float tol);

mcspStatus_t mcspCuinDnnz_compress(mcspHandle_t handle, int m, const mcspMatDescr_t mcsp_descr_A, const double *csr_val_A,
                                 const int *csr_row_A, int *nnz_per_row, int *nnz_C, double tol);

mcspStatus_t mcspCuinCnnz_compress(mcspHandle_t handle, int m, const mcspMatDescr_t mcsp_descr_A,
                                 const mcspComplexFloat *csr_val_A, const int *csr_row_A, int *nnz_per_row, int *nnz_C,
                                 mcspComplexFloat tol);

mcspStatus_t mcspCuinZnnz_compress(mcspHandle_t handle, int m, const mcspMatDescr_t mcsp_descr_A,
                                 const mcspComplexDouble *csr_val_A, const int *csr_row_A, int *nnz_per_row, int *nnz_C,
                                 mcspComplexDouble tol);

/**
 * @brief       Compress the sparse matrix in CSR formate into the compressed CSR format
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_val            [in]        pointer to the value list in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the col offset in CSR matrix A
 * @param nnz_A              [in]        number of nonzeros of A
 * @param nnz_per_row        [in]        pointer to the number of elements greater than tol per row
 * @param csr_val_C          [out]       pointer to the value list in CSR matrix C
 * @param csr_row_C          [out]       pointer to the row offset in CSR matrix C
 * @param csr_col_C          [out]       pointer to the col offset in CSR matrix C
 * @param tol                [in]        non-negative tolerence to determine if a number less than or equal to it
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsr2csrCompress(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                  const float *csr_val_A, const mcspInt *csr_row_A, const mcspInt *csr_col_A,
                                  mcspInt nnz_A, const mcspInt *nnz_per_row, float *csr_val_C, mcspInt *csr_row_C,
                                  mcspInt *csr_col_C, float tol);

mcspStatus_t mcspDcsr2csrCompress(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                  const double *csr_val_A, const mcspInt *csr_row_A, const mcspInt *csr_col_A,
                                  mcspInt nnz_A, const mcspInt *nnz_per_row, double *csr_val_C, mcspInt *csr_row_C,
                                  mcspInt *csr_col_C, double tol);

mcspStatus_t mcspCcsr2csrCompress(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                  const mcspComplexFloat *csr_val_A, const mcspInt *csr_row_A, const mcspInt *csr_col_A,
                                  mcspInt nnz_A, const mcspInt *nnz_per_row, mcspComplexFloat *csr_val_C,
                                  mcspInt *csr_row_C, mcspInt *csr_col_C, mcspComplexFloat tol);

mcspStatus_t mcspZcsr2csrCompress(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                  const mcspComplexDouble *csr_val_A, const mcspInt *csr_row_A,
                                  const mcspInt *csr_col_A, mcspInt nnz_A, const mcspInt *nnz_per_row,
                                  mcspComplexDouble *csr_val_C, mcspInt *csr_row_C, mcspInt *csr_col_C,
                                  mcspComplexDouble tol);
mcspStatus_t mcspCuinScsr2csr_compress(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                                     const float *csr_val_A, const int *csr_col_A, const int *csr_row_A, int nnz_A,
                                     const int *nnz_per_row, float *csr_val_C, int *csr_col_C, int *csr_row_C,
                                     float tol);

mcspStatus_t mcspCuinDcsr2csr_compress(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                                     const double *csr_val_A, const int *csr_col_A, const int *csr_row_A, int nnz_A,
                                     const int *nnz_per_row, double *csr_val_C, int *csr_col_C, int *csr_row_C,
                                     double tol);

mcspStatus_t mcspCuinCcsr2csr_compress(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                                     const mcspComplexFloat *csr_val_A, const int *csr_col_A, const int *csr_row_A,
                                     int nnz_A, const int *nnz_per_row, mcspComplexFloat *csr_val_C, int *csr_col_C,
                                     int *csr_row_C, mcspComplexFloat tol);
mcspStatus_t mcspCuinZcsr2csr_compress(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                                     const mcspComplexDouble *csr_val_A, const int *csr_col_A, const int *csr_row_A,
                                     int nnz_A, const int *nnz_per_row, mcspComplexDouble *csr_val_C, int *csr_col_C,
                                     int *csr_row_C, mcspComplexDouble tol);
/**
 * @brief    Get buffer size for CSR sort function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the col offset in CSR matrix A
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrSortBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspInt *csr_rows,
                                   const mcspInt *csr_cols, size_t *buffer_size);
/**
 * @brief    Sort a matrix in CSR format
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in/out]    pointer to the (sorted) col offset in CSR matrix A
 * @param perm               [in/out]    pointer to the integer array of sorted map indices
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsrSort(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspMatDescr_t mcsp_descr_A,
                         const mcspInt *csr_rows, mcspInt *csr_cols, mcspInt *perm, void *temp_buffer);

mcspStatus_t mcspCuinXcsrsort_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, const int *csr_rows,
                                          const int *csr_cols, size_t *buffer_size);

mcspStatus_t mcspCuinXcsrsort(mcspHandle_t handle, int m, int n, int nnz, mcspMatDescr_t mcsp_descr_A,
                            const int *csr_rows, int *csr_cols, int *perm, void *temp_buffer);

/**
 * @brief   Backward transformation from sorted CSR to unsorted CSR
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals           [in/out]    pointer to the (unsorted) value list in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in/out]    pointer to the (unsorted) col offset in CSR matrix A
 * @param info               [in]        information for csr to unsorted csr
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsr2csru(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           float *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols, mcspCsru2csrInfo_t info,
                           void *temp_buffer);

mcspStatus_t mcspDcsr2csru(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           double *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols, mcspCsru2csrInfo_t info,
                           void *temp_buffer);

mcspStatus_t mcspCcsr2csru(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           mcspComplexFloat *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                           mcspCsru2csrInfo_t info, void *temp_buffer);

mcspStatus_t mcspZcsr2csru(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           mcspComplexDouble *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                           mcspCsru2csrInfo_t info, void *temp_buffer);

mcspStatus_t mcspCuinScsr2csru(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             float *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer);

mcspStatus_t mcspCuinDcsr2csru(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             double *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer);

mcspStatus_t mcspCuinCcsr2csru(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             mcspComplexFloat *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer);

mcspStatus_t mcspCuinZcsr2csru(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             mcspComplexDouble *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer);
/**
 * @brief    Get buffer size for CSC sort function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param csc_cols           [in]        pointer to the col offset in CSC matrix A
 * @param csc_rows           [in]        pointer to the row offset in CSC matrix A
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspCscSortBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspInt *csc_cols,
                                   const mcspInt *csc_rows, size_t *buffer_size);

/**
 * @brief    Sort a matrix in CSC format
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSC matrix A
 * @param csc_cols           [in]        pointer to the col offset in CSC matrix A
 * @param csc_rows           [in/out]    pointer to the (sorted)row offset in CSC matrix A
 * @param perm               [in/out]    pointer to the integer array of sorted map indices
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspCscSort(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspMatDescr_t mcsp_descr_A,
                         const mcspInt *csc_cols, mcspInt *csc_rows, mcspInt *perm, void *temp_buffer);

mcspStatus_t mcspCuinXcscsort_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, const int *csc_cols,
                                          const int *csc_rows, size_t *buffer_size);

mcspStatus_t mcspCuinXcscsort(mcspHandle_t handle, int m, int n, int nnz, mcspMatDescr_t mcsp_descr_A,
                            const int *csc_cols, int *csc_rows, int *perm, void *temp_buffer);

/**
 * @brief    Get buffer size for COO sort function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param coo_rows           [in]        pointer to the row offset in COO matrix A
 * @param coo_cols           [in]        pointer to the col offset in COO matrix A
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspCooSortBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspInt *coo_rows,
                                   const mcspInt *coo_cols, size_t *buffer_size);
/**
 * @brief    Sort a matrix in COO format
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param coo_rows           [in/out]    pointer to the (sorted) row offset in COO matrix A
 * @param coo_cols           [in/out]    pointer to the (sorted) col offset in COO matrix A
 * @param perm               [in/out]    pointer to the integer array of sorted map indices
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspCooSortByRow(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspInt *coo_rows,
                              mcspInt *coo_cols, mcspInt *perm, void *temp_buffer);

mcspStatus_t mcspCooSortByColumn(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspInt *coo_rows,
                                 mcspInt *coo_cols, mcspInt *perm, void *temp_buffer);

mcspStatus_t mcspCuinXcoosort_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, const int *coo_rows,
                                          const int *coo_cols, size_t *buffer_size);

mcspStatus_t mcspCuinXcoosortByRow(mcspHandle_t handle, int m, int n, int nnz, int *coo_rows, int *coo_cols, int *perm,
                                 void *temp_buffer);

mcspStatus_t mcspCuinXcoosortByColumn(mcspHandle_t handle, int m, int n, int nnz, int *coo_rows, int *coo_cols, int *perm,
                                    void *temp_buffer);

/**
 * @brief   Convert a sparse CSR matrix into a sparse ELL matrix, calculate ell_width
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param csr_descr          [in]        descriptor of the sparse CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param ell_descr          [in]        descriptor of the sparse ELL matrix A
 * @param ell_width          [out]       pointer to the number of nnz per row in ELL format
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsr2EllWidth(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr, const mcspInt *csr_rows,
                              const mcspMatDescr_t ell_descr, mcspInt *ell_width);

/**
 * @brief   Convert a sparse CSR matrix into a sparse ELL matrix
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param csr_descr          [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals           [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param ell_descr          [in]        descriptor of the sparse ELL matrix A
 * @param ell_width          [in]        pointer to the number of nnz per row in ELL format
 * @param ell_vals           [out]       pointer to the values of element in ELL
 * @param ell_cols           [out]       pointer to the column indexes of element in ELL
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsr2Ell(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr, const float *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspMatDescr_t ell_descr,
                          mcspInt ell_width, float *ell_vals, mcspInt *ell_cols);
mcspStatus_t mcspDcsr2Ell(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr, const double *csr_vals,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, const mcspMatDescr_t ell_descr,
                          mcspInt ell_width, double *ell_vals, mcspInt *ell_cols);
mcspStatus_t mcspCcsr2Ell(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr,
                          const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                          const mcspMatDescr_t ell_descr, mcspInt ell_width, mcspComplexFloat *ell_vals,
                          mcspInt *ell_cols);
mcspStatus_t mcspZcsr2Ell(mcspHandle_t handle, mcspInt m, const mcspMatDescr_t csr_descr,
                          const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                          const mcspMatDescr_t ell_descr, mcspInt ell_width, mcspComplexDouble *ell_vals,
                          mcspInt *ell_cols);

/**
 * @brief   Creates an identity map,the output p = 0:1:(n-1)
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param n                  [in]        size of the map
 * @param p                  [out]       array of elements n
 * @return mcspStatus_t
 */
mcspStatus_t mcspCreateIdentityPermutation(mcspHandle_t handle, mcspInt n, mcspInt *p);
mcspStatus_t mcspCreateIdentityPermutation64(mcspHandle_t handle, int64_t n, int64_t *p);

mcspStatus_t mcspCuinCreateIdentityPermutation(mcspHandle_t handle, int n, int *p);

/**
 * @brief   computes the number of nonzero elements per row or column and the total number of nonzero elements in a
 * dense matrix.
 *
 * @param handle                    [in]        handle of mcSPARSE library
 * @param dir                       [in]        direction that specifies whether to count nonzero elements
 * @param m                         [in]        number of rows of A
 * @param n                         [in]        number of cols of A
 * @param mcsp_descr_A              [in]        descriptor of the sparse matrix A
 * @param dense_matrix              [in]        array of dense matrix
 * @param lda                       [in]        leading dimension of dense matrix
 * @param nnz_per_row_or_column     [out]       pointer to the number of elements greater than zero per row(column)
 * @param nnz                       [out]       number of nonzeros of dense matrix
 * @return mcspStatus_t
 */

mcspStatus_t mcspSnnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                      const mcspMatDescr_t mcsp_descr_A, const float *dense_matrix, mcspInt lda,
                      mcspInt *nnz_per_row_or_column, mcspInt *nnz);

mcspStatus_t mcspDnnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                      const mcspMatDescr_t mcsp_descr_A, const double *dense_matrix, mcspInt lda,
                      mcspInt *nnz_per_row_or_column, mcspInt *nnz);

mcspStatus_t mcspCnnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                      const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *dense_matrix, mcspInt lda,
                      mcspInt *nnz_per_row_or_column, mcspInt *nnz);

mcspStatus_t mcspZnnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                      const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *dense_matrix, mcspInt lda,
                      mcspInt *nnz_per_row_or_column, mcspInt *nnz);
mcspStatus_t mcspCuinSnnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                        const float *dense_matrix, int lda, int *nnz_per_row_or_column, int *nnz);

mcspStatus_t mcspCuinDnnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                        const double *dense_matrix, int lda, int *nnz_per_row_or_column, int *nnz);

mcspStatus_t mcspCuinCnnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                        const mcspComplexFloat *dense_matrix, int lda, int *nnz_per_row_or_column, int *nnz);

mcspStatus_t mcspCuinZnnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                        const mcspComplexDouble *dense_matrix, int lda, int *nnz_per_row_or_column, int *nnz);

// for generic
mcspStatus_t mcspSgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                             mcspInt n, const mcspMatDescr_t mcsp_descr_A, const float *dense_matrix, mcspInt lda,
                             mcspInt *nnz_per_row_or_column, mcspInt *nnz);

mcspStatus_t mcspDgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                             mcspInt n, const mcspMatDescr_t mcsp_descr_A, const double *dense_matrix, mcspInt lda,
                             mcspInt *nnz_per_row_or_column, mcspInt *nnz);

mcspStatus_t mcspCgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                             mcspInt n, const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *dense_matrix,
                             mcspInt lda, mcspInt *nnz_per_row_or_column, mcspInt *nnz);

mcspStatus_t mcspZgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                             mcspInt n, const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *dense_matrix,
                             mcspInt lda, mcspInt *nnz_per_row_or_column, mcspInt *nnz);

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                                mcspInt n, const mcspMatDescr_t mcsp_descr_A, const __half *dense_matrix, mcspInt lda,
                                mcspInt *nnz_per_row_or_column, mcspInt *nnz);
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                                mcspInt n, const mcspMatDescr_t mcsp_descr_A, const __half2 *dense_matrix, mcspInt lda,
                                mcspInt *nnz_per_row_or_column, mcspInt *nnz);

mcspStatus_t mcspR16BFgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                                 mcspInt n, const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16 *dense_matrix,
                                 mcspInt lda, mcspInt *nnz_per_row_or_column, mcspInt *nnz);

mcspStatus_t mcspC16BFgenericNnz(mcspHandle_t handle, mcsparseOrder_t A_order, mcsparseDirection_t B_dir, mcspInt m,
                                 mcspInt n, const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162 *dense_matrix,
                                 mcspInt lda, mcspInt *nnz_per_row_or_column, mcspInt *nnz);
#endif

/**
 * @brief   converts the matrix A in dense format into a sparse matrix in CSR format.
 *
 * @param handle                    [in]        handle of mcSPARSE library
 * @param m                         [in]        number of rows of A
 * @param n                         [in]        number of cols of A
 * @param mcsp_descr_A              [in]        descriptor of the sparse CSR matrix A
 * @param dense_matrix              [in]        array of dense matrix
 * @param lda                       [in]        leading dimension of dense matrix
 * @param nnz_per_row_or_column     [in]        pointer to the number of elements greater than zero per row(column)
 * @param csr_vals                  [out]       pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows                  [out]       pointer to the row offset in CSR matrix A
 * @param csr_cols                  [out]       pointer to the column indexes of nonzeros in CSR matrix A
 * @return mcspStatus_t
 */
mcspStatus_t mcspSdense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column, float *csr_vals,
                            mcspInt *csr_rows, mcspInt *csr_cols);

mcspStatus_t mcspDdense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column, double *csr_vals,
                            mcspInt *csr_rows, mcspInt *csr_cols);

mcspStatus_t mcspCdense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column,
                            mcspComplexFloat *csr_vals, mcspInt *csr_rows, mcspInt *csr_cols);

mcspStatus_t mcspZdense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column,
                            mcspComplexDouble *csr_vals, mcspInt *csr_rows, mcspInt *csr_cols);

mcspStatus_t mcspCuinSdense2csr(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const float *dense_matrix, int lda, int *nnz_per_row_or_column, float *csr_vals,
                              int *csr_rows, int *csr_cols);

mcspStatus_t mcspCuinDdense2csr(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const double *dense_matrix, int lda, int *nnz_per_row_or_column, double *csr_vals,
                              int *csr_rows, int *csr_cols);

mcspStatus_t mcspCuinCdense2csr(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat *dense_matrix, int lda, int *nnz_per_row_or_column,
                              mcspComplexFloat *csr_vals, int *csr_rows, int *csr_cols);

mcspStatus_t mcspCuinZdense2csr(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble *dense_matrix, int lda, int *nnz_per_row_or_column,
                              mcspComplexDouble *csr_vals, int *csr_rows, int *csr_cols);

/**
 * @brief   converts the matrix A in dense format into a sparse matrix in CSC format.
 *
 * @param handle                    [in]        handle of mcSPARSE library
 * @param m                         [in]        number of rows of A
 * @param n                         [in]        number of cols of A
 * @param mcsp_descr_A              [in]        descriptor of the sparse CSR matrix A
 * @param dense_matrix              [in]        array of dense matrix
 * @param lda                       [in]        leading dimension of dense matrix
 * @param nnz_per_row_or_column     [in]        pointer to the number of elements greater than zero per row(column)
 * @param csc_vals                  [out]       pointer to the values of nonzeros in CSC matrix A
 * @param csc_rows                  [out]       pointer to the row indexes of nonzeros in CSC matrix A
 * @param csc_cols                  [out]       pointer to the column offset in CSC matrix A
 * @return mcspStatus_t
 */
mcspStatus_t mcspSdense2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column, float *csc_vals,
                            mcspInt *csc_rows, mcspInt *csc_cols);

mcspStatus_t mcspDdense2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column, double *csc_vals,
                            mcspInt *csc_rows, mcspInt *csc_cols);

mcspStatus_t mcspCdense2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column,
                            mcspComplexFloat *csc_vals, mcspInt *csc_rows, mcspInt *csc_cols);

mcspStatus_t mcspZdense2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *dense_matrix, mcspInt lda, mcspInt *nnz_per_row_or_column,
                            mcspComplexDouble *csc_vals, mcspInt *csc_rows, mcspInt *csc_cols);

mcspStatus_t mcspCuinSdense2csc(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const float *dense_matrix, int lda, int *nnz_per_row_or_column, float *csc_vals,
                              int *csc_rows, int *csc_cols);

mcspStatus_t mcspCuinDdense2csc(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const double *dense_matrix, int lda, int *nnz_per_row_or_column, double *csc_vals,
                              int *csc_rows, int *csc_cols);

mcspStatus_t mcspCuinCdense2csc(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat *dense_matrix, int lda, int *nnz_per_row_or_column,
                              mcspComplexFloat *csc_vals, int *csc_rows, int *csc_cols);

mcspStatus_t mcspCuinZdense2csc(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble *dense_matrix, int lda, int *nnz_per_row_or_column,
                              mcspComplexDouble *csc_vals, int *csc_rows, int *csc_cols);

// for generic
mcspStatus_t mcspSgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const float *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, float *csr_vals, mcspInt *csr_rows,
                                   mcspInt *csr_cols);

mcspStatus_t mcspDgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const double *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, double *csr_vals, mcspInt *csr_rows,
                                   mcspInt *csr_cols);

mcspStatus_t mcspCgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, mcspComplexFloat *csr_vals, mcspInt *csr_rows,
                                   mcspInt *csr_cols);

mcspStatus_t mcspZgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *dense_matrix,
                                   mcspInt lda, mcspInt *nnz_per_row_or_column, mcspComplexDouble *csr_vals,
                                   mcspInt *csr_rows, mcspInt *csr_cols);

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row_or_column, __half *csr_vals, mcspInt *csr_rows,
                                      mcspInt *csr_cols);
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half2 *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row_or_column, __half2 *csr_vals, mcspInt *csr_rows,
                                      mcspInt *csr_cols);

mcspStatus_t mcspR16BFgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row_or_column, mcsp_bfloat16 *csr_vals,
                                       mcspInt *csr_rows, mcspInt *csr_cols);

mcspStatus_t mcspC16BFgenericDense2Csr(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row_or_column, mcsp_bfloat162 *csr_vals,
                                       mcspInt *csr_rows, mcspInt *csr_cols);
#endif

mcspStatus_t mcspSgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const float *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, float *csc_vals, mcspInt *csc_rows,
                                   mcspInt *csc_cols);

mcspStatus_t mcspDgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const double *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, double *csc_vals, mcspInt *csc_rows,
                                   mcspInt *csc_cols);

mcspStatus_t mcspCgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row_or_column, mcspComplexFloat *csc_vals, mcspInt *csc_rows,
                                   mcspInt *csc_cols);

mcspStatus_t mcspZgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *dense_matrix,
                                   mcspInt lda, mcspInt *nnz_per_row_or_column, mcspComplexDouble *csc_vals,
                                   mcspInt *csc_rows, mcspInt *csc_cols);

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row_or_column, __half *csc_vals, mcspInt *csc_rows,
                                      mcspInt *csc_cols);
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half2 *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row_or_column, __half2 *csc_vals, mcspInt *csc_rows,
                                      mcspInt *csc_cols);

mcspStatus_t mcspR16BFgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row_or_column, mcsp_bfloat16 *csc_vals,
                                       mcspInt *csc_rows, mcspInt *csc_cols);

mcspStatus_t mcspC16BFgenericDense2Csc(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row_or_column, mcsp_bfloat162 *csc_vals,
                                       mcspInt *csc_rows, mcspInt *csc_cols);
#endif

/**
 * @brief   converts the matrix A in dense format into a sparse matrix in COO format.
 *
 * @param handle                    [in]        handle of mcSPARSE library
 * @param m                         [in]        number of rows of A
 * @param n                         [in]        number of cols of A
 * @param mcsp_descr_A              [in]        descriptor of the sparse CSR matrix A
 * @param dense_matrix              [in]        array of dense matrix
 * @param lda                       [in]        leading dimension of dense matrix
 * @param nnz_per_row               [in]        pointer to the number of elements greater than zero per row
 * @param coo_vals                  [out]       pointer to the values of nonzeros in COO matrix A
 * @param coo_rows                  [out]       pointer to the row indexes of nonzeros in COO matrix A
 * @param coo_cols                  [out]       pointer to the column indexes of nonzeros in COO matrix A
 * @return mcspStatus_t
 */

mcspStatus_t mcspSdense2Coo(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *dense_matrix, mcspInt lda, mcspInt *nnz_per_row, float *coo_vals,
                            mcspInt *coo_rows, mcspInt *coo_cols);

mcspStatus_t mcspDdense2Coo(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *dense_matrix, mcspInt lda, mcspInt *nnz_per_row, double *coo_vals,
                            mcspInt *coo_rows, mcspInt *coo_cols);

mcspStatus_t mcspCdense2Coo(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *dense_matrix, mcspInt lda, mcspInt *nnz_per_row,
                            mcspComplexFloat *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols);

mcspStatus_t mcspZdense2Coo(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *dense_matrix, mcspInt lda, mcspInt *nnz_per_row,
                            mcspComplexDouble *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols);

// for generic
mcspStatus_t mcspSgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const float *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row, float *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols);

mcspStatus_t mcspDgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const double *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row, double *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols);

mcspStatus_t mcspCgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *dense_matrix, mcspInt lda,
                                   mcspInt *nnz_per_row, mcspComplexFloat *coo_vals, mcspInt *coo_rows,
                                   mcspInt *coo_cols);

mcspStatus_t mcspZgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *dense_matrix,
                                   mcspInt lda, mcspInt *nnz_per_row, mcspComplexDouble *coo_vals, mcspInt *coo_rows,
                                   mcspInt *coo_cols);

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row, __half *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols);
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t mcsp_descr_A, const __half2 *dense_matrix, mcspInt lda,
                                      mcspInt *nnz_per_row, __half2 *coo_vals, mcspInt *coo_rows, mcspInt *coo_cols);

mcspStatus_t mcspR16BFgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row, mcsp_bfloat16 *coo_vals, mcspInt *coo_rows,
                                       mcspInt *coo_cols);

mcspStatus_t mcspC16BFgenericDense2Coo(mcspHandle_t handle, mcsparseOrder_t A_order, mcspInt m, mcspInt n,
                                       const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162 *dense_matrix,
                                       mcspInt lda, mcspInt *nnz_per_row, mcsp_bfloat162 *coo_vals, mcspInt *coo_rows,
                                       mcspInt *coo_cols);
#endif

/**
 * @brief   converts the sparse matrix in COO format into a matrix in dense format.
 *
 * @param handle                    [in]        handle of mcSPARSE library
 * @param m                         [in]        number of rows of A
 * @param n                         [in]        number of cols of A
 * @param nnz                       [in]        number of nonzeros of dense matrix
 * @param mcsp_descr_A              [in]        descriptor of the sparse COO matrix A
 * @param coo_vals                  [in]        pointer to the values of nonzeros in COO matrix A
 * @param coo_rows                  [in]        pointer to the row indexes of nonzeros in COO matrix A
 * @param coo_cols                  [in]        pointer to the column indexes of nonzeros in COO matrix A
 * @param dense_matrix              [out]       array of dense matrix
 * @param lda                       [in]        leading dimension of dense matrix
 * @return mcspStatus_t
 */
mcspStatus_t mcspScoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                            const float *coo_vals, const mcspInt *coo_rows, const mcspInt *coo_cols,
                            float *dense_matrix, mcspInt lda);

mcspStatus_t mcspDcoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                            const double *coo_vals, const mcspInt *coo_rows, const mcspInt *coo_cols,
                            double *dense_matrix, mcspInt lda);

mcspStatus_t mcspCcoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *coo_vals, const mcspInt *coo_rows, const mcspInt *coo_cols,
                            mcspComplexFloat *dense_matrix, mcspInt lda);

mcspStatus_t mcspZcoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *coo_vals, const mcspInt *coo_rows, const mcspInt *coo_cols,
                            mcspComplexDouble *dense_matrix, mcspInt lda);

// for generic
mcspStatus_t mcspSgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                   const mcspMatDescr_t mcsp_descr_A, const float *coo_vals, const mcspInt *coo_rows,
                                   const mcspInt *coo_cols, float *dense_matrix, mcspInt lda, mcsparseOrder_t order_B);

mcspStatus_t mcspDgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                   const mcspMatDescr_t mcsp_descr_A, const double *coo_vals, const mcspInt *coo_rows,
                                   const mcspInt *coo_cols, double *dense_matrix, mcspInt lda, mcsparseOrder_t order_B);

mcspStatus_t mcspCgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexFloat *coo_vals,
                                   const mcspInt *coo_rows, const mcspInt *coo_cols, mcspComplexFloat *dense_matrix,
                                   mcspInt lda, mcsparseOrder_t order_B);

mcspStatus_t mcspZgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                   const mcspMatDescr_t mcsp_descr_A, const mcspComplexDouble *coo_vals,
                                   const mcspInt *coo_rows, const mcspInt *coo_cols, mcspComplexDouble *dense_matrix,
                                   mcspInt lda, mcsparseOrder_t order_B);

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                      const mcspMatDescr_t mcsp_descr_A, const __half *coo_vals,
                                      const mcspInt *coo_rows, const mcspInt *coo_cols, __half *dense_matrix,
                                      mcspInt lda, mcsparseOrder_t order_B);
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                      const mcspMatDescr_t mcsp_descr_A, const __half2 *coo_vals,
                                      const mcspInt *coo_rows, const mcspInt *coo_cols, __half2 *dense_matrix,
                                      mcspInt lda, mcsparseOrder_t order_B);

mcspStatus_t mcspR16BgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                      const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat16 *coo_vals,
                                      const mcspInt *coo_rows, const mcspInt *coo_cols, mcsp_bfloat16 *dense_matrix,
                                      mcspInt lda, mcsparseOrder_t order_B);

mcspStatus_t mcspC16BgenericCoo2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                      const mcspMatDescr_t mcsp_descr_A, const mcsp_bfloat162 *coo_vals,
                                      const mcspInt *coo_rows, const mcspInt *coo_cols, mcsp_bfloat162 *dense_matrix,
                                      mcspInt lda, mcsparseOrder_t order_B);
#endif
/**
 * @brief   converts the sparse matrix in CSR format into a matrix in dense format.
 *
 * @param handle                    [in]        handle of mcSPARSE library
 * @param m                         [in]        number of rows of A
 * @param n                         [in]        number of cols of A
 * @param mcsp_descr_A              [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals                  [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows                  [in]        pointer to the row indexes of nonzeros in CSR matrix A
 * @param csr_cols                  [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param dense_matrix              [out]       array of dense matrix
 * @param lda                       [in]        leading dimension of dense matrix
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                            float *dense_matrix, mcspInt lda);

mcspStatus_t mcspDcsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                            double *dense_matrix, mcspInt lda);

mcspStatus_t mcspCcsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                            mcspComplexFloat *dense_matrix, mcspInt lda);

mcspStatus_t mcspZcsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                            mcspComplexDouble *dense_matrix, mcspInt lda);

mcspStatus_t mcspCuinScsr2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const float *csr_vals, const int *csr_rows, const int *csr_cols, float *dense_matrix,
                              int lda);

mcspStatus_t mcspCuinDcsr2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const double *csr_vals, const int *csr_rows, const int *csr_cols, double *dense_matrix,
                              int lda);

mcspStatus_t mcspCuinCcsr2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat *csr_vals, const int *csr_rows, const int *csr_cols,
                              mcspComplexFloat *dense_matrix, int lda);

mcspStatus_t mcspCuinZcsr2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble *csr_vals, const int *csr_rows, const int *csr_cols,
                              mcspComplexDouble *dense_matrix, int lda);

// for generic
mcspStatus_t mcspSgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                   float *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspDgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                   double *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspCgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                   mcspComplexFloat *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspZgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                   mcspComplexDouble *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const __half *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                      __half *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const __half2 *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                      __half2 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspR16BgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const mcsp_bfloat16 *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                      mcsp_bfloat16 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspC16BgenericCsr2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const mcsp_bfloat162 *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                      mcsp_bfloat162 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);
#endif

/**
 * @brief   converts the sparse matrix in CSC format into a matrix in dense format.
 *
 * @param handle                    [in]        handle of mcSPARSE library
 * @param m                         [in]        number of rows of A
 * @param n                         [in]        number of cols of A
 * @param mcsp_descr_A              [in]        descriptor of the sparse CSC matrix A
 * @param csc_vals                  [in]        pointer to the values of nonzeros in CSC matrix A
 * @param csc_rows                  [in]        pointer to the row indexes of nonzeros in CSC matrix A
 * @param csc_cols                  [in]        pointer to the column indexes of nonzeros in CSC matrix A
 * @param dense_matrix              [out]       array of dense matrix
 * @param lda                       [in]        leading dimension of dense matrix
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const float *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                            float *dense_matrix, mcspInt lda);

mcspStatus_t mcspDcsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const double *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                            double *dense_matrix, mcspInt lda);

mcspStatus_t mcspCcsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexFloat *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                            mcspComplexFloat *dense_matrix, mcspInt lda);

mcspStatus_t mcspZcsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                            const mcspComplexDouble *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                            mcspComplexDouble *dense_matrix, mcspInt lda);

mcspStatus_t mcspCuinScsc2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const float *csc_vals, const int *csc_rows, const int *csc_cols, float *dense_matrix,
                              int lda);

mcspStatus_t mcspCuinDcsc2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const double *csc_vals, const int *csc_rows, const int *csc_cols, double *dense_matrix,
                              int lda);

mcspStatus_t mcspCuinCcsc2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexFloat *csc_vals, const int *csc_rows, const int *csc_cols,
                              mcspComplexFloat *dense_matrix, int lda);

mcspStatus_t mcspCuinZcsc2dense(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                              const mcspComplexDouble *csc_vals, const int *csc_rows, const int *csc_cols,
                              mcspComplexDouble *dense_matrix, int lda);

// for generic
mcspStatus_t mcspSgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const float *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                   float *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspDgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const double *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                   double *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspCgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const mcspComplexFloat *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                   mcspComplexFloat *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspZgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                   const mcspComplexDouble *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                   mcspComplexDouble *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

#if defined(__MACA__)
mcspStatus_t mcspR16FgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const __half *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                      __half *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);
#endif

#ifdef __MACA__
mcspStatus_t mcspC16FgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const __half2 *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                      __half2 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspR16BgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const mcsp_bfloat16 *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                      mcsp_bfloat16 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);

mcspStatus_t mcspC16BgenericCsc2Dense(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                      const mcsp_bfloat162 *csc_vals, const mcspInt *csc_rows, const mcspInt *csc_cols,
                                      mcsp_bfloat162 *dense_matrix, mcspInt lda, mcsparseOrder_t B_order);
#endif
/**
 * @brief    Get buffer size for prune dense to csr function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param dense_matrix       [in]        array of dense matrix
 * @param lda                [in]        leading dimension of dense matrix
 * @param threshold          [in]        non-negative tolerence to determine if a number less than or equal to it
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals           [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the col offset in CSR matrix A
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneDense2CsrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix,
                                           mcspInt lda, const float *threshold, const mcspMatDescr_t mcsp_descr_A,
                                           const float *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                           size_t *buffer_size);

mcspStatus_t mcspDpruneDense2CsrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix,
                                           mcspInt lda, const double *threshold, const mcspMatDescr_t mcsp_descr_A,
                                           const double *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                           size_t *buffer_size);
/**
 * @brief      Get the number of elements greater than tol per row and the row offset in CSR matrix A
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param dense_matrix       [in]        array of dense matrix
 * @param lda                [in]        leading dimension of dense matrix
 * @param threshold          [in]        non-negative tolerence to determine if a number less than or equal to it
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_rows           [out]       pointer to the row offset in CSR matrix A
 * @param nnz                [out]       number of nonzeros of dense matrix
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneDense2CsrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix, mcspInt lda,
                                    const float *threshold, const mcspMatDescr_t mcsp_descr_A, mcspInt *csr_rows,
                                    mcspInt *nnz, void *temp_buffer);

mcspStatus_t mcspDpruneDense2CsrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix, mcspInt lda,
                                    const double *threshold, const mcspMatDescr_t mcsp_descr_A, mcspInt *csr_rows,
                                    mcspInt *nnz, void *temp_buffer);

/**
 * @brief    converts the matrix A in dense format into a sparse matrix in CSR format with threshold.
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param dense_matrix       [in]        array of dense matrix
 * @param lda                [in]        leading dimension of dense matrix
 * @param threshold          [in]        non-negative tolerence to determine if a number less than or equal to it
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals           [out]       pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [out]       pointer to the col offset in CSR matrix A
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneDense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix, mcspInt lda,
                                 const float *threshold, const mcspMatDescr_t mcsp_descr_A, float *csr_vals,
                                 const mcspInt *csr_rows, mcspInt *csr_cols, void *temp_buffer);

mcspStatus_t mcspDpruneDense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix, mcspInt lda,
                                 const double *threshold, const mcspMatDescr_t mcsp_descr_A, double *csr_vals,
                                 const mcspInt *csr_rows, mcspInt *csr_cols, void *temp_buffer);

mcspStatus_t mcspCuinSpruneDense2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                                 const float *threshold, const mcspMatDescr_t mcsp_descr_A,
                                                 const float *csr_vals, const int *csr_rows, const int *csr_cols,
                                                 size_t *buffer_size);

mcspStatus_t mcspCuinDpruneDense2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                                 const double *threshold, const mcspMatDescr_t mcsp_descr_A,
                                                 const double *csr_vals, const int *csr_rows, const int *csr_cols,
                                                 size_t *buffer_size);

mcspStatus_t mcspCuinSpruneDense2csrNnz(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                      const float *threshold, const mcspMatDescr_t mcsp_descr_A, int *csr_rows,
                                      int *nnz, void *temp_buffer);

mcspStatus_t mcspCuinDpruneDense2csrNnz(mcspHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                      const double *threshold, const mcspMatDescr_t mcsp_descr_A, int *csr_rows,
                                      int *nnz, void *temp_buffer);

mcspStatus_t mcspCuinSpruneDense2csr(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                   const float *threshold, const mcspMatDescr_t mcsp_descr_A, float *csr_vals,
                                   const int *csr_rows, int *csr_cols, void *temp_buffer);

mcspStatus_t mcspCuinDpruneDense2csr(mcspHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                   const double *threshold, const mcspMatDescr_t mcsp_descr_A, double *csr_vals,
                                   const int *csr_rows, int *csr_cols, void *temp_buffer);

#if defined(__MACA__)
mcspStatus_t mcspHpruneDense2CsrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix,
                                           mcspInt lda, const __half *threshold, const mcspMatDescr_t mcsp_descr_A,
                                           const __half *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                                           size_t *buffer_size);

mcspStatus_t mcspHpruneDense2CsrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix, mcspInt lda,
                                    const __half *threshold, const mcspMatDescr_t mcsp_descr_A, mcspInt *csr_rows,
                                    mcspInt *nnz, void *temp_buffer);

mcspStatus_t mcspHpruneDense2Csr(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix, mcspInt lda,
                                 const __half *threshold, const mcspMatDescr_t mcsp_descr_A, __half *csr_vals,
                                 const mcspInt *csr_rows, mcspInt *csr_cols, void *temp_buffer);
#endif

/**
 * @brief    Get buffer size for prune dense to csr by percentage function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param dense_matrix       [in]        array of dense matrix
 * @param lda                [in]        leading dimension of dense matrix
 * @param percentage         [in]        pruning percentage for a dense matrix
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals           [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the col offset in CSR matrix A
 * @param info               [in]
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneDense2CsrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n,
                                                       const float *dense_matrix, mcspInt lda, float percentage,
                                                       const mcspMatDescr_t mcsp_descr_A, const float *csr_vals,
                                                       const mcspInt *csr_rows, const mcspInt *csr_cols,
                                                       const mcspPruneInfo_t info, size_t *buffer_size);

mcspStatus_t mcspDpruneDense2CsrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n,
                                                       const double *dense_matrix, mcspInt lda, double percentage,
                                                       const mcspMatDescr_t mcsp_descr_A, const double *csr_vals,
                                                       const mcspInt *csr_rows, const mcspInt *csr_cols,
                                                       const mcspPruneInfo_t info, size_t *buffer_size);

/**
 * @brief      Get the number of elements greater than tol and the row offset in CSR matrix A
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param dense_matrix       [in]        array of dense matrix
 * @param lda                [in]        leading dimension of dense matrix
 * @param percentage         [in]        pruning percentage for a dense matrix
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_rows           [out]       pointer to the row offset in CSR matrix A
 * @param nnz                [out]       number of nonzeros of dense matrix
 * @param info               [in]
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneDense2CsrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix,
                                                mcspInt lda, float percentage, const mcspMatDescr_t mcsp_descr_A,
                                                mcspInt *csr_rows, mcspInt *nnz, const mcspPruneInfo_t info,
                                                void *temp_buffer);

mcspStatus_t mcspDpruneDense2CsrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix,
                                                mcspInt lda, double percentage, const mcspMatDescr_t mcsp_descr_A,
                                                mcspInt *csr_rows, mcspInt *nnz, const mcspPruneInfo_t info,
                                                void *temp_buffer);

/**
 * @brief    prunes the matrix A in dense format into a sparse matrix in CSR format with threshold.
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param dense_matrix       [in]        array of dense matrix
 * @param lda                [in]        leading dimension of dense matrix
 * @param percentage         [in]        pruning percentage for a dense matrix
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals           [out]       pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [out]       pointer to the col offset in CSR matrix A
 * @param info               [in]
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneDense2CsrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const float *dense_matrix,
                                             mcspInt lda, float percentage, const mcspMatDescr_t mcsp_descr_A,
                                             float *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                                             const mcspPruneInfo_t info, void *temp_buffer);

mcspStatus_t mcspDpruneDense2CsrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const double *dense_matrix,
                                             mcspInt lda, double percentage, const mcspMatDescr_t mcsp_descr_A,
                                             double *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                                             const mcspPruneInfo_t info, void *temp_buffer);

mcspStatus_t mcspCuinSpruneDense2csrByPercentage_bufferSizeExt(mcspHandle_t handle, int m, int n,
                                                             const float *dense_matrix, int lda, float percentage,
                                                             const mcspMatDescr_t mcsp_descr_A, const float *csr_vals,
                                                             const int *csr_rows, const int *csr_cols,
                                                             const mcspPruneInfo_t info, size_t *buffer_size);

mcspStatus_t mcspCuinDpruneDense2csrByPercentage_bufferSizeExt(mcspHandle_t handle, int m, int n,
                                                             const double *dense_matrix, int lda, double percentage,
                                                             const mcspMatDescr_t mcsp_descr_A, const double *csr_vals,
                                                             const int *csr_rows, const int *csr_cols,
                                                             const mcspPruneInfo_t info, size_t *buffer_size);

mcspStatus_t mcspCuinSpruneDense2csrNnzByPercentage(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                                  float percentage, const mcspMatDescr_t mcsp_descr_A, int *csr_rows,
                                                  int *nnz, const mcspPruneInfo_t info, void *temp_buffer);

mcspStatus_t mcspCuinDpruneDense2csrNnzByPercentage(mcspHandle_t handle, int m, int n, const double *dense_matrix,
                                                  int lda, double percentage, const mcspMatDescr_t mcsp_descr_A,
                                                  int *csr_rows, int *nnz, const mcspPruneInfo_t info,
                                                  void *temp_buffer);

mcspStatus_t mcspCuinSpruneDense2csrByPercentage(mcspHandle_t handle, int m, int n, const float *dense_matrix, int lda,
                                               float percentage, const mcspMatDescr_t mcsp_descr_A, float *csr_vals,
                                               const int *csr_rows, int *csr_cols, const mcspPruneInfo_t info,
                                               void *temp_buffer);

mcspStatus_t mcspCuinDpruneDense2csrByPercentage(mcspHandle_t handle, int m, int n, const double *dense_matrix, int lda,
                                               double percentage, const mcspMatDescr_t mcsp_descr_A, double *csr_vals,
                                               const int *csr_rows, int *csr_cols, const mcspPruneInfo_t info,
                                               void *temp_buffer);

#if defined(__MACA__)
mcspStatus_t mcspHpruneDense2CsrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n,
                                                       const __half *dense_matrix, mcspInt lda, float percentage,
                                                       const mcspMatDescr_t mcsp_descr_A, const __half *csr_vals,
                                                       const mcspInt *csr_rows, const mcspInt *csr_cols,
                                                       const mcspPruneInfo_t info, size_t *buffer_size);

mcspStatus_t mcspHpruneDense2CsrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix,
                                                mcspInt lda, float percentage, const mcspMatDescr_t mcsp_descr_A,
                                                mcspInt *csr_rows, mcspInt *nnz, const mcspPruneInfo_t info,
                                                void *temp_buffer);

mcspStatus_t mcspHpruneDense2CsrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, const __half *dense_matrix,
                                             mcspInt lda, float percentage, const mcspMatDescr_t mcsp_descr_A,
                                             __half *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                                             const mcspPruneInfo_t info, void *temp_buffer);
#endif

/**
 * @brief    Get buffer size for prune csr to csr function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz_A              [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_val_A          [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_row_A          [in]        pointer to the row offset in CSR matrix A
 * @param csr_col_A          [in]        pointer to the col offset in CSR matrix A
 * @param tol                [in]        non-negative tolerence to determine if a number less than or equal to it
 * @param mcsp_descr_C       [in]        descriptor of the sparse CSR matrix C
 * @param csr_val_C          [in]        pointer to the values of nonzeros in CSR matrix C
 * @param csr_row_C          [in]        pointer to the row offset in CSR matrix C
 * @param csr_col_C          [in]        pointer to the col offset in CSR matrix C
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneCsr2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                         const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A,
                                         const mcspInt *csr_row_A, const mcspInt *csr_col_A, const float *tol,
                                         const mcspMatDescr_t mcsp_descr_C, const float *csr_val_C,
                                         const mcspInt *csr_row_C, const mcspInt *csr_col_C, size_t *buffer_size);

mcspStatus_t mcspDpruneCsr2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                         const mcspMatDescr_t mcsp_descr_A, const double *csr_val_A,
                                         const mcspInt *csr_row_A, const mcspInt *csr_col_A, const double *tol,
                                         const mcspMatDescr_t mcsp_descr_C, const double *csr_val_C,
                                         const mcspInt *csr_row_C, const mcspInt *csr_col_C, size_t *buffer_size);

/**
 * @brief    Get the number of elements greater than tol per row and the row offset in CSR matrix A
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz_A              [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_val_A          [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_row_A          [in]        pointer to the row offset in CSR matrix A
 * @param csr_col_A          [in]        pointer to the col offset in CSR matrix A
 * @param tol                [in]        non-negative tolerence to determine if a number less than or equal to it
 * @param mcsp_descr_C       [in]        descriptor of the sparse CSR matrix C
 * @param csr_row_C          [out]       pointer to the row offset in CSR matrix C
 * @param nnz_C              [out]       number of nonzeros of C
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneCsr2csrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                  const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A, const mcspInt *csr_row_A,
                                  const mcspInt *csr_col_A, const float *tol, const mcspMatDescr_t mcsp_descr_C,
                                  mcspInt *csr_row_C, mcspInt *nnz_C, void *temp_buffer);

mcspStatus_t mcspDpruneCsr2csrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                  const mcspMatDescr_t mcsp_descr_A, const double *csr_val_A, const mcspInt *csr_row_A,
                                  const mcspInt *csr_col_A, const double *tol, const mcspMatDescr_t mcsp_descr_C,
                                  mcspInt *csr_row_C, mcspInt *nnz_C, void *temp_buffer);

/**
 * @brief    Prune the matrix A in CSR format with threshold.
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz_A              [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_val_A          [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_row_A          [in]        pointer to the row offset in CSR matrix A
 * @param csr_col_A          [in]        pointer to the col offset in CSR matrix A
 * @param tol                [in]        non-negative tolerence to determine if a number less than or equal to it
 * @param mcsp_descr_C       [in]        descriptor of the sparse CSR matrix C
 * @param csr_val_C          [out]       pointer to the values of nonzeros in CSR matrix C
 * @param csr_row_C          [in]        pointer to the row offset in CSR matrix C
 * @param csr_col_C          [out]       pointer to the col offset in CSR matrix C
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneCsr2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                               const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A, const mcspInt *csr_row_A,
                               const mcspInt *csr_col_A, const float *tol, const mcspMatDescr_t mcsp_descr_C,
                               float *csr_val_C, const mcspInt *csr_row_C, mcspInt *csr_col_C, void *temp_buffer);

mcspStatus_t mcspDpruneCsr2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                               const mcspMatDescr_t mcsp_descr_A, const double *csr_val_A, const mcspInt *csr_row_A,
                               const mcspInt *csr_col_A, const double *tol, const mcspMatDescr_t mcsp_descr_C,
                               double *csr_val_C, const mcspInt *csr_row_C, mcspInt *csr_col_C, void *temp_buffer);

mcspStatus_t mcspCuinSpruneCsr2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz_A,
                                               const mcspMatDescr_t mcsp_descr_A, const float *csr_val_A,
                                               const int *csr_row_A, const int *csr_col_A, const float *tol,
                                               const mcspMatDescr_t mcsp_descr_C, const float *csr_val_C,
                                               const int *csr_row_C, const int *csr_col_C, size_t *buffer_size);

mcspStatus_t mcspCuinDpruneCsr2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz_A,
                                               const mcspMatDescr_t mcsp_descr_A, const double *csr_val_A,
                                               const int *csr_row_A, const int *csr_col_A, const double *tol,
                                               const mcspMatDescr_t mcsp_descr_C, const double *csr_val_C,
                                               const int *csr_row_C, const int *csr_col_C, size_t *buffer_size);

mcspStatus_t mcspCuinSpruneCsr2csrNnz(mcspHandle_t handle, int m, int n, int nnz_A, const mcspMatDescr_t mcsp_descr_A,
                                    const float *csr_val_A, const int *csr_row_A, const int *csr_col_A,
                                    const float *tol, const mcspMatDescr_t mcsp_descr_C, int *csr_row_C, int *nnz_C,
                                    void *temp_buffer);

mcspStatus_t mcspCuinDpruneCsr2csrNnz(mcspHandle_t handle, int m, int n, int nnz_A, const mcspMatDescr_t mcsp_descr_A,
                                    const double *csr_val_A, const int *csr_row_A, const int *csr_col_A,
                                    const double *tol, const mcspMatDescr_t mcsp_descr_C, int *csr_row_C, int *nnz_C,
                                    void *temp_buffer);

mcspStatus_t mcspCuinSpruneCsr2csr(mcspHandle_t handle, int m, int n, int nnz_A, const mcspMatDescr_t mcsp_descr_A,
                                 const float *csr_val_A, const int *csr_row_A, const int *csr_col_A, const float *tol,
                                 const mcspMatDescr_t mcsp_descr_C, float *csr_val_C, const int *csr_row_C,
                                 int *csr_col_C, void *temp_buffer);

mcspStatus_t mcspCuinDpruneCsr2csr(mcspHandle_t handle, int m, int n, int nnz_A, const mcspMatDescr_t mcsp_descr_A,
                                 const double *csr_val_A, const int *csr_row_A, const int *csr_col_A, const double *tol,
                                 const mcspMatDescr_t mcsp_descr_C, double *csr_val_C, const int *csr_row_C,
                                 int *csr_col_C, void *temp_buffer);

#if defined(__MACA__)
mcspStatus_t mcspHpruneCsr2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                         const mcspMatDescr_t mcsp_descr_A, const __half *csr_val_A,
                                         const mcspInt *csr_row_A, const mcspInt *csr_col_A, const __half *tol,
                                         const mcspMatDescr_t mcsp_descr_C, const __half *csr_val_C,
                                         const mcspInt *csr_row_C, const mcspInt *csr_col_C, size_t *buffer_size);

mcspStatus_t mcspHpruneCsr2csrNnz(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                  const mcspMatDescr_t mcsp_descr_A, const __half *csr_val_A, const mcspInt *csr_row_A,
                                  const mcspInt *csr_col_A, const __half *tol, const mcspMatDescr_t mcsp_descr_C,
                                  mcspInt *csr_row_C, mcspInt *nnz_C, void *temp_buffer);

mcspStatus_t mcspHpruneCsr2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                               const mcspMatDescr_t mcsp_descr_A, const __half *csr_val_A, const mcspInt *csr_row_A,
                               const mcspInt *csr_col_A, const __half *tol, const mcspMatDescr_t mcsp_descr_C,
                               __half *csr_val_C, const mcspInt *csr_row_C, mcspInt *csr_col_C, void *temp_buffer);
#endif

/**
 * @brief    Get buffer size for prune csr to csr by percentage function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz_A              [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals_A         [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows_A         [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols_A         [in]        pointer to the col offset in CSR matrix A
 * @param percentage         [in]        pruning percentage for the sparse CSR matrix A
 * @param mcsp_descr_C       [in]        descriptor of the sparse CSR matrix C
 * @param csr_vals_C         [in]        pointer to the values of nonzeros in CSR matrix C
 * @param csr_rows_C         [in]        pointer to the row offset in CSR matrix C
 * @param csr_cols_C         [in]        pointer to the col offset in CSR matrix C
 * @param info               [in]
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneCsr2csrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                                     const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                                     const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                                                     float percentage, const mcspMatDescr_t mcsp_descr_C,
                                                     const float *csr_vals_C, const mcspInt *csr_rows_C,
                                                     const mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                                     size_t *buffer_size);

mcspStatus_t mcspDpruneCsr2csrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                                     const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                                     const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                                                     double percentage, const mcspMatDescr_t mcsp_descr_C,
                                                     const double *csr_vals_C, const mcspInt *csr_rows_C,
                                                     const mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                                     size_t *buffer_size);

/**
 * @brief    Get the number of elements greater than tol and the row offset in CSR matrix A
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz_A              [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals_A         [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows_A         [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols_A         [in]        pointer to the col offset in CSR matrix A
 * @param percentage         [in]        pruning percentage for the sparse CSR matrix A
 * @param mcsp_descr_C       [in]        descriptor of the sparse CSR matrix C
 * @param csr_rows_C         [out]       pointer to the row offset in CSR matrix C
 * @param nnz_C              [out]       number of nonzeros of C
 * @param info               [in]
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneCsr2csrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                              const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                              const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, float percentage,
                                              const mcspMatDescr_t mcsp_descr_C, mcspInt *csr_rows_C, mcspInt *nnz_C,
                                              const mcspPruneInfo_t info, void *temp_buffer);

mcspStatus_t mcspDpruneCsr2csrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                              const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                              const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, double percentage,
                                              const mcspMatDescr_t mcsp_descr_C, mcspInt *csr_rows_C, mcspInt *nnz_C,
                                              const mcspPruneInfo_t info, void *temp_buffer);

/**
 * @brief    Prune the matrix A in CSR format with threshold.
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz_A              [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals_A         [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows_A         [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols_A         [in]        pointer to the col offset in CSR matrix A
 * @param percentage         [in]        pruning percentage for the sparse CSR matrix A
 * @param mcsp_descr_C       [in]        descriptor of the sparse CSR matrix C
 * @param csr_vals_C         [out]       pointer to the values of nonzeros in CSR matrix C
 * @param csr_rows_C         [in]        pointer to the row offset in CSR matrix C
 * @param csr_cols_C         [out]       pointer to the col offset in CSR matrix C
 * @param info               [in]
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSpruneCsr2csrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                           const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                           const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, float percentage,
                                           const mcspMatDescr_t mcsp_descr_C, float *csr_vals_C,
                                           const mcspInt *csr_rows_C, mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                           void *temp_buffer);

mcspStatus_t mcspDpruneCsr2csrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                           const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                           const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, double percentage,
                                           const mcspMatDescr_t mcsp_descr_C, double *csr_vals_C,
                                           const mcspInt *csr_rows_C, mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                           void *temp_buffer);

mcspStatus_t mcspCuinSpruneCsr2csrByPercentage_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz_A,
                                                           const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                                           const int *csr_rows_A, const int *csr_cols_A,
                                                           float percentage, const mcspMatDescr_t mcsp_descr_C,
                                                           const float *csr_vals_C, const int *csr_rows_C,
                                                           const int *csr_cols_C, const mcspPruneInfo_t info,
                                                           size_t *buffer_size);

mcspStatus_t mcspCuinDpruneCsr2csrByPercentage_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz_A,
                                                           const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                                           const int *csr_rows_A, const int *csr_cols_A,
                                                           double percentage, const mcspMatDescr_t mcsp_descr_C,
                                                           const double *csr_vals_C, const int *csr_rows_C,
                                                           const int *csr_cols_C, const mcspPruneInfo_t info,
                                                           size_t *buffer_size);

mcspStatus_t mcspCuinSpruneCsr2csrNnzByPercentage(mcspHandle_t handle, int m, int n, int nnz_A,
                                                const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                                const int *csr_rows_A, const int *csr_cols_A, float percentage,
                                                const mcspMatDescr_t mcsp_descr_C, int *csr_rows_C, int *nnz_C,
                                                const mcspPruneInfo_t info, void *temp_buffer);

mcspStatus_t mcspCuinDpruneCsr2csrNnzByPercentage(mcspHandle_t handle, int m, int n, int nnz_A,
                                                const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                                const int *csr_rows_A, const int *csr_cols_A, double percentage,
                                                const mcspMatDescr_t mcsp_descr_C, int *csr_rows_C, int *nnz_C,
                                                const mcspPruneInfo_t info, void *temp_buffer);

mcspStatus_t mcspCuinSpruneCsr2csrByPercentage(mcspHandle_t handle, int m, int n, int nnz_A,
                                             const mcspMatDescr_t mcsp_descr_A, const float *csr_vals_A,
                                             const int *csr_rows_A, const int *csr_cols_A, float percentage,
                                             const mcspMatDescr_t mcsp_descr_C, float *csr_vals_C,
                                             const int *csr_rows_C, int *csr_cols_C, const mcspPruneInfo_t info,
                                             void *temp_buffer);

mcspStatus_t mcspCuinDpruneCsr2csrByPercentage(mcspHandle_t handle, int m, int n, int nnz_A,
                                             const mcspMatDescr_t mcsp_descr_A, const double *csr_vals_A,
                                             const int *csr_rows_A, const int *csr_cols_A, double percentage,
                                             const mcspMatDescr_t mcsp_descr_C, double *csr_vals_C,
                                             const int *csr_rows_C, int *csr_cols_C, const mcspPruneInfo_t info,
                                             void *temp_buffer);

#if defined(__MACA__)
mcspStatus_t mcspHpruneCsr2csrByPercentageBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                                     const mcspMatDescr_t mcsp_descr_A, const __half *csr_vals_A,
                                                     const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                                                     float percentage, const mcspMatDescr_t mcsp_descr_C,
                                                     const __half *csr_vals_C, const mcspInt *csr_rows_C,
                                                     const mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                                     size_t *buffer_size);

mcspStatus_t mcspHpruneCsr2csrNnzByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                              const mcspMatDescr_t mcsp_descr_A, const __half *csr_vals_A,
                                              const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, float percentage,
                                              const mcspMatDescr_t mcsp_descr_C, mcspInt *csr_rows_C, mcspInt *nnz_C,
                                              const mcspPruneInfo_t info, void *temp_buffer);

mcspStatus_t mcspHpruneCsr2csrByPercentage(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz_A,
                                           const mcspMatDescr_t mcsp_descr_A, const __half *csr_vals_A,
                                           const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, float percentage,
                                           const mcspMatDescr_t mcsp_descr_C, __half *csr_vals_C,
                                           const mcspInt *csr_rows_C, mcspInt *csr_cols_C, const mcspPruneInfo_t info,
                                           void *temp_buffer);
#endif
/**
 * @brief    Get buffer size for CSR sort function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param csr_vals           [in]        pointer to the val offset in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the col offset in CSR matrix A
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */

mcspStatus_t mcspScsru2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, float *csr_vals,
                                     const mcspInt *csr_rows, mcspInt *csr_cols, mcspCsru2csrInfo_t info,
                                     size_t *buffer_size);

mcspStatus_t mcspDcsru2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, double *csr_vals,
                                     const mcspInt *csr_rows, mcspInt *csr_cols, mcspCsru2csrInfo_t info,
                                     size_t *buffer_size);

mcspStatus_t mcspCcsru2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspComplexFloat *csr_vals,
                                     const mcspInt *csr_rows, mcspInt *csr_cols, mcspCsru2csrInfo_t info,
                                     size_t *buffer_size);

mcspStatus_t mcspZcsru2csrBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz,
                                     mcspComplexDouble *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                                     mcspCsru2csrInfo_t info, size_t *buffer_size);

mcspStatus_t mcspCuinScsru2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, float *csr_vals,
                                           const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                                           size_t *buffer_size);

mcspStatus_t mcspCuinDcsru2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, double *csr_vals,
                                           const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                                           size_t *buffer_size);

mcspStatus_t mcspCuinCcsru2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, mcspComplexFloat *csr_vals,
                                           const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                                           size_t *buffer_size);

mcspStatus_t mcspCuinZcsru2csr_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, mcspComplexDouble *csr_vals,
                                           const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                                           size_t *buffer_size);
/**
 * @brief    Wrapper of csrsort and gthr
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param m                  [in]        number of rows of A
 * @param n                  [in]        number of cols of A
 * @param nnz                [in]        number of nonzeros of A
 * @param mcsp_descr_A       [in]        descriptor of the sparse CSR matrix A
 * @param csr_vals           [in]        pointer to the val offset in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the col offset in CSR matrix A
 * @param info               [in/out]    info includes the integer array of sorted map indices
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */

mcspStatus_t mcspScsru2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           float *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols, mcspCsru2csrInfo_t info,
                           void *temp_buffer);

mcspStatus_t mcspDcsru2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           double *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols, mcspCsru2csrInfo_t info,
                           void *temp_buffer);

mcspStatus_t mcspCcsru2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           mcspComplexFloat *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                           mcspCsru2csrInfo_t info, void *temp_buffer);

mcspStatus_t mcspZcsru2csr(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspMatDescr_t mcsp_descr_A,
                           mcspComplexDouble *csr_vals, const mcspInt *csr_rows, mcspInt *csr_cols,
                           mcspCsru2csrInfo_t info, void *temp_buffer);

mcspStatus_t mcspCuinScsru2csr(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             float *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer);

mcspStatus_t mcspCuinDcsru2csr(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             double *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer);

mcspStatus_t mcspCuinCcsru2csr(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             mcspComplexFloat *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer);

mcspStatus_t mcspCuinZcsru2csr(mcspHandle_t handle, int m, int n, int nnz, const mcspMatDescr_t mcsp_descr_A,
                             mcspComplexDouble *csr_vals, const int *csr_rows, int *csr_cols, mcspCsru2csrInfo_t info,
                             void *temp_buffer);

/**
 * @brief    Get buffer size for csr to bsr function
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param dir                [in]        storage format of blocks
 * @param m                  [in]        number of rows of CSR matrix
 * @param n                  [in]        number of cols of CSR matrix
 * @param csr_descr          [in]        descriptor of the sparse CSR matrix
 * @param csr_val            [in]        pointer to the val offset in CSR matrix
 * @param csr_row            [in]        pointer to the row offset in CSR matrix
 * @param csr_col            [in]        pointer to the col offset in CSR matrix
 * @param row_block_dim      [in]        number of rows within a block of BSR matrix
 * @param col_block_dim      [in]        number of columns within a block of BSR matrix
 * @param buffer_size        [out]       number of bytes of buffer needed for calculation
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsr2gebsrBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t csr_descr, const float *csr_val, const mcspInt *csr_row,
                                      const mcspInt *csr_col, mcspInt row_block_dim, mcspInt col_block_dim,
                                      size_t *buffer_size);

mcspStatus_t mcspDcsr2gebsrBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t csr_descr, const double *csr_val, const mcspInt *csr_row,
                                      const mcspInt *csr_col, mcspInt row_block_dim, mcspInt col_block_dim,
                                      size_t *buffer_size);

mcspStatus_t mcspCcsr2gebsrBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val,
                                      const mcspInt *csr_row, const mcspInt *csr_col, mcspInt row_block_dim,
                                      mcspInt col_block_dim, size_t *buffer_size);

mcspStatus_t mcspZcsr2gebsrBufferSize(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                                      const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val,
                                      const mcspInt *csr_row, const mcspInt *csr_col, mcspInt row_block_dim,
                                      mcspInt col_block_dim, size_t *buffer_size);

/**
 * @brief    Get the total number of nonzero blocks of BSR matrix
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param dir                [in]        storage format of blocks
 * @param m                  [in]        number of rows of CSR matrix
 * @param n                  [in]        number of cols of CSR matrix
 * @param csr_descr          [in]        descriptor of the sparse CSR matrix
 * @param csr_row            [in]        pointer to the row offset in CSR matrix
 * @param csr_col            [in]        pointer to the col offset in CSR matrix
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_row            [out]       pointer to the row offset in BSR matrix
 * @param row_block_dim      [in]        number of rows within a block of BSR matrix
 * @param col_block_dim      [in]        number of columns within a block of BSR matrix
 * @param nnzb               [out]       total number of nonzero blocks of BSR matrix
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsr2gebsrNnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                              const mcspMatDescr_t csr_descr, const mcspInt *csr_row, const mcspInt *csr_col,
                              const mcspMatDescr_t bsr_descr, mcspInt *bsr_row, mcspInt row_block_dim,
                              mcspInt col_block_dim, mcspInt *nnzb, void *temp_buffer);

/**
 * @brief    Converts a sparse matrix with CSR format into a sparse matrix with general BSR format
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param dir                [in]        storage format of blocks
 * @param m                  [in]        number of rows of CSR matrix
 * @param n                  [in]        number of cols of CSR matrix
 * @param csr_descr          [in]        descriptor of the sparse CSR matrix
 * @param csr_val            [in]        pointer to the val offset in CSR matrix
 * @param csr_row            [in]        pointer to the row offset in CSR matrix
 * @param csr_col            [in]        pointer to the col offset in CSR matrix
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_val            [out]       pointer to the val offset in BSR matrix
 * @param bsr_row            [out]       pointer to the row offset in BSR matrix
 * @param bsr_col            [out]       pointer to the col offset in BSR matrix
 * @param row_block_dim      [in]        number of rows within a block of BSR matrix
 * @param col_block_dim      [in]        number of columns within a block of BSR matrix
 * @param temp_buffer        [in]        buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const float *csr_val, const mcspInt *csr_row,
                            const mcspInt *csr_col, const mcspMatDescr_t bsr_descr, float *bsr_val, mcspInt *bsr_row,
                            mcspInt *bsr_col, mcspInt row_block_dim, mcspInt col_block_dim, void *temp_buffer);

mcspStatus_t mcspDcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const double *csr_val, const mcspInt *csr_row,
                            const mcspInt *csr_col, const mcspMatDescr_t bsr_descr, double *bsr_val, mcspInt *bsr_row,
                            mcspInt *bsr_col, mcspInt row_block_dim, mcspInt col_block_dim, void *temp_buffer);

mcspStatus_t mcspCcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val, const mcspInt *csr_row,
                            const mcspInt *csr_col, const mcspMatDescr_t bsr_descr, mcspComplexFloat *bsr_val,
                            mcspInt *bsr_row, mcspInt *bsr_col, mcspInt row_block_dim, mcspInt col_block_dim,
                            void *temp_buffer);

mcspStatus_t mcspZcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val, const mcspInt *csr_row,
                            const mcspInt *csr_col, const mcspMatDescr_t bsr_descr, mcspComplexDouble *bsr_val,
                            mcspInt *bsr_row, mcspInt *bsr_col, mcspInt row_block_dim, mcspInt col_block_dim,
                            void *temp_buffer);

mcspStatus_t mcspCuinScsr2gebsr_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                         const mcspMatDescr_t csr_descr, const float *csr_val, const int *csr_row,
                                         const int *csr_col, int row_block_dim, int col_block_dim, int *buffer_size);

mcspStatus_t mcspCuinDcsr2gebsr_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                         const mcspMatDescr_t csr_descr, const double *csr_val, const int *csr_row,
                                         const int *csr_col, int row_block_dim, int col_block_dim, int *buffer_size);

mcspStatus_t mcspCuinCcsr2gebsr_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                         const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val,
                                         const int *csr_row, const int *csr_col, int row_block_dim, int col_block_dim,
                                         int *buffer_size);

mcspStatus_t mcspCuinZcsr2gebsr_bufferSize(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                         const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val,
                                         const int *csr_row, const int *csr_col, int row_block_dim, int col_block_dim,
                                         int *buffer_size);

mcspStatus_t mcspCuinScsr2gebsr_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                            const mcspMatDescr_t csr_descr, const float *csr_val, const int *csr_row,
                                            const int *csr_col, int row_block_dim, int col_block_dim,
                                            size_t *buffer_size);

mcspStatus_t mcspCuinDcsr2gebsr_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                            const mcspMatDescr_t csr_descr, const double *csr_val, const int *csr_row,
                                            const int *csr_col, int row_block_dim, int col_block_dim,
                                            size_t *buffer_size);

mcspStatus_t mcspCuinCcsr2gebsr_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                            const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val,
                                            const int *csr_row, const int *csr_col, int row_block_dim,
                                            int col_block_dim, size_t *buffer_size);

mcspStatus_t mcspCuinZcsr2gebsr_bufferSizeExt(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                            const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val,
                                            const int *csr_row, const int *csr_col, int row_block_dim,
                                            int col_block_dim, size_t *buffer_size);

mcspStatus_t mcspCuinXcsr2gebsrNnz(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                 const mcspMatDescr_t csr_descr, const int *csr_row, const int *csr_col,
                                 const mcspMatDescr_t bsr_descr, int *bsr_row, int row_block_dim, int col_block_dim,
                                 int *nnzb, void *temp_buffer);

mcspStatus_t mcspCuinScsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcspMatDescr_t csr_descr, const float *csr_val, const int *csr_row,
                              const int *csr_col, const mcspMatDescr_t bsr_descr, float *bsr_val, int *bsr_row,
                              int *bsr_col, int row_block_dim, int col_block_dim, void *temp_buffer);

mcspStatus_t mcspCuinDcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcspMatDescr_t csr_descr, const double *csr_val, const int *csr_row,
                              const int *csr_col, const mcspMatDescr_t bsr_descr, double *bsr_val, int *bsr_row,
                              int *bsr_col, int row_block_dim, int col_block_dim, void *temp_buffer);

mcspStatus_t mcspCuinCcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val, const int *csr_row,
                              const int *csr_col, const mcspMatDescr_t bsr_descr, mcspComplexFloat *bsr_val,
                              int *bsr_row, int *bsr_col, int row_block_dim, int col_block_dim, void *temp_buffer);

mcspStatus_t mcspCuinZcsr2gebsr(mcspHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val, const int *csr_row,
                              const int *csr_col, const mcspMatDescr_t bsr_descr, mcspComplexDouble *bsr_val,
                              int *bsr_row, int *bsr_col, int row_block_dim, int col_block_dim, void *temp_buffer);

/**
 * @brief    Get the total number of nonzero blocks of BSR matrix
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param dir                [in]        storage format of blocks
 * @param m                  [in]        number of rows of CSR matrix
 * @param n                  [in]        number of cols of CSR matrix
 * @param csr_descr          [in]        descriptor of the sparse CSR matrix
 * @param csr_row            [in]        pointer to the row offset in CSR matrix
 * @param csr_col            [in]        pointer to the col offset in CSR matrix
 * @param block_dim          [in]        block dimension of sparse BSR matrix
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_row            [out]       pointer to the row offset in BSR matrix
 * @param nnzb               [out]       total number of nonzero blocks of BSR matrix
 * @return mcspStatus_t
 */
mcspStatus_t mcspCsr2bsrNnz(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                            const mcspMatDescr_t csr_descr, const mcspInt *csr_row, const mcspInt *csr_col,
                            mcspInt block_dim, const mcspMatDescr_t bsr_descr, mcspInt *bsr_row, mcspInt *nnzb);

/**
 * @brief    Converts a sparse matrix with CSR format into a sparse matrix with BSR format
 *
 * @param handle             [in]        handle of mcSPARSE library
 * @param dir                [in]        storage format of blocks
 * @param m                  [in]        number of rows of CSR matrix
 * @param n                  [in]        number of cols of CSR matrix
 * @param csr_descr          [in]        descriptor of the sparse CSR matrix
 * @param csr_val            [in]        pointer to the val offset in CSR matrix
 * @param csr_row            [in]        pointer to the row offset in CSR matrix
 * @param csr_col            [in]        pointer to the col offset in CSR matrix
 * @param block_dim          [in]        number of rows within a block of BSR matrix
 * @param bsr_descr          [in]        descriptor of the sparse BSR matrix
 * @param bsr_val            [out]       pointer to the val offset in BSR matrix
 * @param bsr_row            [out]       pointer to the row offset in BSR matrix
 * @param bsr_col            [out]       pointer to the col offset in BSR matrix
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsr2bsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                          const mcspMatDescr_t csr_descr, const float *csr_val, const mcspInt *csr_row,
                          const mcspInt *csr_col, mcspInt block_dim, const mcspMatDescr_t bsr_descr, float *bsr_val,
                          mcspInt *bsr_row, mcspInt *bsr_col);

mcspStatus_t mcspDcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                          const mcspMatDescr_t csr_descr, const double *csr_val, const mcspInt *csr_row,
                          const mcspInt *csr_col, mcspInt block_dim, const mcspMatDescr_t bsr_descr, double *bsr_val,
                          mcspInt *bsr_row, mcspInt *bsr_col);

mcspStatus_t mcspCcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                          const mcspMatDescr_t csr_descr, const mcspComplexFloat *csr_val, const mcspInt *csr_row,
                          const mcspInt *csr_col, mcspInt block_dim, const mcspMatDescr_t bsr_descr,
                          mcspComplexFloat *bsr_val, mcspInt *bsr_row, mcspInt *bsr_col);

mcspStatus_t mcspZcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dir, mcspInt m, mcspInt n,
                          const mcspMatDescr_t csr_descr, const mcspComplexDouble *csr_val, const mcspInt *csr_row,
                          const mcspInt *csr_col, mcspInt block_dim, const mcspMatDescr_t bsr_descr,
                          mcspComplexDouble *bsr_val, mcspInt *bsr_row, mcspInt *bsr_col);

mcspStatus_t mcspCuinXcsr2bsrNnz(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                               const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
                               const mcspMatDescr_t descrC, int *bsrSortedRowPtrC, int *nnzTotalDevHostPtr);

mcspStatus_t mcspCuinScsr2bsr(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                            const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                            int blockDim, const mcspMatDescr_t descrC, float *bsrSortedValC, int *bsrSortedRowPtrC,
                            int *bsrSortedColIndC);

mcspStatus_t mcspCuinDcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                            const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                            int blockDim, const mcspMatDescr_t descrC, double *bsrSortedValC, int *bsrSortedRowPtrC,
                            int *bsrSortedColIndC);

mcspStatus_t mcspCuinCcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                            const mcFloatComplex *csrSortedValA, const int *csrSortedRowPtrA,
                            const int *csrSortedColIndA, int blockDim, const mcspMatDescr_t descrC,
                            mcFloatComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC);

mcspStatus_t mcspCuinZcsr2bsr(mcspHandle_t handle, mcsparseDirection_t dirA, int m, int n, const mcspMatDescr_t descrA,
                            const mcDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
                            const int *csrSortedColIndA, int blockDim, const mcspMatDescr_t descrC,
                            mcDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC);

// gebsr2gebsc
mcspStatus_t mcspSgebsr2gebsc_bufferSize(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                         const float *bsr_vals, const mcspInt *bsr_rows, const mcspInt *bsr_cols,
                                         mcspInt row_block_dim, mcspInt col_block_dim, int *buffer_size);

mcspStatus_t mcspDgebsr2gebsc_bufferSize(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                         const double *bsr_vals, const mcspInt *bsr_rows, const mcspInt *bsr_cols,
                                         mcspInt row_block_dim, mcspInt col_block_dim, int *buffer_size);

mcspStatus_t mcspCgebsr2gebsc_bufferSize(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                         const mcFloatComplex *bsr_vals, const mcspInt *bsr_rows,
                                         const mcspInt *bsr_cols, mcspInt row_block_dim, mcspInt col_block_dim,
                                         int *buffer_size);

mcspStatus_t mcspZgebsr2gebsc_bufferSize(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                         const mcDoubleComplex *bsr_vals, const mcspInt *bsr_rows,
                                         const mcspInt *bsr_cols, mcspInt row_block_dim, mcspInt col_block_dim,
                                         int *buffer_size);

mcspStatus_t mcspSgebsr2gebsc_bufferSizeExt(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                            const float *bsr_vals, const mcspInt *bsr_rows, const mcspInt *bsr_cols,
                                            mcspInt row_block_dim, mcspInt col_block_dim, size_t *buffer_size);

mcspStatus_t mcspDgebsr2gebsc_bufferSizeExt(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                            const double *bsr_vals, const mcspInt *bsr_rows, const mcspInt *bsr_cols,
                                            mcspInt row_block_dim, mcspInt col_block_dim, size_t *buffer_size);

mcspStatus_t mcspCgebsr2gebsc_bufferSizeExt(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                            const mcFloatComplex *bsr_vals, const mcspInt *bsr_rows,
                                            const mcspInt *bsr_cols, mcspInt row_block_dim, mcspInt col_block_dim,
                                            size_t *buffer_size);

mcspStatus_t mcspZgebsr2gebsc_bufferSizeExt(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                                            const mcDoubleComplex *bsr_vals, const mcspInt *bsr_rows,
                                            const mcspInt *bsr_cols, mcspInt row_block_dim, mcspInt col_block_dim,
                                            size_t *buffer_size);

mcspStatus_t mcspSgebsr2gebsc(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb, const float *bsr_vals,
                              const mcspInt *bsr_rows, const mcspInt *bsr_cols, mcspInt row_block_dim,
                              mcspInt col_block_dim, float *bsc_vals, mcspInt *bsc_rows, mcspInt *bsc_cols,
                              mcsparseAction_t copy_value, mcsparseIndexBase_t idxBase, void *buffer);

mcspStatus_t mcspDgebsr2gebsc(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb, const double *bsr_vals,
                              const mcspInt *bsr_rows, const mcspInt *bsr_cols, mcspInt row_block_dim,
                              mcspInt col_block_dim, double *bsc_vals, mcspInt *bsc_rows, mcspInt *bsc_cols,
                              mcsparseAction_t copy_value, mcsparseIndexBase_t idxBase, void *buffer);

mcspStatus_t mcspCgebsr2gebsc(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb, const mcFloatComplex *bsr_vals,
                              const mcspInt *bsr_rows, const mcspInt *bsr_cols, mcspInt row_block_dim,
                              mcspInt col_block_dim, mcFloatComplex *bsc_vals, mcspInt *bsc_rows, mcspInt *bsc_cols,
                              mcsparseAction_t copy_value, mcsparseIndexBase_t idxBase, void *buffer);

mcspStatus_t mcspZgebsr2gebsc(mcspHandle_t handle, mcspInt mb, mcspInt nb, mcspInt nnzb,
                              const mcDoubleComplex *bsr_vals, const mcspInt *bsr_rows, const mcspInt *bsr_cols,
                              mcspInt row_block_dim, mcspInt col_block_dim, mcDoubleComplex *bsc_vals,
                              mcspInt *bsc_rows, mcspInt *bsc_cols, mcsparseAction_t copy_value,
                              mcsparseIndexBase_t idxBase, void *buffer);

#ifdef __cplusplus
}
#endif

#endif
