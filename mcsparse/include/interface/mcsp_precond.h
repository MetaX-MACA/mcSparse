#ifndef INTERFACE_MCSPARSE_PRECOND_H_
#define INTERFACE_MCSPARSE_PRECOND_H_

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// csrilu02
mcsparseStatus_t mcsparseXcsrilu02_zeroPivot(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int* position);

mcsparseStatus_t mcsparseScsrilu02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                              float* csrSortedValA, const int* csrSortedRowPtrA,
                                              const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                              int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseDcsrilu02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                              double* csrSortedValA, const int* csrSortedRowPtrA,
                                              const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                              int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseCcsrilu02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                              mcComplex* csrSortedValA, const int* csrSortedRowPtrA,
                                              const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                              int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseZcsrilu02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                              mcDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
                                              const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                              int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseScsrilu02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz,
                                                 const mcsparseMatDescr_t descrA, float* csrSortedVal,
                                                 const int* csrSortedRowPtr, const int* csrSortedColInd,
                                                 mcsparseCsrilu02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseDcsrilu02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz,
                                                 const mcsparseMatDescr_t descrA, double* csrSortedVal,
                                                 const int* csrSortedRowPtr, const int* csrSortedColInd,
                                                 mcsparseCsrilu02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseCcsrilu02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz,
                                                 const mcsparseMatDescr_t descrA, mcComplex* csrSortedVal,
                                                 const int* csrSortedRowPtr, const int* csrSortedColInd,
                                                 mcsparseCsrilu02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseZcsrilu02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz,
                                                 const mcsparseMatDescr_t descrA, mcDoubleComplex* csrSortedVal,
                                                 const int* csrSortedRowPtr, const int* csrSortedColInd,
                                                 mcsparseCsrilu02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseScsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int enable_boost,
                                                double* tol, float* boost_val);

mcsparseStatus_t mcsparseDcsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int enable_boost,
                                                double* tol, double* boost_val);

mcsparseStatus_t mcsparseCcsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int enable_boost,
                                                double* tol, mcComplex* boost_val);

mcsparseStatus_t mcsparseZcsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int enable_boost,
                                                double* tol, mcDoubleComplex* boost_val);

mcsparseStatus_t mcsparseScsrilu02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                            const float* csrSortedValA, const int* csrSortedRowPtrA,
                                            const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                            mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseDcsrilu02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                            const double* csrSortedValA, const int* csrSortedRowPtrA,
                                            const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                            mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseCcsrilu02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                            const mcComplex* csrSortedValA, const int* csrSortedRowPtrA,
                                            const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                            mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseZcsrilu02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                            const mcDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
                                            const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                            mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseScsrilu02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                   float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   mcsparseCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseDcsrilu02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                   double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   mcsparseCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseCcsrilu02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                   mcComplex* csrSortedValA_valM, const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseZcsrilu02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                   mcDoubleComplex* csrSortedValA_valM, const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* pBuffer);

// csric02
mcsparseStatus_t mcsparseScsric02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                             const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                             mcsparseCsric02Info_t info, int* buffersize);

mcsparseStatus_t mcsparseDcsric02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                             const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                             mcsparseCsric02Info_t info, int* buffersize);

mcsparseStatus_t mcsparseCcsric02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                             const mcComplex* csr_vals, const int* csr_rows, const int* csr_cols,
                                             mcsparseCsric02Info_t info, int* buffersize);

mcsparseStatus_t mcsparseZcsric02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                             const mcDoubleComplex* csr_vals, const int* csr_rows, const int* csr_cols,
                                             mcsparseCsric02Info_t info, int* buffersize);

mcsparseStatus_t mcsparseScsric02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                                const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                                mcsparseCsric02Info_t info, size_t* buffersize);

mcsparseStatus_t mcsparseDcsric02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                                const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                                mcsparseCsric02Info_t info, size_t* buffersize);

mcsparseStatus_t mcsparseCcsric02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                                const mcComplex* csr_vals, const int* csr_rows, const int* csr_cols,
                                                mcsparseCsric02Info_t info, size_t* buffersize);

mcsparseStatus_t mcsparseZcsric02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                                const mcDoubleComplex* csr_vals, const int* csr_rows,
                                                const int* csr_cols, mcsparseCsric02Info_t info, size_t* buffersize);

mcsparseStatus_t mcsparseScsric02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                           const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                           mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy,
                                           void* temp_buffer);

mcsparseStatus_t mcsparseDcsric02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                           const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                           mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy,
                                           void* temp_buffer);

mcsparseStatus_t mcsparseCcsric02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                           const mcComplex* csr_vals, const int* csr_rows, const int* csr_cols,
                                           mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy,
                                           void* temp_buffer);

mcsparseStatus_t mcsparseZcsric02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                           const mcDoubleComplex* csr_vals, const int* csr_rows, const int* csr_cols,
                                           mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy,
                                           void* temp_buffer);

mcsparseStatus_t mcsparseXcsric02_zeroPivot(mcsparseHandle_t handle, mcsparseCsric02Info_t info, int* position);

mcsparseStatus_t mcsparseScsric02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                  float* csr_vals, const int* csr_rows, const int* csr_cols, mcsparseCsric02Info_t info,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcsparseStatus_t mcsparseDcsric02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                  double* csr_vals, const int* csr_rows, const int* csr_cols,
                                  mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcsparseStatus_t mcsparseCcsric02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                  mcComplex* csr_vals, const int* csr_rows, const int* csr_cols,
                                  mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

mcsparseStatus_t mcsparseZcsric02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                  mcDoubleComplex* csr_vals, const int* csr_rows, const int* csr_cols,
                                  mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer);

// gtsv2
mcsparseStatus_t mcsparseSgtsv2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d,
                                              const float* du, const float* B, int ldb, size_t* bufferSizeInBytes);

mcsparseStatus_t mcsparseDgtsv2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                                              const double* du, const double* B, int ldb, size_t* bufferSizeInBytes);

mcsparseStatus_t mcsparseCgtsv2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcComplex* dl,
                                              const mcComplex* d, const mcComplex* du, const mcComplex* B, int ldb,
                                              size_t* bufferSizeInBytes);

mcsparseStatus_t mcsparseZgtsv2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* dl,
                                              const mcDoubleComplex* d, const mcDoubleComplex* du,
                                              const mcDoubleComplex* B, int ldb, size_t* bufferSizeInBytes);

mcsparseStatus_t mcsparseSgtsv2(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du,
                                float* B, int ldb, void* pBuffer);

mcsparseStatus_t mcsparseDgtsv2(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                                const double* du, double* B, int ldb, void* pBuffer);

mcsparseStatus_t mcsparseCgtsv2(mcsparseHandle_t handle, int m, int n, const mcComplex* dl, const mcComplex* d,
                                const mcComplex* du, mcComplex* B, int ldb, void* pBuffer);

mcsparseStatus_t mcsparseZgtsv2(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* dl,
                                const mcDoubleComplex* d, const mcDoubleComplex* du, mcDoubleComplex* B, int ldb,
                                void* pBuffer);

// gtsv2_nopivot
mcsparseStatus_t mcsparseSgtsv2_nopivot_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const float* dl,
                                                      const float* d, const float* du, const float* B, int ldb,
                                                      size_t* buffer_size);

mcsparseStatus_t mcsparseDgtsv2_nopivot_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const double* dl,
                                                      const double* d, const double* du, const double* B, int ldb,
                                                      size_t* buffer_size);

mcsparseStatus_t mcsparseCgtsv2_nopivot_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcComplex* dl,
                                                      const mcComplex* d, const mcComplex* du, const mcComplex* B,
                                                      int ldb, size_t* buffer_size);

mcsparseStatus_t mcsparseZgtsv2_nopivot_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* dl,
                                                      const mcDoubleComplex* d, const mcDoubleComplex* du,
                                                      const mcDoubleComplex* B, int ldb, size_t* buffer_size);

mcsparseStatus_t mcsparseSgtsv2_nopivot(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d,
                                        const float* du, float* B, int ldb, void* temp_buffer);

mcsparseStatus_t mcsparseDgtsv2_nopivot(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                                        const double* du, double* B, int ldb, void* temp_buffer);

mcsparseStatus_t mcsparseCgtsv2_nopivot(mcsparseHandle_t handle, int m, int n, const mcComplex* dl, const mcComplex* d,
                                        const mcComplex* du, mcComplex* B, int ldb, void* temp_buffer);

mcsparseStatus_t mcsparseZgtsv2_nopivot(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* dl,
                                        const mcDoubleComplex* d, const mcDoubleComplex* du, mcDoubleComplex* B,
                                        int ldb, void* temp_buffer);

// gtsv2StridedBatch
mcsparseStatus_t mcsparseSgtsv2StridedBatch_bufferSizeExt(mcsparseHandle_t handle, int m, const float* dl,
                                                          const float* d, const float* du, const float* x,
                                                          int batchCount, int batchStride, size_t* bufferSizeInBytes);

mcsparseStatus_t mcsparseDgtsv2StridedBatch_bufferSizeExt(mcsparseHandle_t handle, int m, const double* dl,
                                                          const double* d, const double* du, const double* x,
                                                          int batchCount, int batchStride, size_t* bufferSizeInBytes);

mcsparseStatus_t mcsparseCgtsv2StridedBatch_bufferSizeExt(mcsparseHandle_t handle, int m, const mcComplex* dl,
                                                          const mcComplex* d, const mcComplex* du, const mcComplex* x,
                                                          int batchCount, int batchStride, size_t* bufferSizeInBytes);

mcsparseStatus_t mcsparseZgtsv2StridedBatch_bufferSizeExt(mcsparseHandle_t handle, int m, const mcDoubleComplex* dl,
                                                          const mcDoubleComplex* d, const mcDoubleComplex* du,
                                                          const mcDoubleComplex* x, int batchCount, int batchStride,
                                                          size_t* bufferSizeInBytes);

mcsparseStatus_t mcsparseSgtsv2StridedBatch(mcsparseHandle_t handle, int m, const float* dl, const float* d,
                                            const float* du, float* x, int batchCount, int batchStride, void* pBuffer);

mcsparseStatus_t mcsparseDgtsv2StridedBatch(mcsparseHandle_t handle, int m, const double* dl, const double* d,
                                            const double* du, double* x, int batchCount, int batchStride,
                                            void* pBuffer);

mcsparseStatus_t mcsparseCgtsv2StridedBatch(mcsparseHandle_t handle, int m, const mcComplex* dl, const mcComplex* d,
                                            const mcComplex* du, mcComplex* x, int batchCount, int batchStride,
                                            void* pBuffer);

mcsparseStatus_t mcsparseZgtsv2StridedBatch(mcsparseHandle_t handle, int m, const mcDoubleComplex* dl,
                                            const mcDoubleComplex* d, const mcDoubleComplex* du, mcDoubleComplex* x,
                                            int batchCount, int batchStride, void* pBuffer);

// gtsvInterleavedBatch
mcsparseStatus_t mcsparseSgtsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             const float* dl, const float* d, const float* du,
                                                             const float* x, int batch_count, size_t* buffer_size);

mcsparseStatus_t mcsparseDgtsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             const double* dl, const double* d, const double* du,
                                                             const double* x, int batch_count, size_t* buffer_size);

mcsparseStatus_t mcsparseCgtsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             const mcComplex* dl, const mcComplex* d,
                                                             const mcComplex* du, const mcComplex* x, int batch_count,
                                                             size_t* buffer_size);

mcsparseStatus_t mcsparseZgtsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             const mcDoubleComplex* dl, const mcDoubleComplex* d,
                                                             const mcDoubleComplex* du, const mcDoubleComplex* x,
                                                             int batch_count, size_t* buffer_size);

mcsparseStatus_t mcsparseSgtsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, float* dl, float* d,
                                               float* du, float* x, int batch_count, void* temp_buffer);

mcsparseStatus_t mcsparseDgtsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, double* dl, double* d,
                                               double* du, double* x, int batch_count, void* temp_buffer);

mcsparseStatus_t mcsparseCgtsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, mcComplex* dl,
                                               mcComplex* d, mcComplex* du, mcComplex* x, int batch_count,
                                               void* temp_buffer);

mcsparseStatus_t mcsparseZgtsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, mcDoubleComplex* dl,
                                               mcDoubleComplex* d, mcDoubleComplex* du, mcDoubleComplex* x,
                                               int batch_count, void* temp_buffer);

// gpsvInterleavedBatch
mcsparseStatus_t mcsparseSgpsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num, float* ds,
                                                             float* dl, float* d, float* du, float* dw, float* x,
                                                             int batch_count, size_t* buffer_size);

mcsparseStatus_t mcsparseDgpsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num, double* ds,
                                                             double* dl, double* d, double* du, double* dw, double* x,
                                                             int batch_count, size_t* buffer_size);

mcsparseStatus_t mcsparseCgpsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             mcComplex* ds, mcComplex* dl, mcComplex* d, mcComplex* du,
                                                             mcComplex* dw, mcComplex* x, int batch_count,
                                                             size_t* buffer_size);

mcsparseStatus_t mcsparseZgpsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             mcDoubleComplex* ds, mcDoubleComplex* dl,
                                                             mcDoubleComplex* d, mcDoubleComplex* du,
                                                             mcDoubleComplex* dw, mcDoubleComplex* x, int batch_count,
                                                             size_t* buffer_size);

mcsparseStatus_t mcsparseSgpsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, float* ds, float* dl,
                                               float* d, float* du, float* dw, float* x, int batch_count,
                                               void* temp_buffer);

mcsparseStatus_t mcsparseDgpsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, double* ds, double* dl,
                                               double* d, double* du, double* dw, double* x, int batch_count,
                                               void* temp_buffer);

mcsparseStatus_t mcsparseCgpsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, mcComplex* ds,
                                               mcComplex* dl, mcComplex* d, mcComplex* du, mcComplex* dw, mcComplex* x,
                                               int batch_count, void* temp_buffer);

mcsparseStatus_t mcsparseZgpsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, mcDoubleComplex* ds,
                                               mcDoubleComplex* dl, mcDoubleComplex* d, mcDoubleComplex* du,
                                               mcDoubleComplex* dw, mcDoubleComplex* x, int batch_count,
                                               void* temp_buffer);

// bsrilu0
mcsparseStatus_t mcsparseSbsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int enable_boost,
                                                double* tol, float* boost_val);

mcsparseStatus_t mcsparseDbsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int enable_boost,
                                                double* tol, double* boost_val);

mcsparseStatus_t mcsparseCbsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int enable_boost,
                                                double* tol, mcComplex* boost_val);

mcsparseStatus_t mcsparseZbsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int enable_boost,
                                                double* tol, mcDoubleComplex* boost_val);

mcsparseStatus_t mcsparseXbsrilu02_zeroPivot(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int* position);

mcsparseStatus_t mcsparseSbsrilu02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                              const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                              const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                              mcsparseBsrilu02Info_t info, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseDbsrilu02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                              const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                              const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                              mcsparseBsrilu02Info_t info, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseCbsrilu02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                              const mcsparseMatDescr_t descrA, mcComplex* bsrSortedVal,
                                              const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                              mcsparseBsrilu02Info_t info, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseZbsrilu02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                              const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                              const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                              mcsparseBsrilu02Info_t info, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseSbsrilu02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                 const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                                 const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                 mcsparseBsrilu02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseDbsrilu02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                 const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                                 const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                 mcsparseBsrilu02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseCbsrilu02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                 const mcsparseMatDescr_t descrA, mcComplex* bsrSortedVal,
                                                 const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                 mcsparseBsrilu02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseZbsrilu02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                 const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                                 const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                 mcsparseBsrilu02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseSbsrilu02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                            mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseDbsrilu02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                            mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseCbsrilu02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, mcComplex* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                            mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseZbsrilu02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                            mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseSbsrilu02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                   const mcsparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr,
                                   const int* bsrSortedColInd, int blockDim, mcsparseBsrilu02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseDbsrilu02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                   const mcsparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr,
                                   const int* bsrSortedColInd, int blockDim, mcsparseBsrilu02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseCbsrilu02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                   const mcsparseMatDescr_t descrA, mcComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                   const int* bsrSortedColInd, int blockDim, mcsparseBsrilu02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseZbsrilu02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                   const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                   const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                   mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

// bsric0
mcsparseStatus_t mcsparseXbsric02_zeroPivot(mcsparseHandle_t handle, mcsparseBsric02Info_t info, int* position);

mcsparseStatus_t mcsparseSbsric02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                             const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                             const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                             mcsparseBsric02Info_t info, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseDbsric02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                             const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                             const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                             mcsparseBsric02Info_t info, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseCbsric02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                             const mcsparseMatDescr_t descrA, mcComplex* bsrSortedVal,
                                             const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                             mcsparseBsric02Info_t info, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseZbsric02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                             const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                             const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                             mcsparseBsric02Info_t info, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseSbsric02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                                const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                mcsparseBsric02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseDbsric02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                                const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                mcsparseBsric02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseCbsric02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                const mcsparseMatDescr_t descrA, mcComplex* bsrSortedVal,
                                                const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                mcsparseBsric02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseZbsric02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                                const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                mcsparseBsric02Info_t info, size_t* pBufferSize);

mcsparseStatus_t mcsparseSbsric02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                           const mcsparseMatDescr_t descrA, const float* bsrSortedVal,
                                           const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                           mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy,
                                           void* pInputBuffer);

mcsparseStatus_t mcsparseDbsric02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                           const mcsparseMatDescr_t descrA, const double* bsrSortedVal,
                                           const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                           mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy,
                                           void* pInputBuffer);

mcsparseStatus_t mcsparseCbsric02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                           const mcsparseMatDescr_t descrA, const mcComplex* bsrSortedVal,
                                           const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                           mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy,
                                           void* pInputBuffer);

mcsparseStatus_t mcsparseZbsric02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                           const mcsparseMatDescr_t descrA, const mcDoubleComplex* bsrSortedVal,
                                           const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                           mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy,
                                           void* pInputBuffer);

mcsparseStatus_t mcsparseSbsric02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                  const mcsparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr,
                                  const int* bsrSortedColInd, int blockDim, mcsparseBsric02Info_t info,
                                  mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseDbsric02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                  const mcsparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr,
                                  const int* bsrSortedColInd, int blockDim, mcsparseBsric02Info_t info,
                                  mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseCbsric02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                  const mcsparseMatDescr_t descrA, mcComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                  const int* bsrSortedColInd, int blockDim, mcsparseBsric02Info_t info,
                                  mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseZbsric02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                  const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                  const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                  mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer);

mcsparseStatus_t mcsparseSgtsv(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du,
                               float* B, int ldb);

mcsparseStatus_t mcsparseDgtsv(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                               const double* du, double* B, int ldb);

mcsparseStatus_t mcsparseCgtsv(mcsparseHandle_t handle, int m, int n, const mcComplex* dl, const mcComplex* d,
                               const mcComplex* du, mcComplex* B, int ldb);

mcsparseStatus_t mcsparseZgtsv(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* dl,
                               const mcDoubleComplex* d, const mcDoubleComplex* du, mcDoubleComplex* B, int ldb);

mcsparseStatus_t mcsparseSgtsv_nopivot(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d,
                                       const float* du, float* B, int ldb);

mcsparseStatus_t mcsparseDgtsv_nopivot(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                                       const double* du, double* B, int ldb);

mcsparseStatus_t mcsparseCgtsv_nopivot(mcsparseHandle_t handle, int m, int n, const mcComplex* dl, const mcComplex* d,
                                       const mcComplex* du, mcComplex* B, int ldb);

mcsparseStatus_t mcsparseZgtsv_nopivot(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* dl,
                                       const mcDoubleComplex* d, const mcDoubleComplex* du, mcDoubleComplex* B,
                                       int ldb);

mcsparseStatus_t mcsparseSgtsvStridedBatch(mcsparseHandle_t handle, int m, const float* dl, const float* d,
                                           const float* du, float* x, int batch_count, int batch_stride);

mcsparseStatus_t mcsparseDgtsvStridedBatch(mcsparseHandle_t handle, int m, const double* dl, const double* d,
                                           const double* du, double* x, int batch_count, int batch_stride);

mcsparseStatus_t mcsparseCgtsvStridedBatch(mcsparseHandle_t handle, int m, const mcComplex* dl, const mcComplex* d,
                                           const mcComplex* du, mcComplex* x, int batch_count, int batch_stride);

mcsparseStatus_t mcsparseZgtsvStridedBatch(mcsparseHandle_t handle, int m, const mcDoubleComplex* dl,
                                           const mcDoubleComplex* d, const mcDoubleComplex* du, mcDoubleComplex* x,
                                           int batch_count, int batch_stride);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSPARSE_PRECOND_H_
