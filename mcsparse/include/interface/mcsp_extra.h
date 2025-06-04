#ifndef INTERFACE_MCSPARSE_EXTRA_H_
#define INTERFACE_MCSPARSE_EXTRA_H_

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// csrgeam2
mcsparseStatus_t mcsparseScsrgeam2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const float* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                                 const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                 const float* beta, const mcsparseMatDescr_t descrB, int nnzB,
                                                 const float* csrSortedValB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                                 const float* csrSortedValC, const int* csrSortedRowPtrC,
                                                 const int* csrSortedColIndC, size_t* pBufferSizeInBytes);

mcsparseStatus_t mcsparseDcsrgeam2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const double* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                                 const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                 const double* beta, const mcsparseMatDescr_t descrB, int nnzB,
                                                 const double* csrSortedValB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                                 const double* csrSortedValC, const int* csrSortedRowPtrC,
                                                 const int* csrSortedColIndC, size_t* pBufferSizeInBytes);

mcsparseStatus_t mcsparseCcsrgeam2_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, const mcComplex* alpha, const mcsparseMatDescr_t descrA, int nnzA,
    const mcComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcComplex* beta,
    const mcsparseMatDescr_t descrB, int nnzB, const mcComplex* csrSortedValB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const mcsparseMatDescr_t descrC, const mcComplex* csrSortedValC,
    const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);

mcsparseStatus_t mcsparseZcsrgeam2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA,
                                                 const mcDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA,
                                                 const int* csrSortedColIndA, const mcDoubleComplex* beta,
                                                 const mcsparseMatDescr_t descrB, int nnzB,
                                                 const mcDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                                 const mcDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC,
                                                 const int* csrSortedColIndC, size_t* pBufferSizeInBytes);

mcsparseStatus_t mcsparseXcsrgeam2Nnz(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t descrA, int nnzA,
                                      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      const mcsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
                                      const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                      int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace);

mcsparseStatus_t mcsparseScsrgeam2(mcsparseHandle_t handle, int m, int n, const float* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta,
                                   const mcsparseMatDescr_t descrB, int nnzB, const float* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const mcsparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, void* pBuffer);

mcsparseStatus_t mcsparseDcsrgeam2(mcsparseHandle_t handle, int m, int n, const double* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta,
                                   const mcsparseMatDescr_t descrB, int nnzB, const double* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const mcsparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, void* pBuffer);

mcsparseStatus_t mcsparseCcsrgeam2(mcsparseHandle_t handle, int m, int n, const mcComplex* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const mcComplex* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcComplex* beta,
                                   const mcsparseMatDescr_t descrB, int nnzB, const mcComplex* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const mcsparseMatDescr_t descrC, mcComplex* csrSortedValC, int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, void* pBuffer);

mcsparseStatus_t mcsparseZcsrgeam2(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const mcDoubleComplex* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcDoubleComplex* beta, const mcsparseMatDescr_t descrB, int nnzB,
                                   const mcDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB,
                                   const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                   mcDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,
                                   void* pBuffer);

// csrgemm2
mcsparseStatus_t mcsparseScsrgemm2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int k, const float* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
                                                 const int* csrSortedColIndA, const mcsparseMatDescr_t descrB, int nnzB,
                                                 const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                                 const float* beta, const mcsparseMatDescr_t descrD, int nnzD,
                                                 const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                                 mcsparseCsrgemm2Info_t info, size_t* pBufferSizeInBytes);

mcsparseStatus_t mcsparseDcsrgemm2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int k, const double* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
                                                 const int* csrSortedColIndA, const mcsparseMatDescr_t descrB, int nnzB,
                                                 const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                                 const double* beta, const mcsparseMatDescr_t descrD, int nnzD,
                                                 const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                                 mcsparseCsrgemm2Info_t info, size_t* pBufferSizeInBytes);

mcsparseStatus_t mcsparseCcsrgemm2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int k, const mcComplex* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
                                                 const int* csrSortedColIndA, const mcsparseMatDescr_t descrB, int nnzB,
                                                 const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                                 const mcComplex* beta, const mcsparseMatDescr_t descrD, int nnzD,
                                                 const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                                 mcsparseCsrgemm2Info_t info, size_t* pBufferSizeInBytes);

mcsparseStatus_t mcsparseZcsrgemm2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int k,
                                                 const mcDoubleComplex* alpha, const mcsparseMatDescr_t descrA,
                                                 int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                 const mcsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcDoubleComplex* beta,
                                                 const mcsparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD,
                                                 const int* csrSortedColIndD, mcsparseCsrgemm2Info_t info,
                                                 size_t* pBufferSizeInBytes);

mcsparseStatus_t mcsparseXcsrgemm2Nnz(mcsparseHandle_t handle, int m, int n, int k, const mcsparseMatDescr_t descrA,
                                      int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      const mcsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
                                      const int* csrSortedColIndB, const mcsparseMatDescr_t descrD, int nnzD,
                                      const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                      const mcsparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr,
                                      const mcsparseCsrgemm2Info_t info, void* pBuffer);

mcsparseStatus_t mcsparseScsrgemm2(mcsparseHandle_t handle, int m, int n, int k, const float* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcsparseMatDescr_t descrB, int nnzB, const float* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB, const float* beta,
                                   const mcsparseMatDescr_t descrD, int nnzD, const float* csrSortedValD,
                                   const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                   const mcsparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, const mcsparseCsrgemm2Info_t info, void* pBuffer);

mcsparseStatus_t mcsparseDcsrgemm2(mcsparseHandle_t handle, int m, int n, int k, const double* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcsparseMatDescr_t descrB, int nnzB, const double* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB, const double* beta,
                                   const mcsparseMatDescr_t descrD, int nnzD, const double* csrSortedValD,
                                   const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                   const mcsparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, const mcsparseCsrgemm2Info_t info, void* pBuffer);

mcsparseStatus_t mcsparseCcsrgemm2(mcsparseHandle_t handle, int m, int n, int k, const mcComplex* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const mcComplex* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcsparseMatDescr_t descrB, int nnzB, const mcComplex* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB, const mcComplex* beta,
                                   const mcsparseMatDescr_t descrD, int nnzD, const mcComplex* csrSortedValD,
                                   const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                   const mcsparseMatDescr_t descrC, mcComplex* csrSortedValC,
                                   const int* csrSortedRowPtrC, int* csrSortedColIndC,
                                   const mcsparseCsrgemm2Info_t info, void* pBuffer);

mcsparseStatus_t mcsparseZcsrgemm2(mcsparseHandle_t handle, int m, int n, int k, const mcDoubleComplex* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const mcDoubleComplex* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcsparseMatDescr_t descrB, int nnzB, const mcDoubleComplex* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const mcDoubleComplex* beta, const mcsparseMatDescr_t descrD, int nnzD,
                                   const mcDoubleComplex* csrSortedValD, const int* csrSortedRowPtrD,
                                   const int* csrSortedColIndD, const mcsparseMatDescr_t descrC,
                                   mcDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC,
                                   const mcsparseCsrgemm2Info_t info, void* pBuffer);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSPARSE_EXTRA_H_
