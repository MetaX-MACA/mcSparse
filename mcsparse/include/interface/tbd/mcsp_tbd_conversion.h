#ifndef MCSPARSE_TBD_CONVERSION_H
#define MCSPARSE_TBD_CONVERSION_H

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

mcsparseStatus_t mcsparseSbsr2csr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                  const mcsparseMatDescr_t descrA, const float* bsrSortedValA,
                                  const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC,
                                  int* csrSortedColIndC);

mcsparseStatus_t mcsparseDbsr2csr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                  const mcsparseMatDescr_t descrA, const double* bsrSortedValA,
                                  const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC,
                                  int* csrSortedColIndC);

mcsparseStatus_t mcsparseCbsr2csr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                  const mcsparseMatDescr_t descrA, const mcFloatComplex* bsrSortedValA,
                                  const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, mcFloatComplex* csrSortedValC, int* csrSortedRowPtrC,
                                  int* csrSortedColIndC);

mcsparseStatus_t mcsparseZbsr2csr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                  const mcsparseMatDescr_t descrA, const mcDoubleComplex* bsrSortedValA,
                                  const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, mcDoubleComplex* csrSortedValC,
                                  int* csrSortedRowPtrC, int* csrSortedColIndC);

mcsparseStatus_t mcsparseXgebsr2csr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                    const mcsparseMatDescr_t descrA, const int* bsrSortedRowPtrA,
                                    const int* bsrSortedColIndA, int rowBlockDim, int colBlockDim,
                                    const mcsparseMatDescr_t descrC, int* csrSortedRowPtrC, int* csrSortedColIndC);

mcsparseStatus_t mcsparseSgebsr2csr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                    const mcsparseMatDescr_t descrA, const float* bsrSortedValA,
                                    const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim,
                                    int colBlockDim, const mcsparseMatDescr_t descrC, float* csrSortedValC,
                                    int* csrSortedRowPtrC, int* csrSortedColIndC);

mcsparseStatus_t mcsparseDgebsr2csr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                    const mcsparseMatDescr_t descrA, const double* bsrSortedValA,
                                    const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim,
                                    int colBlockDim, const mcsparseMatDescr_t descrC, double* csrSortedValC,
                                    int* csrSortedRowPtrC, int* csrSortedColIndC);

mcsparseStatus_t mcsparseCgebsr2csr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                    const mcsparseMatDescr_t descrA, const mcComplex* bsrSortedValA,
                                    const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim,
                                    int colBlockDim, const mcsparseMatDescr_t descrC, mcComplex* csrSortedValC,
                                    int* csrSortedRowPtrC, int* csrSortedColIndC);

mcsparseStatus_t mcsparseZgebsr2csr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                    const mcsparseMatDescr_t descrA, const mcDoubleComplex* bsrSortedValA,
                                    const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim,
                                    int colBlockDim, const mcsparseMatDescr_t descrC, mcDoubleComplex* csrSortedValC,
                                    int* csrSortedRowPtrC, int* csrSortedColIndC);

mcsparseStatus_t mcsparseSgebsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                                 int nnzb, const mcsparseMatDescr_t descrA, const float* bsrSortedValA,
                                                 const int* bsrSortedRowPtrA, const int* bsrSortedColIndA,
                                                 int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC,
                                                 int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseDgebsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                                 int nnzb, const mcsparseMatDescr_t descrA, const double* bsrSortedValA,
                                                 const int* bsrSortedRowPtrA, const int* bsrSortedColIndA,
                                                 int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC,
                                                 int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseCgebsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                                 int nnzb, const mcsparseMatDescr_t descrA,
                                                 const mcComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                                 const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                                 int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseZgebsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                                 int nnzb, const mcsparseMatDescr_t descrA,
                                                 const mcDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                                 const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                                 int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes);

mcsparseStatus_t mcsparseSgebsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                                    int nnzb, const mcsparseMatDescr_t descrA,
                                                    const float* bsrSortedValA, const int* bsrSortedRowPtrA,
                                                    const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                                    int rowBlockDimC, int colBlockDimC, size_t* pBufferSize);

mcsparseStatus_t mcsparseDgebsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                                    int nnzb, const mcsparseMatDescr_t descrA,
                                                    const double* bsrSortedValA, const int* bsrSortedRowPtrA,
                                                    const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                                    int rowBlockDimC, int colBlockDimC, size_t* pBufferSize);

mcsparseStatus_t mcsparseCgebsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                                    int nnzb, const mcsparseMatDescr_t descrA,
                                                    const mcComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                                    const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                                    int rowBlockDimC, int colBlockDimC, size_t* pBufferSize);

mcsparseStatus_t mcsparseZgebsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb,
                                                    int nnzb, const mcsparseMatDescr_t descrA,
                                                    const mcDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                                    const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                                    int rowBlockDimC, int colBlockDimC, size_t* pBufferSize);

mcsparseStatus_t mcsparseXgebsr2gebsrNnz(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb, int nnzb,
                                         const mcsparseMatDescr_t descrA, const int* bsrSortedRowPtrA,
                                         const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA,
                                         const mcsparseMatDescr_t descrC, int* bsrSortedRowPtrC, int rowBlockDimC,
                                         int colBlockDimC, int* nnzTotalDevHostPtr, void* pBuffer);

mcsparseStatus_t mcsparseSgebsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb, int nnzb,
                                      const mcsparseMatDescr_t descrA, const float* bsrSortedValA,
                                      const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA,
                                      int colBlockDimA, const mcsparseMatDescr_t descrC, float* bsrSortedValC,
                                      int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC,
                                      void* pBuffer);

mcsparseStatus_t mcsparseDgebsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb, int nnzb,
                                      const mcsparseMatDescr_t descrA, const double* bsrSortedValA,
                                      const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA,
                                      int colBlockDimA, const mcsparseMatDescr_t descrC, double* bsrSortedValC,
                                      int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC,
                                      void* pBuffer);

mcsparseStatus_t mcsparseCgebsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb, int nnzb,
                                      const mcsparseMatDescr_t descrA, const mcComplex* bsrSortedValA,
                                      const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA,
                                      int colBlockDimA, const mcsparseMatDescr_t descrC, mcComplex* bsrSortedValC,
                                      int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC,
                                      void* pBuffer);

mcsparseStatus_t mcsparseZgebsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nb, int nnzb,
                                      const mcsparseMatDescr_t descrA, const mcDoubleComplex* bsrSortedValA,
                                      const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA,
                                      int colBlockDimA, const mcsparseMatDescr_t descrC, mcDoubleComplex* bsrSortedValC,
                                      int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC,
                                      void* pBuffer);

#ifdef __cplusplus
}
#endif

#endif