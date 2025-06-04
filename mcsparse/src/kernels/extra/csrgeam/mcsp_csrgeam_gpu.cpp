#include <assert.h>
#include <stdio.h>

#include <cstddef>
#include <numeric>
#include <vector>

#include "common/mcsp_types.h"
#include "csrgeam_device.hpp"
#include "device_radix_sort.hpp"
#include "internal_interface/mcsp_internal_conversion.h"
#include "internal_interface/mcsp_internal_extra.h"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType>
mcspStatus_t mcspCsrGeamNnzTemplate(mcspHandle_t handle, idxType m, idxType n, const mcspMatDescr_t descr_A,
                                    idxType nnz_A, const idxType* csr_rows_A, const idxType* csr_cols_A,
                                    const mcspMatDescr_t descr_B, idxType nnz_B, const idxType* csr_rows_B,
                                    const idxType* csr_cols_B, const mcspMatDescr_t descr_C, idxType* csr_rows_C,
                                    idxType* nnz_C) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz_A < 0 || nnz_B < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (csr_rows_A == nullptr || csr_cols_A == nullptr || csr_rows_B == nullptr || csr_cols_B == nullptr ||
        csr_rows_C == nullptr || descr_A == nullptr || descr_B == nullptr || descr_C == nullptr || nnz_C == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_A->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_A->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT || descr_B->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr_B->fill_mode != MCSPARSE_FILL_MODE_FULL || descr_B->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT ||
        descr_C->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_C->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_C->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    mcStream_t stream = handle->stream;

#define CSRGEAM_BLOCK_SIZE 512
    int block_row = CSRGEAM_BLOCK_SIZE;
    int n_block = (m + block_row - 1) / block_row;
    mcLaunchKernelGGL((mcspCsrGeamNnzKernel), dim3(n_block), dim3(CSRGEAM_BLOCK_SIZE), 0, stream, m, n, csr_rows_A,
                       csr_cols_A, csr_rows_B, csr_cols_B, csr_rows_C, descr_A->base, descr_B->base);
#undef CSRGEAM_BLOCK_SIZE
    MACA_ASSERT(mcStreamSynchronize(stream));

    // exclusive scan get csr_rows_C
    idxType prim_buffer_size;
    void* scan_buffer = nullptr;
    mcprim::exclusive_scan(nullptr, prim_buffer_size, csr_rows_C, csr_rows_C, m + 1,
                           reinterpret_cast<idxType*>(&descr_C->base), stream);
    bool use_buffer_pool = handle->mcspUsePoolBuffer((void**)&scan_buffer, prim_buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc((void**)&scan_buffer, prim_buffer_size));
    }
    mcprim::exclusive_scan(scan_buffer, prim_buffer_size, csr_rows_C, csr_rows_C, m + 1,
                           reinterpret_cast<idxType*>(&descr_C->base), stream);
    MACA_ASSERT(mcStreamSynchronize(stream));
    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(scan_buffer));
    }

    idxType h_nnz_C = 0;
    MACA_ASSERT(mcMemcpyAsync(&h_nnz_C, csr_rows_C + m, sizeof(idxType), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    h_nnz_C -= descr_C->base;
    h_nnz_C = std::max((idxType)1, h_nnz_C);
    if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
        *nnz_C = h_nnz_C;
    } else {
        MACA_ASSERT(mcMemcpyAsync(nnz_C, &h_nnz_C, sizeof(idxType), mcMemcpyHostToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsrGeamTemplate(mcspHandle_t handle, idxType m, idxType n, const valType* alpha,
                                 const mcspMatDescr_t descr_A, idxType nnz_A, const valType* csr_vals_A,
                                 const idxType* csr_rows_A, const idxType* csr_cols_A, const valType* beta,
                                 const mcspMatDescr_t descr_B, idxType nnz_B, const valType* csr_vals_B,
                                 const idxType* csr_rows_B, const idxType* csr_cols_B, const mcspMatDescr_t descr_C,
                                 valType* csr_vals_C, const idxType* csr_rows_C, idxType* csr_cols_C,
                                 void* pBuffer = nullptr) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz_A < 0 || nnz_B < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    // currently csr C matrix is not supported
    if (alpha == nullptr || beta == nullptr || csr_rows_A == nullptr || csr_cols_A == nullptr ||
        csr_vals_A == nullptr || csr_rows_B == nullptr || csr_cols_B == nullptr || csr_vals_B == nullptr ||
        csr_rows_C == nullptr || csr_cols_C == nullptr || csr_vals_C == nullptr || descr_A == nullptr ||
        descr_B == nullptr || descr_C == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (nnz_A == 0 || nnz_B == 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_A->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_A->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT || descr_B->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr_B->fill_mode != MCSPARSE_FILL_MODE_FULL || descr_B->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT ||
        descr_C->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_C->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_C->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0)) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    mcStream_t stream = handle->stream;
    valType h_beta = getScalarToHost(beta, handle->ptr_mode);
    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
#define CSRGEAM_BLOCK_SIZE 512
    int block_row = CSRGEAM_BLOCK_SIZE;
    int n_block = (m + block_row - 1) / block_row;
    mcLaunchKernelGGL((mcspCsrGeamCalKernel), dim3(n_block), dim3(CSRGEAM_BLOCK_SIZE), 0, stream, m, n, h_alpha,
                       csr_rows_A, csr_cols_A, csr_vals_A, h_beta, csr_rows_B, csr_cols_B, csr_vals_B, csr_rows_C,
                       csr_cols_C, csr_vals_C, descr_A->base, descr_B->base, descr_C->base);
#undef CSRGEAM_BLOCK_SIZE

    return MCSP_STATUS_SUCCESS;
}

mcspStatus_t mcspXcsrgeam2NnzImpl(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t descrA, mcspInt nnzA,
                                  const mcspInt* csrSortedRowPtrA, const mcspInt* csrSortedColIndA,
                                  const mcspMatDescr_t descrB, mcspInt nnzB, const mcspInt* csrSortedRowPtrB,
                                  const mcspInt* csrSortedColIndB, const mcspMatDescr_t descrC,
                                  mcspInt* csrSortedRowPtrC, mcspInt* nnzTotalDevHostPtr, void* workspace) {
    return mcspCsrGeamNnzTemplate(handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                                  csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr);
}

template <typename idxType, typename valType>
mcspStatus_t mcspXcsrgeam2_bufferSizeExtImpl(mcspHandle_t handle, idxType m, idxType n, const valType* alpha,
                                             const mcspMatDescr_t descrA, idxType nnzA, const valType* csrSortedValA,
                                             const idxType* csrSortedRowPtrA, const idxType* csrSortedColIndA,
                                             const valType* beta, const mcspMatDescr_t descrB, idxType nnzB,
                                             const valType* csrSortedValB, const idxType* csrSortedRowPtrB,
                                             const idxType* csrSortedColIndB, const mcspMatDescr_t descrC,
                                             const valType* csrSortedValC, const idxType* csrSortedRowPtrC,
                                             const idxType* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    *pBufferSizeInBytes = MIN_BUFFER_SIZE;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspXcsrgeam2Impl(mcspHandle_t handle, idxType m, idxType n, const valType* alpha,
                               const mcspMatDescr_t descrA, idxType nnzA, const valType* csrSortedValA,
                               const idxType* csrSortedRowPtrA, const idxType* csrSortedColIndA, const valType* beta,
                               const mcspMatDescr_t descrB, idxType nnzB, const valType* csrSortedValB,
                               const idxType* csrSortedRowPtrB, const idxType* csrSortedColIndB,
                               const mcspMatDescr_t descrC, valType* csrSortedValC, idxType* csrSortedRowPtrC,
                               idxType* csrSortedColIndC, void* pBuffer) {
    return mcspCsrGeamTemplate(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                               beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                               csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspCsrGeamNnz(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t descr_A, mcspInt nnz_A,
                            const mcspInt* csr_rows_A, const mcspInt* csr_cols_A, const mcspMatDescr_t descr_B,
                            mcspInt nnz_B, const mcspInt* csr_rows_B, const mcspInt* csr_cols_B,
                            const mcspMatDescr_t descr_C, mcspInt* csr_rows_C, mcspInt* nnz_C) {
    return mcspCsrGeamNnzTemplate(handle, m, n, descr_A, nnz_A, csr_rows_A, csr_cols_A, descr_B, nnz_B, csr_rows_B,
                                  csr_cols_B, descr_C, csr_rows_C, nnz_C);
}

mcspStatus_t mcspScsrGeam(mcspHandle_t handle, mcspInt m, mcspInt n, const float* alpha, const mcspMatDescr_t descr_A,
                          mcspInt nnz_A, const float* csr_vals_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                          const float* beta, const mcspMatDescr_t descr_B, mcspInt nnz_B, const float* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspMatDescr_t descr_C,
                          float* csr_vals_C, const mcspInt* csr_rows_C, mcspInt* csr_cols_C) {
    return mcspCsrGeamTemplate(handle, m, n, alpha, descr_A, nnz_A, csr_vals_A, csr_rows_A, csr_cols_A, beta, descr_B,
                               nnz_B, csr_vals_B, csr_rows_B, csr_cols_B, descr_C, csr_vals_C, csr_rows_C, csr_cols_C);
}

mcspStatus_t mcspDcsrGeam(mcspHandle_t handle, mcspInt m, mcspInt n, const double* alpha, const mcspMatDescr_t descr_A,
                          mcspInt nnz_A, const double* csr_vals_A, const mcspInt* csr_rows_A, const mcspInt* csr_cols_A,
                          const double* beta, const mcspMatDescr_t descr_B, mcspInt nnz_B, const double* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspMatDescr_t descr_C,
                          double* csr_vals_C, const mcspInt* csr_rows_C, mcspInt* csr_cols_C) {
    return mcspCsrGeamTemplate(handle, m, n, alpha, descr_A, nnz_A, csr_vals_A, csr_rows_A, csr_cols_A, beta, descr_B,
                               nnz_B, csr_vals_B, csr_rows_B, csr_cols_B, descr_C, csr_vals_C, csr_rows_C, csr_cols_C);
}

mcspStatus_t mcspCcsrGeam(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexFloat* alpha,
                          const mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspComplexFloat* csr_vals_A,
                          const mcspInt* csr_rows_A, const mcspInt* csr_cols_A, const mcspComplexFloat* beta,
                          const mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspComplexFloat* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspMatDescr_t descr_C,
                          mcspComplexFloat* csr_vals_C, const mcspInt* csr_rows_C, mcspInt* csr_cols_C) {
    return mcspCsrGeamTemplate(handle, m, n, alpha, descr_A, nnz_A, csr_vals_A, csr_rows_A, csr_cols_A, beta, descr_B,
                               nnz_B, csr_vals_B, csr_rows_B, csr_cols_B, descr_C, csr_vals_C, csr_rows_C, csr_cols_C);
}

mcspStatus_t mcspZcsrGeam(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspComplexDouble* alpha,
                          const mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspComplexDouble* csr_vals_A,
                          const mcspInt* csr_rows_A, const mcspInt* csr_cols_A, const mcspComplexDouble* beta,
                          const mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspComplexDouble* csr_vals_B,
                          const mcspInt* csr_rows_B, const mcspInt* csr_cols_B, const mcspMatDescr_t descr_C,
                          mcspComplexDouble* csr_vals_C, const mcspInt* csr_rows_C, mcspInt* csr_cols_C) {
    return mcspCsrGeamTemplate(handle, m, n, alpha, descr_A, nnz_A, csr_vals_A, csr_rows_A, csr_cols_A, beta, descr_B,
                               nnz_B, csr_vals_B, csr_rows_B, csr_cols_B, descr_C, csr_vals_C, csr_rows_C, csr_cols_C);
}

// This function performs following matrix-matrix operation C=alpha*A+beta*B, where A, B, and C are m×n sparse matrices
// (defined in CSR storage format), alpha and beta are scalars.
mcspStatus_t mcspCuinScsrgeam2_bufferSizeExt(mcspHandle_t handle, int m, int n, const float* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta,
                                           const mcspMatDescr_t descrB, int nnzB, const float* csrSortedValB,
                                           const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                           const mcspMatDescr_t descrC, const float* csrSortedValC,
                                           const int* csrSortedRowPtrC, const int* csrSortedColIndC,
                                           size_t* pBufferSizeInBytes) {
    return mcspXcsrgeam2_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)n, alpha, descrA, (mcspInt)nnzA, csrSortedValA,
                                           (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, beta, descrB,
                                           (mcspInt)nnzB, csrSortedValB, (mcspInt*)csrSortedRowPtrB,
                                           (mcspInt*)csrSortedColIndB, descrC, csrSortedValC,
                                           (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinDcsrgeam2_bufferSizeExt(mcspHandle_t handle, int m, int n, const double* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta,
                                           const mcspMatDescr_t descrB, int nnzB, const double* csrSortedValB,
                                           const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                           const mcspMatDescr_t descrC, const double* csrSortedValC,
                                           const int* csrSortedRowPtrC, const int* csrSortedColIndC,
                                           size_t* pBufferSizeInBytes) {
    return mcspXcsrgeam2_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)n, alpha, descrA, (mcspInt)nnzA, csrSortedValA,
                                           (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, beta, descrB,
                                           (mcspInt)nnzB, csrSortedValB, (mcspInt*)csrSortedRowPtrB,
                                           (mcspInt*)csrSortedColIndB, descrC, csrSortedValC,
                                           (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinCcsrgeam2_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcFloatComplex* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const mcFloatComplex* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                           const mcFloatComplex* beta, const mcspMatDescr_t descrB, int nnzB,
                                           const mcFloatComplex* csrSortedValB, const int* csrSortedRowPtrB,
                                           const int* csrSortedColIndB, const mcspMatDescr_t descrC,
                                           const mcFloatComplex* csrSortedValC, const int* csrSortedRowPtrC,
                                           const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    return mcspXcsrgeam2_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)n, alpha, descrA, (mcspInt)nnzA, csrSortedValA,
                                           (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, beta, descrB,
                                           (mcspInt)nnzB, csrSortedValB, (mcspInt*)csrSortedRowPtrB,
                                           (mcspInt*)csrSortedColIndB, descrC, csrSortedValC,
                                           (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinZcsrgeam2_bufferSizeExt(mcspHandle_t handle, int m, int n, const mcDoubleComplex* alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const mcDoubleComplex* csrSortedValA,
                                           const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                           const mcDoubleComplex* beta, const mcspMatDescr_t descrB, int nnzB,
                                           const mcDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB,
                                           const int* csrSortedColIndB, const mcspMatDescr_t descrC,
                                           const mcDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC,
                                           const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    return mcspXcsrgeam2_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)n, alpha, descrA, (mcspInt)nnzA, csrSortedValA,
                                           (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, beta, descrB,
                                           (mcspInt)nnzB, csrSortedValB, (mcspInt*)csrSortedRowPtrB,
                                           (mcspInt*)csrSortedColIndB, descrC, csrSortedValC,
                                           (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinXcsrgeam2Nnz(mcspHandle_t handle, int m, int n, const mcspMatDescr_t descrA, int nnzA,
                                const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcspMatDescr_t descrB,
                                int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                const mcspMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr,
                                void* workspace) {
    return mcspXcsrgeam2NnzImpl(handle, (mcspInt)m, (mcspInt)n, descrA, (mcspInt)nnzA, (mcspInt*)csrSortedRowPtrA,
                                (mcspInt*)csrSortedColIndA, descrB, (mcspInt)nnzB, (mcspInt*)csrSortedRowPtrB,
                                (mcspInt*)csrSortedColIndB, descrC, (mcspInt*)csrSortedRowPtrC,
                                (mcspInt*)nnzTotalDevHostPtr, workspace);
}

mcspStatus_t mcspCuinScsrgeam2(mcspHandle_t handle, int m, int n, const float* alpha, const mcspMatDescr_t descrA,
                             int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA, const float* beta, const mcspMatDescr_t descrB, int nnzB,
                             const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                             const mcspMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC,
                             int* csrSortedColIndC, void* pBuffer) {
    return mcspXcsrgeam2Impl(handle, (mcspInt)m, (mcspInt)n, alpha, descrA, (mcspInt)nnzA, csrSortedValA,
                             (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, beta, descrB, (mcspInt)nnzB,
                             csrSortedValB, (mcspInt*)csrSortedRowPtrB, (mcspInt*)csrSortedColIndB, descrC,
                             csrSortedValC, (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBuffer);
}

mcspStatus_t mcspCuinDcsrgeam2(mcspHandle_t handle, int m, int n, const double* alpha, const mcspMatDescr_t descrA,
                             int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA,
                             const int* csrSortedColIndA, const double* beta, const mcspMatDescr_t descrB, int nnzB,
                             const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                             const mcspMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC,
                             int* csrSortedColIndC, void* pBuffer) {
    return mcspXcsrgeam2Impl(handle, (mcspInt)m, (mcspInt)n, alpha, descrA, (mcspInt)nnzA, csrSortedValA,
                             (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, beta, descrB, (mcspInt)nnzB,
                             csrSortedValB, (mcspInt*)csrSortedRowPtrB, (mcspInt*)csrSortedColIndB, descrC,
                             csrSortedValC, (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBuffer);
}

mcspStatus_t mcspCuinCcsrgeam2(mcspHandle_t handle, int m, int n, const mcFloatComplex* alpha,
                             const mcspMatDescr_t descrA, int nnzA, const mcFloatComplex* csrSortedValA,
                             const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcFloatComplex* beta,
                             const mcspMatDescr_t descrB, int nnzB, const mcFloatComplex* csrSortedValB,
                             const int* csrSortedRowPtrB, const int* csrSortedColIndB, const mcspMatDescr_t descrC,
                             mcFloatComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,
                             void* pBuffer) {
    return mcspXcsrgeam2Impl(handle, (mcspInt)m, (mcspInt)n, alpha, descrA, (mcspInt)nnzA, csrSortedValA,
                             (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, beta, descrB, (mcspInt)nnzB,
                             csrSortedValB, (mcspInt*)csrSortedRowPtrB, (mcspInt*)csrSortedColIndB, descrC,
                             csrSortedValC, (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBuffer);
}

mcspStatus_t mcspCuinZcsrgeam2(mcspHandle_t handle, int m, int n, const mcDoubleComplex* alpha,
                             const mcspMatDescr_t descrA, int nnzA, const mcDoubleComplex* csrSortedValA,
                             const int* csrSortedRowPtrA, const int* csrSortedColIndA, const mcDoubleComplex* beta,
                             const mcspMatDescr_t descrB, int nnzB, const mcDoubleComplex* csrSortedValB,
                             const int* csrSortedRowPtrB, const int* csrSortedColIndB, const mcspMatDescr_t descrC,
                             mcDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,
                             void* pBuffer) {
    return mcspXcsrgeam2Impl(handle, (mcspInt)m, (mcspInt)n, alpha, descrA, (mcspInt)nnzA, csrSortedValA,
                             (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, beta, descrB, (mcspInt)nnzB,
                             csrSortedValB, (mcspInt*)csrSortedRowPtrB, (mcspInt*)csrSortedColIndB, descrC,
                             csrSortedValC, (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBuffer);
}

#ifdef __cplusplus
}
#endif
