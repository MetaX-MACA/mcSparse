#include "common/mcsp_types.h"
#include "csr_spgemm_host.hpp"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType, typename valType>
mcspStatus_t mcspXcsrgemm2_bufferSizeExtImpl(mcspHandle_t handle, idxType m, idxType n, idxType k, const valType *alpha,
                                             const mcspMatDescr_t descrA, idxType nnzA, const idxType *csrRowPtrA,
                                             const idxType *csrColIndA, const mcspMatDescr_t descrB, idxType nnzB,
                                             const idxType *csrRowPtrB, const idxType *csrColIndB, const valType *beta,
                                             const mcspMatDescr_t descrD, idxType nnzD, const idxType *csrRowPtrD,
                                             const idxType *csrColIndD, mcspCsrgemm2Info_t info,
                                             size_t *pBufferSizeInBytes) {
    if (info == nullptr || info->mat_info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (alpha != nullptr) {
        info->alpha_null = false;
    }
    if (beta != nullptr) {
        info->beta_null = false;
    }
    return mcspCsrgemmBuffersizeTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, MCSPARSE_OPERATION_NON_TRANSPOSE, m,
                                         n, k, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB,
                                         csrColIndB, descrD, nnzD, csrRowPtrD, csrColIndD, info->mat_info,
                                         pBufferSizeInBytes);
}

mcspStatus_t mcspXcsrgemm2NnzImpl(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt k, const mcspMatDescr_t descrA,
                                  mcspInt nnzA, const mcspInt *csrRowPtrA, const mcspInt *csrColIndA,
                                  const mcspMatDescr_t descrB, mcspInt nnzB, const mcspInt *csrRowPtrB,
                                  const mcspInt *csrColIndB, const mcspMatDescr_t descrD, mcspInt nnzD,
                                  const mcspInt *csrRowPtrD, const mcspInt *csrColIndD, const mcspMatDescr_t descrC,
                                  mcspInt *csrRowPtrC, mcspInt *nnzTotalDevHostPtr, const mcspCsrgemm2Info_t info,
                                  void *pBuffer) {
    if (info == nullptr || info->mat_info == nullptr || (info->alpha_null && info->beta_null)) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    if (info->alpha_null) {
        if (csrRowPtrD == nullptr || csrColIndD == nullptr) {
            return MCSP_STATUS_INVALID_POINTER;
        }
        *nnzTotalDevHostPtr = nnzD;
        mcStream_t stream = mcspGetStreamInternal(handle);
        MACA_ASSERT(
            mcMemcpyAsync(csrRowPtrC, csrRowPtrD, (m + 1) * sizeof(*csrRowPtrC), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        return MCSP_STATUS_SUCCESS;
    }
    if (info->beta_null) {
        bool include_addition = false;
        return mcspCsrgemmNnzTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, MCSPARSE_OPERATION_NON_TRANSPOSE, m, n,
                                      k, descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB,
                                      descrD, nnzD, csrRowPtrD, csrColIndD, descrC, csrRowPtrC, nnzTotalDevHostPtr,
                                      info->mat_info, pBuffer, include_addition);
    }

    // according to api doc, D and C must have save sparsity pattern.
    return mcspCsrgemmNnzTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, MCSPARSE_OPERATION_NON_TRANSPOSE, m, n, k,
                                  descrA, nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrD,
                                  nnzD, csrRowPtrD, csrColIndD, descrC, csrRowPtrC, nnzTotalDevHostPtr, info->mat_info,
                                  pBuffer);
}

template <typename idxType, typename valType>
mcspStatus_t mcspXcsrgemm2Impl(mcspHandle_t handle, idxType m, idxType n, idxType k, const valType *alpha,
                               const mcspMatDescr_t descrA, idxType nnzA, const valType *csrValA,
                               const idxType *csrRowPtrA, const idxType *csrColIndA, const mcspMatDescr_t descrB,
                               idxType nnzB, const valType *csrValB, const idxType *csrRowPtrB,
                               const idxType *csrColIndB, const valType *beta, const mcspMatDescr_t descrD,
                               idxType nnzD, const valType *csrValD, const idxType *csrRowPtrD,
                               const idxType *csrColIndD, const mcspMatDescr_t descrC, valType *csrValC,
                               const idxType *csrRowPtrC, idxType *csrColIndC, const mcspCsrgemm2Info_t info,
                               void *pBuffer) {
    if (info == nullptr || info->mat_info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    return mcspCsrgemmTemplate(handle, MCSPARSE_OPERATION_NON_TRANSPOSE, MCSPARSE_OPERATION_NON_TRANSPOSE, m, n, k,
                               alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB,
                               csrColIndB, beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC,
                               csrRowPtrC, csrColIndC, info->mat_info, pBuffer);
}

#ifdef __cplusplus
extern "C" {
#endif

// This function performs following matrix-matrix operation: D=alpha*A*B+beta*C, where A, B, D and C are m×k, k×n, m×n
// and m×n sparse matrices (defined in CSR storage format)
mcspStatus_t mcspCuinScsrgemm2_bufferSizeExt(mcspHandle_t handle, int m, int n, int k, const float *alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const int *csrRowPtrA,
                                           const int *csrColIndA, const mcspMatDescr_t descrB, int nnzB,
                                           const int *csrRowPtrB, const int *csrColIndB, const float *beta,
                                           const mcspMatDescr_t descrD, int nnzD, const int *csrRowPtrD,
                                           const int *csrColIndD, mcspCsrgemm2Info_t info, size_t *pBufferSizeInBytes) {
    return mcspXcsrgemm2_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, alpha, descrA, (mcspInt)nnzA,
                                           (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, descrB, (mcspInt)nnzB,
                                           (mcspInt *)csrRowPtrB, (mcspInt *)csrColIndB, beta, descrD, (mcspInt)nnzD,
                                           (mcspInt *)csrRowPtrD, (mcspInt *)csrColIndD, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinDcsrgemm2_bufferSizeExt(mcspHandle_t handle, int m, int n, int k, const double *alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const int *csrRowPtrA,
                                           const int *csrColIndA, const mcspMatDescr_t descrB, int nnzB,
                                           const int *csrRowPtrB, const int *csrColIndB, const double *beta,
                                           const mcspMatDescr_t descrD, int nnzD, const int *csrRowPtrD,
                                           const int *csrColIndD, mcspCsrgemm2Info_t info, size_t *pBufferSizeInBytes) {
    return mcspXcsrgemm2_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, alpha, descrA, (mcspInt)nnzA,
                                           (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, descrB, (mcspInt)nnzB,
                                           (mcspInt *)csrRowPtrB, (mcspInt *)csrColIndB, beta, descrD, (mcspInt)nnzD,
                                           (mcspInt *)csrRowPtrD, (mcspInt *)csrColIndD, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinCcsrgemm2_bufferSizeExt(mcspHandle_t handle, int m, int n, int k, const mcFloatComplex *alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const int *csrRowPtrA,
                                           const int *csrColIndA, const mcspMatDescr_t descrB, int nnzB,
                                           const int *csrRowPtrB, const int *csrColIndB, const mcFloatComplex *beta,
                                           const mcspMatDescr_t descrD, int nnzD, const int *csrRowPtrD,
                                           const int *csrColIndD, mcspCsrgemm2Info_t info, size_t *pBufferSizeInBytes) {
    return mcspXcsrgemm2_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, alpha, descrA, (mcspInt)nnzA,
                                           (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, descrB, (mcspInt)nnzB,
                                           (mcspInt *)csrRowPtrB, (mcspInt *)csrColIndB, beta, descrD, (mcspInt)nnzD,
                                           (mcspInt *)csrRowPtrD, (mcspInt *)csrColIndD, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinZcsrgemm2_bufferSizeExt(mcspHandle_t handle, int m, int n, int k, const mcDoubleComplex *alpha,
                                           const mcspMatDescr_t descrA, int nnzA, const int *csrRowPtrA,
                                           const int *csrColIndA, const mcspMatDescr_t descrB, int nnzB,
                                           const int *csrRowPtrB, const int *csrColIndB, const mcDoubleComplex *beta,
                                           const mcspMatDescr_t descrD, int nnzD, const int *csrRowPtrD,
                                           const int *csrColIndD, mcspCsrgemm2Info_t info, size_t *pBufferSizeInBytes) {
    return mcspXcsrgemm2_bufferSizeExtImpl(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, alpha, descrA, (mcspInt)nnzA,
                                           (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, descrB, (mcspInt)nnzB,
                                           (mcspInt *)csrRowPtrB, (mcspInt *)csrColIndB, beta, descrD, (mcspInt)nnzD,
                                           (mcspInt *)csrRowPtrD, (mcspInt *)csrColIndD, info, pBufferSizeInBytes);
}

mcspStatus_t mcspCuinXcsrgemm2Nnz(mcspHandle_t handle, int m, int n, int k, const mcspMatDescr_t descrA, int nnzA,
                                const int *csrRowPtrA, const int *csrColIndA, const mcspMatDescr_t descrB, int nnzB,
                                const int *csrRowPtrB, const int *csrColIndB, const mcspMatDescr_t descrD, int nnzD,
                                const int *csrRowPtrD, const int *csrColIndD, const mcspMatDescr_t descrC,
                                int *csrRowPtrC, int *nnzTotalDevHostPtr, const mcspCsrgemm2Info_t info,
                                void *pBuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    mcspInt nnzTotal = getScalarToHost(nnzTotalDevHostPtr, handle->ptr_mode);
    mcspStatus_t stat = mcspXcsrgemm2NnzImpl(
        handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, descrA, (mcspInt)nnzA, (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA,
        descrB, (mcspInt)nnzB, (mcspInt *)csrRowPtrB, (mcspInt *)csrColIndB, descrD, (mcspInt)nnzD,
        (mcspInt *)csrRowPtrD, (mcspInt *)csrColIndD, descrC, (mcspInt *)csrRowPtrC, &nnzTotal, info, pBuffer);
    if (handle->ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
        *nnzTotalDevHostPtr = nnzTotal;
    } else {
        MACA_ASSERT(mcMemcpy(nnzTotalDevHostPtr, &nnzTotal, sizeof(int), mcMemcpyHostToDevice));
    }
    return stat;
}

mcspStatus_t mcspCuinScsrgemm2(mcspHandle_t handle, int m, int n, int k, const float *alpha, const mcspMatDescr_t descrA,
                             int nnzA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                             const mcspMatDescr_t descrB, int nnzB, const float *csrValB, const int *csrRowPtrB,
                             const int *csrColIndB, const float *beta, const mcspMatDescr_t descrD, int nnzD,
                             const float *csrValD, const int *csrRowPtrD, const int *csrColIndD,
                             const mcspMatDescr_t descrC, float *csrValC, const int *csrRowPtrC, int *csrColIndC,
                             const mcspCsrgemm2Info_t info, void *pBuffer) {
    return mcspXcsrgemm2Impl(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, alpha, descrA, (mcspInt)nnzA, csrValA,
                             (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, descrB, (mcspInt)nnzB, csrValB,
                             (mcspInt *)csrRowPtrB, (mcspInt *)csrColIndB, beta, descrD, (mcspInt)nnzD, csrValD,
                             (mcspInt *)csrRowPtrD, (mcspInt *)csrColIndD, descrC, csrValC, (mcspInt *)csrRowPtrC,
                             (mcspInt *)csrColIndC, info, pBuffer);
}

mcspStatus_t mcspCuinDcsrgemm2(mcspHandle_t handle, int m, int n, int k, const double *alpha, const mcspMatDescr_t descrA,
                             int nnzA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                             const mcspMatDescr_t descrB, int nnzB, const double *csrValB, const int *csrRowPtrB,
                             const int *csrColIndB, const double *beta, const mcspMatDescr_t descrD, int nnzD,
                             const double *csrValD, const int *csrRowPtrD, const int *csrColIndD,
                             const mcspMatDescr_t descrC, double *csrValC, const int *csrRowPtrC, int *csrColIndC,
                             const mcspCsrgemm2Info_t info, void *pBuffer) {
    return mcspXcsrgemm2Impl(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, alpha, descrA, (mcspInt)nnzA, csrValA,
                             (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, descrB, (mcspInt)nnzB, csrValB,
                             (mcspInt *)csrRowPtrB, (mcspInt *)csrColIndB, beta, descrD, (mcspInt)nnzD, csrValD,
                             (mcspInt *)csrRowPtrD, (mcspInt *)csrColIndD, descrC, csrValC, (mcspInt *)csrRowPtrC,
                             (mcspInt *)csrColIndC, info, pBuffer);
}

mcspStatus_t mcspCuinCcsrgemm2(mcspHandle_t handle, int m, int n, int k, const mcFloatComplex *alpha,
                             const mcspMatDescr_t descrA, int nnzA, const mcFloatComplex *csrValA,
                             const int *csrRowPtrA, const int *csrColIndA, const mcspMatDescr_t descrB, int nnzB,
                             const mcFloatComplex *csrValB, const int *csrRowPtrB, const int *csrColIndB,
                             const mcFloatComplex *beta, const mcspMatDescr_t descrD, int nnzD,
                             const mcFloatComplex *csrValD, const int *csrRowPtrD, const int *csrColIndD,
                             const mcspMatDescr_t descrC, mcFloatComplex *csrValC, const int *csrRowPtrC,
                             int *csrColIndC, const mcspCsrgemm2Info_t info, void *pBuffer) {
    return mcspXcsrgemm2Impl(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, alpha, descrA, (mcspInt)nnzA, csrValA,
                             (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, descrB, (mcspInt)nnzB, csrValB,
                             (mcspInt *)csrRowPtrB, (mcspInt *)csrColIndB, beta, descrD, (mcspInt)nnzD, csrValD,
                             (mcspInt *)csrRowPtrD, (mcspInt *)csrColIndD, descrC, csrValC, (mcspInt *)csrRowPtrC,
                             (mcspInt *)csrColIndC, info, pBuffer);
}

mcspStatus_t mcspCuinZcsrgemm2(mcspHandle_t handle, int m, int n, int k, const mcDoubleComplex *alpha,
                             const mcspMatDescr_t descrA, int nnzA, const mcDoubleComplex *csrValA,
                             const int *csrRowPtrA, const int *csrColIndA, const mcspMatDescr_t descrB, int nnzB,
                             const mcDoubleComplex *csrValB, const int *csrRowPtrB, const int *csrColIndB,
                             const mcDoubleComplex *beta, const mcspMatDescr_t descrD, int nnzD,
                             const mcDoubleComplex *csrValD, const int *csrRowPtrD, const int *csrColIndD,
                             const mcspMatDescr_t descrC, mcDoubleComplex *csrValC, const int *csrRowPtrC,
                             int *csrColIndC, const mcspCsrgemm2Info_t info, void *pBuffer) {
    return mcspXcsrgemm2Impl(handle, (mcspInt)m, (mcspInt)n, (mcspInt)k, alpha, descrA, (mcspInt)nnzA, csrValA,
                             (mcspInt *)csrRowPtrA, (mcspInt *)csrColIndA, descrB, (mcspInt)nnzB, csrValB,
                             (mcspInt *)csrRowPtrB, (mcspInt *)csrColIndB, beta, descrD, (mcspInt)nnzD, csrValD,
                             (mcspInt *)csrRowPtrD, (mcspInt *)csrColIndD, descrC, csrValC, (mcspInt *)csrRowPtrC,
                             (mcspInt *)csrColIndC, info, pBuffer);
}

#ifdef __cplusplus
}
#endif
