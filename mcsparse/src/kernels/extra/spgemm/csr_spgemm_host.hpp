#ifndef KERNELS_EXTRA_SPGEMM_CSR_SPGEMM_HOST_HPP__
#define KERNELS_EXTRA_SPGEMM_CSR_SPGEMM_HOST_HPP__

#include "common/mcsp_types.h"
#include "internal_interface/mcsp_internal_helper.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"

template <typename idxType>
mcspStatus_t mcspCsrgemmBuffersizeTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A,
                                           mcsparseOperation_t trans_B, idxType m, idxType n, idxType k,
                                           mcspMatDescr_t descr_A, idxType nnz_A, const idxType *csr_rows_A,
                                           const idxType *csr_cols_A, mcspMatDescr_t descr_B, idxType nnz_B,
                                           const idxType *csr_rows_B, const idxType *csr_cols_B, mcspMatDescr_t descr_C,
                                           idxType nnz_C, const idxType *csr_rows_C, const idxType *csr_cols_C,
                                           mcspMatInfo_t info_D, size_t *buffer_size);

template <typename idxType>
mcspStatus_t mcspCsrgemmNnzTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    idxType m, idxType n, idxType k, mcspMatDescr_t descr_A, idxType nnz_A,
                                    const idxType *csr_rows_A, const idxType *csr_cols_A, mcspMatDescr_t descr_B,
                                    idxType nnz_B, const idxType *csr_rows_B, const idxType *csr_cols_B,
                                    mcspMatDescr_t descr_C, idxType nnz_C, const idxType *csr_rows_C,
                                    const idxType *csr_cols_C, mcspMatDescr_t descr_D, idxType *csr_rows_D,
                                    idxType *nnz_D, mcspMatInfo_t info_D, void *buffer, bool include_addition = true);

template <typename idxType, typename valType>
mcspStatus_t mcspCsrgemmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                 idxType m, idxType n, idxType k, const valType *alpha, mcspMatDescr_t descr_A,
                                 idxType nnz_A, const valType *csr_vals_A, const idxType *csr_rows_A,
                                 const idxType *csr_cols_A, mcspMatDescr_t descr_B, idxType nnz_B,
                                 const valType *csr_vals_B, const idxType *csr_rows_B, const idxType *csr_cols_B,
                                 const valType *beta, mcspMatDescr_t descr_C, idxType nnz_C, const valType *csr_vals_C,
                                 const idxType *csr_rows_C, const idxType *csr_cols_C, mcspMatDescr_t descr_D,
                                 valType *csr_vals_D, const idxType *csr_rows_D, idxType *csr_cols_D,
                                 mcspMatInfo_t info_D, void *buffer, bool include_addition = true);

#endif