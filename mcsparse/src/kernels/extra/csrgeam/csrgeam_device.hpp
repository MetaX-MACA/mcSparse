#ifndef KERNELS_EXTRA_CSRGEAM_CSRGEAM_DEVICE_HPP__
#define KERNELS_EXTRA_CSRGEAM_CSRGEAM_DEVICE_HPP__

#include <stdio.h>

#include "mcsp_config.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType>
__global__ void mcspCsrGeamNnzKernel(idxType m, idxType n, const idxType *csr_rows_A, const idxType *csr_cols_A,
                                     const idxType *csr_rows_B, const idxType *csr_cols_B, idxType *csr_rows_C,
                                     mcsparseIndexBase_t idx_base_A, mcsparseIndexBase_t idx_base_B) {
    const idxType row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= m) {
        return;
    }

    idxType row_nnz_C = 0;

    idxType row_start_A = csr_rows_A[row] - idx_base_A;
    idxType row_end_A = csr_rows_A[row + 1] - idx_base_A;

    idxType row_start_B = csr_rows_B[row] - idx_base_B;
    idxType row_end_B = csr_rows_B[row + 1] - idx_base_B;

    idxType col_A = (row_start_A < row_end_A) ? csr_cols_A[row_start_A] - idx_base_A : n;
    idxType col_B = (row_start_B < row_end_B) ? csr_cols_B[row_start_B] - idx_base_B : n;

    while (col_A != n || col_B != n) {
        idxType col_C = (col_A < col_B) ? col_A : col_B;
        if (col_A == col_C) {
            ++row_start_A;
            if (row_start_A < row_end_A) {
                col_A = csr_cols_A[row_start_A] - idx_base_A;
            } else {
                col_A = n;
            }
        }
        if (col_B == col_C) {
            ++row_start_B;
            if (row_start_B < row_end_B) {
                col_B = csr_cols_B[row_start_B] - idx_base_B;
            } else {
                col_B = n;
            }
        }

        ++row_nnz_C;
    }

    csr_rows_C[row] = row_nnz_C;
}

template <typename idxType, typename valType>
__global__ void mcspCsrGeamCalKernel(idxType m, idxType n, valType alpha, const idxType *csr_rows_A,
                                     const idxType *csr_cols_A, const valType *csr_vals_A, valType beta,
                                     const idxType *csr_rows_B, const idxType *csr_cols_B, const valType *csr_vals_B,
                                     const idxType *csr_rows_C, idxType *csr_cols_C, valType *csr_vals_C,
                                     mcsparseIndexBase_t idx_base_A, mcsparseIndexBase_t idx_base_B,
                                     mcsparseIndexBase_t idx_base_C) {
    const idxType row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= m) {
        return;
    }

    idxType row_start_A = csr_rows_A[row] - idx_base_A;
    idxType row_end_A = csr_rows_A[row + 1] - idx_base_A;

    idxType row_start_B = csr_rows_B[row] - idx_base_B;
    idxType row_end_B = csr_rows_B[row + 1] - idx_base_B;

    idxType row_start_C = csr_rows_C[row] - idx_base_C;

    idxType col_A = (row_start_A < row_end_A) ? csr_cols_A[row_start_A] - idx_base_A : n;
    idxType col_B = (row_start_B < row_end_B) ? csr_cols_B[row_start_B] - idx_base_B : n;

    while (col_A != n || col_B != n) {
        idxType col_C = (col_A < col_B) ? col_A : col_B;
        valType val_C = static_cast<valType>(0);
        if (col_A == col_C) {
            val_C += alpha * csr_vals_A[row_start_A];
            ++row_start_A;
            if (row_start_A < row_end_A) {
                col_A = csr_cols_A[row_start_A] - idx_base_A;
            } else {
                col_A = n;
            }
        }
        if (col_B == col_C) {
            val_C += beta * csr_vals_B[row_start_B];
            ++row_start_B;
            if (row_start_B < row_end_B) {
                col_B = csr_cols_B[row_start_B] - idx_base_B;
            } else {
                col_B = n;
            }
        }

        csr_cols_C[row_start_C] = col_C + idx_base_C;
        csr_vals_C[row_start_C] = val_C;
        ++row_start_C;
    }
}

#endif
