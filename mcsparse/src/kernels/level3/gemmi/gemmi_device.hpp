#ifndef KERNELS_BLAS_GEMMI_DEVICE_HPP__
#define KERNELS_BLAS_GEMMI_DEVICE_HPP__

#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspGemmiNTKernel(idxType m, idxType n, idxType k, const valType alpha, const valType *A, idxType lda,
                                  const valType *csr_val, const idxType *csr_row, const idxType *csr_col,
                                  const valType beta, valType *C, idxType ldc, mcsparseIndexBase_t idx_base) {
    idxType row = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    idxType col = blockIdx.y;
    if (row >= m || col >= n) {
        return;
    }

    idxType start = csr_row[col] - idx_base;
    idxType end = csr_row[col + 1] - idx_base;
    valType tmp = beta * C[row + col * ldc];
    // @TODO(zhiming): optimize to one warp calculate one nnz row using register, then warp reduce
    for (idxType i = start; i < end; i++) {
        if (csr_col[i] - idx_base >= k) {
            continue;
        }
        tmp += alpha * A[(csr_col[i] - idx_base) * lda + row] * csr_val[i];
    }
    C[row + col * ldc] = tmp;
}

#endif  // KERNELS_BLAS_GEMMI_DEVICE_HPP__
