#ifndef KERNELS_LEVEL2_SPMV_CSR_SCALAR_H__
#define KERNELS_LEVEL2_SPMV_CSR_SCALAR_H__

#include "mcsp_runtime_wrapper.h"
/**
 * @brief   compute CSR-based SpMV in double precision in GPU, csr_scalar algorithm
 *          y = alpha * A * x + beta * y
 *
 * @param row_num       [in]        number of rows
 * @param alpha         [in]        alpha
 * @param beta          [in]        beta
 * @param csr_row_ptr   [in]        pointer to the row offset in CSR
 * @param csr_columns   [in]        pointer to the column indexes of nonzeros in CSR
 * @param csr_values    [in]        pointer to the values of nonzeros in CSR
 * @param x             [in]        pointer to the vector x
 * @param y             [in/out]    pointer to the vector y
 * @return void
 */
template <typename idxType, typename computeType>
__global__ void mcspSpmvCsrScalarKernel(const idxType row_num, const computeType alpha, const computeType beta,
                                        const idxType *csr_row_ptr, const idxType *csr_columns,
                                        const computeType *csr_values, const computeType *x, computeType *y) {
    const idxType ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < row_num) {
        computeType ret = 0.0;
        idxType row_start = csr_row_ptr[ix];
        idxType row_end = csr_row_ptr[ix + 1];
        for (idxType j = row_start; j < row_end; j++) {
            ret += csr_values[j] * x[csr_columns[j]];
        }
        y[ix] = beta * y[ix] + alpha * ret;
    }
}

#endif
