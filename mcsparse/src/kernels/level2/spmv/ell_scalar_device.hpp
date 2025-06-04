#ifndef KERNELS_LEVEL2_SPMV_ELL_SCALAR_H__
#define KERNELS_LEVEL2_SPMV_ELL_SCALAR_H__

#include <cmath>

#include "mcsp_runtime_wrapper.h"
/**
 * @brief   compute ELL-based SpMV in double precision in GPU, ell_scalar algorithm
 *          y = alpha * A * x + beta * y
 *
 * @param row_num       [in]        number of rows
 * @param col_num       [in]        number of columns
 * @param ell_k         [in]        threshold value of ELL matrix
 * @param alpha         [in]        alpha
 * @param beta          [in]        beta
 * @param ell_cols      [in]        pointer to the column indexes of element in ELL
 * @param ell_vals      [in]        pointer to the values of element in ELL
 * @param x             [in]        pointer to the vector x
 * @param y             [in/out]    pointer to the vector y
 * @return void
 */
template <typename idxType, typename valType>
__global__ void mcspSpmvEllScalarKernel(const idxType row_num, const idxType col_num, const idxType ell_k,
                                        const valType alpha, const valType beta, const idxType *ell_cols,
                                        const valType *ell_vals, const valType *x, valType *y) {
    const idxType ix = threadIdx.x + blockIdx.x * blockDim.x;
    idxType col;

    if (ix < row_num) {
        valType ret = 0.0;
        for (idxType j = 0; j < ell_k; j++) {
            idxType ell_idx = j * row_num + ix;
            col = ell_cols[ell_idx];
            if (col != (idxType)(-1)) {
                ret += ell_vals[ell_idx] * x[col];
            }
        }
        y[ix] = beta * y[ix] + alpha * ret;
    }
}

/**
 * @brief   compute ELL-based SpMV in double precision in GPU, ell_vector algorithm
 *          y = alpha * A * x + beta * y
 *
 * @param row_num       [in]        number of rows
 * @param col_num       [in]        number of columns
 * @param ell_k         [in]        threshold value of ELL matrix
 * @param alpha         [in]        alpha
 * @param beta          [in]        beta
 * @param ell_cols      [in]        pointer to the column indexes of element in ELL
 * @param ell_vals      [in]        pointer to the values of element in ELL
 * @param x             [in]        pointer to the vector x
 * @param y             [in/out]    pointer to the vector y
 * @return void
 */
template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspSpmvEllVectorKernel(const idxType row_num, const idxType col_num, const idxType ell_k,
                                        const valType alpha, const valType beta, const idxType *ell_cols,
                                        const valType *ell_vals, const valType *x, valType *y) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);

    const idxType global_id = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType lane = threadIdx.x & (WARP_SIZE - 1);
    const idxType group = threadIdx.x / WARP_SIZE;
    const idxType group_num = BLOCK_SIZE / WARP_SIZE;
    const idxType row = blockIdx.x * WARP_SIZE + lane;
    const idxType idx = threadIdx.x;

    idxType col;

    if (row < row_num) {
        vals[idx] = (valType)0;
        idxType row_start = group * row_num + row;
        idxType row_end = ell_k * row_num;
        for (idxType j = row_start; j < row_end; j += group_num * row_num) {
            col = ell_cols[j];
            if (col != (idxType)(-1)) {
                vals[idx] += ell_vals[j] * x[col];
            }
        }
        __syncthreads();

#pragma unroll
        for (unsigned int i = BLOCK_SIZE >> 1; i >= WARP_SIZE; i >>= 1) {
            if (threadIdx.x < i) {
                vals[idx] += vals[idx + i];
            }
            __syncthreads();
        }

        idxType output_id = group + group_num * lane;
        idxType output_row = blockIdx.x * WARP_SIZE + output_id;
        if (output_id < WARP_SIZE) {
            y[output_row] = beta * y[output_row] + alpha * vals[output_id];
        }
    }
}

#endif
