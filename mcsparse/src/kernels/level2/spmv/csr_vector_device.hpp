#ifndef KERNELS_LEVEL2_SPMV_CSR_VECTOR_H__
#define KERNELS_LEVEL2_SPMV_CSR_VECTOR_H__

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"
/**
 * @brief   compute CSR-based SpMV in double precision in GPU, csr_vector algorithm
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

template <typename idxType, typename computeType, typename inputType, typename outputType>
__global__ void mcspSpmvCsrVectorKernel(const idxType row_num, const computeType alpha, const computeType beta,
                                        const idxType *csr_row_ptr, const idxType *csr_columns,
                                        const inputType *csr_values, const inputType *x, outputType *y,
                                        mcsparseIndexBase_t idx_base) {
    extern __shared__ __align__(sizeof(computeType)) unsigned char smem[];
    computeType *vals = reinterpret_cast<computeType *>(smem);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType row = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType idx = threadIdx.x;

    if (row < row_num) {
        idxType row_start = csr_row_ptr[row] - idx_base;
        idxType row_end = csr_row_ptr[row + 1] - idx_base;

        vals[idx] = 0;
        for (idxType j = row_start + lane; j < row_end; j += WARP_SIZE) {
            if constexpr (!std::is_same_v<computeType, inputType>) {
#if defined(__MACA__)
                if constexpr (std::is_same_v<inputType, __half> || std::is_same_v<inputType, mcsp_bfloat16>) {
                    vals[idx] +=
                        GetFloatFromLowReal(csr_values[j]) * GetFloatFromLowReal(x[(csr_columns[j] - idx_base)]);
                } else if constexpr (std::is_same_v<inputType, __half2> || std::is_same_v<inputType, mcsp_bfloat162>) {
                    vals[idx] +=
                        GetCf32FromLowComplex(csr_values[j]) * GetCf32FromLowComplex(x[(csr_columns[j] - idx_base)]);
                } else if constexpr (std::is_same_v<inputType, int8_t>) {
                    vals[idx] += static_cast<computeType>(csr_values[j]) *
                                 static_cast<computeType>(x[(csr_columns[j] - idx_base)]);
                }
#endif
            } else {
                vals[idx] += csr_values[j] * x[(csr_columns[j] - idx_base)];
            }
        }
#pragma unroll
        for (int i = WARP_SIZE >> 1; i != 0; i >>= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (lane < i) vals[idx] += vals[idx + i];
        }

        if (lane == 0) {
            if constexpr (!std::is_same_v<computeType, outputType>) {
                if constexpr (std::is_same_v<outputType, __half> || std::is_same_v<outputType, mcsp_bfloat16>) {
                    y[row] = GetTypedValue<outputType>(beta * GetFloatFromLowReal(y[row]) + alpha * vals[idx]);
                } else if constexpr (std::is_same_v<outputType, __half2> ||
                                     std::is_same_v<outputType, mcsp_bfloat162>) {
                    y[row] = GetLowComplexType<outputType>(beta * GetCf32FromLowComplex(y[row]) + alpha * vals[idx]);
                }
            } else {
                y[row] = beta * y[row] + alpha * vals[idx];
            }
        }
    }
}

template <typename idxType, typename sparseType, typename computeType>
__global__ void mcspSpmvCsrVectorMixedRealComplexKernel(const idxType row_num, const computeType alpha,
                                                        const computeType beta, const idxType *csr_row_ptr,
                                                        const idxType *csr_columns, const sparseType *csr_values,
                                                        const computeType *x, computeType *y,
                                                        mcsparseIndexBase_t idx_base) {
    extern __shared__ __align__(sizeof(computeType)) unsigned char smem[];
    computeType *vals = reinterpret_cast<computeType *>(smem);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType row = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType idx = threadIdx.x;

    if (row < row_num) {
        idxType row_start = csr_row_ptr[row] - idx_base;
        idxType row_end = csr_row_ptr[row + 1] - idx_base;

        vals[idx] = 0;
        for (idxType j = row_start + lane; j < row_end; j += WARP_SIZE) {
            vals[idx].x += csr_values[j] * x[(csr_columns[j] - idx_base)].x;
            vals[idx].y += csr_values[j] * x[(csr_columns[j] - idx_base)].y;
        }
#pragma unroll
        for (int i = WARP_SIZE >> 1; i != 0; i >>= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (lane < i) vals[idx] += vals[idx + i];
        }

        if (lane == 0) {
            y[row] = beta * y[row] + alpha * vals[idx];
        }
    }
}

// only for half complex and bfloat complex data type
template <typename idxType, typename computeType>
__global__ void mcspSpmvCsrVectorKernelLowPrecisionComplex(const idxType row_num, const computeType alpha,
                                                           const computeType beta, const idxType *csr_row_ptr,
                                                           const idxType *csr_columns, const computeType *csr_values,
                                                           const computeType *x, computeType *y,
                                                           mcsparseIndexBase_t idx_base) {
    extern __shared__ __align__(sizeof(computeType)) unsigned char smem[];
    computeType *vals = reinterpret_cast<computeType *>(smem);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType row = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType idx = threadIdx.x;

    if (row < row_num) {
        idxType row_start = csr_row_ptr[row] - idx_base;
        idxType row_end = csr_row_ptr[row + 1] - idx_base;
        vals[idx] = GetTypedValue<computeType>(0);
        for (idxType j = row_start + lane; j < row_end; j += WARP_SIZE) {
            vals[idx] += complex_mul(csr_values[j], x[(csr_columns[j] - idx_base)]);
        }
#pragma unroll
        for (int i = WARP_SIZE >> 1; i != 0; i >>= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (lane < i) vals[idx] += vals[idx + i];
        }

        if (lane == 0) {
            y[row] = complex_mul(beta, y[row]) + complex_mul(alpha, vals[idx]);
        }
    }
}

#endif
