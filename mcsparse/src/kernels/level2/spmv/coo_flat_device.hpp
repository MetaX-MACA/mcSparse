#ifndef KERNELS_LEVEL2_SPMV_COO_FLAT_H__
#define KERNELS_LEVEL2_SPMV_COO_FLAT_H__

#include "common/internal/mcsp_bfloat16.hpp"
#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

#ifdef __MACA__
#define COO_BLOCK_SIZE 512
#else
#define COO_BLOCK_SIZE 1024
#endif

template <typename idxType, typename computeType, typename dataType>
__global__ void mcspCooScalingKernel(idxType size, computeType beta, dataType *data) {
    idxType tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        if constexpr (!std::is_same_v<computeType, dataType>) {
#if defined(__MACA__)
            if constexpr (std::is_same_v<dataType, __half> || std::is_same_v<dataType, mcsp_bfloat16>) {
                data[tid] = GetTypedValue<dataType>(beta * GetFloatFromLowReal(data[tid]));
            } else if constexpr (std::is_same_v<dataType, __half2> || std::is_same_v<dataType, mcsp_bfloat162>) {
                data[tid] = GetLowComplexType<dataType>(beta * GetCf32FromLowComplex(data[tid]));
            } else if constexpr (std::is_same_v<dataType, int8_t>) {
                data[tid] = static_cast<dataType>(beta * static_cast<computeType>(data[tid]));
            }
#endif
        } else {
            data[tid] *= beta;
        }
    }
}

template <typename idxType, typename computeType>
__global__ void mcspCooScalingKernelLowPrecisionComplex(idxType size, computeType beta, computeType *data) {
    idxType tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        data[tid] = complex_mul(data[tid], beta);
    }
}

/**
 * @brief   compute COO-based SpMV in double precision in GPU, coo_flat algorithm
 *          This algorithm needs A in row-major order
 *          1st step: warp-wise block body calculation and write out carries
 *          ref: Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors,
 *               2009
 *          y = alpha * A * x + beta * y
 *
 * @param interval_size [in]        number of non-zeros handled by one warp
 * @param nnz           [in]        number of non-zeros
 * @param alpha         [in]        alpha
 * @param beta          [in]        beta
 * @param coo_rows      [in]        pointer to the row indexes of nonzeros in COO
 * @param coo_columns   [in]        pointer to the column indexes of nonzeros in COO
 * @param coo_values    [in]        pointer to the values of nonzeros in COO
 * @param x             [in]        pointer to the vector x
 * @param y             [in/out]    pointer to the vector y
 * @param carries_rows  [out]       pointer to the row indexes of carries to be handled in 2nd step
 * @param carries       [out]       pointer to the value of carries to be hanled in 2nd step
 * @return void
 */
template <typename idxType, typename computeType, typename inputType, typename outputType>
__global__ void mcspSpmvCooBodyKernel(const idxType interval_size, const idxType nnz, const computeType alpha,
                                      const idxType *coo_rows, const idxType *coo_columns, const inputType *coo_values,
                                      const inputType *x, outputType *y, idxType *carries_rows, computeType *carries,
                                      mcsparseIndexBase_t idx_base) {
    __shared__ __align__(sizeof(idxType)) unsigned char smem1[COO_BLOCK_SIZE * 3 / 2 * sizeof(idxType)];
    idxType *rows_inner = reinterpret_cast<idxType *>(smem1);
    __shared__ __align__(sizeof(computeType)) unsigned char smem2[COO_BLOCK_SIZE * sizeof(computeType)];
    computeType *vals = reinterpret_cast<computeType *>(smem2);

    const idxType tid = COO_BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType warp_id = tid / WARP_SIZE;

    const idxType interval_begin = warp_id * interval_size;
    const idxType interval_end = (interval_begin + interval_size < nnz) ? interval_begin + interval_size : nnz;

    const idxType inner_idx = WARP_SIZE / 2 * (threadIdx.x / WARP_SIZE + 1) + threadIdx.x;

    rows_inner[inner_idx - WARP_SIZE / 2] = (idxType)-1;

    if (lane == WARP_SIZE - 1) {
        rows_inner[inner_idx] = (idxType)(-1);
        vals[threadIdx.x] = GetTypedValue<computeType>(0);
    }

    idxType row;
    computeType val;
    __syncthreads();
    for (idxType global_idx = interval_begin + lane; global_idx < interval_end; global_idx += WARP_SIZE) {
        row = coo_rows[global_idx] - idx_base;
        if constexpr (!std::is_same_v<computeType, inputType>) {
#if defined(__MACA__)
            if constexpr (std::is_same_v<inputType, __half> || std::is_same_v<inputType, mcsp_bfloat16>) {
                val = alpha * GetFloatFromLowReal(coo_values[global_idx]) *
                      GetFloatFromLowReal(x[coo_columns[global_idx] - idx_base]);
            } else if constexpr (std::is_same_v<inputType, __half2> || std::is_same_v<inputType, mcsp_bfloat162>) {
                val = alpha * GetCf32FromLowComplex(coo_values[global_idx]) *
                      GetCf32FromLowComplex(x[coo_columns[global_idx] - idx_base]);
            } else if constexpr (std::is_same_v<inputType, int8_t>) {
                val = alpha * static_cast<computeType>(coo_values[global_idx]) *
                      static_cast<computeType>(x[coo_columns[global_idx] - idx_base]);
            }
#endif
        } else {
            val = alpha * coo_values[global_idx] * x[coo_columns[global_idx] - idx_base];
        }

        if (lane == 0) {
            if (row == rows_inner[inner_idx + WARP_SIZE - 1]) {
                val += vals[threadIdx.x + WARP_SIZE - 1];
            } else if (rows_inner[inner_idx + WARP_SIZE - 1] != (idxType)(-1)) {
                if constexpr (!std::is_same_v<computeType, outputType>) {
#if defined(__MACA__)
                    if constexpr (std::is_same_v<outputType, __half> || std::is_same_v<outputType, mcsp_bfloat16>) {
                        y[rows_inner[inner_idx + WARP_SIZE - 1]] =
                            GetTypedValue<outputType>(vals[threadIdx.x + WARP_SIZE - 1]) +
                            y[rows_inner[inner_idx + WARP_SIZE - 1]];
                    } else if constexpr (std::is_same_v<outputType, __half2> ||
                                         std::is_same_v<outputType, mcsp_bfloat162>) {
                        y[rows_inner[inner_idx + WARP_SIZE - 1]] = GetLowComplexType<outputType>(
                            vals[threadIdx.x + WARP_SIZE - 1] +
                            GetCf32FromLowComplex(y[rows_inner[inner_idx + WARP_SIZE - 1]]));
                    }
#endif
                } else {
                    y[rows_inner[inner_idx + WARP_SIZE - 1]] += vals[threadIdx.x + WARP_SIZE - 1];
                }
            }
        }

        rows_inner[inner_idx] = row;
        vals[threadIdx.x] = val;

        __syncthreads();
#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (row == rows_inner[inner_idx - i]) {
                vals[threadIdx.x] += vals[threadIdx.x - i];
            }
        }

        __syncthreads();
        if ((lane < WARP_SIZE - 1) && (row != rows_inner[inner_idx + 1])) {
            if constexpr (!std::is_same_v<computeType, outputType>) {
#if defined(__MACA__)
                if constexpr (std::is_same_v<outputType, __half> || std::is_same_v<outputType, mcsp_bfloat16>) {
                    y[row] = GetTypedValue<outputType>(vals[threadIdx.x]) + y[row];
                } else if constexpr (std::is_same_v<outputType, __half2> ||
                                     std::is_same_v<outputType, mcsp_bfloat162>) {
                    y[row] = GetLowComplexType<outputType>(vals[threadIdx.x] + GetCf32FromLowComplex(y[row]));
                }
#endif
            } else {
                y[row] += vals[threadIdx.x];
            }
        }
    }

    if (lane == WARP_SIZE - 1) {
        carries_rows[warp_id] = row;
        carries[warp_id] = vals[threadIdx.x];
    }
}

template <typename idxType, typename sparseType, typename computeType>
__global__ void mcspSpmvCooBodyMixedRealComplexKernel(const idxType interval_size, const idxType nnz,
                                                      const computeType alpha, const idxType *coo_rows,
                                                      const idxType *coo_columns, const sparseType *coo_values,
                                                      const computeType *x, computeType *y, idxType *carries_rows,
                                                      computeType *carries, mcsparseIndexBase_t idx_base) {
    __shared__ __align__(sizeof(idxType)) unsigned char smem1[COO_BLOCK_SIZE * 3 / 2 * sizeof(idxType)];
    idxType *rows_inner = reinterpret_cast<idxType *>(smem1);
    __shared__ __align__(sizeof(computeType)) unsigned char smem2[COO_BLOCK_SIZE * sizeof(computeType)];
    computeType *vals = reinterpret_cast<computeType *>(smem2);

    const idxType tid = COO_BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType warp_id = tid / WARP_SIZE;

    const idxType interval_begin = warp_id * interval_size;
    const idxType interval_end = (interval_begin + interval_size < nnz) ? interval_begin + interval_size : nnz;

    const idxType inner_idx = WARP_SIZE / 2 * (threadIdx.x / WARP_SIZE + 1) + threadIdx.x;

    rows_inner[inner_idx - WARP_SIZE / 2] = (idxType)-1;

    if (lane == WARP_SIZE - 1) {
        rows_inner[inner_idx] = (idxType)(-1);
        vals[threadIdx.x] = GetTypedValue<computeType>(0);
    }

    idxType row;
    computeType val;
    __syncthreads();
    for (idxType global_idx = interval_begin + lane; global_idx < interval_end; global_idx += WARP_SIZE) {
        row = coo_rows[global_idx] - idx_base;

        val.x = coo_values[global_idx] * (alpha * x[coo_columns[global_idx] - idx_base]).x;
        val.y = coo_values[global_idx] * (alpha * x[coo_columns[global_idx] - idx_base]).y;

        if (lane == 0) {
            if (row == rows_inner[inner_idx + WARP_SIZE - 1]) {
                val += vals[threadIdx.x + WARP_SIZE - 1];
            } else if (rows_inner[inner_idx + WARP_SIZE - 1] != (idxType)(-1)) {
                y[rows_inner[inner_idx + WARP_SIZE - 1]] += vals[threadIdx.x + WARP_SIZE - 1];
            }
        }

        rows_inner[inner_idx] = row;
        vals[threadIdx.x] = val;

        __syncthreads();
#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (row == rows_inner[inner_idx - i]) {
                vals[threadIdx.x] += vals[threadIdx.x - i];
            }
        }

        __syncthreads();
        if ((lane < WARP_SIZE - 1) && (row != rows_inner[inner_idx + 1])) {
            y[row] += vals[threadIdx.x];
        }
    }

    if (lane == WARP_SIZE - 1) {
        carries_rows[warp_id] = row;
        carries[warp_id] = vals[threadIdx.x];
    }
}

template <typename idxType, typename computeType>
__global__ void mcspSpmvCooBodyKernelLowPrecisionComplex(const idxType interval_size, const idxType nnz,
                                                         const computeType alpha, const idxType *coo_rows,
                                                         const idxType *coo_columns, const computeType *coo_values,
                                                         const computeType *x, computeType *y, idxType *carries_rows,
                                                         computeType *carries, mcsparseIndexBase_t idx_base) {
    __shared__ __align__(sizeof(idxType)) unsigned char smem1[COO_BLOCK_SIZE * 3 / 2 * sizeof(idxType)];
    idxType *rows_inner = reinterpret_cast<idxType *>(smem1);
    __shared__ __align__(sizeof(computeType)) unsigned char smem2[COO_BLOCK_SIZE * sizeof(computeType)];
    computeType *vals = reinterpret_cast<computeType *>(smem2);

    const idxType tid = COO_BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType warp_id = tid / WARP_SIZE;

    const idxType interval_begin = warp_id * interval_size;
    const idxType interval_end = (interval_begin + interval_size < nnz) ? interval_begin + interval_size : nnz;

    const idxType inner_idx = WARP_SIZE / 2 * (threadIdx.x / WARP_SIZE + 1) + threadIdx.x;

    rows_inner[inner_idx - WARP_SIZE / 2] = (idxType)-1;

    if (lane == WARP_SIZE - 1) {
        rows_inner[inner_idx] = (idxType)(-1);
        vals[threadIdx.x] = GetTypedValue<computeType>(0);
    }

    idxType row;
    computeType val;
    __syncthreads();
    for (idxType global_idx = interval_begin + lane; global_idx < interval_end; global_idx += WARP_SIZE) {
        row = coo_rows[global_idx] - idx_base;
        val = complex_mul(alpha, complex_mul(coo_values[global_idx], x[coo_columns[global_idx] - idx_base]));

        if (lane == 0) {
            if (row == rows_inner[inner_idx + WARP_SIZE - 1]) {
                val += vals[threadIdx.x + WARP_SIZE - 1];
            } else if (rows_inner[inner_idx + WARP_SIZE - 1] != (idxType)(-1)) {
                y[rows_inner[inner_idx + WARP_SIZE - 1]] += vals[threadIdx.x + WARP_SIZE - 1];
            }
        }

        rows_inner[inner_idx] = row;
        vals[threadIdx.x] = val;

        __syncthreads();
#pragma unroll
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (row == rows_inner[inner_idx - i]) {
                vals[threadIdx.x] += vals[threadIdx.x - i];
            }
        }

        __syncthreads();
        if ((lane < WARP_SIZE - 1) && (row != rows_inner[inner_idx + 1])) {
            y[row] += vals[threadIdx.x];
        }
    }

    if (lane == WARP_SIZE - 1) {
        carries_rows[warp_id] = row;
        carries[warp_id] = vals[threadIdx.x];
    }
}

/**
 * @brief   compute COO-based SpMV in double precision in GPU, coo_flat algorithm
 *          This algorithm needs A in row-major order
 *          2nd step: carries reduction by one single block
 *          ref: Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors,
 *               2009
 *          y = alpha * A * x + beta * y
 *
 * @param warp_num      [in]        number of warps/carries to be reduced
 * @param carries_rows  [out]       pointer to the row indexes of carries
 * @param carries       [out]       pointer to the value of carries
 * @param y             [in/out]    pointer to the vector y
 * @return void
 */
template <typename idxType, typename computeType, typename outputType>
__global__ void mcspSpmvCooCarriesReduceKernel(const idxType warp_num, const idxType *carries_rows,
                                               const computeType *carries, outputType *y) {
    __shared__ __align__(sizeof(idxType)) unsigned char smem1[(COO_BLOCK_SIZE * 3 / 2 + 1) * sizeof(idxType)];
    idxType *rows = reinterpret_cast<idxType *>(smem1);
    __shared__ __align__(sizeof(computeType)) unsigned char smem2[(COO_BLOCK_SIZE * 3 / 2 + 1) * sizeof(computeType)];
    computeType *vals = reinterpret_cast<computeType *>(smem2);

    const idxType inner_idx = threadIdx.x + COO_BLOCK_SIZE / 2;
    const idxType tail_start = (warp_num / COO_BLOCK_SIZE) * COO_BLOCK_SIZE;

    computeType left = GetTypedValue<computeType>(0);

    rows[inner_idx - COO_BLOCK_SIZE / 2] = -1;
    vals[inner_idx - COO_BLOCK_SIZE / 2] = GetTypedValue<computeType>(0);

    if (threadIdx.x == 0) {
        rows[COO_BLOCK_SIZE * 3 / 2] = -1;
        vals[COO_BLOCK_SIZE * 3 / 2] = GetTypedValue<computeType>(0);
    }

    idxType row;
    computeType val;
    idxType end_idx;
    idxType global_idx;

    __syncthreads();

    for (global_idx = threadIdx.x; global_idx < tail_start; global_idx += COO_BLOCK_SIZE) {
        row = carries_rows[global_idx];
        val = carries[global_idx];

        rows[inner_idx] = row;
        vals[inner_idx] = val;
        __syncthreads();

#pragma unroll
        for (int i = 1; i < COO_BLOCK_SIZE; i <<= 1) {
            if (row == rows[inner_idx - i]) {
                left = vals[inner_idx - i];
            }
            __syncthreads();
            vals[inner_idx] += left;
            left = GetTypedValue<computeType>(0);
            __syncthreads();
        }

        if (row != rows[inner_idx + 1]) {
            if constexpr (!std::is_same_v<computeType, outputType>) {
#if defined(__MACA__)
                if constexpr (std::is_same_v<outputType, __half> || std::is_same_v<outputType, mcsp_bfloat16>) {
                    y[rows[inner_idx]] = GetTypedValue<outputType>(vals[inner_idx]) + y[rows[inner_idx]];
                } else if constexpr (std::is_same_v<outputType, __half2> ||
                                     std::is_same_v<outputType, mcsp_bfloat162>) {
                    y[rows[inner_idx]] =
                        GetLowComplexType<outputType>(vals[inner_idx] + GetCf32FromLowComplex(y[rows[inner_idx]]));
                }
#endif
            } else {
                y[rows[inner_idx]] += vals[inner_idx];
            }
        }
        __syncthreads();
    }

    if (tail_start == warp_num) {
        return;
    }

    global_idx = tail_start + threadIdx.x;
    if (global_idx < warp_num) {
        row = carries_rows[global_idx];
        val = carries[global_idx];
    } else {
        row = -1;
        val = GetTypedValue<computeType>(0);
    }
    rows[inner_idx] = row;
    vals[inner_idx] = val;
    end_idx = warp_num - tail_start - 1;
    __syncthreads();

#pragma unroll
    for (int i = 1; i < COO_BLOCK_SIZE; i <<= 1) {
        if (row == rows[inner_idx - i]) {
            left = vals[inner_idx - i];
        }
        __syncthreads();
        vals[inner_idx] += left;
        left = GetTypedValue<computeType>(0);
        __syncthreads();
    }

    if (global_idx < warp_num && row != rows[inner_idx + 1]) {
        if constexpr (!std::is_same_v<computeType, outputType>) {
#if defined(__MACA__)
            if constexpr (std::is_same_v<outputType, __half> || std::is_same_v<outputType, mcsp_bfloat16>) {
                y[rows[inner_idx]] = GetTypedValue<outputType>(vals[inner_idx]) + y[rows[inner_idx]];
            } else if constexpr (std::is_same_v<outputType, __half2> || std::is_same_v<outputType, mcsp_bfloat162>) {
                y[rows[inner_idx]] =
                    GetLowComplexType<outputType>(vals[inner_idx] + GetCf32FromLowComplex(y[rows[inner_idx]]));
            }
#endif
        } else {
            y[rows[inner_idx]] += vals[inner_idx];
        }
    }
}

/**
 * @brief   compute COO-based SpMV in double precision in GPU, coo_flat algorithm
 *          This algorithm needs A in row-major order
 *          3rd step: tail calculation by one block
 *          ref: Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors,
 *               2009
 *          y = alpha * A * x + beta * y
 *
 * @param tail_num      [in]        number of remaining non-zeros
 * @param alpha         [in]        alpha
 * @param beta          [in]        beta
 * @param rows_tail     [in]        pointer to the tail row indexes of nonzeros in COO
 * @param columns_tail  [in]        pointer to the tail column indexes of nonzeros in COO
 * @param values_tail   [in]        pointer to the tail values of nonzeros in COO
 * @param x             [in]        pointer to the vector x
 * @param y             [in/out]    pointer to the vector y
 * @return void
 */
template <typename idxType, typename computeType, typename inputType, typename outputType>
__global__ void mcspSpmvCooTailKernel(idxType tail_num, const computeType alpha, const idxType *row_tail,
                                      const idxType *columns_tail, const inputType *values_tail, const inputType *x,
                                      outputType *y, mcsparseIndexBase_t idx_base) {
    __shared__ __align__(sizeof(idxType)) unsigned char smem1[(COO_BLOCK_SIZE * 3 / 2 + 1) * sizeof(idxType)];
    idxType *rows = reinterpret_cast<idxType *>(smem1);
    __shared__ __align__(sizeof(computeType)) unsigned char smem2[(COO_BLOCK_SIZE * 3 / 2 + 1) * sizeof(computeType)];
    computeType *vals = reinterpret_cast<computeType *>(smem2);

    const idxType inner_idx = threadIdx.x + COO_BLOCK_SIZE / 2;
    const idxType tid = threadIdx.x;

    computeType left = GetTypedValue<computeType>(0);

    rows[inner_idx - COO_BLOCK_SIZE / 2] = -1;
    vals[inner_idx - COO_BLOCK_SIZE / 2] = GetTypedValue<computeType>(0);

    if (threadIdx.x == 0) {
        rows[COO_BLOCK_SIZE * 3 / 2] = -1;
        vals[COO_BLOCK_SIZE * 3 / 2] = GetTypedValue<computeType>(0);
    }

    const idxType tail_start = (tail_num / COO_BLOCK_SIZE) * COO_BLOCK_SIZE;
    const idxType tail_cnt = tail_num - tail_start;
    idxType global_idx;
    idxType row;
    computeType val;

    __syncthreads();
    for (global_idx = threadIdx.x; global_idx < tail_start; global_idx += COO_BLOCK_SIZE) {
        row = row_tail[global_idx] - idx_base;
        if constexpr (!std::is_same_v<computeType, inputType>) {
#if defined(__MACA__)
            if constexpr (std::is_same_v<inputType, __half> || std::is_same_v<inputType, mcsp_bfloat16>) {
                val = alpha * GetFloatFromLowReal(values_tail[global_idx]) *
                      GetFloatFromLowReal(x[columns_tail[global_idx] - idx_base]);
            } else if constexpr (std::is_same_v<inputType, __half2> || std::is_same_v<inputType, mcsp_bfloat162>) {
                val = alpha * GetCf32FromLowComplex(values_tail[global_idx]) *
                      GetCf32FromLowComplex(x[columns_tail[global_idx] - idx_base]);
            } else if constexpr (std::is_same_v<inputType, int8_t>) {
                val = alpha * static_cast<computeType>(values_tail[global_idx]) *
                      static_cast<computeType>(x[columns_tail[global_idx] - idx_base]);
            }
#endif
        } else {
            val = alpha * values_tail[global_idx] * x[columns_tail[global_idx] - idx_base];
        }
        rows[inner_idx] = row;
        vals[inner_idx] = val;

        __syncthreads();
#pragma unroll
        for (int i = 1; i < COO_BLOCK_SIZE; i <<= 1) {
            if (row == rows[inner_idx - i]) {
                left = vals[inner_idx - i];
            }
            __syncthreads();
            vals[inner_idx] += left;
            left = GetTypedValue<computeType>(0);
            __syncthreads();
        }

        if (row != rows[inner_idx + 1]) {
            if constexpr (!std::is_same_v<computeType, outputType>) {
#if defined(__MACA__)
                if constexpr (std::is_same_v<outputType, __half> || std::is_same_v<outputType, mcsp_bfloat16>) {
                    y[row] = GetTypedValue<outputType>(vals[inner_idx]) + y[row];
                } else if constexpr (std::is_same_v<outputType, __half2> ||
                                     std::is_same_v<outputType, mcsp_bfloat162>) {
                    y[row] = GetLowComplexType<outputType>(vals[inner_idx] + GetCf32FromLowComplex(y[row]));
                }
#endif
            } else {
                y[row] += vals[inner_idx];
            }
        }
        __syncthreads();
    }

    if (tail_start == tail_num) {
        return;
    }

    global_idx = tail_start + threadIdx.x;
    if (global_idx < tail_num) {
        row = row_tail[global_idx] - idx_base;
        if constexpr (!std::is_same_v<computeType, inputType>) {
#if defined(__MACA__)
            if constexpr (std::is_same_v<inputType, __half> || std::is_same_v<inputType, mcsp_bfloat16>) {
                val = alpha * GetFloatFromLowReal(values_tail[global_idx]) *
                      GetFloatFromLowReal(x[columns_tail[global_idx] - idx_base]);
            } else if constexpr (std::is_same_v<inputType, __half2> || std::is_same_v<inputType, mcsp_bfloat162>) {
                val = alpha * GetCf32FromLowComplex(values_tail[global_idx]) *
                      GetCf32FromLowComplex(x[columns_tail[global_idx] - idx_base]);
            } else if constexpr (std::is_same_v<inputType, int8_t>) {
                val = alpha * static_cast<computeType>(values_tail[global_idx]) *
                      static_cast<computeType>(x[columns_tail[global_idx] - idx_base]);
            }
#endif
        } else {
            val = alpha * values_tail[global_idx] * x[columns_tail[global_idx] - idx_base];
        }
    } else {
        row = -1;
        val = GetTypedValue<computeType>(0);
    }
    rows[inner_idx] = row;
    vals[inner_idx] = val;

    __syncthreads();
#pragma unroll
    for (int i = 1; i < COO_BLOCK_SIZE; i <<= 1) {
        if (row == rows[inner_idx - i]) {
            left = vals[inner_idx - i];
        }
        __syncthreads();
        vals[inner_idx] += left;
        left = GetTypedValue<computeType>(0);
        __syncthreads();
    }

    if (global_idx < tail_num && row != rows[inner_idx + 1]) {
        if constexpr (!std::is_same_v<computeType, outputType>) {
#if defined(__MACA__)
            if constexpr (std::is_same_v<outputType, __half> || std::is_same_v<outputType, mcsp_bfloat16>) {
                y[rows[inner_idx]] = GetTypedValue<outputType>(vals[inner_idx]) + y[rows[inner_idx]];
            } else if constexpr (std::is_same_v<outputType, __half2> || std::is_same_v<outputType, mcsp_bfloat162>) {
                y[rows[inner_idx]] =
                    GetLowComplexType<outputType>(vals[inner_idx] + GetCf32FromLowComplex(y[rows[inner_idx]]));
            }
#endif
        } else {
            y[rows[inner_idx]] += vals[inner_idx];
        }
    }
}

template <typename idxType, typename sparseType, typename computeType>
__global__ void mcspSpmvCooTailMixedRealComplexKernel(idxType tail_num, const computeType alpha,
                                                      const idxType *row_tail, const idxType *columns_tail,
                                                      const sparseType *values_tail, const computeType *x,
                                                      computeType *y, mcsparseIndexBase_t idx_base) {
    __shared__ __align__(sizeof(idxType)) unsigned char smem1[(COO_BLOCK_SIZE * 3 / 2 + 1) * sizeof(idxType)];
    idxType *rows = reinterpret_cast<idxType *>(smem1);
    __shared__ __align__(sizeof(computeType)) unsigned char smem2[(COO_BLOCK_SIZE * 3 / 2 + 1) * sizeof(computeType)];
    computeType *vals = reinterpret_cast<computeType *>(smem2);

    const idxType inner_idx = threadIdx.x + COO_BLOCK_SIZE / 2;
    const idxType tid = threadIdx.x;

    computeType left = GetTypedValue<computeType>(0);

    rows[inner_idx - COO_BLOCK_SIZE / 2] = -1;
    vals[inner_idx - COO_BLOCK_SIZE / 2] = GetTypedValue<computeType>(0);

    if (threadIdx.x == 0) {
        rows[COO_BLOCK_SIZE * 3 / 2] = -1;
        vals[COO_BLOCK_SIZE * 3 / 2] = GetTypedValue<computeType>(0);
    }

    const idxType tail_start = (tail_num / COO_BLOCK_SIZE) * COO_BLOCK_SIZE;
    const idxType tail_cnt = tail_num - tail_start;
    idxType global_idx;
    idxType row;
    computeType val;

    __syncthreads();
    for (global_idx = threadIdx.x; global_idx < tail_start; global_idx += COO_BLOCK_SIZE) {
        row = row_tail[global_idx] - idx_base;

        val.x = values_tail[global_idx] * (alpha * x[columns_tail[global_idx] - idx_base]).x;
        val.y = values_tail[global_idx] * (alpha * x[columns_tail[global_idx] - idx_base]).y;

        rows[inner_idx] = row;
        vals[inner_idx] = val;

        __syncthreads();
#pragma unroll
        for (int i = 1; i < COO_BLOCK_SIZE; i <<= 1) {
            if (row == rows[inner_idx - i]) {
                left = vals[inner_idx - i];
            }
            __syncthreads();
            vals[inner_idx] += left;
            left = GetTypedValue<computeType>(0);
            __syncthreads();
        }

        if (row != rows[inner_idx + 1]) {
            y[row] += vals[inner_idx];
        }
        __syncthreads();
    }

    if (tail_start == tail_num) {
        return;
    }

    global_idx = tail_start + threadIdx.x;
    if (global_idx < tail_num) {
        row = row_tail[global_idx] - idx_base;

        val.x = values_tail[global_idx] * (alpha * x[columns_tail[global_idx] - idx_base]).x;
        val.y = values_tail[global_idx] * (alpha * x[columns_tail[global_idx] - idx_base]).y;

    } else {
        row = -1;
        val = GetTypedValue<computeType>(0);
    }
    rows[inner_idx] = row;
    vals[inner_idx] = val;

    __syncthreads();
#pragma unroll
    for (int i = 1; i < COO_BLOCK_SIZE; i <<= 1) {
        if (row == rows[inner_idx - i]) {
            left = vals[inner_idx - i];
        }
        __syncthreads();
        vals[inner_idx] += left;
        left = GetTypedValue<computeType>(0);
        __syncthreads();
    }

    if (global_idx < tail_num && row != rows[inner_idx + 1]) {
        y[rows[inner_idx]] += vals[inner_idx];
    }
}

template <typename idxType, typename computeType>
__global__ void mcspSpmvCooTailKernelLowPrecisionComplex(idxType tail_num, const computeType alpha,
                                                         const idxType *row_tail, const idxType *columns_tail,
                                                         const computeType *values_tail, const computeType *x,
                                                         computeType *y, mcsparseIndexBase_t idx_base) {
    __shared__ __align__(sizeof(idxType)) unsigned char smem1[(COO_BLOCK_SIZE * 3 / 2 + 1) * sizeof(idxType)];
    idxType *rows = reinterpret_cast<idxType *>(smem1);
    __shared__ __align__(sizeof(computeType)) unsigned char smem2[(COO_BLOCK_SIZE * 3 / 2 + 1) * sizeof(computeType)];
    computeType *vals = reinterpret_cast<computeType *>(smem2);

    const idxType inner_idx = threadIdx.x + COO_BLOCK_SIZE / 2;
    const idxType tid = threadIdx.x;

    computeType left = GetTypedValue<computeType>(0);

    rows[inner_idx - COO_BLOCK_SIZE / 2] = -1;
    vals[inner_idx - COO_BLOCK_SIZE / 2] = GetTypedValue<computeType>(0);

    if (threadIdx.x == 0) {
        rows[COO_BLOCK_SIZE * 3 / 2] = -1;
        vals[COO_BLOCK_SIZE * 3 / 2] = GetTypedValue<computeType>(0);
    }

    const idxType tail_start = (tail_num / COO_BLOCK_SIZE) * COO_BLOCK_SIZE;
    const idxType tail_cnt = tail_num - tail_start;
    idxType global_idx;
    idxType row;
    computeType val;

    __syncthreads();
    for (global_idx = threadIdx.x; global_idx < tail_start; global_idx += COO_BLOCK_SIZE) {
        row = row_tail[global_idx] - idx_base;
        val = complex_mul(alpha, complex_mul(values_tail[global_idx], x[columns_tail[global_idx] - idx_base]));

        rows[inner_idx] = row;
        vals[inner_idx] = val;

        __syncthreads();
#pragma unroll
        for (int i = 1; i < COO_BLOCK_SIZE; i <<= 1) {
            if (row == rows[inner_idx - i]) {
                left = vals[inner_idx - i];
            }
            __syncthreads();
            vals[inner_idx] += left;
            left = GetTypedValue<computeType>(0);
            __syncthreads();
        }

        if (row != rows[inner_idx + 1]) {
            y[row] += vals[inner_idx];
        }
        __syncthreads();
    }

    if (tail_start == tail_num) {
        return;
    }

    global_idx = tail_start + threadIdx.x;
    if (global_idx < tail_num) {
        row = row_tail[global_idx] - idx_base;
        val = complex_mul(alpha, complex_mul(values_tail[global_idx], x[columns_tail[global_idx] - idx_base]));
    } else {
        row = -1;
        val = GetTypedValue<computeType>(0);
    }
    rows[inner_idx] = row;
    vals[inner_idx] = val;

    __syncthreads();
#pragma unroll
    for (int i = 1; i < COO_BLOCK_SIZE; i <<= 1) {
        if (row == rows[inner_idx - i]) {
            left = vals[inner_idx - i];
        }
        __syncthreads();
        vals[inner_idx] += left;
        left = GetTypedValue<computeType>(0);
        __syncthreads();
    }

    if (global_idx < tail_num && row != rows[inner_idx + 1]) {
        y[rows[inner_idx]] += vals[inner_idx];
    }
}

template <typename idxType>
__global__ void mcspSpmvAosCoo2CooKernel(int64_t nnz, const idxType *coo_ind, idxType *coo_rows, idxType *coo_cols) {
    idxType tid = threadIdx.x + blockIdx.x * COO_BLOCK_SIZE;
    if (tid >= nnz) {
        return;
    }
    coo_rows[tid] = coo_ind[2 * tid];
    coo_cols[tid] = coo_ind[2 * tid + 1];
}
#endif
