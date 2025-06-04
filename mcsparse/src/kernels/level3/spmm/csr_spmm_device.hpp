#ifndef KERNELS_LEVEL3_SPMM_CSR_SPMM_DEVICE_HPP__
#define KERNELS_LEVEL3_SPMM_CSR_SPMM_DEVICE_HPP__

#include <stdio.h>

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspSpmmCsrNnKernel(idxType m, idxType n, const valType alpha, const valType *csr_vals,
                                    const idxType *csr_rows, const idxType *csr_cols, const valType *mtx_B, idxType ldb,
                                    const valType beta, valType *mtx_C, idxType ldc, mcsparseIndexBase_t idx_base,
                                    mcsparseOrder_t orderC) {
    const idxType row_id = blockIdx.x;
    const idxType col_gid = blockIdx.y;
    const idxType lane = threadIdx.x & (WARP_SIZE - 1);
    const idxType col_inner_id = threadIdx.x / WARP_SIZE;
    const idxType col_id = col_gid * (BLOCK_SIZE / WARP_SIZE) + col_inner_id;

    idxType mtx_offset;
    valType tmp_val;
    idxType b_offset;
    if ((row_id < m) && (col_id < n)) {
        idxType row_start = csr_rows[row_id] - idx_base;
        idxType row_end = csr_rows[row_id + 1] - idx_base;

        tmp_val = GetTypedValue<valType>(0);
        for (idxType j = row_start + lane; j < row_end; j += WARP_SIZE) {
            b_offset = col_id * ldb + (csr_cols[j] - idx_base);
            tmp_val += csr_vals[j] * mtx_B[b_offset];
        }
        tmp_val = warpReduceSum<WARP_SIZE>(tmp_val);

        if (lane == (WARP_SIZE - 1)) {
            mtx_offset = (orderC == MCSPARSE_ORDER_COL) ? col_id * ldc + row_id : col_id + row_id * ldc;
            mtx_C[mtx_offset] = beta * mtx_C[mtx_offset] + alpha * tmp_val;
        }
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspSpmmCsrNtKernel(idxType m, idxType n, const valType alpha, const valType *csr_vals,
                                    const idxType *csr_rows, const idxType *csr_cols, const valType *mtx_B, idxType ldb,
                                    const valType beta, valType *mtx_C, idxType ldc, mcsparseIndexBase_t idx_base,
                                    mcsparseOrder_t orderC) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);

    const idxType row_id = blockIdx.x;
    const idxType col_gid = blockIdx.y;

    const idxType lane = threadIdx.x & (WARP_SIZE - 1);
    const idxType col_inner_id = threadIdx.x / WARP_SIZE;
    const idxType idx = threadIdx.x;

    const idxType col_id = col_gid * (BLOCK_SIZE / WARP_SIZE) + col_inner_id;

    idxType mtx_offset;
    valType tmp_val;
    idxType b_offset;

    if ((row_id < m) && (col_id < n)) {
        idxType row_start = csr_rows[row_id] - idx_base;
        idxType row_end = csr_rows[row_id + 1] - idx_base;

        vals[idx] = 0;
        tmp_val = 0.0;
        for (idxType j = row_start + lane; j < row_end; j += WARP_SIZE) {
            b_offset = (csr_cols[j] - idx_base) * ldb + col_id;
            tmp_val += csr_vals[j] * mtx_B[b_offset];
        }
        vals[idx] = tmp_val;
#pragma unroll
        for (int i = WARP_SIZE >> 1; i != 0; i >>= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (lane < i) {
                vals[idx] += vals[idx + i];
            }
        }

        if (lane == 0) {
            mtx_offset = (orderC == MCSPARSE_ORDER_COL) ? col_id * ldc + row_id : col_id + row_id * ldc;
            mtx_C[mtx_offset] = beta * mtx_C[mtx_offset] + alpha * vals[idx];
        }
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename computeType, typename inputType, typename outputType>
__global__ void mcspBatchedSpmmCsrNnKernel(idxType m, idxType n, const computeType alpha, const inputType *csr_vals,
                                           const idxType *csr_rows, const idxType *csr_cols, int64_t row_stride,
                                           int64_t col_stride, const inputType *mtx_B, idxType ldb, int64_t stride_B,
                                           const computeType beta, outputType *mtx_C, idxType ldc, int64_t stride_C,
                                           mcsparseIndexBase_t idx_base, mcsparseOrder_t orderC) {
    extern __shared__ __align__(sizeof(computeType)) unsigned char smem[];
    computeType *vals = reinterpret_cast<computeType *>(smem);
    const idxType idx = threadIdx.x;
    const idxType row_id = blockIdx.x;
    const idxType col_gid = blockIdx.y;
    const idxType batch_id = blockIdx.z;
    const idxType *cur_csr_rows = csr_rows + batch_id * row_stride;
    const idxType *cur_csr_cols = csr_cols + batch_id * col_stride;
    const inputType *cur_csr_vals = csr_vals + batch_id * col_stride;

    const inputType *cur_mtx_B = mtx_B + batch_id * stride_B;
    outputType *cur_mtx_C = mtx_C + batch_id * stride_C;
    const idxType lane = threadIdx.x & (WARP_SIZE - 1);
    const idxType col_inner_id = threadIdx.x / WARP_SIZE;
    const idxType col_id = col_gid * (BLOCK_SIZE / WARP_SIZE) + col_inner_id;

    idxType mtx_offset;
    idxType b_offset;
    computeType tmp_val;
    if ((row_id < m) && (col_id < n)) {
        idxType row_start = cur_csr_rows[row_id] - idx_base;
        idxType row_end = cur_csr_rows[row_id + 1] - idx_base;

        tmp_val = GetTypedValue<computeType>(0);
        for (idxType j = row_start + lane; j < row_end; j += WARP_SIZE) {
            b_offset = col_id * ldb + (cur_csr_cols[j] - idx_base);
            if constexpr (!std::is_same_v<computeType, inputType>) {
#if defined(__MACA__)
                if constexpr (std::is_same_v<inputType, __half> || std::is_same_v<inputType, mcsp_bfloat16>) {
                    tmp_val += GetFloatFromLowReal(cur_csr_vals[j]) * GetFloatFromLowReal(cur_mtx_B[b_offset]);
                } else if constexpr (std::is_same_v<inputType, __half2> || std::is_same_v<inputType, mcsp_bfloat162>) {
                    tmp_val += GetCf32FromLowComplex(cur_csr_vals[j]) * GetCf32FromLowComplex(cur_mtx_B[b_offset]);
                } else if constexpr (std::is_same_v<inputType, int8_t>) {
                    tmp_val +=
                        static_cast<computeType>(cur_csr_vals[j]) * static_cast<computeType>(cur_mtx_B[b_offset]);
                }
#endif
            } else {
                tmp_val += cur_csr_vals[j] * cur_mtx_B[b_offset];
            }
        }
        vals[idx] = tmp_val;
#pragma unroll
        for (int i = WARP_SIZE >> 1; i != 0; i >>= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (lane < i) {
                vals[idx] += vals[idx + i];
            }
        }

        if (lane == 0) {
            mtx_offset = (orderC == MCSPARSE_ORDER_COL) ? col_id * ldc + row_id : col_id + row_id * ldc;
            if constexpr (!std::is_same_v<computeType, outputType>) {
#if defined(__MACA__)
                if constexpr (std::is_same_v<outputType, __half> || std::is_same_v<outputType, mcsp_bfloat16>) {
                    cur_mtx_C[mtx_offset] = GetTypedValue<outputType>(
                        beta * GetFloatFromLowReal(cur_mtx_C[mtx_offset]) + alpha * vals[idx]);
                } else if constexpr (std::is_same_v<outputType, __half2> ||
                                     std::is_same_v<outputType, mcsp_bfloat162>) {
                    cur_mtx_C[mtx_offset] = GetLowComplexType<outputType>(
                        beta * GetCf32FromLowComplex(cur_mtx_C[mtx_offset]) + alpha * vals[idx]);
                }
#endif
            } else {
                cur_mtx_C[mtx_offset] = beta * cur_mtx_C[mtx_offset] + alpha * vals[idx];
            }
        }
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspBatchedSpmmCsrNnKernelLowPrecisionComplex(idxType m, idxType n, const valType alpha,
                                                              const valType *csr_vals, const idxType *csr_rows,
                                                              const idxType *csr_cols, int64_t row_stride,
                                                              int64_t col_stride, const valType *mtx_B, idxType ldb,
                                                              int64_t stride_B, const valType beta, valType *mtx_C,
                                                              idxType ldc, int64_t stride_C,
                                                              mcsparseIndexBase_t idx_base, mcsparseOrder_t orderC) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);

    const idxType idx = threadIdx.x;
    const idxType row_id = blockIdx.x;
    const idxType col_gid = blockIdx.y;
    const idxType batch_id = blockIdx.z;
    const idxType *cur_csr_rows = csr_rows + batch_id * row_stride;
    const idxType *cur_csr_cols = csr_cols + batch_id * col_stride;
    const valType *cur_csr_vals = csr_vals + batch_id * col_stride;

    const valType *cur_mtx_B = mtx_B + batch_id * stride_B;
    valType *cur_mtx_C = mtx_C + batch_id * stride_C;
    const idxType lane = threadIdx.x & (WARP_SIZE - 1);
    const idxType col_inner_id = threadIdx.x / WARP_SIZE;
    const idxType col_id = col_gid * (BLOCK_SIZE / WARP_SIZE) + col_inner_id;

    idxType mtx_offset;
    valType tmp_val;
    idxType b_offset;
    if ((row_id < m) && (col_id < n)) {
        idxType row_start = cur_csr_rows[row_id] - idx_base;
        idxType row_end = cur_csr_rows[row_id + 1] - idx_base;

        tmp_val = GetTypedValue<valType>(0);
        for (idxType j = row_start + lane; j < row_end; j += WARP_SIZE) {
            b_offset = col_id * ldb + (cur_csr_cols[j] - idx_base);
            tmp_val += complex_mul(cur_csr_vals[j], cur_mtx_B[b_offset]);
        }
        vals[idx] = tmp_val;
#pragma unroll
        for (int i = WARP_SIZE >> 1; i != 0; i >>= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (lane < i) {
                vals[idx] += vals[idx + i];
            }
        }

        if (lane == 0) {
            mtx_offset = (orderC == MCSPARSE_ORDER_COL) ? col_id * ldc + row_id : col_id + row_id * ldc;
            cur_mtx_C[mtx_offset] = complex_mul(beta, cur_mtx_C[mtx_offset]) + complex_mul(alpha, vals[idx]);
        }
    }
}

#endif
