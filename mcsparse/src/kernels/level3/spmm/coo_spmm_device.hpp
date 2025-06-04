#ifndef KERNELS_LEVEL3_COO_SPMM_DEVICE_HPP__
#define KERNELS_LEVEL3_COO_SPMM_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_hashtable_device.hpp"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE, typename idxType, typename computeType, typename dataType>
__global__ void mcspSpmmCooScalingKernel(idxType m, idxType n, idxType ld, int64_t stride, computeType beta,
                                         dataType *data, mcsparseOrder_t order = MCSPARSE_ORDER_COL) {
    idxType tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    idxType batch = blockIdx.y;
    if (tid >= m * n) {
        return;
    }
    idxType wid = tid / m;
    idxType lid = tid % m;
    idxType offset = (order == MCSPARSE_ORDER_COL) ? wid * ld + lid : wid + lid * ld;
    if constexpr (!std::is_same_v<computeType, dataType>) {
#if defined(__MACA__)
        if constexpr (std::is_same_v<dataType, __half> || std::is_same_v<dataType, mcsp_bfloat16>) {
            data[offset + stride * batch] =
                GetTypedValue<dataType>(beta * GetFloatFromLowReal(data[offset + stride * batch]));
        } else if constexpr (std::is_same_v<dataType, __half2> || std::is_same_v<dataType, mcsp_bfloat162>) {
            data[offset + stride * batch] =
                GetLowComplexType<dataType>(beta * GetCf32FromLowComplex(data[offset + stride * batch]));
        } else if constexpr (std::is_same_v<dataType, int8_t>) {
            data[offset + stride * batch] =
                static_cast<dataType>(beta * static_cast<computeType>(data[offset + stride * batch]));
        }
#endif
    } else {
        data[offset + stride * batch] *= beta;
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename dataType>
__global__ void mcspSpmmCooLowPrecisionComplexScalingKernel(idxType m, idxType n, idxType ld, int64_t stride,
                                                            dataType beta, dataType *data,
                                                            mcsparseOrder_t order = MCSPARSE_ORDER_COL) {
    idxType tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    idxType batch = blockIdx.y;
    if (tid >= m * n) {
        return;
    }
    idxType wid = tid / m;
    idxType lid = tid % m;
    idxType offset = (order == MCSPARSE_ORDER_COL) ? wid * ld + lid : wid + lid * ld;
    data[offset + stride * batch] = complex_mul(beta, data[offset + stride * batch]);
}

template <unsigned int BLOCK_SIZE, typename idxType, typename computeType, typename inputType, typename outputType>
__global__ void mcspSpmmCooBodyKernel(idxType n, idxType nnz, computeType alpha, const idxType *coo_rows,
                                      const idxType *coo_cols, const inputType *coo_vals, int64_t row_stride,
                                      int64_t col_stride, const inputType *mtx_B, idxType ldb, int64_t stride_B,
                                      outputType *mtx_C, idxType ldc, int64_t stride_C, mcsparseIndexBase_t idx_base,
                                      mcsparseOrder_t orderC) {
    idxType tid = threadIdx.x;
    idxType gid = BLOCK_SIZE * blockIdx.x + tid;
    idxType lid = tid & (WARP_SIZE - 1);
    idxType wid = tid / WARP_SIZE;

    idxType batch_id = blockIdx.y;

    __shared__ __align__(sizeof(idxType)) unsigned char smem1[BLOCK_SIZE * sizeof(idxType)];
    idxType *shared_row = reinterpret_cast<idxType *>(smem1);
    __shared__ __align__(sizeof(computeType)) unsigned char smem2[BLOCK_SIZE * sizeof(computeType)];
    computeType *shared_val = reinterpret_cast<computeType *>(smem2);

    idxType coo_row = (gid < nnz) ? (coo_rows[gid + batch_id * col_stride] - idx_base) : idxType(-1);
    idxType coo_col = (gid < nnz) ? (coo_cols[gid + batch_id * col_stride] - idx_base) : idxType(0);
    computeType coo_val;
    if constexpr (!std::is_same_v<computeType, inputType>) {
#if defined(__MACA__)
        if constexpr (std::is_same_v<inputType, __half> || std::is_same_v<inputType, mcsp_bfloat16>) {
            coo_val = (gid < nnz) ? alpha * GetFloatFromLowReal(coo_vals[gid + batch_id * col_stride]) : computeType(0);
        } else if constexpr (std::is_same_v<inputType, __half2> || std::is_same_v<inputType, mcsp_bfloat162>) {
            coo_val =
                (gid < nnz) ? alpha * GetCf32FromLowComplex(coo_vals[gid + batch_id * col_stride]) : computeType(0);
        } else if constexpr (std::is_same_v<inputType, int8_t>) {
            coo_val =
                (gid < nnz) ? alpha * static_cast<computeType>(coo_vals[gid + batch_id * col_stride]) : computeType(0);
        }
#endif
    } else {
        coo_val = (gid < nnz) ? alpha * coo_vals[gid + batch_id * col_stride] : computeType(0);
    }

    for (idxType B_col_id = 0; B_col_id < n; B_col_id += WARP_SIZE) {
        idxType B_col = B_col_id + lid;

        computeType sum = computeType(0);
#if defined(__MACA__)
        idxType cur_row = __shfl_sync(UINT64_BIT_MASK, coo_row, 0, WARP_SIZE);
#else
        idxType cur_row = __shfl(coo_row, 0, WARP_SIZE);
#endif
        for (idxType idx = 0; idx < WARP_SIZE; ++idx) {
#if defined(__MACA__)
            idxType shlf_row = __shfl_sync(UINT64_BIT_MASK, coo_row, idx, WARP_SIZE);
            idxType shlf_col = __shfl_sync(UINT64_BIT_MASK, coo_col, idx, WARP_SIZE);
            computeType shfl_val;
            if constexpr (std::is_same_v<computeType, float> || std::is_same_v<computeType, double> ||
                          std::is_same_v<computeType, __half> || std::is_same_v<computeType, mcsp_bfloat16> ||
                          std::is_same_v<computeType, int32_t>) {
                shfl_val = __shfl_sync(UINT64_BIT_MASK, coo_val, idx, WARP_SIZE);
            } else {
                shfl_val.x = __shfl_sync(UINT64_BIT_MASK, coo_val.x, idx, WARP_SIZE);
                shfl_val.y = __shfl_sync(UINT64_BIT_MASK, coo_val.y, idx, WARP_SIZE);
            }
#else
            idxType shlf_row = __shfl(coo_row, idx, WARP_SIZE);
            idxType shlf_col = __shfl(coo_col, idx, WARP_SIZE);
            computeType shfl_val;
            if constexpr (std::is_same_v<computeType, float> || std::is_same_v<computeType, double> ||
                          std::is_same_v<computeType, __half> || std::is_same_v<computeType, mcsp_bfloat16>) {
                shfl_val = __shfl(coo_val, idx, WARP_SIZE);
            } else {
                shfl_val.x = __shfl(coo_val.x, idx, WARP_SIZE);
                shfl_val.y = __shfl(coo_val.y, idx, WARP_SIZE);
            }
#endif
            if (shlf_row != cur_row) {
                idxType mtx_c_offset = (orderC == MCSPARSE_ORDER_COL) ? B_col * ldc + cur_row + stride_C * batch_id
                                                                      : B_col + cur_row * ldc + stride_C * batch_id;
                if (B_col < n) {
                    if constexpr (std::is_same_v<outputType, float> || std::is_same_v<outputType, double> ||
                                  std::is_same_v<outputType, __half> || std::is_same_v<outputType, mcsp_bfloat16> ||
                                  std::is_same_v<computeType, int32_t>) {
                        atomicAdd(&mtx_C[mtx_c_offset], GetTypedValue<outputType>(sum));
                    } else {
                        complexAtomicAddByPart_(&mtx_C[mtx_c_offset], GetLowComplexType<outputType>(sum));
                    }
                }
                sum = computeType(0);
                cur_row = shlf_row;
            }

            if (B_col < n) {
                if constexpr (!std::is_same_v<computeType, inputType>) {
#if defined(__MACA__)
                    if constexpr (std::is_same_v<inputType, __half> || std::is_same_v<inputType, mcsp_bfloat16>) {
                        sum = shfl_val * GetFloatFromLowReal(mtx_B[B_col * ldb + shlf_col + stride_B * batch_id]) + sum;
                    } else if constexpr (std::is_same_v<inputType, __half2> ||
                                         std::is_same_v<inputType, mcsp_bfloat162>) {
                        sum =
                            shfl_val * GetCf32FromLowComplex(mtx_B[B_col * ldb + shlf_col + stride_B * batch_id]) + sum;
                    } else if constexpr (std::is_same_v<inputType, int8_t>) {
                        sum = shfl_val * static_cast<computeType>(mtx_B[B_col * ldb + shlf_col + stride_B * batch_id]) +
                              sum;
                    }
#endif
                } else {
                    sum = shfl_val * mtx_B[B_col * ldb + shlf_col + stride_B * batch_id] + sum;
                }
            }
        }

        __syncthreads();
        shared_row[(BLOCK_SIZE / WARP_SIZE) * lid + wid] = cur_row;
        shared_val[(BLOCK_SIZE / WARP_SIZE) * lid + wid] = sum;
        __syncthreads();

        cur_row = shared_row[tid];
        sum = shared_val[tid];

        idxType cur_lid = tid & ((BLOCK_SIZE / WARP_SIZE) - 1);
        idxType cur_wid = tid / (BLOCK_SIZE / WARP_SIZE);

        for (idxType j = 1; j < (BLOCK_SIZE / WARP_SIZE); j <<= 1) {
            if (cur_lid >= j) {
                if (cur_row == shared_row[cur_lid - j]) {
                    sum += shared_val[(BLOCK_SIZE / WARP_SIZE) * cur_wid + cur_lid - j];
                }
            }
            __syncthreads();
            shared_val[(BLOCK_SIZE / WARP_SIZE) * cur_wid + cur_lid] = sum;
            __syncthreads();
        }
        if (cur_lid < ((BLOCK_SIZE / WARP_SIZE) - 1)) {
            if (cur_row != shared_row[cur_lid + 1] && (int)cur_row >= 0) {
                if ((B_col_id + cur_wid) < n) {
                    idxType mtx_c_offset = (orderC == MCSPARSE_ORDER_COL)
                                               ? (B_col_id + cur_wid) * ldc + cur_row + stride_C * batch_id
                                               : (B_col_id + cur_wid) + cur_row * ldc + stride_C * batch_id;
                    if constexpr (std::is_same_v<computeType, float> || std::is_same_v<computeType, double> ||
                                  std::is_same_v<computeType, __half> || std::is_same_v<computeType, mcsp_bfloat16> ||
                                  std::is_same_v<computeType, int32_t>) {
                        atomicAdd(&mtx_C[mtx_c_offset], GetTypedValue<outputType>(sum));
                    } else {
                        complexAtomicAddByPart_(&mtx_C[mtx_c_offset], GetLowComplexType<outputType>(sum));
                    }
                }
            }
        }

        if (cur_lid == ((BLOCK_SIZE / WARP_SIZE) - 1)) {
            if ((int)cur_row >= 0) {
                if ((B_col_id + cur_wid) < n) {
                    idxType mtx_c_offset = (orderC == MCSPARSE_ORDER_COL)
                                               ? (B_col_id + cur_wid) * ldc + cur_row + stride_C * batch_id
                                               : (B_col_id + cur_wid) + cur_row * ldc + stride_C * batch_id;
                    if constexpr (std::is_same_v<computeType, float> || std::is_same_v<computeType, double> ||
                                  std::is_same_v<computeType, __half> || std::is_same_v<computeType, mcsp_bfloat16> ||
                                  std::is_same_v<computeType, int32_t>) {
                        atomicAdd(&mtx_C[mtx_c_offset], GetTypedValue<outputType>(sum));
                    } else {
                        complexAtomicAddByPart_(&mtx_C[mtx_c_offset], GetLowComplexType<outputType>(sum));
                    }
                }
            }
        }
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename dataType>
__global__ void mcspSpmmCooLowPrecisionComplexBodyKernel(
    idxType n, idxType nnz, dataType alpha, const idxType *coo_rows, const idxType *coo_cols, const dataType *coo_vals,
    int64_t row_stride, int64_t col_stride, const dataType *mtx_B, idxType ldb, int64_t stride_B, dataType *mtx_C,
    idxType ldc, int64_t stride_C, mcsparseIndexBase_t idx_base, mcsparseOrder_t orderC) {
    idxType tid = threadIdx.x;
    idxType gid = BLOCK_SIZE * blockIdx.x + tid;
    idxType lid = tid & (WARP_SIZE - 1);
    idxType wid = tid / WARP_SIZE;

    idxType batch_id = blockIdx.y;

    __shared__ __align__(sizeof(idxType)) unsigned char smem1[BLOCK_SIZE * sizeof(idxType)];
    idxType *shared_row = reinterpret_cast<idxType *>(smem1);
    __shared__ __align__(sizeof(dataType)) unsigned char smem2[BLOCK_SIZE * sizeof(dataType)];
    dataType *shared_val = reinterpret_cast<dataType *>(smem2);

    idxType coo_row = (gid < nnz) ? (coo_rows[gid + batch_id * col_stride] - idx_base) : idxType(-1);
    idxType coo_col = (gid < nnz) ? (coo_cols[gid + batch_id * col_stride] - idx_base) : idxType(0);
    dataType coo_val =
        (gid < nnz) ? complex_mul(alpha, coo_vals[gid + batch_id * col_stride]) : GetTypedValue<dataType>(0);

    for (idxType B_col_id = 0; B_col_id < n; B_col_id += WARP_SIZE) {
        idxType B_col = B_col_id + lid;
        dataType sum = GetTypedValue<dataType>(0);
#if defined(__MACA__)
        idxType cur_row = __shfl_sync(UINT64_BIT_MASK, coo_row, 0, WARP_SIZE);
#else
        idxType cur_row = __shfl(coo_row, 0, WARP_SIZE);
#endif
        for (idxType idx = 0; idx < WARP_SIZE; ++idx) {
            dataType shfl_val;
#if defined(__MACA__)
            idxType shlf_row = __shfl_sync(UINT64_BIT_MASK, coo_row, idx, WARP_SIZE);
            idxType shlf_col = __shfl_sync(UINT64_BIT_MASK, coo_col, idx, WARP_SIZE);
            shfl_val.x = __shfl_sync(UINT64_BIT_MASK, coo_val.x, idx, WARP_SIZE);
            shfl_val.y = __shfl_sync(UINT64_BIT_MASK, coo_val.y, idx, WARP_SIZE);
#else
            idxType shlf_row = __shfl(coo_row, idx, WARP_SIZE);
            idxType shlf_col = __shfl(coo_col, idx, WARP_SIZE);
            shfl_val.x = __shfl(coo_val.x, idx, WARP_SIZE);
            shfl_val.y = __shfl(coo_val.y, idx, WARP_SIZE);
#endif
            if (shlf_row != cur_row) {
                idxType mtx_c_offset = (orderC == MCSPARSE_ORDER_COL) ? B_col * ldc + cur_row + stride_C * batch_id
                                                                      : B_col + cur_row * ldc + stride_C * batch_id;
                if (B_col < n) {
                    complexAtomicAddByPart_(&mtx_C[mtx_c_offset], sum);
                }
                sum = GetTypedValue<dataType>(0);
                cur_row = shlf_row;
            }

            if (B_col < n) {
                sum = complex_mul(shfl_val, mtx_B[B_col * ldb + shlf_col + stride_B * batch_id]) + sum;
            }
        }

        __syncthreads();
        shared_row[(BLOCK_SIZE / WARP_SIZE) * lid + wid] = cur_row;
        shared_val[(BLOCK_SIZE / WARP_SIZE) * lid + wid] = sum;
        __syncthreads();

        cur_row = shared_row[tid];
        sum = shared_val[tid];

        idxType cur_lid = tid & ((BLOCK_SIZE / WARP_SIZE) - 1);
        idxType cur_wid = tid / (BLOCK_SIZE / WARP_SIZE);

        for (idxType j = 1; j < (BLOCK_SIZE / WARP_SIZE); j <<= 1) {
            if (cur_lid >= j) {
                if (cur_row == shared_row[cur_lid - j]) {
                    sum += shared_val[(BLOCK_SIZE / WARP_SIZE) * cur_wid + cur_lid - j];
                }
            }
            __syncthreads();
            shared_val[(BLOCK_SIZE / WARP_SIZE) * cur_wid + cur_lid] = sum;
            __syncthreads();
        }

        idxType mtx_c_offset = (orderC == MCSPARSE_ORDER_COL)
                                   ? (B_col_id + cur_wid) * ldc + cur_row + stride_C * batch_id
                                   : (B_col_id + cur_wid) + cur_row * ldc + stride_C * batch_id;
        if ((cur_lid < ((BLOCK_SIZE / WARP_SIZE) - 1)) && (cur_row != shared_row[cur_lid + 1] && (int)cur_row >= 0) &&
            ((B_col_id + cur_wid) < n)) {
            complexAtomicAddByPart_(&mtx_C[mtx_c_offset], sum);
        }

        if ((cur_lid == ((BLOCK_SIZE / WARP_SIZE) - 1)) && ((int)cur_row >= 0) && ((B_col_id + cur_wid) < n)) {
            complexAtomicAddByPart_(&mtx_C[mtx_c_offset], sum);
        }
    }
}
#endif