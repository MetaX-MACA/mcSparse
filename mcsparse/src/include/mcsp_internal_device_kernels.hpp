#ifndef COMMON_MCSP_INTERNAL_DEVICE_KERNELS_HPP__
#define COMMON_MCSP_INTERNAL_DEVICE_KERNELS_HPP__

#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_runtime_wrapper.h"

// inject low precision abs to namespace std
namespace std {

#if defined(__MACA__)
__device__ __forceinline__ __half abs(__half x) {
    return __habs(x);
}
#endif

#ifdef __MACA__
__device__ __forceinline__ __half2 abs(__half2 x) {
    return __habs2(x);
}

__device__ __forceinline__ mcsp_bfloat16 abs(mcsp_bfloat16 x) {
    return __habs(x);
}

__device__ __forceinline__ mcsp_bfloat162 abs(mcsp_bfloat162 x) {
    return __habs2(x);
}
#endif
};  // namespace std

// initialize values by integer
template <typename valType, typename dataType>
__device__ __host__ __forceinline__ valType GetTypedValue(dataType val) {
    valType typed_val;
#if defined(__MACA__)
    if constexpr (std::is_same_v<valType, __half>) {
        typed_val = __float2half(val);
    } else if constexpr (std::is_same_v<valType, __half2>) {
        typed_val = __half2(__float2half(val), __float2half(0));
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
        typed_val = __float2bfloat16(val);
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat162>) {
        typed_val = mcsp_bfloat162(__float2bfloat16(val), __float2bfloat16(0));
    } else {
        typed_val = valType(val);
    }
#else
    typed_val = val;
#endif
    return typed_val;
}

// complex float to low precision value
template <typename valType, typename dataType>
__device__ __host__ __forceinline__ valType GetLowComplexType(dataType val) {
    valType typed_val;
#if defined(__MACA__)
    if constexpr (std::is_same_v<valType, __half2>) {
        typed_val = __half2(__float2half(val.x), __float2half(val.y));
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat162>) {
        typed_val = mcsp_bfloat162(__float2bfloat16(val.x), __float2bfloat16(val.y));
    } else {
        typed_val = valType(val);
    }
#else
    typed_val = val;
#endif
    return typed_val;
}

// low precision value to float
template <typename valType>
__device__ __host__ __forceinline__ float GetFloatFromLowReal(const valType val) {
    float target_val = 0.0f;
#if defined(__MACA__)
    if constexpr (std::is_same_v<valType, __half>) {
        target_val = __half2float(val);
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
        target_val = __bfloat162float(val);
    } else {
        target_val = valType(val);
    }
#endif
    return target_val;
}

// low precision value to complex float
template <typename valType>
__device__ __host__ __forceinline__ mcspComplexFloat GetCf32FromLowComplex(const valType val) {
    mcspComplexFloat target_val;
#if defined(__MACA__)
    if constexpr (std::is_same_v<valType, __half2>) {
        target_val = {__half2float(val.x), __half2float(val.y)};
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat162>) {
        target_val = {__bfloat162float(val.x), __bfloat162float(val.y)};
    }
#endif
    return target_val;
}

__device__ __forceinline__ float mcsp_conj(const float &x) {
    return x;
}
__device__ __forceinline__ double mcsp_conj(const double &x) {
    return x;
}
__device__ __forceinline__ mcFloatComplex mcsp_conj(const mcFloatComplex &x) {
    return std::conj(x);
}
__device__ __forceinline__ mcDoubleComplex mcsp_conj(const mcDoubleComplex &x) {
    return std::conj(x);
}

#if defined(__MACA__)
__device__ __forceinline__ __half2 c16f_mul(const __half2 &a, const __half2 &b) {
    float a_low = GetFloatFromLowReal(__low2half(a));
    float a_high = GetFloatFromLowReal(__high2half(a));
    float b_low = GetFloatFromLowReal(__low2half(b));
    float b_high = GetFloatFromLowReal(__high2half(b));
    __half re = GetTypedValue<__half>(a_low * b_low - a_high * b_high);
    __half im = GetTypedValue<__half>(a_low * b_high + a_high * b_low);
    return __half2(re, im);
}

__device__ __forceinline__ mcsp_bfloat162 c16bf_mul(const mcsp_bfloat162 &a, const mcsp_bfloat162 &b) {
#ifdef __MACA__
    mcsp_bfloat16 re = __low2bfloat16(a) * __low2bfloat16(b) - __high2bfloat16(a) * __high2bfloat16(b);
    mcsp_bfloat16 im = __low2bfloat16(a) * __high2bfloat16(b) + __high2bfloat16(a) * __low2bfloat16(b);
    return mcsp_bfloat162(re, im);
#else
    return GetTypedValue<mcsp_bfloat162>(0);
#endif
}

template <typename ComplexType>
__device__ __forceinline__ ComplexType complex_mul(const ComplexType &a, const ComplexType &b) {
    return GetTypedValue<ComplexType>(0);
}

template <>
__device__ __forceinline__ __half2 complex_mul(const __half2 &a, const __half2 &b) {
    return c16f_mul(a, b);
}

template <>
__device__ __forceinline__ mcsp_bfloat162 complex_mul(const mcsp_bfloat162 &a, const mcsp_bfloat162 &b) {
    return c16bf_mul(a, b);
}
#endif

template <uint32_t blockSize, typename idxType>
__device__ __forceinline__ idxType warpReduceSum(idxType sum) {
#ifndef __MACA__
    if (blockSize >= 2) sum += __shfl_up_sync(UINT32_BIT_MASK, sum, 1);
    __syncwarp();
#else
    if (blockSize >= 2) sum += __shfl_up(sum, 1);
#endif

#ifndef __MACA__
    if (blockSize >= 4) sum += __shfl_up_sync(UINT32_BIT_MASK, sum, 2);
    __syncwarp();
#else
    if (blockSize >= 4) sum += __shfl_up(sum, 2);
#endif

#ifndef __MACA__
    if (blockSize >= 8) sum += __shfl_up_sync(UINT32_BIT_MASK, sum, 4);
    __syncwarp();
#else
    if (blockSize >= 8) sum += __shfl_up(sum, 4);
#endif

#ifndef __MACA__
    if (blockSize >= 16) sum += __shfl_up_sync(UINT32_BIT_MASK, sum, 8);
    __syncwarp();
#else
    if (blockSize >= 16) sum += __shfl_up(sum, 8);
#endif

#ifndef __MACA__
    if (blockSize >= 32) sum += __shfl_up_sync(UINT32_BIT_MASK, sum, 16);
    __syncwarp();
#else
    if (blockSize >= 32) sum += __shfl_up(sum, 16);
#endif

#ifndef __MACA__
    if (blockSize >= 64) sum += __shfl_up_sync(UINT32_BIT_MASK, sum, 32);
    __syncwarp();
#else
    if (blockSize >= 64) sum += __shfl_up(sum, 32);
#endif
    return sum;
}

template <uint32_t blockSize>
__device__ __forceinline__ mcspComplexFloat warpReduceSum(mcspComplexFloat sum) {
    sum.x = warpReduceSum<blockSize>(sum.x);
    sum.y = warpReduceSum<blockSize>(sum.y);
    return sum;
}

template <uint32_t blockSize>
__device__ __forceinline__ mcspComplexDouble warpReduceSum(mcspComplexDouble sum) {
    sum.x = warpReduceSum<blockSize>(sum.x);
    sum.y = warpReduceSum<blockSize>(sum.y);
    return sum;
}

template <typename idxType>
__global__ void selfIncreaseInplaceKernel(idxType size, idxType *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] += idxType(1);
    }
}

template <typename idxType>
__global__ void oppositeImagePartInplaceKernel(idxType size, mcspComplexFloat *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid].y = -data[tid].y;
    }
}

template <typename idxType>
__global__ void oppositeImagePartInplaceKernel(idxType size, mcspComplexDouble *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid].y = -data[tid].y;
    }
}

#if defined(__MACA__)
template <typename idxType>
__global__ void oppositeImagePartInplaceKernel(idxType size, __half2 *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid].y = -data[tid].y;
    }
}

template <typename idxType>
__global__ void oppositeImagePartInplaceKernel(idxType size, mcsp_bfloat162 *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid].y = -data[tid].y;
    }
}
#endif
// input and output matrix should be in column-major
template <typename idxType, typename valType>
__global__ void transferDenseMatrixKernel(idxType m_in, idxType n_in, idxType ld_in, idxType m_out, idxType n_out,
                                          idxType ld_out, const valType *mat_in, valType *mat_out) {
    idxType row = blockIdx.x * blockDim.x + threadIdx.x;
    idxType col = blockIdx.y;

    if (row < m_out && col < n_out && row < m_in && col < n_in) {
        mat_out[row + ld_out * col] = mat_in[row + ld_in * col];
    }
}

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void transposeBsrBlockKernel(idxType nnz, idxType block_nnz, idxType row_block_dim, idxType col_block_dim,
                                        valType *bsrt_vals, mcsparseDirection_t bsrt_dir) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx >= nnz) {
        return;
    }
    idxType row_idx = idx / block_nnz;
    idxType val_idx = idx % block_nnz;
    idxType i = 0;
    idxType j = 0;
    if (bsrt_dir == MCSPARSE_DIRECTION_ROW) {
        i = val_idx / row_block_dim;
        j = val_idx % row_block_dim;
        val_idx = row_idx * block_nnz + i + j * col_block_dim;
    } else {
        i = val_idx % col_block_dim;
        j = val_idx / col_block_dim;
        val_idx = row_idx * block_nnz + i * row_block_dim + j;
    }
    if (i > j) {
        valType val = bsrt_vals[val_idx];
        bsrt_vals[val_idx] = bsrt_vals[idx];
        bsrt_vals[idx] = val;
    }
}

template <uint32_t BLOCKSIZE, typename idxType, typename computeType, typename valueType = computeType>
__global__ void denseAxpbyNonZeroBetaKernel(idxType size, computeType alpha, const valueType *d_input, computeType beta,
                                            valueType *d_output) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx < size) {
        if constexpr (!std::is_same_v<computeType, valueType>) {
            if constexpr (std::is_same_v<valueType, __half> || std::is_same_v<valueType, mcsp_bfloat16>) {
#if defined(__MACA__)
                d_output[idx] = GetTypedValue<valueType>(alpha * GetFloatFromLowReal(d_input[idx]) +
                                                         beta * GetFloatFromLowReal(d_output[idx]));
#endif
            }
        } else {
            d_output[idx] = alpha * d_input[idx] + beta * d_output[idx];
        }
    }
}

template <uint32_t BLOCKSIZE, typename idxType, typename computeType, typename valueType = computeType>
__global__ void denseAxpbyZeroBetaKernel(idxType size, computeType alpha, const valueType *d_input,
                                         valueType *d_output) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx < size) {
        if constexpr (!std::is_same_v<computeType, valueType>) {
            if constexpr (std::is_same_v<valueType, __half> || std::is_same_v<valueType, mcsp_bfloat16>) {
#if defined(__MACA__)
                d_output[idx] = GetTypedValue<valueType>(alpha * GetFloatFromLowReal(d_input[idx]));
#endif
            }
        } else {
            d_output[idx] = alpha * d_input[idx];
        }
    }
}

template <uint32_t BLOCKSIZE, typename idxType, typename computeType, typename valueType = computeType>
void denseAxpby(mcStream_t stream, int n_block, idxType size, computeType alpha, const valueType *d_input,
                computeType beta, valueType *d_output) {
    if (beta != GetTypedValue<computeType>(0)) {
        mcLaunchKernelGGL((denseAxpbyNonZeroBetaKernel<BLOCKSIZE>), dim3(n_block), dim3(BLOCKSIZE), 0, stream, size,
                           alpha, d_input, beta, d_output);
    } else {
        mcLaunchKernelGGL((denseAxpbyZeroBetaKernel<BLOCKSIZE>), dim3(n_block), dim3(BLOCKSIZE), 0, stream, size,
                           alpha, d_input, d_output);
    }
}

template <uint32_t BLOCKSIZE, typename idxType, typename valType>
__global__ void denseAxpbyNonZeroBetaKernelLowPrecisionComplex(idxType size, valType alpha, const valType *d_input,
                                                               valType beta, valType *d_output) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx < size) {
        d_output[idx] = complex_mul(alpha, d_input[idx]) + complex_mul(beta, d_output[idx]);
    }
}

template <uint32_t BLOCKSIZE, typename idxType, typename valType>
__global__ void denseAxpbyZeroBetaKernelLowPrecisionComplex(idxType size, valType alpha, const valType *d_input,
                                                            valType *d_output) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx < size) {
        d_output[idx] = complex_mul(alpha, d_input[idx]);
    }
}

template <uint32_t BLOCKSIZE, typename idxType, typename computeType, typename valueType = computeType>
void denseAxpbyLowPrecisionComplex(mcStream_t stream, int n_block, idxType size, computeType alpha,
                                   const valueType *d_input, computeType beta, valueType *d_output) {
    if (beta != GetTypedValue<computeType>(0)) {
        mcLaunchKernelGGL((denseAxpbyNonZeroBetaKernelLowPrecisionComplex<BLOCKSIZE>), dim3(n_block), dim3(BLOCKSIZE),
                           0, stream, size, alpha, d_input, beta, d_output);
    } else {
        mcLaunchKernelGGL((denseAxpbyZeroBetaKernelLowPrecisionComplex<BLOCKSIZE>), dim3(n_block), dim3(BLOCKSIZE), 0,
                           stream, size, alpha, d_input, d_output);
    }
}

#endif
