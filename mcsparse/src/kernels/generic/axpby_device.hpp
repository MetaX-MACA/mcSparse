#ifndef KERNELS_GENERIC_AXPBY_DEVICE_HPP__
#define KERNELS_GENERIC_AXPBY_DEVICE_HPP__

#include "common/mcsp_types.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE, typename idxType, typename computeType, typename inoutType>
__global__ void axpby_scale_kernel(computeType alpha, inoutType* x, idxType size) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx >= size) return;
#if defined(__MACA__)
    if constexpr (!std::is_same_v<computeType, inoutType>) {
        if constexpr (std::is_same_v<inoutType, __half> || std::is_same_v<inoutType, mcsp_bfloat16>) {
            x[idx] = GetTypedValue<inoutType>(alpha * GetFloatFromLowReal(x[idx]));
        } else if constexpr (std::is_same_v<inoutType, __half2> || std::is_same_v<inoutType, mcsp_bfloat162>) {
            x[idx] = GetLowComplexType<inoutType>(alpha * GetCf32FromLowComplex(x[idx]));
        }
    } else {
        x[idx] *= alpha;
    }
#else
    x[idx] *= alpha;
#endif
}

template <unsigned int BLOCKSIZE, typename idxType, typename computeType, typename inoutType>
__global__ void axpby_kernel(idxType nnz, computeType alpha, const inoutType* x_val, const idxType* x_ind, inoutType* y,
                             mcsparseIndexBase_t idx_base) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx >= nnz) {
        return;
    }
    idxType i = x_ind[idx] - idx_base;
#if defined(__MACA__)
    if constexpr (!std::is_same_v<computeType, inoutType>) {
        if constexpr (std::is_same_v<inoutType, __half> || std::is_same_v<inoutType, mcsp_bfloat16>) {
            y[i] = GetTypedValue<inoutType>((alpha * GetFloatFromLowReal(x_val[idx]) + GetFloatFromLowReal(y[i])));
        } else if constexpr (std::is_same_v<inoutType, __half2> || std::is_same_v<inoutType, mcsp_bfloat162>) {
            y[i] =
                GetLowComplexType<inoutType>((alpha * GetCf32FromLowComplex(x_val[idx]) + GetCf32FromLowComplex(y[i])));
        }
    } else {
        y[i] = alpha * x_val[idx] + y[i];
    }
#else
    y[i] = alpha * x_val[idx] + y[i];
#endif
}

#endif