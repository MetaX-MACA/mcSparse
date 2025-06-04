#ifndef KERNELS_LEVEL1_ROTI_DEVICE_HPP__
#define KERNELS_LEVEL1_ROTI_DEVICE_HPP__

#include "common/mcsp_types.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCKSIZE, typename idxType, typename computeType, typename inoutType>
__global__ void mcspRotiKernel(idxType nnz, inoutType* x_val, const idxType* x_ind, inoutType* y, computeType c,
                               computeType s, mcsparseIndexBase_t idx_base) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx < nnz) {
        idxType col = x_ind[idx] - idx_base;
        if constexpr (!std::is_same_v<computeType, inoutType>) {
#if defined(__MACA__)
            if constexpr (std::is_same_v<inoutType, __half> || std::is_same_v<inoutType, mcsp_bfloat16>) {
                computeType x_v = GetFloatFromLowReal(x_val[idx]);
                computeType y_v = GetFloatFromLowReal(y[col]);
                x_val[idx] = GetTypedValue<inoutType>(c * x_v + s * y_v);
                y[col] = GetTypedValue<inoutType>(-s * x_v + c * y_v);
            } else if constexpr (std::is_same_v<inoutType, __half2> || std::is_same_v<inoutType, mcsp_bfloat162>) {
                computeType x_v = GetCf32FromLowComplex(x_val[idx]);
                computeType y_v = GetCf32FromLowComplex(y[col]);
                x_val[idx] = GetLowComplexType<inoutType>(c * x_v + s * y_v);
                y[col] = GetLowComplexType<inoutType>(-s * x_v + c * y_v);
            }
#endif
        } else {
            computeType x_v = x_val[idx];
            computeType y_v = y[col];
            x_val[idx] = c * x_v + s * y_v;
            y[col] = -s * x_v + c * y_v;
        }
    }
}

#endif