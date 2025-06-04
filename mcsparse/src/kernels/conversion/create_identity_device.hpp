#ifndef KERNELS_CONVERSION_CREATE_IDENTITY_DEVICE_HPP__
#define KERNELS_CONVERSION_CREATE_IDENTITY_DEVICE_HPP__

#include "mcsp_runtime_wrapper.h"

template <typename idxType, std::enable_if_t<std::is_integral_v<idxType>, int> = 0>
__global__ void mcspCreateIdentityKernel(idxType m, idxType *ibuffer) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        ibuffer[tid] = tid;
    }
}

#endif