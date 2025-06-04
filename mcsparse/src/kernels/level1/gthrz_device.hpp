#ifndef KERNELS_LEVEL1_GTHRZ_DEVICE_HPP__
#define KERNELS_LEVEL1_GTHRZ_DEVICE_HPP__

#include "common/mcsp_types.h"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspGthrzKernel(idxType nnz, valType* y, valType* x_val, const idxType* x_ind,
                                mcsparseIndexBase_t idx_base) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx < nnz) {
        idxType i = x_ind[idx] - idx_base;
        x_val[idx] = y[i];
        y[i] = static_cast<valType>(0);
    }
}

#endif