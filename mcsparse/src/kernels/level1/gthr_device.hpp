#ifndef KERNELS_LEVEL1_GTHR_DEVICE_HPP__
#define KERNELS_LEVEL1_GTHR_DEVICE_HPP__

#include "common/mcsp_types.h"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspGthrKernel(idxType nnz, const valType* y, valType* x_val, const idxType* x_ind,
                               mcsparseIndexBase_t idx_base) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx < nnz) {
        x_val[idx] = y[x_ind[idx] - idx_base];
    }
}

#endif