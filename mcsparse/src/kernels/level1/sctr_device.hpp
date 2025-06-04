#ifndef KERNELS_LEVEL1_SCTR_DEVICE_HPP__
#define KERNELS_LEVEL1_SCTR_DEVICE_HPP__

#include "common/mcsp_types.h"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspSctrKernel(idxType nnz, const valType* x_val, const idxType* x_ind, valType* y,
                               mcsparseIndexBase_t idx_base) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx < nnz) {
        y[x_ind[idx] - idx_base] = x_val[idx];
    }
}

#endif