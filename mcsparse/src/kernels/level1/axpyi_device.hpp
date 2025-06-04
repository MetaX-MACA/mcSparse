#ifndef KERNELS_LEVEL1_AXPYI_DEVICE_HPP__
#define KERNELS_LEVEL1_AXPYI_DEVICE_HPP__

#include "common/mcsp_types.h"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void axpyi_kernel(idxType nnz, valType alpha, const valType* x_val, const idxType* x_ind, valType* y,
                             mcsparseIndexBase_t idx_base) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx >= nnz) {
        return;
    }

    idxType i = x_ind[idx] - idx_base;
    y[i] = alpha * x_val[idx] + y[i];
}

#endif