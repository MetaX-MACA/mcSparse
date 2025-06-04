#ifndef KERNELS_CONVERSION_GEBSR2GEBSC_DEVICE_HPP__
#define KERNELS_CONVERSION_GEBSR2GEBSC_DEVICE_HPP__

#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspBsr2bscKernel(idxType nnz, idxType block_nnz, const idxType *bsr_rows, const valType *bsr_vals,
                                  const idxType *map, idxType *bsc_rows, valType *bsc_vals) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx >= nnz) {
        return;
    }
    idxType row_idx = idx / block_nnz;
    idxType val_idx = idx % block_nnz;
    idxType i = map[row_idx];
    bsc_rows[row_idx] = bsr_rows[i];
    bsc_vals[idx] = bsr_vals[i * block_nnz + val_idx];
}

#endif