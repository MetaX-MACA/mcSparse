#ifndef KERNELS_CONVERSION_CSR2CSC_DEVICE_HPP__
#define KERNELS_CONVERSION_CSR2CSC_DEVICE_HPP__

#include "common/internal/mcsp_bfloat16.hpp"
#include "common/internal/mcsp_half.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspCsr2CscKernel(idxType nnz, const idxType *in1, const valType *in2, const idxType *map,
                                  idxType *out1, valType *out2) {
    idxType idx = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (idx >= nnz) {
        return;
    }
    idxType i = map[idx];
    out1[idx] = in1[i];
    out2[idx] = in2[i];
}

#endif