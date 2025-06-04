#ifndef KERNELS_LEVEL1_DOTCI_DEVICE_HPP__
#define KERNELS_LEVEL1_DOTCI_DEVICE_HPP__

#include "block_reduce.hpp"
#include "common/mcsp_types.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE, typename idxType, typename computeType, typename inoutType>
__global__ void doti_block_kernel(idxType nnz, const inoutType *x_val, const idxType *x_ind, const inoutType *y,
                                  computeType *buffer, mcsparseIndexBase_t idx_base) {
    __shared__ computeType sdata[BLOCKSIZE];
    unsigned int tid = threadIdx.x;
    idxType gid = blockIdx.x * BLOCKSIZE + tid;
    if constexpr (!std::is_same_v<computeType, inoutType>) {
#if defined(__MACA__)
        if constexpr (std::is_same_v<inoutType, __half> || std::is_same_v<inoutType, mcsp_bfloat16>) {
            sdata[tid] = gid < nnz ? GetFloatFromLowReal(y[x_ind[gid] - idx_base]) * GetFloatFromLowReal(x_val[gid])
                                   : static_cast<computeType>(0);
        } else if constexpr (std::is_same_v<inoutType, int8_t>) {
            sdata[tid] = gid < nnz
                             ? static_cast<computeType>(y[x_ind[gid] - idx_base]) * static_cast<computeType>(x_val[gid])
                             : static_cast<computeType>(0);
        } else if constexpr (std::is_same_v<inoutType, __half2> || std::is_same_v<inoutType, mcsp_bfloat162>) {
            sdata[tid] = gid < nnz ? GetCf32FromLowComplex(y[x_ind[gid] - idx_base]) * GetCf32FromLowComplex(x_val[gid])
                                   : static_cast<computeType>(0);
        }
#endif
    } else {
        sdata[tid] = gid < nnz ? y[x_ind[gid] - idx_base] * x_val[gid] : static_cast<computeType>(0);
    }
    __syncthreads();

    mcprim::intra_block_reduce<BLOCKSIZE>(tid, sdata, mcprim::plus<computeType>());

    if (tid == 0) {
        buffer[blockIdx.x] = sdata[0];
    }
}

#endif