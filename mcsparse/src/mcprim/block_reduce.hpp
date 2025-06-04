#ifndef MCPRIM_BLOCK_REDUCE_HPP_
#define MCPRIM_BLOCK_REDUCE_HPP_

#include <assert.h>

#include "prim_types.h"

namespace mcprim {

template <unsigned int BLOCKSIZE, typename idxType, typename T, class binaryOp>
__device__ __forceinline__ void intra_block_reduce(idxType i, T *data, binaryOp op) {
    if (BLOCKSIZE > 512) {
        if (i < 512 && i + 512 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 512]);
        }
        __syncthreads();
    }
    if (BLOCKSIZE > 256) {
        if (i < 256 && i + 256 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 256]);
        }
        __syncthreads();
    }
    if (BLOCKSIZE > 128) {
        if (i < 128 && i + 128 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 128]);
        }
        __syncthreads();
    }
    if (BLOCKSIZE > 64) {
        if (i < 64 && i + 64 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 64]);
        }
        __syncthreads();
    }
    if (BLOCKSIZE > 32) {
        if (i < 32 && i + 32 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 32]);
        }
        __syncthreads();
    }
    if (BLOCKSIZE > 16) {
        if (i < 16 && i + 16 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 16]);
        }
        __syncthreads();
    }
    if (BLOCKSIZE > 8) {
        if (i < 8 && i + 8 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 8]);
        }
        __syncthreads();
    }
    if (BLOCKSIZE > 4) {
        if (i < 4 && i + 4 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 4]);
        }
        __syncthreads();
    }
    if (BLOCKSIZE > 2) {
        if (i < 2 && i + 2 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 2]);
        }
        __syncthreads();
    }
    if (BLOCKSIZE > 1) {
        if (i < 1 && i + 1 < BLOCKSIZE) {
            data[i] = op(data[i], data[i + 1]);
        }
        __syncthreads();
    }
}

// requirement: size of output >= [(data_size -1) / BLOCKSIZE +1]
template <unsigned int BLOCKSIZE, typename idxType, typename valType, class binaryOp>
MCPRIM_KERNEL void block_reduce_kernel(valType *input, valType *output, idxType data_size, valType unitary_val,
                                       binaryOp op) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);
    const idxType idx = threadIdx.x;
    vals[idx] = unitary_val;
    const idxType tid = threadIdx.x + blockIdx.x * blockDim.x;

    const idxType stride = BLOCKSIZE * gridDim.x;
    for (idxType global_idx = tid; global_idx < data_size; global_idx += stride) {
        vals[idx] = op(input[global_idx], vals[idx]);
    }
    __syncthreads();

    intra_block_reduce<BLOCKSIZE, idxType, valType, binaryOp>(idx, vals, op);

    if (idx == 0) {
        output[blockIdx.x] = vals[idx];
    }
}

template <typename idxType, typename valType, class binaryOp>
mcprimStatus_t block_reduce(valType *input, valType *output, idxType data_size, binaryOp op,
                            mcStream_t stream = nullptr) {
    constexpr unsigned int n_repeat = 1024;
#ifdef __MACA__
    constexpr unsigned int n_elem = 512;
#else
    constexpr unsigned int n_elem = 1024;
#endif
    valType unitary_val = (valType)0;
    // if valType is complex class, the following will not take effect and unitary_val = 0
    if constexpr (op((valType)1, std::numeric_limits<valType>::max() - (valType)1) == (valType)1) {
        // op is min(), ensure op(unitary_val, x) == x;
        unitary_val = std::numeric_limits<valType>::max();
    } else if constexpr (op((valType)1, std::numeric_limits<valType>::lowest()) == (valType)1) {
        // op is max(), or lowest() == 0, ensure op(unitary_val, x) == x
        unitary_val = std::numeric_limits<valType>::lowest();
    }

    unsigned int n_block = (data_size + n_elem * n_repeat - 1) / (n_elem * n_repeat);
    mcLaunchKernelGGL((block_reduce_kernel<n_elem>), dim3(n_block), dim3(n_elem), n_elem * sizeof(*input), stream,
                       input, output, data_size, unitary_val, op);

    return MCPRIM_STATUS_SUCCESS;
}

}  // namespace mcprim

#endif
