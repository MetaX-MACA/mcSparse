#ifndef MCPRIM_BLOCK_SCAN_HPP_
#define MCPRIM_BLOCK_SCAN_HPP_

#include <assert.h>

#include "prim_types.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

namespace mcprim {

template <unsigned int BLOCKSIZE, unsigned int PARTSIZE, typename idxType, typename valType>
MCPRIM_KERNEL void block_exclusive_scan_plus_kernel(valType *part, valType *output, idxType n) {
    if (blockIdx.x == 0) return;
    idxType index = blockIdx.x * PARTSIZE + threadIdx.x;

#pragma unroll
    for (idxType i = 0; i < PARTSIZE / BLOCKSIZE; i++) {
        if (index + i * BLOCKSIZE < n) {
            output[index + i * BLOCKSIZE] += part[blockIdx.x];
        }
        __syncthreads();
    }
}

template <unsigned int BLOCKSIZE, unsigned int PARTSIZE, typename idxType, typename valType>
MCPRIM_KERNEL void block_exclusive_scan_kernel(const valType *dev_in, valType *dev_part, valType *dev_output,
                                               idxType n) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *shm = reinterpret_cast<valType *>(smem);
    unsigned int tid = threadIdx.x;

#pragma unroll
    for (idxType i = 0; i < PARTSIZE / BLOCKSIZE; i++) {
        idxType index = blockIdx.x * PARTSIZE + i * BLOCKSIZE + threadIdx.x;
        idxType id = tid + i * BLOCKSIZE;
        idxType offset = CONFLICT_FREE_OFFSET(id);
        shm[id + offset] = index < n ? dev_in[index] : 0;
    }
    __syncthreads();

    valType tmp1, tmp2;
#pragma unroll
    for (idxType stride = 1; stride < PARTSIZE; stride *= 2) {
        for (idxType r_idx = 2 * stride * (tid + 1) - 1; r_idx < PARTSIZE; r_idx += 2 * stride * BLOCKSIZE) {
            idxType l_idx = r_idx - stride;
            idxType l_offset = CONFLICT_FREE_OFFSET(l_idx);
            idxType r_offset = CONFLICT_FREE_OFFSET(r_idx);

            tmp1 = shm[l_idx + l_offset];
            shm[r_idx + r_offset] += tmp1;
        }
        __syncthreads();
    }

    if (tid == 0) {
        dev_part[blockIdx.x] = shm[PARTSIZE - 1 + CONFLICT_FREE_OFFSET(PARTSIZE - 1)];
        shm[PARTSIZE - 1 + CONFLICT_FREE_OFFSET(PARTSIZE - 1)] = 0;
    }
    __syncthreads();

#pragma unroll
    for (idxType stride = PARTSIZE / 2; stride > 0; stride /= 2) {
        for (idxType r_idx = 2 * stride * (tid + 1) - 1; r_idx < PARTSIZE; r_idx += 2 * stride * BLOCKSIZE) {
            idxType l_idx = r_idx - stride;
            idxType l_offset = CONFLICT_FREE_OFFSET(l_idx);
            idxType r_offset = CONFLICT_FREE_OFFSET(r_idx);

            tmp1 = shm[l_idx + l_offset];
            tmp2 = shm[r_idx + r_offset];
            shm[l_idx + l_offset] = tmp2;
            shm[r_idx + r_offset] += tmp1;
        }
        __syncthreads();
    }

#pragma unorll
    for (idxType i = 0; i < PARTSIZE / BLOCKSIZE; i++) {
        idxType index = blockIdx.x * PARTSIZE + i * BLOCKSIZE + threadIdx.x;
        if (index < n) {
            dev_output[index] = shm[tid + i * BLOCKSIZE + CONFLICT_FREE_OFFSET(tid + i * BLOCKSIZE)];
        }
    }
}

template <unsigned int BLOCKSIZE, unsigned int PARTSIZE, typename idxType, typename valType>
MCPRIM_KERNEL void block_inclusive_scan_plus_kernel(valType *part, valType *output, idxType n) {
    if (blockIdx.x == 0) return;
    idxType index = blockIdx.x * PARTSIZE + threadIdx.x;

    for (idxType i = 0; i < PARTSIZE / BLOCKSIZE; i++) {
        if (index + i * BLOCKSIZE < n) {
            output[index + i * BLOCKSIZE] += part[blockIdx.x - 1];
        }
        __syncthreads();
    }
}

template <unsigned int BLOCKSIZE, unsigned int PARTSIZE, typename idxType, typename valType>
MCPRIM_KERNEL void block_inclusive_scan_kernel(const valType *dev_in, valType *dev_part, valType *dev_output,
                                               idxType n) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *shm = reinterpret_cast<valType *>(smem);
    unsigned int tid = threadIdx.x;

#pragma unroll
    for (idxType i = 0; i < PARTSIZE / BLOCKSIZE; i++) {
        idxType index = blockIdx.x * PARTSIZE + i * BLOCKSIZE + threadIdx.x;
        idxType idx = tid + i * BLOCKSIZE;
        idxType offset = CONFLICT_FREE_OFFSET(idx);
        shm[idx + offset] = index < n ? dev_in[index] : 0;
    }
    __syncthreads();

    valType tmp1, tmp2;
#pragma unroll
    for (idxType stride = 1; stride < PARTSIZE; stride *= 2) {
        for (idxType r_idx = 2 * stride * (tid + 1) - 1; r_idx < PARTSIZE; r_idx += 2 * stride * BLOCKSIZE) {
            idxType r_offset = CONFLICT_FREE_OFFSET(r_idx);
            idxType l_idx = r_idx - stride;
            idxType l_offset = CONFLICT_FREE_OFFSET(l_idx);

            tmp1 = shm[l_idx + l_offset];
            shm[r_idx + r_offset] += tmp1;
        }
        __syncthreads();
    }

    valType part_sum = 0;
    if (tid == 0) {
        idxType offset = CONFLICT_FREE_OFFSET(PARTSIZE - 1);
        idxType idx = PARTSIZE - 1 + offset;
        part_sum = shm[idx];
        dev_part[blockIdx.x] = shm[idx];

        shm[idx] = 0;
    }
    __syncthreads();

#pragma unroll
    for (idxType stride = PARTSIZE / 2; stride > 0; stride /= 2) {
        for (idxType r_idx = 2 * stride * (tid + 1) - 1; r_idx < PARTSIZE; r_idx += 2 * stride * BLOCKSIZE) {
            idxType l_idx = r_idx - stride;
            idxType l_offset = CONFLICT_FREE_OFFSET(l_idx);
            idxType r_offset = CONFLICT_FREE_OFFSET(r_idx);

            tmp1 = shm[l_idx + l_offset];
            tmp2 = shm[r_idx + r_offset];
            shm[l_idx + l_offset] = tmp2;
            shm[r_idx + r_offset] += tmp1;
        }
        __syncthreads();
    }

#pragma unroll
    for (idxType i = 0; i < PARTSIZE / BLOCKSIZE; i++) {
        idxType index = blockIdx.x * PARTSIZE + i * BLOCKSIZE + threadIdx.x;
        if (index < n - 1) {
            idxType idx = tid + i * BLOCKSIZE + 1;
            idx += CONFLICT_FREE_OFFSET(idx);
            dev_output[index] = shm[idx];
        }
    }

    __syncthreads();
    if (tid == 0) {
        if ((blockIdx.x + 1) * PARTSIZE < n)
            dev_output[(blockIdx.x + 1) * PARTSIZE - 1] = part_sum;
        else
            dev_output[n - 1] = part_sum;
    }
}

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
MCPRIM_KERNEL void add_initial_value_kernel(valType *output, idxType n, valType init_val) {
    idxType index = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (index < n) {
        output[index] += init_val;
    }
}

}  // namespace mcprim

#endif
