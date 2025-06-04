#ifndef KERNELS_REORDERING_CSRCOLOR_DEVICE_HPP__
#define KERNELS_REORDERING_CSRCOLOR_DEVICE_HPP__

#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_runtime_wrapper.h"

static __forceinline__ __device__ uint32_t murmur3_32(uint32_t key, uint32_t seed) {
    uint32_t h = seed;
    uint32_t k = key;
    k *= 0xcc9e2d51;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;

    h ^= 4;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

static __forceinline__ __device__ uint32_t murmur3_32_s(uint32_t key, uint32_t seed) {
    uint32_t h = key;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

template <typename idxType>
__global__ void mcspCsrColorJplWarpKernel(idxType m, idxType color_base, const idxType *csr_rows,
                                          const idxType *csr_cols, mcsparseIndexBase_t csr_base, idxType *colors,
                                          idxType uncolored_tag, uint32_t seed) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;

    const idxType row = gid;
    if (row >= m) {
        return;
    }
    if (colors[row] != uncolored_tag) {
        return;
    }

    const idxType row_start = csr_rows[row] - csr_base;
    const idxType row_end = csr_rows[row + 1] - csr_base;
    idxType col;

    uint32_t hash_m = murmur3_32_s(row, seed);
    uint32_t hash_col;

    uint32_t min_flag = 1;
    uint32_t max_flag = 1;

    for (idxType i = row_start + lane; i < row_end; i += WARP_SIZE) {
        col = csr_cols[i] - csr_base;
        if (col == row) {
            continue;
        }
        idxType color_col = colors[col];
        if ((color_col != uncolored_tag) && (color_col != color_base) && (color_col != (color_base + 1))) {
            continue;
        }
        hash_col = murmur3_32_s(col, seed);

        if (hash_m >= hash_col) {
            min_flag = 0;
        } else if (hash_m <= hash_col) {
            max_flag = 0;
        }
    }

    // intra-warp Reduce, Op and
    for (uint32_t i = 1; i < WARP_SIZE; i <<= 1) {
#ifndef __MACA__
        max_flag &= __shfl_xor_sync(UINT32_BIT_MASK, max_flag, i);
        min_flag &= __shfl_xor_sync(UINT32_BIT_MASK, min_flag, i);
#else
        max_flag &= __shfl_xor(max_flag, i);
        min_flag &= __shfl_xor(min_flag, i);
#endif
    }

    if (lane == 0) {
        if (max_flag) {
            colors[row] = color_base;
        } else if (min_flag) {
            colors[row] = color_base + 1;
        }
    }
}

template <typename idxType>
__global__ void mcspCsrTagUncoloredKernel(idxType m, const idxType *colors, idxType *tags, idxType uncolored_tag) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        tags[tid] = (colors[tid] == uncolored_tag);
    }
}

template <typename idxType>
__global__ void mcspCsrClearUncoloredKernel(idxType m, const idxType *colors, idxType *tags, idxType uncolored_tag,
                                            idxType clear_val) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        if (colors[tid] == uncolored_tag) {
            tags[tid] = clear_val;
        } else {
            tags[tid] = colors[tid];
        }
    }
}

template <typename idxType>
__global__ void mcspCsrAssignUncoloredKernel(idxType m, idxType *colors, const idxType *tags, idxType uncolored_tag,
                                             idxType num_colored) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        if (colors[tid] == uncolored_tag) {
            colors[tid] = tags[tid] + num_colored;
        }
    }
}

template <typename idxType>
__global__ void mcspCsrColorCheckKernel(idxType m, const idxType *csr_rows, const idxType *csr_cols,
                                        mcsparseIndexBase_t csr_base, const idxType *colors, idxType *color_check) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;

    const idxType row = gid;
    if (row >= m) {
        return;
    }
    idxType cur_color = colors[row];

    const idxType row_start = csr_rows[row] - csr_base;
    const idxType row_end = csr_rows[row + 1] - csr_base;
    idxType col;
    for (idxType i = row_start + lane; i < row_end; i += WARP_SIZE) {
        col = csr_cols[i] - csr_base;
        if (col == row) {
            continue;
        }
        idxType color_col = colors[col];
        if (cur_color == color_col) {
            color_check[0] = 1;
        }
    }
}

#endif
