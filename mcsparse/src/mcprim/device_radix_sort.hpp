#ifndef MCPRIM_DEVICE_RADIX_SORT_HPP_
#define MCPRIM_DEVICE_RADIX_SORT_HPP_

#include "device_scan.hpp"
#include "mcsp_config.h"
#include "prim_types.h"

namespace mcprim {

#ifndef __MACA__
#define WARP_THREADS 32
#define WARP_THREADS_EXP 5
#else
#define WARP_THREADS 64
#define WARP_THREADS_EXP 6
#endif

MCPRIM_DEVICE MCPRIM_FORCE_INLINE unsigned int get_lane_id(unsigned int thread_id) {
    return thread_id & (WARP_THREADS - 1);
}

MCPRIM_DEVICE MCPRIM_FORCE_INLINE unsigned int get_warp_id(unsigned int thread_id) {
    return thread_id >> WARP_THREADS_EXP;
}

#if defined(__MACA__)
MCPRIM_DEVICE MCPRIM_FORCE_INLINE unsigned int masked_bit_count(uint64_t x, unsigned int add = 0) {
    int c;
#ifdef __MACA__
    c = __builtin_mxc_mbcnt_lo(static_cast<int>(x), add);
    c = __builtin_mxc_mbcnt_hi(static_cast<int>(x >> 32), c);
#else
    c = __mbcnt_lo(static_cast<int>(x), add);
    c = __mbcnt_hi(static_cast<int>(x >> 32), c);
#endif

    return c;
}
#endif

template <typename valType>
MCPRIM_DEVICE MCPRIM_FORCE_INLINE valType warp_scan(valType val) {
    valType lane = get_lane_id(threadIdx.x);

#pragma unroll
    for (unsigned int i = 1; i < WARP_THREADS; i *= 2) {
#if defined(__MACA__)
        valType tmp = __shfl_up(val, i);
        if (lane >= i) {
            val += tmp;
        }
#else
        valType tmp = __shfl_up_sync(UINT32_BIT_MASK, val, i);
        if (lane >= i) {
            val += tmp;
        }
        __syncwarp();
#endif
    }

    return val;
}

template <unsigned int WARP_NUM, typename valType>
MCPRIM_DEVICE MCPRIM_FORCE_INLINE int block_scan(valType val) {
    int warp_id = get_warp_id(threadIdx.x);
    int lane = get_lane_id(threadIdx.x);
    __shared__ int warp_sum[WARP_NUM];

    val = warp_scan(val);
    __syncthreads();

    if (lane == WARP_THREADS - 1) {
        warp_sum[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        warp_sum[lane] = warp_scan(warp_sum[lane]);
    }
    __syncthreads();

    if (warp_id > 0) {
        val += warp_sum[warp_id - 1];
    }
    __syncthreads();
    return val;
}

template <unsigned int BLOCKSIZE, unsigned int CHUNKSIZE, typename idxType, typename keyType, typename valType>
MCPRIM_KERNEL void intra_block_radix_sort_pairs_kernel(void *buffer, const keyType *key_input, keyType *key_output,
                                                       const valType *val_input, valType *val_output, idxType data_size,
                                                       idxType start_bit, idxType block_num) {
    extern __shared__ __align__(sizeof(keyType)) unsigned char smem[];
    keyType *vals = reinterpret_cast<keyType *>(smem);
    keyType *prefix = vals;
    keyType *shuffle = prefix + BLOCKSIZE + BLOCKSIZE / NUM_BANKS + 1;
    keyType *sum_buffer = shuffle + BLOCKSIZE + BLOCKSIZE / NUM_BANKS + 1;

    keyType *block_sum_buffer = (keyType *)buffer;

    constexpr unsigned int warp_num = (BLOCKSIZE + WARP_THREADS - 1) / WARP_THREADS;
    __shared__ keyType warp_prefix[warp_num];

    unsigned int warp_id = get_warp_id(threadIdx.x);
    unsigned int lane_id = get_lane_id(threadIdx.x);
    const idxType idx = threadIdx.x;
    const idxType tid = threadIdx.x + blockIdx.x * blockDim.x;

    keyType skey;
    valType sval;
    if (tid < data_size) {
        skey = key_input[tid];
        if (val_input != nullptr) {
            sval = val_input[tid];
        }
    } else {
        skey = static_cast<keyType>(0);
        sval = static_cast<valType>(0);
    }

    unsigned int radix = (skey >> start_bit) & (CHUNKSIZE - 1);
    __syncthreads();

    keyType tmp1, tmp2;
#pragma unroll
    for (unsigned int i = 0; i < CHUNKSIZE; i++) {
        unsigned int val = tid < data_size ? static_cast<unsigned int>(radix == i) : 0;

#if defined(__MACA__)
        uint64_t bits = __ballot(val);
        int warp_reduction = __popcll(bits);
        if (lane_id == 0) {
            warp_prefix[warp_id] = warp_reduction;
        }
        __syncthreads();

        if (idx < warp_num) {
            warp_prefix[idx] = warp_scan(warp_prefix[idx]);
        }

        // lane prefix sum
        unsigned int lane_prefix = 0;
        lane_prefix = masked_bit_count(bits, lane_prefix);
        __syncthreads();

        if (tid < data_size) {
            if (idx == BLOCKSIZE - 1 || tid == data_size - 1) {
                sum_buffer[i] = warp_prefix[warp_id];
            }
            if (radix == i) {
                shuffle[idx + CONFLICT_FREE_OFFSET(idx)] =
                    warp_id == 0 ? lane_prefix : lane_prefix + warp_prefix[warp_id - 1];
            }
        }
        __syncthreads();
#else
        prefix[idx + CONFLICT_FREE_OFFSET(idx)] = block_scan<warp_num>(val);
        __syncthreads();
        if (tid < data_size) {
            if (idx == BLOCKSIZE - 1 || tid == data_size - 1) {
                sum_buffer[i] = prefix[idx + CONFLICT_FREE_OFFSET(idx)];
            }
            if (radix == i) {
                shuffle[idx + CONFLICT_FREE_OFFSET(idx)] =
                    idx == 0 ? 0 : prefix[idx - 1 + CONFLICT_FREE_OFFSET(idx - 1)];
            }
        }
        __syncthreads();

#endif
    }

    // exclusive-scan of sum_buffer using Hillis-Steele algorithm
    if (idx == 0) {
        sum_buffer[idx + CHUNKSIZE] = 0;
    } else if (idx < CHUNKSIZE) {
        sum_buffer[idx + CHUNKSIZE] = sum_buffer[idx - 1];
    }
    __syncthreads();
#pragma unroll
    for (unsigned int i = 1; i < CHUNKSIZE; i <<= 1) {
        if (idx + i < CHUNKSIZE) {
            sum_buffer[idx + i + CHUNKSIZE] += sum_buffer[idx + CHUNKSIZE];
        }
        __syncthreads();
    }

    if (tid < data_size) {
        tmp1 = shuffle[idx + CONFLICT_FREE_OFFSET(idx)];
        tmp2 = sum_buffer[radix + CHUNKSIZE];
        tmp1 += (tmp2 + blockIdx.x * BLOCKSIZE);
        key_output[tmp1] = skey;
        if (val_output != nullptr) val_output[tmp1] = sval;
    }
    if (idx < CHUNKSIZE) {
        tmp1 = sum_buffer[idx];
        block_sum_buffer[idx * block_num + blockIdx.x] = tmp1;
    }
}

template <unsigned int BLOCKSIZE, unsigned int CHUNKSIZE, typename idxType, typename keyType, typename valType>
MCPRIM_KERNEL void radix_sort_pairs_global_reorder_kernel(void *buffer, keyType *key_input, keyType *key_output,
                                                          valType *val_input, valType *val_output, idxType data_size,
                                                          idxType start_bit, idxType block_num) {
    extern __shared__ __align__(sizeof(keyType)) unsigned char smem[];
    keyType *vals = reinterpret_cast<keyType *>(smem);

    const idxType idx = threadIdx.x;
    const idxType tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < CHUNKSIZE) {
        vals[idx] = 0;
    }
    __syncthreads();

    if (tid < data_size) {
        keyType skey = key_input[tid];
        keyType skey_pre = 0;
        valType sval;
        if (val_input != nullptr) sval = val_input[tid];
        unsigned int radix = (skey >> start_bit) & (CHUNKSIZE - 1);
        if (idx > 0) {
            skey_pre = key_input[tid - 1];
        }
        unsigned int radix_pre = (skey_pre >> start_bit) & (CHUNKSIZE - 1);
        if (radix != radix_pre) {
            vals[radix] = idx;
        }
        __syncthreads();

        keyType *block_sum_buffer = (keyType *)buffer;
        keyType dest_pos = block_sum_buffer[radix * block_num + blockIdx.x];
        dest_pos += idx - vals[radix];

        key_output[dest_pos] = skey;
        if (val_output != nullptr) val_output[dest_pos] = sval;
    }
}

// Restriction: when (end_bit - start_bit) is odd number, bits higher than end_bit must be 0
template <typename idxType, typename keyType, typename valType>
mcprimStatus_t radix_sort_pairs(void *temp_buffer, idxType &buffer_size, const keyType *key_input, keyType *key_output,
                                const valType *val_input, valType *val_output, idxType data_size,
                                mcStream_t stream = nullptr, uint32_t start_bit = 0,
                                uint32_t end_bit = 8 * sizeof(keyType)) {
#ifdef __MACA__
    constexpr unsigned int block_size = 512;
#else
    constexpr unsigned int block_size = 1024;
#endif
    constexpr unsigned int chunk_size = 4;
    constexpr unsigned int chunk_size_exp = 2;

    unsigned int block_num = (data_size + block_size - 1) / block_size;
    unsigned int scan_buffer_size;
    exclusive_scan(nullptr, scan_buffer_size, (keyType *)nullptr, (keyType *)nullptr, 4 * block_num, (keyType *)nullptr,
                   stream);
    if (temp_buffer == nullptr) {
        buffer_size = chunk_size * block_num * sizeof(keyType) + scan_buffer_size + data_size * sizeof(keyType) +
                      data_size * sizeof(valType);
        return MCPRIM_STATUS_SUCCESS;
    }

    void *buffer_head = temp_buffer;

    keyType *buffer = (keyType *)buffer_head;
    buffer_head = (void *)(buffer + chunk_size * block_num);

    keyType *prefix_sums_buffer = (keyType *)buffer_head;
    buffer_head = (void *)(prefix_sums_buffer + scan_buffer_size / sizeof(keyType));

    keyType *key_tmp = (keyType *)buffer_head;
    buffer_head = (void *)(key_tmp + data_size);

    valType *val_tmp = nullptr;
    if (val_input != nullptr) {
        val_tmp = (valType *)buffer_head;
        buffer_head = (void *)(val_tmp + data_size);
    }
    unsigned int shm_size = (block_size * 2 + 4 * chunk_size + (block_size * 2) / NUM_BANKS + 3) * sizeof(keyType);

    idxType bit_loc = start_bit;
    mcLaunchKernelGGL((intra_block_radix_sort_pairs_kernel<block_size, chunk_size>), dim3(block_num), dim3(block_size),
                       shm_size, stream, buffer, key_input, key_tmp, val_input, val_tmp, data_size, bit_loc,
                       (idxType)block_num);

    exclusive_scan(prefix_sums_buffer, scan_buffer_size, buffer, buffer, chunk_size * block_num, (keyType *)nullptr,
                   stream);

    mcLaunchKernelGGL((radix_sort_pairs_global_reorder_kernel<block_size, chunk_size>), dim3(block_num),
                       dim3(block_size), chunk_size * sizeof(*key_input), stream, buffer, key_tmp, key_output, val_tmp,
                       val_output, data_size, bit_loc, (idxType)block_num);

    for (bit_loc = start_bit + 2; bit_loc < end_bit; bit_loc += chunk_size_exp) {
        mcLaunchKernelGGL((intra_block_radix_sort_pairs_kernel<block_size, chunk_size>), dim3(block_num),
                           dim3(block_size), shm_size, stream, buffer, key_output, key_tmp, val_output, val_tmp,
                           data_size, bit_loc, (idxType)block_num);

        exclusive_scan(prefix_sums_buffer, scan_buffer_size, buffer, buffer, chunk_size * block_num, (keyType *)nullptr,
                       stream);

        mcLaunchKernelGGL((radix_sort_pairs_global_reorder_kernel<block_size, chunk_size>), dim3(block_num),
                           dim3(block_size), chunk_size * sizeof(*key_input), stream, buffer, key_tmp, key_output,
                           val_tmp, val_output, data_size, bit_loc, (idxType)block_num);
    }

    return MCPRIM_STATUS_SUCCESS;
}

template <typename idxType, typename keyType>
mcprimStatus_t radix_sort_keys(void *temp_buffer, idxType &buffer_size, const keyType *key_input, keyType *key_output,
                               idxType data_size, mcStream_t stream = nullptr, uint32_t start_bit = 0,
                               uint32_t end_bit = 8 * sizeof(keyType))
                               {
    return radix_sort_pairs(temp_buffer, buffer_size, key_input, key_output, (keyType *)nullptr, (keyType *)nullptr,
                            data_size, stream, start_bit, end_bit);
}

template <typename idxType, typename keyType, typename valType>
mcprimStatus_t radix_sort_pairs_range8(void *temp_buffer, idxType &buffer_size, const keyType *key_input,
                                       keyType *key_output, const valType *val_input, valType *val_output,
                                       idxType data_size, mcStream_t stream = nullptr) {
    constexpr unsigned int block_size = 512;
    constexpr unsigned int bit_count = 8;

    unsigned int block_num = (data_size + block_size - 1) / block_size;
    unsigned int scan_buffer_size;
    exclusive_scan(nullptr, scan_buffer_size, (keyType *)nullptr, (keyType *)nullptr, 8 * block_num, (keyType *)nullptr,
                   stream);
    if (temp_buffer == nullptr) {
        buffer_size = bit_count * block_num * sizeof(keyType) + scan_buffer_size + data_size * sizeof(keyType) +
                      data_size * sizeof(valType);
        return MCPRIM_STATUS_SUCCESS;
    }

    void *buffer_head = temp_buffer;

    keyType *buffer = (keyType *)buffer_head;
    buffer_head = (void *)(buffer + bit_count * block_num);

    keyType *prefix_sums_buffer = (keyType *)buffer_head;
    buffer_head = (void *)(prefix_sums_buffer + scan_buffer_size / sizeof(keyType));

    keyType *key_tmp = (keyType *)buffer_head;
    buffer_head = (void *)(key_tmp + data_size);

    valType *val_tmp = (valType *)buffer_head;
    buffer_head = (void *)(val_tmp + data_size);

    idxType bit_loc = 0;
    unsigned int shm_size = (block_size * 2 + 2 * bit_count + (block_size * 2) / NUM_BANKS + 3) * sizeof(keyType);
    mcLaunchKernelGGL((intra_block_radix_sort_pairs_kernel<block_size, bit_count>), dim3(block_num), dim3(block_size),
                       shm_size, stream, buffer, key_input, key_tmp, val_input, val_tmp, data_size, bit_loc,
                       (idxType)block_num);

    exclusive_scan(prefix_sums_buffer, scan_buffer_size, buffer, buffer, bit_count * block_num, (keyType *)nullptr,
                   stream);

    shm_size = bit_count * sizeof(keyType);
    mcLaunchKernelGGL((radix_sort_pairs_global_reorder_kernel<block_size, bit_count>), dim3(block_num),
                       dim3(block_size), shm_size, stream, buffer, key_tmp, key_output, val_tmp, val_output, data_size,
                       bit_loc, (idxType)block_num);

    return MCPRIM_STATUS_SUCCESS;
}

}  // namespace mcprim

#endif
