#ifndef MCPRIM_BLOCK_SEGMENTED_SCAN_HPP_
#define MCPRIM_BLOCK_SEGMENTED_SCAN_HPP_

#include <assert.h>

#include "prim_types.h"

namespace mcprim {

template <unsigned int BLOCKSIZE, typename idxType, typename flagType, typename valType>
MCPRIM_KERNEL void block_segmented_exclusive_scan_kernel(valType *buffer, flagType *flag_buffer, valType *input,
                                                         valType *output, flagType *head_flags, idxType data_size) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);
    flagType *flags = reinterpret_cast<flagType *>(vals + BLOCKSIZE);
    flagType *flags_init = flags + BLOCKSIZE;

    valType sval;
    flagType sflag;
    const idxType idx = threadIdx.x;
    const idxType tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < data_size) {
        vals[idx] = input[tid];
        flags[idx] = head_flags[tid];
        if ((idx == BLOCKSIZE - 1) || (tid == data_size - 1)) {
            sval = vals[idx];
        }
    } else {
        vals[idx] = (valType)0;
        flags[idx] = (flagType)0;
    }
    flags_init[idx] = flags[idx];

    __syncthreads();
    unsigned int r_idx, l_idx;
    valType tmp1, tmp2;
    flagType flag1, flag2, flag3;
#pragma unroll
    for (unsigned int stride = 1; stride < BLOCKSIZE; stride <<= 1) {
        r_idx = (idx + 1) * 2 * stride - 1;
        l_idx = r_idx - stride;
        if (r_idx < BLOCKSIZE) {
            tmp1 = vals[l_idx];
            tmp2 = vals[r_idx];
            flag1 = flags[l_idx];
            flag2 = flags[r_idx];
            flags[r_idx] = flag1 | flag2;
            if (flag2 == 0) {
                vals[r_idx] = tmp1 + tmp2;
            }
        }
        __syncthreads();
    }
    if (idx == BLOCKSIZE - 1) {
        vals[idx] = (valType)0;
        flags[idx] = (flagType)0;
    }
    __syncthreads();
#pragma unroll
    for (unsigned int stride = BLOCKSIZE / 2; stride != 0; stride >>= 1) {
        r_idx = (idx + 1) * 2 * stride - 1;
        l_idx = r_idx - stride;
        if (r_idx < BLOCKSIZE) {
            flag3 = flags_init[l_idx + 1];
            tmp1 = vals[r_idx];
            flag1 = flags[r_idx];
            tmp2 = vals[l_idx];
            flag2 = flags[l_idx];

            vals[l_idx] = tmp1;
            flags[l_idx] = flag1;
            if (flag3) {
                flags[r_idx] = 0;
                vals[r_idx] = 0;
            } else {
                flags[r_idx] = flag1 | flag2;
                if (flag2 == 0) {
                    vals[r_idx] = tmp1 + tmp2;
                } else {
                    vals[r_idx] = tmp2;
                }
            }
        }
        __syncthreads();
    }
    if (tid < data_size) {
        output[tid] = vals[idx];
        if ((idx == BLOCKSIZE - 1) || (tid == data_size - 1)) {
            if (buffer != nullptr) {
                buffer[blockIdx.x] = sval;
                flag_buffer[blockIdx.x] = flags_init[idx];
            }
        }
    }
}

template <unsigned int BLOCKSIZE, typename idxType, typename flagType, typename valType>
MCPRIM_KERNEL void block_segmented_exclusive_scan_upsweep_kernel(valType *buffer, flagType *flag_buffer, valType *input,
                                                                 valType *output, flagType *head_flags,
                                                                 flagType *flag_output, idxType data_size) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);
    flagType *flags = reinterpret_cast<flagType *>(vals + BLOCKSIZE);

    valType sval;
    flagType sflag;
    const idxType idx = threadIdx.x;
    const idxType tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < data_size) {
        vals[idx] = input[tid];
        flags[idx] = head_flags[tid];
        if ((idx == BLOCKSIZE - 1) || (tid == data_size - 1)) {
            sval = vals[idx];
        }
    } else {
        vals[idx] = (valType)0;
        flags[idx] = (flagType)0;
    }

    __syncthreads();
    unsigned int r_idx, l_idx;
    valType tmp1, tmp2;
    flagType flag1, flag2, flag3;
#pragma unroll
    for (unsigned int stride = 1; stride < BLOCKSIZE; stride <<= 1) {
        r_idx = (idx + 1) * 2 * stride - 1;
        l_idx = r_idx - stride;
        if (r_idx < BLOCKSIZE) {
            tmp1 = vals[l_idx];
            tmp2 = vals[r_idx];
            flag1 = flags[l_idx];
            flag2 = flags[r_idx];
            flags[r_idx] = flag1 | flag2;
            if (flag2 == 0) {
                vals[r_idx] = tmp1 + tmp2;
            }
        }
        __syncthreads();
    }

    if (tid < data_size) {
        output[tid] = vals[idx];
        flag_output[tid] = flags[idx];
    }
    if (idx == BLOCKSIZE - 1) {
        if (buffer != nullptr) {
            buffer[blockIdx.x] = vals[idx];
            flag_buffer[blockIdx.x] = flags[idx];
        }
    }
}

template <unsigned int BLOCKSIZE, typename idxType, typename idaType, typename flagType, typename valType>
MCPRIM_KERNEL void block_segmented_exclusive_scan_downsweep_kernel(valType *buffer, flagType *flag_buffer,
                                                                   valType *input, valType *output,
                                                                   flagType *flags_input, flagType *flags_output,
                                                                   flagType *init_flags, idxType data_size,
                                                                   idxType total_data_size, idaType lda, bool is_final,
                                                                   bool is_root) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);
    flagType *flags = reinterpret_cast<flagType *>(vals + BLOCKSIZE);
    flagType *flags_init = flags + BLOCKSIZE;

    flagType sflag;
    const idxType idx = threadIdx.x;
    const idxType tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < data_size) {
        vals[idx] = input[tid];
        flags[idx] = flags_input[tid];
    } else {
        vals[idx] = (valType)0;
        flags[idx] = (flagType)0;
    }
    idxType global_idx = (tid + 1) * lda;
    if (global_idx < total_data_size) {
        sflag = init_flags[global_idx];
    } else {
        sflag = 0;
    }
    flags_init[idx] = sflag;

    __syncthreads();
    if (idx == BLOCKSIZE - 1) {
        if (is_root) {
            vals[idx] = (valType)0;
            flags[idx] = (flagType)0;
        } else {
            vals[idx] = buffer[blockIdx.x];
            flags[idx] = flag_buffer[blockIdx.x];
        }
    }

    unsigned int r_idx, l_idx;
    valType tmp1, tmp2;
    flagType flag1, flag2, flag3;

    __syncthreads();
#pragma unroll
    for (unsigned int stride = BLOCKSIZE / 2; stride != 0; stride >>= 1) {
        r_idx = (idx + 1) * 2 * stride - 1;
        l_idx = r_idx - stride;
        if (r_idx < BLOCKSIZE) {
            flag3 = flags_init[l_idx];
            tmp1 = vals[r_idx];
            flag1 = flags[r_idx];
            tmp2 = vals[l_idx];
            flag2 = flags[l_idx];

            vals[l_idx] = tmp1;
            flags[l_idx] = flag1;
            if (flag3) {
                flags[r_idx] = 0;
                vals[r_idx] = 0;
            } else {
                flags[r_idx] = flag1 | flag2;
                if (flag2 == 0) {
                    vals[r_idx] = tmp1 + tmp2;
                } else {
                    vals[r_idx] = tmp2;
                }
            }
        }
        __syncthreads();
    }
    if (tid < data_size) {
        output[tid] = vals[idx];
        if (!is_final) {
            flags_output[tid] = flags[idx];
        }
    }
}

template <unsigned int BLOCKSIZE, typename idxType, typename flagType, typename valType>
MCPRIM_KERNEL void block_segmented_inclusive_scan_downsweep_kernel(valType *buffer, flagType *flag_buffer,
                                                                   valType *input, valType *output,
                                                                   flagType *flags_input, flagType *flags_output,
                                                                   valType *init_val, flagType *init_flags,
                                                                   idxType data_size, idxType total_data_size,
                                                                   idxType lda, bool is_final, bool is_root) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);
    flagType *flags = reinterpret_cast<flagType *>(vals + BLOCKSIZE);
    flagType *flags_init = flags + BLOCKSIZE;

    // valType sval;
    flagType sflag;
    flagType gflag;
    valType gval;
    const idxType idx = threadIdx.x;
    const idxType tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < data_size) {
        gval = input[tid];
        gflag = flags_input[tid];
        vals[idx] = gval;
        flags[idx] = gflag;
        gval = init_val[tid];
    } else {
        vals[idx] = (valType)0;
        flags[idx] = (flagType)0;
    }
    idxType global_idx = (tid + 1) * lda;
    if (global_idx < total_data_size) {
        sflag = init_flags[global_idx];
    } else {
        sflag = 0;
    }
    flags_init[idx] = sflag;

    __syncthreads();
    if (idx == BLOCKSIZE - 1) {
        if (is_root) {
            vals[idx] = (valType)0;
            flags[idx] = (flagType)0;
        } else {
            vals[idx] = buffer[blockIdx.x];
            flags[idx] = flag_buffer[blockIdx.x];
        }
    }

    unsigned int r_idx, l_idx;
    valType tmp1, tmp2;
    flagType flag1, flag2, flag3;

    __syncthreads();
#pragma unroll
    for (unsigned int stride = BLOCKSIZE / 2; stride != 0; stride >>= 1) {
        r_idx = (idx + 1) * 2 * stride - 1;
        l_idx = r_idx - stride;
        if (r_idx < BLOCKSIZE) {
            flag3 = flags_init[l_idx];
            tmp1 = vals[r_idx];
            flag1 = flags[r_idx];
            tmp2 = vals[l_idx];
            flag2 = flags[l_idx];

            vals[l_idx] = tmp1;
            flags[l_idx] = flag1;
            if (flag3) {
                flags[r_idx] = 0;
                vals[r_idx] = 0;
            } else {
                flags[r_idx] = flag1 | flag2;
                if (flag2 == 0) {
                    vals[r_idx] = tmp1 + tmp2;
                } else {
                    vals[r_idx] = tmp2;
                }
            }
        }
        __syncthreads();
    }
    if (tid < data_size) {
        if (!is_final) {
            flags_output[tid] = flags[idx];
            output[tid] = vals[idx];
        } else {
            output[tid] = vals[idx] + gval;
        }
    }
}

template <typename idxType, typename flagType, typename valType>
mcprimStatus_t block_segmented_exclusive_scan(valType *input, valType *output, flagType *flags, idxType data_size,
                                              mcStream_t stream = nullptr) {
#ifdef __MACA__
    constexpr unsigned int n_elem = 512;
#else
    constexpr unsigned int n_elem = 1024;
#endif
    unsigned int n_block = (data_size + n_elem - 1) / n_elem;
    mcLaunchKernelGGL(block_segmented_exclusive_scan_kernel<n_elem>, dim3(n_block), dim3(n_elem),
                       n_elem * (sizeof(*input) + sizeof(*flags) * 2), stream, (valType *)nullptr, (flagType *)nullptr,
                       input, output, flags, data_size);

    return MCPRIM_STATUS_SUCCESS;
}

template <typename idxType, typename flagType, typename valType>
mcprimStatus_t block_segmented_exclusive_scan(void *temp_buffer, idxType &buffer_size, valType *input, valType *output,
                                              flagType *flags, idxType data_size, mcStream_t stream = nullptr) {
#ifdef __MACA__
    constexpr unsigned int n_elem = 512;
#else
    constexpr unsigned int n_elem = 1024;
#endif
    unsigned int n_block = (data_size + n_elem - 1) / n_elem;
    assert(buffer_size >= n_block);
    valType *val_buffer = (valType *)temp_buffer;
    flagType *flag_buffer = (flagType *)(val_buffer + n_elem);
    mcLaunchKernelGGL(block_segmented_exclusive_scan_kernel<n_elem>, dim3(n_block), dim3(n_elem),
                       n_elem * (sizeof(*input) + sizeof(*flags) * 2), stream, val_buffer, flag_buffer, input, output,
                       data_size);

    return MCPRIM_STATUS_SUCCESS;
}

template <typename idxType, typename flagType, typename valType>
mcprimStatus_t block_segmented_exclusive_scan_upsweep(valType *buffer, flagType *flag_buffer, idxType &buffer_size,
                                                      valType *input, valType *output, flagType *head_flags,
                                                      flagType *flag_output, idxType data_size,
                                                      mcStream_t stream = nullptr) {
#ifdef __MACA__
    constexpr unsigned int n_elem = 512;
#else
    constexpr unsigned int n_elem = 1024;
#endif
    unsigned int n_block = (data_size + n_elem - 1) / n_elem;
    mcLaunchKernelGGL(block_segmented_exclusive_scan_upsweep_kernel<n_elem>, dim3(n_block), dim3(n_elem),
                       n_elem * (sizeof(*input) + sizeof(*head_flags)), stream, buffer, flag_buffer, input, output,
                       head_flags, flag_output, data_size);

    return MCPRIM_STATUS_SUCCESS;
}

template <typename idxType, typename idaType, typename flagType, typename valType>
mcprimStatus_t block_segmented_exclusive_scan_downsweep(valType *buffer, flagType *flag_buffer, valType *input,
                                                        valType *output, flagType *flags_input, flagType *flags_output,
                                                        flagType *init_flags, idxType data_size,
                                                        idxType total_data_size, idaType lda, bool is_final,
                                                        bool is_root, mcStream_t stream = nullptr) {
#ifdef __MACA__
    constexpr unsigned int n_elem = 512;
#else
    constexpr unsigned int n_elem = 1024;
#endif
    unsigned int n_block = (data_size + n_elem - 1) / n_elem;
    mcLaunchKernelGGL(block_segmented_exclusive_scan_downsweep_kernel<n_elem>, dim3(n_block), dim3(n_elem),
                       n_elem * (sizeof(valType) + sizeof(flagType) * 2), stream, buffer, flag_buffer, input, output,
                       flags_input, flags_output, init_flags, data_size, total_data_size, lda, is_final, is_root);

    return MCPRIM_STATUS_SUCCESS;
}

template <typename idxType, typename flagType, typename valType>
mcprimStatus_t block_segmented_inclusive_scan_downsweep(valType *buffer, flagType *flag_buffer, valType *input,
                                                        valType *output, flagType *flags_input, flagType *flags_output,
                                                        valType *init_val, flagType *init_flags, idxType data_size,
                                                        idxType total_data_size, idxType lda, bool is_final,
                                                        bool is_root, mcStream_t stream = nullptr) {
#ifdef __MACA__
    constexpr unsigned int n_elem = 512;
#else
    constexpr unsigned int n_elem = 1024;
#endif
    unsigned int n_block = (data_size + n_elem - 1) / n_elem;
    mcLaunchKernelGGL(block_segmented_inclusive_scan_downsweep_kernel<n_elem>, dim3(n_block), dim3(n_elem),
                       n_elem * (sizeof(valType) + sizeof(flagType) * 2), stream, buffer, flag_buffer, input, output,
                       flags_input, flags_output, init_val, init_flags, data_size, total_data_size, lda, is_final,
                       is_root);

    return MCPRIM_STATUS_SUCCESS;
}

}  // namespace mcprim

#endif
