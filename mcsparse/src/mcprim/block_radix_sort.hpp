#ifndef MCPRIM_BLOCK_RADIX_SORT_HPP_
#define MCPRIM_BLOCK_RADIX_SORT_HPP_

#include <assert.h>

#include "prim_types.h"

namespace mcprim {

template <unsigned int BLOCKSIZE, unsigned int NBITS, typename idxType, typename keyType, typename valType>
MCPRIM_KERNEL void block_radix_sort_pairs_kernel(void *buffer, keyType *key_input, keyType *key_output,
                                                 valType *val_input, valType *val_output, idxType data_size,
                                                 idxType start_bit, idxType n_bits) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);

    valType sval;
    const idxType idx = threadIdx.x;
    const idxType tid = threadIdx.x + blockIdx.x * blockDim.x;
}

template <typename idxType, typename valType>
mcprimStatus_t block_radix_scan(valType *input, valType *output, idxType data_size, mcStream_t stream = nullptr) {
#ifdef __MACA__
    constexpr unsigned int n_elem = 512;
#else
    constexpr unsigned int n_elem = 1024;
#endif
    unsigned int n_block = (data_size + n_elem - 1) / n_elem;
    mcLaunchKernelGGL(block_exclusive_scan_kernel<n_elem>, dim3(n_block), dim3(n_elem), n_elem * sizeof(*input),
                       stream, (valType *)nullptr, input, output, data_size);

    return MCPRIM_STATUS_SUCCESS;
}

}  // namespace mcprim

#endif
