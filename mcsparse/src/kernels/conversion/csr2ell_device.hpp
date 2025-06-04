#ifndef KERNELS_CONVERSION_CSR2ELL_DEVICE_HPP__
#define KERNELS_CONVERSION_CSR2ELL_DEVICE_HPP__

#include "block_reduce.hpp"
#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE, unsigned int LOOP_CNT, typename idxType>
__global__ void mcspCsr2EllWidthKernel(idxType m, const idxType *csr_rows, idxType *buffer) {
    extern __shared__ __align__(sizeof(idxType)) unsigned char smem[];
    idxType *vals = reinterpret_cast<idxType *>(smem);

    const idxType gid = LOOP_CNT * blockDim.x * blockIdx.x + threadIdx.x;
    const idxType nid = min(m, LOOP_CNT * blockDim.x * (blockIdx.x + 1) + threadIdx.x);
    const idxType tid = threadIdx.x;

    vals[tid] = (idxType)0;
    for (idxType i = gid; i < nid; i += blockDim.x) {
        vals[tid] = max(vals[tid], (csr_rows[i + 1] - csr_rows[i]));
    }
    __syncthreads();

    mcprim::intra_block_reduce<BLOCK_SIZE>(tid, vals, mcprim::maximum<idxType>());

    if (tid == 0) {
        buffer[blockIdx.x] = vals[0];
    }
}

template <typename idxType, typename valType>
__global__ void mcspCsr2EllKernel(idxType m, const valType *csr_vals, const idxType *csr_rows, const idxType *csr_cols,
                                  mcsparseIndexBase_t csr_base, idxType ell_k, valType *ell_vals, idxType *ell_cols,
                                  mcsparseIndexBase_t ell_base) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);

    idxType row = gid;
    if (row >= m) {
        return;
    }
    const idxType row_start = csr_rows[row] - csr_base;
    const idxType row_end = csr_rows[row + 1] - csr_base;
    idxType csr_idx;
    idxType ell_idx;
    for (idxType i = lane; i < ell_k; i += WARP_SIZE) {
        csr_idx = row_start + i;
        ell_idx = i * m + row;
        if (csr_idx < row_end) {
            ell_vals[ell_idx] = csr_vals[csr_idx];
            ell_cols[ell_idx] = csr_cols[csr_idx] - csr_base + ell_base;
        } else {
            ell_vals[ell_idx] = (valType)0.0;
            ell_cols[ell_idx] = (idxType)(-1);
        }
    }
}

#endif
