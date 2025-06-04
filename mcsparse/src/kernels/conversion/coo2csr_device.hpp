#ifndef KERNELS_CONVERSION_COO2CSR_DEVICE_HPP__
#define KERNELS_CONVERSION_COO2CSR_DEVICE_HPP__

#include "common/mcsp_types.h"
#include "mcsp_config.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType>
__global__ void mcspCoo2CsrKernel(idxType nnz, idxType m, const idxType *coo_rows, idxType *csr_rows,
                                  mcsparseIndexBase_t csr_base) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;

    idxType cur_row;
    idxType next_row;

    if (tid <= nnz) {
        cur_row = (tid == 0) ? 0 : (coo_rows[tid - 1] - csr_base + 1);
        next_row = (tid == nnz) ? (m + 1) : (coo_rows[tid] - csr_base + 1);

        for (idxType i = cur_row; i < next_row; i++) {
            csr_rows[i] = tid + csr_base;
        }
    }
}

template <typename idxType>
__global__ void mcspCsr2CooKernel(idxType m, const idxType *csr_rows, idxType *coo_rows, mcsparseIndexBase_t csr_base) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType row = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;

    if (row < m) {
        idxType row_start = csr_rows[row] - csr_base;
        idxType row_end = csr_rows[row + 1] - csr_base;

        for (idxType j = row_start + lane; j < row_end; j += WARP_SIZE) {
            coo_rows[j] = row + csr_base;
        }
    }
}

#endif
