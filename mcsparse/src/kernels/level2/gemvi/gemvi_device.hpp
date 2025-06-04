#ifndef KERNELS_BLAS_GEMVI_DEVICE_HPP__
#define KERNELS_BLAS_GEMVI_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_runtime_wrapper.h"

template <uint32_t BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspGemviKernel(idxType m, idxType n, valType alpha, const valType *A, idxType lda, idxType nnz,
                                const valType *x_val, const idxType *x_ind, valType beta, valType *y,
                                mcsparseIndexBase_t idx_base, void *temp_buffer) {
    extern __shared__ __align__(sizeof(valType)) unsigned char smem[];
    valType *vals = reinterpret_cast<valType *>(smem);

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int warp_num_in_block = BLOCK_SIZE / WARP_SIZE;
    idxType columns_tail = nnz % (static_cast<idxType>(warp_num_in_block));

    idxType row = WARP_SIZE * blockIdx.x + lane;
    if (row >= m) {
        return;
    }

    vals[tid] = 0;
    // handling tail
    idxType j = warp_id;
    if (j < columns_tail) {
        vals[tid] += alpha * A[(x_ind[j] - idx_base) * lda + row] * x_val[j];
    }
    __syncthreads();

    // @TODO(zhiming): optimize nnz loop when y is large
    j += columns_tail;
    for (; j < nnz; j += warp_num_in_block) {
        vals[tid] += alpha * A[(x_ind[j] - idx_base) * lda + row] * x_val[j];
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int i = 1; i < warp_num_in_block; ++i) {
            vals[lane] += vals[lane + WARP_SIZE * i];
        }
        y[row] = vals[lane] + beta * y[row];
    }
}

#endif  // KERNELS_BLAS_GEMVI_DEVICE_HPP__
