#ifndef COMMON_MCSP_DENSE_TRANSPOSE_DEVICE_HPP_
#define COMMON_MCSP_DENSE_TRANSPOSE_DEVICE_HPP_

#include "common/mcsp_types.h"
#include "mcsp_runtime_wrapper.h"

// Perform dense matrix transposition
template <unsigned int DIMX, unsigned int DIMY, typename idxType, typename valType>
__global__ void mcspDenseTransposeKernel(idxType m, idxType n, const valType* A, idxType lda, valType* B, idxType ldb,
                                         int64_t batch_stride = 0) {
    int lid = threadIdx.x & (DIMX - 1);
    int wid = threadIdx.x / DIMX;
    idxType row_A = blockIdx.x * DIMX + lid;
    idxType row_B = blockIdx.x * DIMX + wid;
    idxType batch_idx = blockIdx.y;
    const valType* cur_A_ptr = &A[batch_idx * batch_stride];
    valType* cur_B_ptr = &B[batch_idx * batch_stride];

    __shared__ valType sdata[DIMX][DIMX];
    for (idxType j = 0; j < n; j += DIMX) {
        __syncthreads();

        idxType col_A = j + wid;
        for (idxType k = 0; k < DIMX; k += DIMY) {
            if (row_A < m && col_A + k < n) {
                sdata[wid + k][lid] = cur_A_ptr[row_A + lda * (col_A + k)];
            }
        }
        __syncthreads();

        idxType col_B = j + lid;
        for (idxType k = 0; k < DIMX; k += DIMY) {
            if (col_B < n && row_B + k < m) {
                cur_B_ptr[col_B + ldb * (row_B + k)] = sdata[lid][wid + k];
            }
        }
    }
}

// Perform dense matrix transposition back after transposition
template <unsigned int DIMX, unsigned int DIMY, typename idxType, typename valType>
__global__ void mcspDenseTransposeBackKernel(idxType m, idxType n, const valType* A, idxType lda, valType* B,
                                             idxType ldb) {
    int lid = threadIdx.x & (DIMX - 1);
    int wid = threadIdx.x / DIMX;

    idxType row_A = blockIdx.x * DIMX + wid;
    idxType row_B = blockIdx.x * DIMX + lid;

    __shared__ valType sdata[DIMX][DIMX];

    for (idxType j = 0; j < n; j += DIMX) {
        __syncthreads();

        idxType col_A = j + lid;
        for (idxType k = 0; k < DIMX; k += DIMY) {
            if (col_A < n && row_A + k < m) {
                sdata[wid + k][lid] = A[col_A + lda * (row_A + k)];
            }
        }
        __syncthreads();

        idxType col_B = j + wid;
        for (idxType k = 0; k < DIMX; k += DIMY) {
            if (row_B < m && col_B + k < n) {
                B[row_B + ldb * (col_B + k)] = sdata[lid][wid + k];
            }
        }
    }
}

#endif
