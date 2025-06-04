#ifndef GTSV_NO_PIVOT_DEVICE_H_
#define GTSV_NO_PIVOT_DEVICE_H_

#include "mcsp_config.h"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspGtsvPcrKernel(idxType m, idxType n, idxType ldb, idxType iter, idxType pow2m, const valType *dl,
                                  const valType *d, const valType *du, valType *B, valType *temp_dl, valType *temp_d,
                                  valType *temp_du, valType *temp_B, valType *temp_x) {
    idxType row = threadIdx.x;
    idxType col = blockIdx.x;
    idxType idx = col * m + row;
    idxType stride = 1;
    if (row < m) {
        temp_dl[idx] = dl[row];
        temp_d[idx] = d[row];
        temp_du[idx] = du[row];
        temp_B[idx] = B[ldb * col + row];
        __syncthreads();

        for (idxType i = 0; i < iter; i++) {
            idxType left = row > stride ? (row - stride) : 0;
            idxType right = (row + stride) < m ? (row + stride) : (m - 1);
            right = col * m + right;
            left = col * m + left;

            valType k1 = temp_dl[idx] / temp_d[left];
            valType k2 = temp_du[idx] / temp_d[right];

            valType tdm = temp_d[idx] - temp_du[left] * k1 - temp_dl[right] * k2;
            valType tB = temp_B[idx] - temp_B[left] * k1 - temp_B[right] * k2;
            valType tdl = -temp_dl[left] * k1;
            valType tdu = -temp_du[right] * k2;
            __syncthreads();

            temp_d[idx] = tdm;
            temp_B[idx] = tB;
            temp_dl[idx] = tdl;
            temp_du[idx] = tdu;
            stride *= 2;
            __syncthreads();
        }

        if (row < pow2m / 2) {
            idxType left = row;
            idxType right = row + stride;
            if (right < m) {
                left = col * m + row;
                right = col * m + row + stride;
                valType det = temp_d[right] * temp_d[left] - temp_du[left] * temp_dl[right];
                temp_x[left] = (temp_d[right] * temp_B[left] - temp_du[left] * temp_B[right]) / det;
                temp_x[right] = (temp_B[right] * temp_d[left] - temp_B[left] * temp_dl[right]) / det;
            } else {
                left = col * m + row;
                temp_x[left] = temp_B[left] / temp_d[left];
            }
        }
        __syncthreads();

        B[ldb * col + row] = temp_x[col * m + row];
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspGtsvPcrInitValKernel(idxType m, idxType n, idxType ldb, const valType *dl, const valType *d,
                                         const valType *du, const valType *B, valType *temp_dl, valType *temp_d,
                                         valType *temp_du, valType *temp_B) {
    idxType idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    idxType row = idx % m;
    idxType col = idx / m;
    if (col < n && row < m) {
        temp_dl[idx] = dl[row];
        temp_d[idx] = d[row];
        temp_du[idx] = du[row];
        temp_B[idx] = B[ldb * col + row];
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspGtsvPcrIterKernel(idxType m, idxType n, idxType ldb, idxType stride, const valType *dl_from,
                                      const valType *d_from, const valType *du_from, const valType *B_from,
                                      valType *dl_to, valType *d_to, valType *du_to, valType *B_to) {
    idxType idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    idxType row = idx % m;
    idxType col = idx / m;
    if (col < n && row < m) {
        idxType left = row > stride ? (row - stride) : 0;
        idxType right = (row + stride) < m ? (row + stride) : (m - 1);
        right = col * m + right;
        left = col * m + left;

        valType k1 = dl_from[idx] / d_from[left];
        valType k2 = du_from[idx] / d_from[right];

        d_to[idx] = d_from[idx] - du_from[left] * k1 - dl_from[right] * k2;
        B_to[idx] = B_from[idx] - B_from[left] * k1 - B_from[right] * k2;
        dl_to[idx] = -dl_from[left] * k1;
        du_to[idx] = -du_from[right] * k2;
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspGtsvPcrGetXKernel(idxType m, idxType n, idxType ldb, idxType stride, idxType pow2m,
                                      const valType *temp_dl, const valType *temp_d, const valType *temp_du,
                                      const valType *temp_B, valType *temp_x) {
    idxType idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    idxType row = idx % m;
    idxType col = idx / m;
    if (col < n && row < m && row < pow2m / 2) {
        idxType left = row;
        idxType right = row + stride;
        if (right < m) {
            left = col * m + row;
            right = col * m + row + stride;
            valType det = temp_d[right] * temp_d[left] - temp_du[left] * temp_dl[right];
            temp_x[left] = (temp_d[right] * temp_B[left] - temp_du[left] * temp_B[right]) / det;
            temp_x[right] = (temp_B[right] * temp_d[left] - temp_B[left] * temp_dl[right]) / det;
        } else {
            left = col * m + row;
            temp_x[left] = temp_B[left] / temp_d[left];
        }
    }
}

template <unsigned int BLOCK_SIZE, typename idxType, typename valType>
__global__ void mcspGtsvPcrSetXKernel(idxType m, idxType n, idxType ldb, valType *B, const valType *temp_x) {
    idxType idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    idxType row = idx % m;
    idxType col = idx / m;
    if (row < m && col < n) {
        B[ldb * col + row] = temp_x[col * m + row];
    }
}
#endif