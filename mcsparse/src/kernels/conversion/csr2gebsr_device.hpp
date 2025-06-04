#ifndef KERNELS_CONVERSION_CSR2BSR_DEVICE_HPP__
#define KERNELS_CONVERSION_CSR2BSR_DEVICE_HPP__

#include "mcsp_config.h"
#include "mcsp_general_utility.h"
#include "mcsp_runtime_wrapper.h"

template <unsigned int BLOCKSIZE, typename idxType>
__global__ void mcspCsr2BsrNnzKernel(idxType m, idxType n, idxType bsr_mb, idxType bsr_nb, mcsparseIndexBase_t csr_base,
                                     const idxType *csr_row_ind, const idxType *csr_col_ind,
                                     mcsparseIndexBase_t bsr_base, idxType *bsr_row_ind, idxType row_block_dim,
                                     idxType col_block_dim, void *temp_buffer) {
    idxType bsr_row = BLOCKSIZE * blockIdx.x + threadIdx.x;

    if (bsr_row >= bsr_mb) {
        return;
    }

    idxType nnzb_per_row = 0;
    char *ptr = reinterpret_cast<char *>(temp_buffer);
    ptr += 2 * bsr_row * ALIGN(sizeof(idxType) * row_block_dim, ALIGNED_SIZE);
    idxType *csr_row_starts = reinterpret_cast<idxType *>(ptr);
    ptr += ALIGN(sizeof(idxType) * row_block_dim, ALIGNED_SIZE);
    idxType *csr_row_ends = reinterpret_cast<idxType *>(ptr);

    for (idxType j = 0; j < row_block_dim; j++) {
        idxType csr_row = bsr_row * row_block_dim + j;
        csr_row_starts[j] = 0;
        csr_row_ends[j] = 0;
        if (csr_row < m) {
            csr_row_starts[j] = csr_row_ind[csr_row] - csr_base;
            csr_row_ends[j] = csr_row_ind[csr_row + 1] - csr_base;
        }
    }

    idxType bsr_col = 0;
    while (bsr_col < bsr_nb) {
        idxType min_bsr_col = bsr_nb;
        for (idxType j = 0; j < row_block_dim; j++) {
            idxType start = csr_row_starts[j];
            idxType end = csr_row_ends[j];
            for (idxType i = start; i < end; ++i) {
                idxType cur_bsr_col = (csr_col_ind[i] - csr_base) / col_block_dim;
                if (cur_bsr_col >= bsr_col) {
                    if (cur_bsr_col <= min_bsr_col) {
                        min_bsr_col = cur_bsr_col;
                    }
                    csr_row_starts[j] = i;
                    break;
                }
            }
        }
        bsr_col = min_bsr_col + 1;
        if (min_bsr_col < bsr_nb) {
            nnzb_per_row++;
        }
    }
    bsr_row_ind[bsr_row + 1] = nnzb_per_row;
    bsr_row_ind[0] = bsr_base;
}

template <unsigned int BLOCKSIZE, typename idxType, typename valType>
__global__ void mcspCsr2BsrKernel(idxType m, idxType n, idxType bsr_mb, idxType bsr_nb, mcsparseIndexBase_t csr_base,
                                  const valType *csr_vals, const idxType *csr_row_ind, const idxType *csr_col_ind,
                                  mcsparseIndexBase_t bsr_base, valType *bsr_vals, idxType *bsr_row_ind,
                                  idxType *bsr_col_ind, idxType row_block_dim, idxType col_block_dim,
                                  mcsparseDirection_t dir, void *temp_buffer) {
    idxType bsr_row = BLOCKSIZE * blockIdx.x + threadIdx.x;

    if (bsr_row >= bsr_mb) {
        return;
    }

    idxType bsr_col_index = 0;
    idxType bsr_row_start = bsr_row_ind[bsr_row] - bsr_base;

    char *ptr = reinterpret_cast<char *>(temp_buffer);
    ptr += 3 * bsr_row * ALIGN(sizeof(idxType) * row_block_dim, ALIGNED_SIZE) +
           bsr_row * ALIGN(sizeof(valType) * row_block_dim, ALIGNED_SIZE);

    idxType *csr_row_starts = reinterpret_cast<idxType *>(ptr);
    ptr += ALIGN(sizeof(idxType) * row_block_dim, ALIGNED_SIZE);
    idxType *csr_row_ends = reinterpret_cast<idxType *>(ptr);
    ptr += ALIGN(sizeof(idxType) * row_block_dim, ALIGNED_SIZE);
    idxType *cur_csr_col_ind = reinterpret_cast<idxType *>(ptr);
    ptr += ALIGN(sizeof(idxType) * row_block_dim, ALIGNED_SIZE);
    valType *cur_csr_vals = reinterpret_cast<valType *>(ptr);

    for (idxType j = 0; j < row_block_dim; j++) {
        idxType csr_row = bsr_row * row_block_dim + j;
        csr_row_starts[j] = 0;
        csr_row_ends[j] = 0;
        if (csr_row < m) {
            csr_row_starts[j] = csr_row_ind[csr_row] - csr_base;
            csr_row_ends[j] = csr_row_ind[csr_row + 1] - csr_base;
        }
    }

    idxType csr_col = 0;
    idxType bsr_col = 0;

    while (csr_col < n) {
        idxType min_csr_col = n;
        for (idxType j = 0; j < row_block_dim; j++) {
            idxType start = csr_row_starts[j];
            idxType end = csr_row_ends[j];
            cur_csr_vals[j] = 0;
            cur_csr_col_ind[j] = n;

            for (idxType i = start; i < end; i++) {
                idxType cur_csr_col = csr_col_ind[i] - csr_base;
                valType cur_csr_val = csr_vals[i];
                cur_csr_col_ind[j] = cur_csr_col;
                cur_csr_vals[j] = cur_csr_val;

                if (cur_csr_col >= csr_col) {
                    if (cur_csr_col <= min_csr_col) {
                        min_csr_col = cur_csr_col;
                    }
                    csr_row_starts[j] = i;
                    break;
                }
            }
        }

        if (min_csr_col < n && min_csr_col / col_block_dim >= bsr_col) {
            bsr_col_ind[bsr_col_index + bsr_row_start] = min_csr_col / col_block_dim + bsr_base;
            bsr_col_index++;
            bsr_col = (min_csr_col / col_block_dim) + 1;
        }

        for (idxType j = 0; j < row_block_dim; j++) {
            idxType cur_csr_col = cur_csr_col_ind[j];
            if (cur_csr_col < n && cur_csr_col / col_block_dim == min_csr_col / col_block_dim) {
                idxType bsr_index;
                if (dir == MCSPARSE_DIRECTION_ROW) {
                    bsr_index = (bsr_col_index + bsr_row_start - 1) * col_block_dim * row_block_dim +
                                j * col_block_dim + cur_csr_col % col_block_dim;
                } else {
                    bsr_index = (bsr_col_index + bsr_row_start - 1) * col_block_dim * row_block_dim +
                                (cur_csr_col % col_block_dim) * row_block_dim + j;
                }
                bsr_vals[bsr_index] = cur_csr_vals[j];
            }
        }
        csr_col = min_csr_col + 1;
    }
}
#endif