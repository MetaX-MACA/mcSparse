#ifndef KERNELS_EXTRA_SPGEMM_CSR_SPGEMM_DEVICE_HPP__
#define KERNELS_EXTRA_SPGEMM_CSR_SPGEMM_DEVICE_HPP__

#include <stdio.h>

#include "mcsp_config.h"
#include "mcsp_hashtable_device.hpp"
#include "mcsp_internal_device_kernels.hpp"
#include "mcsp_runtime_wrapper.h"

template <typename idxType>
__global__ void mcspSpgemmGroupOffsetKernel(idxType m, idxType *ibuffer, idxType *obuffer) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid == 0) {
        obuffer[0] = 0;
    } else if (tid < m + 1) {
        idxType cid;
        idxType pid = ibuffer[tid - 1];
        if (tid == m) {
            cid = 9;
        } else {
            cid = ibuffer[tid];
        }

        for (idxType i = pid; i < cid; i++) {
            obuffer[i + 1] = tid;
        }
    }
}

template <typename idxType>
__global__ void mcspSpgemmNnzSizeGroupKernel(idxType m, const idxType *csr_rows_D, idxType *buffer) {
    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        idxType row_nnz = csr_rows_D[tid + 1] - csr_rows_D[tid];

        if (row_nnz <= 16) {
            buffer[tid] = 0;
        } else if (row_nnz <= 32) {
            buffer[tid] = 1;
        } else if (row_nnz <= 256) {
            buffer[tid] = 2;
        } else if (row_nnz <= 512) {
            buffer[tid] = 3;
        } else if (row_nnz <= 1024) {
            buffer[tid] = 4;
        } else if (row_nnz <= 2048) {
            buffer[tid] = 5;
        } else if (row_nnz <= 4096) {
            buffer[tid] = 6;
        } else {
            buffer[tid] = 7;
        }
    }
}

template <typename idxType>
__global__ void mcspSpgemmExpandSizeKernel(idxType m, idxType n, const idxType *csr_rows_A, const idxType *csr_cols_A,
                                           const idxType *csr_rows_B, const idxType *csr_rows_C, idxType *buffer,
                                           bool include_addition, mcsparseIndexBase_t idx_baseA) {
    extern __shared__ __align__(sizeof(idxType)) unsigned char smem[];
    idxType *vals = reinterpret_cast<idxType *>(smem);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType row = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType idx = threadIdx.x;

    if (row < m) {
        idxType row_start = csr_rows_A[row] - idx_baseA;
        idxType row_end = csr_rows_A[row + 1] - idx_baseA;

        if (include_addition) {
            vals[idx] = csr_rows_C[row + 1] - csr_rows_C[row];
        } else {
            vals[idx] = 0;
        }
        for (idxType j = row_start + lane; j < row_end; j += WARP_SIZE) {
            idxType row_id_B = csr_cols_A[j] - idx_baseA;
            vals[idx] += csr_rows_B[row_id_B + 1] - csr_rows_B[row_id_B];
        }
#pragma unroll
        for (int i = WARP_SIZE >> 1; i != 0; i >>= 1) {
#ifndef __MACA__
            __syncwarp();
#endif
            if (lane < i) vals[idx] += vals[idx + i];
        }

        if (lane == 0) {
            buffer[row] = vals[idx];

            if (vals[idx] <= 32) {
                buffer[m + row] = 0;
            } else if (vals[idx] <= 64) {
                buffer[m + row] = 1;
            } else if (vals[idx] <= 512) {
                buffer[m + row] = 2;
            } else if (vals[idx] <= 1024) {
                buffer[m + row] = 3;
            } else if (vals[idx] <= 2048) {
                buffer[m + row] = 4;
            } else if (vals[idx] <= 4096) {
                buffer[m + row] = 5;
            } else if (vals[idx] <= 8192) {
                buffer[m + row] = 6;
            } else {
                buffer[m + row] = 7;
            }
        }
    }
}

// calculate row_nnz by intra-block hash-table, 1 warp 1 row_D
template <unsigned int HASH_SIZE, typename idxType>
__global__ void mcspSpgemmNnzWarpRowAKernel(idxType group_size, idxType group_offset, const idxType *csr_rows_A,
                                            const idxType *csr_cols_A, const idxType *csr_rows_B,
                                            const idxType *csr_cols_B, const idxType *csr_rows_C,
                                            const idxType *csr_cols_C, idxType *csr_rows_D, idxType *identity_buffer,
                                            bool include_addition, mcsparseIndexBase_t idx_baseA,
                                            mcsparseIndexBase_t idx_baseB, mcsparseIndexBase_t idx_baseC) {
    // ,idxType *test_buffer1=nullptr,idxType *test_buffer2=nullptr
    extern __shared__ __align__(sizeof(idxType)) unsigned char smem[];
    idxType *table = reinterpret_cast<idxType *>(smem);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;

    idxType *cur_table = table + wid * HASH_SIZE;
    for (idxType i = lane; i < HASH_SIZE; i += WARP_SIZE) {
        cur_table[i] = (idxType)HASH_MAGIC_NULL_VAL;
    }
    __threadfence_block();

    idxType row_nnz = 0;
    idxType row_id_A;
    if (gid < group_size) {
        row_id_A = identity_buffer[group_offset + gid];

        idxType row_start_A = csr_rows_A[row_id_A] - idx_baseA;
        idxType row_end_A = csr_rows_A[row_id_A + 1] - idx_baseA;

        for (idxType j = row_start_A + lane; j < row_end_A; j += WARP_SIZE) {
            idxType row_id_B = csr_cols_A[j] - idx_baseA;
            idxType row_start_B = csr_rows_B[row_id_B] - idx_baseB;
            idxType row_end_B = csr_rows_B[row_id_B + 1] - idx_baseB;
            for (idxType k = row_start_B; k < row_end_B; k++) {
                row_nnz += mcspInsertHashTableKey<HASH_SIZE, 79>((csr_cols_B[k] - idx_baseB), cur_table);
            }
        }
        __threadfence_block();

        if (include_addition) {
            idxType row_start_C = csr_rows_C[row_id_A] - idx_baseC;
            idxType row_end_C = csr_rows_C[row_id_A + 1] - idx_baseC;
            for (idxType j = row_start_C + lane; j < row_end_C; j += WARP_SIZE) {
                row_nnz += mcspInsertHashTableKey<HASH_SIZE, 79>((csr_cols_C[j] - idx_baseC), cur_table);
            }
            __threadfence_block();
        }

#pragma unroll
        // intra-warp reduce
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
#ifndef __MACA__
            row_nnz += __shfl_xor_sync(UINT32_BIT_MASK, row_nnz, i);
#else
            row_nnz += __shfl_xor(row_nnz, i);
#endif
        }
        if (lane == 0) {
            csr_rows_D[row_id_A] = row_nnz;
        }
    }
}

// calculate row_nnz by intra-block hash-table, 1 block 1 row_D
template <unsigned int BLOCK_SIZE, unsigned int HASH_SIZE, typename idxType>
__global__ void mcspSpgemmNnzBlockRowAKernel(idxType group_size, idxType group_offset, const idxType *csr_rows_A,
                                             const idxType *csr_cols_A, const idxType *csr_rows_B,
                                             const idxType *csr_cols_B, const idxType *csr_rows_C,
                                             const idxType *csr_cols_C, idxType *csr_rows_D, idxType *identity_buffer,
                                             bool include_addition, mcsparseIndexBase_t idx_baseA,
                                             mcsparseIndexBase_t idx_baseB, mcsparseIndexBase_t idx_baseC) {
    extern __shared__ __align__(sizeof(idxType)) unsigned char smem[];
    idxType *table = reinterpret_cast<idxType *>(smem);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = blockIdx.x;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType idx = threadIdx.x;
    const idxType wid = threadIdx.x / WARP_SIZE;

    idxType *cur_table = table;
    for (idxType i = idx; i < HASH_SIZE; i += BLOCK_SIZE) {
        cur_table[i] = (idxType)HASH_MAGIC_NULL_VAL;
    }
    __syncthreads();

    idxType row_nnz = 0;
    idxType row_id_A;
    if (gid < group_size) {
        row_id_A = identity_buffer[group_offset + gid];
        idxType row_start_A = csr_rows_A[row_id_A] - idx_baseA;
        idxType row_end_A = csr_rows_A[row_id_A + 1] - idx_baseA;

        for (idxType j = row_start_A + wid; j < row_end_A; j += BLOCK_SIZE / WARP_SIZE) {
            idxType row_id_B = csr_cols_A[j] - idx_baseA;
            idxType row_start_B = csr_rows_B[row_id_B] - idx_baseB;
            idxType row_end_B = csr_rows_B[row_id_B + 1] - idx_baseB;
            for (idxType k = row_start_B + lane; k < row_end_B; k += WARP_SIZE) {
                row_nnz += mcspInsertHashTableKey<HASH_SIZE, 79>((csr_cols_B[k] - idx_baseB), cur_table);
            }
        }
        __syncthreads();
        if (include_addition) {
            idxType row_start_C = csr_rows_C[row_id_A] - idx_baseC;
            idxType row_end_C = csr_rows_C[row_id_A + 1] - idx_baseC;
            for (idxType j = row_start_C + idx; j < row_end_C; j += BLOCK_SIZE) {
                row_nnz += mcspInsertHashTableKey<HASH_SIZE, 79>((csr_cols_C[j] - idx_baseC), cur_table);
            }
            __syncthreads();
        }

#pragma unroll
        // intra-warp reduce
        for (int i = 1; i < WARP_SIZE; i <<= 1) {
#ifndef __MACA__
            row_nnz += __shfl_xor_sync(UINT32_BIT_MASK, row_nnz, i, WARP_SIZE);
#else
            row_nnz += __shfl_xor(row_nnz, i, WARP_SIZE);
#endif
        }

        if (lane == 0) {
            table[wid] = row_nnz;
        }
        __syncthreads();

#pragma unroll
        for (unsigned int i = 1; i < BLOCK_SIZE / WARP_SIZE; i <<= 1) {
            unsigned int left_id = 2 * i * idx;
            unsigned int right_id = left_id + i;
            if (right_id < BLOCK_SIZE / WARP_SIZE) {
                table[left_id] += table[right_id];
            }
            __syncthreads();
        }

        if (idx == 0) {
            csr_rows_D[row_id_A] = table[0];
        }
    }
}

// When shared-memory is not enough for hash-table, divide col_B as different chunks.
template <unsigned int BLOCK_SIZE, unsigned int CHUNK_SIZE, typename idxType>
__global__ void mcspSpgemmNnzMultiRowAKernel(idxType group_size, idxType group_offset, idxType kb,
                                             const idxType *csr_rows_A, const idxType *csr_cols_A,
                                             const idxType *csr_rows_B, const idxType *csr_cols_B,
                                             const idxType *csr_rows_C, const idxType *csr_cols_C, idxType *csr_rows_D,
                                             idxType *identity_buffer, idxType *multi_front_buffer,
                                             bool include_addition, mcsparseIndexBase_t idx_baseA,
                                             mcsparseIndexBase_t idx_baseB, mcsparseIndexBase_t idx_baseC) {
    volatile __shared__ short table[CHUNK_SIZE];
    volatile __shared__ idxType thread_nnz[BLOCK_SIZE];

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = blockIdx.x;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType idx = threadIdx.x;
    const idxType wid = threadIdx.x / WARP_SIZE;

    idxType row_id_A;
    idxType chunk_begin = 0;
    idxType row_nnz = 0;
    if (gid < group_size) {
        row_id_A = identity_buffer[group_offset + gid];
        idxType row_start_A = csr_rows_A[row_id_A] - idx_baseA;
        idxType row_end_A = csr_rows_A[row_id_A + 1] - idx_baseA;
        idxType row_start_C;
        idxType row_end_C;
        if (include_addition) {
            row_start_C = csr_rows_C[row_id_A] - idx_baseC;
            row_end_C = csr_rows_C[row_id_A + 1] - idx_baseC;
        }

        bool chunk_start = true;
        for (idxType chunk_offset = 0; chunk_offset < kb; chunk_offset += CHUNK_SIZE) {
            thread_nnz[idx] = 0;
            for (idxType k = idx; k < CHUNK_SIZE; k += BLOCK_SIZE) {
                table[k] = 0;
            }
            __syncthreads();
            for (idxType j = row_start_A + wid; j < row_end_A; j += BLOCK_SIZE / WARP_SIZE) {
                idxType row_id_B = csr_cols_A[j] - idx_baseA;
                idxType row_start_B = (chunk_start) ? (csr_rows_B[row_id_B] - idx_baseB) : multi_front_buffer[j];
                idxType row_end_B = csr_rows_B[row_id_B + 1] - idx_baseB;

                idxType next_k = row_start_B + lane;
                for (idxType k = next_k; k < row_end_B; k += WARP_SIZE) {
                    idxType col_idx_B = csr_cols_B[k] - idx_baseB;
                    if (col_idx_B >= chunk_offset && col_idx_B < chunk_offset + CHUNK_SIZE) {
                        table[col_idx_B - chunk_offset] = 1;
                    } else if (col_idx_B >= chunk_offset + CHUNK_SIZE) {
                        next_k = k;
                        break;
                    }
                }

#ifndef __MACA__
                __syncwarp();
#endif

#pragma unroll
                // intra-warp reduce
                for (int i = 1; i < WARP_SIZE; i <<= 1) {
#ifndef __MACA__
                    next_k = min(next_k, __shfl_xor_sync(UINT32_BIT_MASK, next_k, i, WARP_SIZE));
#else
                    next_k = min(next_k, __shfl_xor(next_k, i, WARP_SIZE));
#endif
                }
                if (lane == 0) {
                    multi_front_buffer[j] = next_k;
                }
            }
            __syncthreads();

            if (include_addition) {
                for (idxType j = row_start_C + idx; j < row_end_C; j += BLOCK_SIZE) {
                    idxType col_idx_C = csr_cols_C[j] - idx_baseC;
                    if (col_idx_C >= chunk_offset && col_idx_C < chunk_offset + CHUNK_SIZE) {
                        table[col_idx_C - chunk_offset] = 1;
                    }
                }
                __syncthreads();
            }

#pragma unroll
            // block-level reduce in current CHUNK
            for (idxType stride = 1; stride < CHUNK_SIZE; stride <<= 1) {
                idxType l_idx = 2 * idx * stride;
                idxType r_idx = l_idx + stride;
                if (r_idx < CHUNK_SIZE) {
                    table[l_idx] += table[r_idx];
                }
                for (unsigned int j = 1; j < CHUNK_SIZE / (2 * BLOCK_SIZE * stride); j++) {
                    r_idx += BLOCK_SIZE * 2 * stride;
                    l_idx += BLOCK_SIZE * 2 * stride;
                    if (r_idx < CHUNK_SIZE) {
                        table[l_idx] += table[r_idx];
                    }
                }
                __syncthreads();
            }

            if (idx == 0) {
                row_nnz += table[0];
            }
            chunk_start = false;
        }

        if (idx == 0) {
            csr_rows_D[row_id_A] = row_nnz;
        }
    }
}

// calculate columns and values array of output matrix D by intra-block hash-table, 1 warp 1 row_D
template <unsigned int BLOCK_SIZE, unsigned int HASH_SIZE, typename idxType, typename valType,
          typename smType = valType>
__global__ void mcspSpgemmCalcWarpRowAKernel(idxType group_size, idxType group_offset, valType alpha,
                                             const valType *csr_vals_A, const idxType *csr_rows_A,
                                             const idxType *csr_cols_A, const valType *csr_vals_B,
                                             const idxType *csr_rows_B, const idxType *csr_cols_B, valType beta,
                                             const valType *csr_vals_C, const idxType *csr_rows_C,
                                             const idxType *csr_cols_C, valType *csr_vals_D, const idxType *csr_rows_D,
                                             idxType *csr_cols_D, const idxType *identity_buffer, bool include_addition,
                                             mcsparseIndexBase_t idx_baseA, mcsparseIndexBase_t idx_baseB,
                                             mcsparseIndexBase_t idx_baseC, mcsparseIndexBase_t idx_baseD) {
    extern __shared__ __align__(sizeof(idxType)) unsigned char smem[];
    idxType *idx_table = reinterpret_cast<idxType *>(smem);
    smType *val_table = reinterpret_cast<smType *>(idx_table + HASH_SIZE * BLOCK_SIZE / WARP_SIZE);
    short *radix_table = reinterpret_cast<short *>(val_table + HASH_SIZE * BLOCK_SIZE / WARP_SIZE);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = tid / WARP_SIZE;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType wid = threadIdx.x / WARP_SIZE;

    idxType *cur_idx_table = idx_table + wid * HASH_SIZE;
    smType *cur_val_table = val_table + wid * HASH_SIZE;
    short *cur_radix_table = radix_table + 2 * wid * HASH_SIZE;
    for (idxType i = lane; i < HASH_SIZE; i += WARP_SIZE) {
        cur_idx_table[i] = (idxType)HASH_MAGIC_NULL_VAL;
        cur_val_table[i] = GetTypedValue<smType>(0);
    }
#ifndef __MACA__
    __syncwarp();
#endif

    idxType row_id_A;
    if (gid < group_size) {
        row_id_A = identity_buffer[group_offset + gid];
        idxType row_start_D = csr_rows_D[row_id_A] - idx_baseD;
        idxType row_start_A = csr_rows_A[row_id_A] - idx_baseA;
        idxType row_end_A = csr_rows_A[row_id_A + 1] - idx_baseA;

        for (idxType j = row_start_A + lane; j < row_end_A; j += WARP_SIZE) {
            idxType row_id_B = csr_cols_A[j] - idx_baseA;
            valType val_A = csr_vals_A[j];
            idxType row_start_B = csr_rows_B[row_id_B] - idx_baseB;
            idxType row_end_B = csr_rows_B[row_id_B + 1] - idx_baseB;
            if constexpr (std::is_same_v<valType, __half>) {
                for (idxType k = row_start_B; k < row_end_B; k++) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>(
                        (csr_cols_B[k] - idx_baseB), __half2float(alpha * val_A * csr_vals_B[k]), cur_idx_table,
                        cur_val_table);
                }
            } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
                for (idxType k = row_start_B; k < row_end_B; k++) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>(
                        (csr_cols_B[k] - idx_baseB), __bfloat162float(alpha * val_A * csr_vals_B[k]), cur_idx_table,
                        cur_val_table);
                }
            } else if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
                for (idxType k = row_start_B; k < row_end_B; k++) {
                    valType ori_val = complex_mul(alpha, complex_mul(val_A, csr_vals_B[k]));
                    smType sm_val = smType(__low2float(ori_val), __high2float(ori_val));
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>((csr_cols_B[k] - idx_baseB), sm_val,
                                                                             cur_idx_table, cur_val_table);
                }
            } else {
                for (idxType k = row_start_B; k < row_end_B; k++) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>(
                        (csr_cols_B[k] - idx_baseB), alpha * val_A * csr_vals_B[k], cur_idx_table, cur_val_table);
                }
            }
        }
#ifndef __MACA__
        __syncwarp();
#endif
        if (include_addition) {
            idxType row_start_C = csr_rows_C[row_id_A] - idx_baseC;
            idxType row_end_C = csr_rows_C[row_id_A + 1] - idx_baseC;
            if constexpr (std::is_same_v<valType, __half>) {
                for (idxType j = row_start_C + lane; j < row_end_C; j += WARP_SIZE) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>(
                        (csr_cols_C[j] - idx_baseC), __half2float(beta * csr_vals_C[j]), cur_idx_table, cur_val_table);
                }
            } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
                for (idxType j = row_start_C + lane; j < row_end_C; j += WARP_SIZE) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>((csr_cols_C[j] - idx_baseC),
                                                                             __bfloat162float(beta * csr_vals_C[j]),
                                                                             cur_idx_table, cur_val_table);
                }
            } else if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
                for (idxType j = row_start_C + lane; j < row_end_C; j += WARP_SIZE) {
                    valType ori_val = complex_mul(beta, csr_vals_C[j]);
                    smType sm_val = smType(__low2float(ori_val), __high2float(ori_val));
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>((csr_cols_C[j] - idx_baseC), sm_val,
                                                                             cur_idx_table, cur_val_table);
                }
            } else {
                for (idxType j = row_start_C + lane; j < row_end_C; j += WARP_SIZE) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>(
                        (csr_cols_C[j] - idx_baseC), beta * csr_vals_C[j], cur_idx_table, cur_val_table);
                }
            }
#ifndef __MACA__
            __syncwarp();
#endif
        }

        // warp-level radix-sort
        idxType tmp_idx;
        smType tmp_val;
        short radix;
        for (unsigned int i = 0; i < 8 * sizeof(idxType); i++) {
            if (lane < HASH_SIZE) {
                tmp_idx = cur_idx_table[lane];
                tmp_val = cur_val_table[lane];
                radix = short((tmp_idx >> i) & 1);
                cur_radix_table[radix * HASH_SIZE + lane] = 1;
                cur_radix_table[(radix ^ 1) * HASH_SIZE + lane] = 0;
            }

#ifndef __MACA__
            __syncwarp();
#endif
            // exclusive scan cur_radix_table
            unsigned int r_idx, l_idx;
            short tmp1, tmp2;
#pragma unroll
            for (unsigned int stride = 1; stride < HASH_SIZE * 2; stride <<= 1) {
                r_idx = (lane + 1) * 2 * stride - 1;
                l_idx = r_idx - stride;
                if (r_idx < HASH_SIZE * 2) {
                    tmp1 = cur_radix_table[l_idx];
                    cur_radix_table[r_idx] += tmp1;
                }
#ifndef __MACA__
                __syncwarp();
#endif
            }
            if (lane == 0) {
                cur_radix_table[2 * HASH_SIZE - 1] = 0;
            }

#ifndef __MACA__
            __syncwarp();
#endif

#pragma unroll
            for (unsigned int stride = HASH_SIZE; stride != 0; stride >>= 1) {
                r_idx = (lane + 1) * 2 * stride - 1;
                l_idx = r_idx - stride;
                if (r_idx < HASH_SIZE * 2) {
                    tmp1 = cur_radix_table[r_idx];
                    tmp2 = cur_radix_table[l_idx];
                    cur_radix_table[r_idx] += tmp2;
                    cur_radix_table[l_idx] = tmp1;
                }
#ifndef __MACA__
                __syncwarp();
#endif
            }
            if (lane < HASH_SIZE) {
                tmp1 = cur_radix_table[radix * HASH_SIZE + lane];
                cur_idx_table[tmp1] = tmp_idx;
                cur_val_table[tmp1] = tmp_val;
            }
#ifndef __MACA__
            __syncwarp();
#endif
        }
        // radix sort done

        // write final result to global memory
        if (lane < HASH_SIZE && cur_idx_table[lane] != (idxType)HASH_MAGIC_NULL_VAL) {
            csr_cols_D[row_start_D + lane] = cur_idx_table[lane];
            if constexpr (std::is_same_v<valType, __half>) {
                csr_vals_D[row_start_D + lane] = __float2half(cur_val_table[lane]);
            } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
                csr_vals_D[row_start_D + lane] = __float2bfloat16(cur_val_table[lane]);
            } else if constexpr (std::is_same_v<valType, __half2>) {
                csr_vals_D[row_start_D + lane] =
                    __half2(__float2half(cur_val_table[lane].x), __float2half(cur_val_table[lane].y));
            } else if constexpr (std::is_same_v<valType, mcsp_bfloat162>) {
                csr_vals_D[row_start_D + lane] =
                    mcsp_bfloat162(__float2bfloat16(cur_val_table[lane].x), __float2bfloat16(cur_val_table[lane].y));
            } else {
                csr_vals_D[row_start_D + lane] = cur_val_table[lane];
            }
        }
    }
}

// calculate columns and values array of output matrix D by intra-block hash-table, 1 block 1 row_D
template <unsigned int BLOCK_SIZE, unsigned int HASH_SIZE, typename idxType, typename valType,
          typename smType = valType>
__global__ void mcspSpgemmCalcBlockRowAKernel(
    idxType group_size, idxType group_offset, valType alpha, const valType *csr_vals_A, const idxType *csr_rows_A,
    const idxType *csr_cols_A, const valType *csr_vals_B, const idxType *csr_rows_B, const idxType *csr_cols_B,
    valType beta, const valType *csr_vals_C, const idxType *csr_rows_C, const idxType *csr_cols_C, valType *csr_vals_D,
    const idxType *csr_rows_D, idxType *csr_cols_D, const idxType *identity_buffer, bool include_addition,
    mcsparseIndexBase_t idx_baseA, mcsparseIndexBase_t idx_baseB, mcsparseIndexBase_t idx_baseC,
    mcsparseIndexBase_t idx_baseD) {
    extern __shared__ __align__(sizeof(idxType)) unsigned char smem[];
    idxType *idx_table = reinterpret_cast<idxType *>(smem);
    smType *val_table = reinterpret_cast<smType *>(idx_table + HASH_SIZE);
    short *radix_table = reinterpret_cast<short *>(val_table + HASH_SIZE);

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = blockIdx.x;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType idx = threadIdx.x;
    const idxType wid = threadIdx.x / WARP_SIZE;

    idxType *cur_idx_table = idx_table;
    smType *cur_val_table = val_table;
    short *cur_radix_table = radix_table;
    for (idxType i = idx; i < HASH_SIZE; i += BLOCK_SIZE) {
        cur_idx_table[i] = (idxType)HASH_MAGIC_NULL_VAL;
        cur_val_table[i] = GetTypedValue<smType>(0);
    }
    __syncthreads();

    idxType row_id_A;
    if (gid < group_size) {
        row_id_A = identity_buffer[group_offset + gid];
        idxType row_start_D = csr_rows_D[row_id_A] - idx_baseD;
        idxType row_start_A = csr_rows_A[row_id_A] - idx_baseA;
        idxType row_end_A = csr_rows_A[row_id_A + 1] - idx_baseA;

        for (idxType j = row_start_A + wid; j < row_end_A; j += BLOCK_SIZE / WARP_SIZE) {
            idxType row_id_B = csr_cols_A[j] - idx_baseA;
            valType val_A = csr_vals_A[j];
            idxType row_start_B = csr_rows_B[row_id_B] - idx_baseB;
            idxType row_end_B = csr_rows_B[row_id_B + 1] - idx_baseB;
            if constexpr (std::is_same_v<valType, __half>) {
                for (idxType k = row_start_B + lane; k < row_end_B; k += WARP_SIZE) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>(
                        (csr_cols_B[k] - idx_baseB), __half2float(alpha * val_A * csr_vals_B[k]), cur_idx_table,
                        cur_val_table);
                }
            } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
                for (idxType k = row_start_B + lane; k < row_end_B; k += WARP_SIZE) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>(
                        (csr_cols_B[k] - idx_baseB), __bfloat162float(alpha * val_A * csr_vals_B[k]), cur_idx_table,
                        cur_val_table);
                }
            } else if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
                for (idxType k = row_start_B + lane; k < row_end_B; k += WARP_SIZE) {
                    valType ori_val = complex_mul(alpha, complex_mul(val_A, csr_vals_B[k]));
                    smType sm_val = smType(__low2float(ori_val), __high2float(ori_val));
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>((csr_cols_B[k] - idx_baseB), sm_val,
                                                                             cur_idx_table, cur_val_table);
                }
            } else {
                for (idxType k = row_start_B + lane; k < row_end_B; k += WARP_SIZE) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79>((csr_cols_B[k] - idx_baseB), alpha * val_A * csr_vals_B[k],
                                                            cur_idx_table, cur_val_table);
                }
            }
        }
        __syncthreads();

        if (include_addition) {
            idxType row_start_C = csr_rows_C[row_id_A] - idx_baseC;
            idxType row_end_C = csr_rows_C[row_id_A + 1] - idx_baseC;
            if constexpr (std::is_same_v<valType, __half>) {
                for (idxType j = row_start_C + idx; j < row_end_C; j += BLOCK_SIZE) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>(
                        (csr_cols_C[j] - idx_baseC), __half2float(beta * csr_vals_C[j]), cur_idx_table, cur_val_table);
                }
            } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
                for (idxType j = row_start_C + idx; j < row_end_C; j += BLOCK_SIZE) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>((csr_cols_C[j] - idx_baseC),
                                                                             __bfloat162float(beta * csr_vals_C[j]),
                                                                             cur_idx_table, cur_val_table);
                }
            } else if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
                for (idxType j = row_start_C + idx; j < row_end_C; j += BLOCK_SIZE) {
                    valType ori_val = complex_mul(beta, csr_vals_C[j]);
                    smType sm_val = smType(__low2float(ori_val), __high2float(ori_val));
                    mcspInsertHashTablePairs<HASH_SIZE, 79, idxType, smType>((csr_cols_C[j] - idx_baseC), sm_val,
                                                                             cur_idx_table, cur_val_table);
                }
            } else {
                for (idxType j = row_start_C + idx; j < row_end_C; j += BLOCK_SIZE) {
                    mcspInsertHashTablePairs<HASH_SIZE, 79>((csr_cols_C[j] - idx_baseC), beta * csr_vals_C[j],
                                                            cur_idx_table, cur_val_table);
                }
            }
            __syncthreads();
        }

        // block-level radix-sort
        idxType tmp_idx[HASH_SIZE / BLOCK_SIZE];
        smType tmp_val[HASH_SIZE / BLOCK_SIZE];
        for (unsigned int i = 0; i < 8 * sizeof(idxType); i++) {
#pragma unroll
            for (unsigned int j = 0; j < HASH_SIZE / BLOCK_SIZE; j++) {
                tmp_idx[j] = cur_idx_table[idx + j * BLOCK_SIZE];
                tmp_val[j] = cur_val_table[idx + j * BLOCK_SIZE];

                short radix = short((tmp_idx[j] >> i) & 1);
                cur_radix_table[radix * HASH_SIZE + idx + j * BLOCK_SIZE] = 1;
                cur_radix_table[(radix ^ 1) * HASH_SIZE + idx + j * BLOCK_SIZE] = 0;
            }
            __syncthreads();
            // exclusive scan cur_radix_table
            unsigned int r_idx, l_idx;
            short tmp1, tmp2;
#pragma unroll
            // up-sweep
            for (unsigned int stride = 1; stride < HASH_SIZE * 2; stride <<= 1) {
                r_idx = (idx + 1) * 2 * stride - 1;
                l_idx = r_idx - stride;
                if (r_idx < HASH_SIZE * 2) {
                    tmp1 = cur_radix_table[l_idx];
                    cur_radix_table[r_idx] += tmp1;
                }
                for (unsigned int j = 1; j < HASH_SIZE / (BLOCK_SIZE * stride); j++) {
                    r_idx += BLOCK_SIZE * 2 * stride;
                    l_idx += BLOCK_SIZE * 2 * stride;
                    if (r_idx < HASH_SIZE * 2) {
                        tmp1 = cur_radix_table[l_idx];
                        cur_radix_table[r_idx] += tmp1;
                    }
                }
                __syncthreads();
            }

            if (idx == BLOCK_SIZE - 1) {
                cur_radix_table[2 * HASH_SIZE - 1] = 0;
            }
            __syncthreads();

#pragma unroll
            // down-sweep
            for (unsigned int stride = HASH_SIZE; stride != 0; stride >>= 1) {
                r_idx = (idx + 1) * 2 * stride - 1;
                l_idx = r_idx - stride;
                if (r_idx < HASH_SIZE * 2) {
                    tmp1 = cur_radix_table[r_idx];
                    tmp2 = cur_radix_table[l_idx];
                    cur_radix_table[r_idx] += tmp2;
                    cur_radix_table[l_idx] = tmp1;
                }
                for (unsigned int j = 1; j < HASH_SIZE / (BLOCK_SIZE * stride); j++) {
                    r_idx += BLOCK_SIZE * 2 * stride;
                    l_idx += BLOCK_SIZE * 2 * stride;
                    if (r_idx < HASH_SIZE * 2) {
                        tmp1 = cur_radix_table[r_idx];
                        tmp2 = cur_radix_table[l_idx];
                        cur_radix_table[r_idx] += tmp2;
                        cur_radix_table[l_idx] = tmp1;
                    }
                }
                __syncthreads();
            }

#pragma unroll
            for (unsigned int j = 0; j < HASH_SIZE / BLOCK_SIZE; j++) {
                short radix = short((tmp_idx[j] >> i) & 1);
                tmp1 = cur_radix_table[radix * HASH_SIZE + j * BLOCK_SIZE + idx];
                cur_idx_table[tmp1] = tmp_idx[j];
                cur_val_table[tmp1] = tmp_val[j];
            }
            __syncthreads();
        }
        // radix sort done

        // write final result to global memory
        for (unsigned int j = idx; j < HASH_SIZE; j += BLOCK_SIZE) {
            if (cur_idx_table[j] != (idxType)HASH_MAGIC_NULL_VAL) {
                csr_cols_D[row_start_D + j] = cur_idx_table[j];
                if constexpr (std::is_same_v<valType, __half>) {
                    csr_vals_D[row_start_D + j] = __float2half(cur_val_table[j]);
                } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
                    csr_vals_D[row_start_D + j] = __float2bfloat16(cur_val_table[j]);
                } else if constexpr (std::is_same_v<valType, __half2>) {
                    csr_vals_D[row_start_D + j] =
                        __half2(__float2half(cur_val_table[j].x), __float2half(cur_val_table[j].y));
                } else if constexpr (std::is_same_v<valType, mcsp_bfloat162>) {
                    csr_vals_D[row_start_D + j] =
                        mcsp_bfloat162(__float2bfloat16(cur_val_table[j].x), __float2bfloat16(cur_val_table[j].y));
                } else {
                    csr_vals_D[row_start_D + j] = cur_val_table[j];
                }
            }
        }
    }
}

// When shared-memory is not enough for hash-table, divide col_B as different chunks.
template <unsigned int BLOCK_SIZE, unsigned int CHUNK_SIZE, typename idxType, typename valType,
          typename smType = valType>
__global__ void mcspSpgemmCalcMultiRowAKernel(
    idxType group_size, idxType group_offset, idxType kb, valType alpha, const valType *csr_vals_A,
    const idxType *csr_rows_A, const idxType *csr_cols_A, const valType *csr_vals_B, const idxType *csr_rows_B,
    const idxType *csr_cols_B, valType beta, const valType *csr_vals_C, const idxType *csr_rows_C,
    const idxType *csr_cols_C, valType *csr_vals_D, const idxType *csr_rows_D, idxType *csr_cols_D,
    idxType *identity_buffer, idxType *multi_front_buffer, bool include_addition, mcsparseIndexBase_t idx_baseA,
    mcsparseIndexBase_t idx_baseB, mcsparseIndexBase_t idx_baseC, mcsparseIndexBase_t idx_baseD) {
    volatile __shared__ short table[CHUNK_SIZE];
    volatile __shared__ short table_bak[CHUNK_SIZE];
    __shared__ smType val_table[CHUNK_SIZE];
    volatile __shared__ idxType thread_nnz[BLOCK_SIZE];

    const idxType tid = blockDim.x * blockIdx.x + threadIdx.x;
    const idxType gid = blockIdx.x;
    const idxType lane = tid & (WARP_SIZE - 1);
    const idxType idx = threadIdx.x;
    const idxType wid = threadIdx.x / WARP_SIZE;

    idxType row_id_A;
    idxType chunk_begin = 0;
    if (gid < group_size) {
        row_id_A = identity_buffer[group_offset + gid];
        idxType row_start_A = csr_rows_A[row_id_A] - idx_baseA;
        idxType row_end_A = csr_rows_A[row_id_A + 1] - idx_baseA;
        idxType row_start_C;
        idxType row_end_C;
        if (include_addition) {
            row_start_C = csr_rows_C[row_id_A] - idx_baseC;
            row_end_C = csr_rows_C[row_id_A + 1] - idx_baseC;
        }
        bool chunk_start = true;

        idxType row_start_D = csr_rows_D[row_id_A] - idx_baseD;
        idxType row_nnz = 0;
        for (idxType chunk_offset = 0; chunk_offset < kb; chunk_offset += CHUNK_SIZE) {
            thread_nnz[idx] = 0;
            for (idxType k = idx; k < CHUNK_SIZE; k += BLOCK_SIZE) {
                table[k] = 0;
                val_table[k] = GetTypedValue<smType>(0);
            }
            __syncthreads();

            for (idxType j = row_start_A + wid; j < row_end_A; j += BLOCK_SIZE / WARP_SIZE) {
                idxType row_id_B = csr_cols_A[j] - idx_baseA;
                valType val_A = csr_vals_A[j];
                idxType row_start_B = (chunk_start) ? (csr_rows_B[row_id_B] - idx_baseB) : multi_front_buffer[j];
                idxType row_end_B = csr_rows_B[row_id_B + 1] - idx_baseB;

                idxType next_k = row_start_B + lane;
                for (idxType k = next_k; k < row_end_B; k += WARP_SIZE) {
                    idxType col_idx_B = csr_cols_B[k] - idx_baseB;
                    if (col_idx_B >= chunk_offset && col_idx_B < chunk_offset + CHUNK_SIZE) {
                        table[col_idx_B - chunk_offset] = 1;
                        valType ori_val = alpha * val_A * csr_vals_B[k];
                        smType sm_val = GetTypedValue<smType>(0);
                        if constexpr (std::is_same_v<valType, __half>) {
                            sm_val = __half2float(ori_val);
                        } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
                            sm_val = __bfloat162float(ori_val);
                        } else if constexpr (std::is_same_v<valType, __half2> ||
                                             std::is_same_v<valType, mcsp_bfloat162>) {
                            ori_val = complex_mul(alpha, complex_mul(val_A, csr_vals_B[k]));
                            sm_val = smType(__low2float(ori_val), __high2float(ori_val));
                        } else {
                            sm_val = ori_val;
                        }

                        if constexpr (std::is_same_v<smType, mcspComplexDouble> ||
                                      std::is_same_v<smType, mcspComplexFloat>) {
                            complexAtomicAddByPart_(&(val_table[col_idx_B - chunk_offset]), sm_val);
                        } else {
                            atomicAdd_(&(val_table[col_idx_B - chunk_offset]), sm_val);
                        }
                    } else if (col_idx_B >= chunk_offset + CHUNK_SIZE) {
                        next_k = k;
                        break;
                    }
                }

#ifndef __MACA__
                __syncwarp();
#endif

#pragma unroll
                for (int i = 1; i < WARP_SIZE; i <<= 1) {
#ifndef __MACA__
                    next_k = min(next_k, __shfl_xor_sync(UINT32_BIT_MASK, next_k, i, WARP_SIZE));
#else
                    next_k = min(next_k, __shfl_xor(next_k, i, WARP_SIZE));
#endif
                }
                if (lane == 0) {
                    multi_front_buffer[j] = next_k;
                }
            }
            __syncthreads();

            if (include_addition) {
                for (idxType j = row_start_C + idx; j < row_end_C; j += BLOCK_SIZE) {
                    idxType col_idx_C = csr_cols_C[j] - idx_baseC;
                    if (col_idx_C >= chunk_offset && col_idx_C < chunk_offset + CHUNK_SIZE) {
                        table[col_idx_C - chunk_offset] = 1;
                        valType ori_val = beta * csr_vals_C[j];
                        smType sm_val = GetTypedValue<smType>(0);
                        if constexpr (std::is_same_v<valType, __half>) {
                            sm_val = __half2float(ori_val);
                        } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
                            sm_val = __bfloat162float(ori_val);
                        } else if constexpr (std::is_same_v<valType, __half2> ||
                                             std::is_same_v<valType, mcsp_bfloat162>) {
                            ori_val = complex_mul(beta, csr_vals_C[j]);
                            sm_val = smType(__low2float(ori_val), __high2float(ori_val));
                        } else {
                            sm_val = ori_val;
                        }

                        if constexpr (std::is_same_v<smType, mcspComplexDouble> ||
                                      std::is_same_v<smType, mcspComplexFloat>) {
                            complexAtomicAddByPart_(&(val_table[col_idx_C - chunk_offset]), sm_val);
                        } else {
                            atomicAdd_(&(val_table[col_idx_C - chunk_offset]), sm_val);
                        }
                    }
                }
                __syncthreads();
            }

            for (idxType k = idx; k < CHUNK_SIZE; k += BLOCK_SIZE) {
                table_bak[k] = table[k];
            }
            __syncthreads();
            // block exclusive scan
            unsigned int r_idx, l_idx;
            short tmp1, tmp2;
#pragma unroll
            // up-sweep
            for (unsigned int stride = 1; stride < CHUNK_SIZE; stride <<= 1) {
                r_idx = (idx + 1) * 2 * stride - 1;
                l_idx = r_idx - stride;
                if (r_idx < CHUNK_SIZE) {
                    tmp1 = table[l_idx];
                    table[r_idx] += tmp1;
                }
                for (unsigned int j = 1; j < CHUNK_SIZE / (2 * BLOCK_SIZE * stride); j++) {
                    r_idx += BLOCK_SIZE * 2 * stride;
                    l_idx += BLOCK_SIZE * 2 * stride;
                    if (r_idx < CHUNK_SIZE) {
                        tmp1 = table[l_idx];
                        table[r_idx] += tmp1;
                    }
                }
                __syncthreads();
            }

            if (idx == BLOCK_SIZE - 1) {
                table[CHUNK_SIZE - 1] = 0;
            }
            __syncthreads();

#pragma unroll
            // down-sweep
            for (unsigned int stride = CHUNK_SIZE / 2; stride != 0; stride >>= 1) {
                r_idx = (idx + 1) * 2 * stride - 1;
                l_idx = r_idx - stride;
                if (r_idx < CHUNK_SIZE) {
                    tmp1 = table[r_idx];
                    tmp2 = table[l_idx];
                    table[r_idx] += tmp2;
                    table[l_idx] = tmp1;
                }
                for (unsigned int j = 1; j < CHUNK_SIZE / (2 * BLOCK_SIZE * stride); j++) {
                    r_idx += BLOCK_SIZE * 2 * stride;
                    l_idx += BLOCK_SIZE * 2 * stride;
                    if (r_idx < CHUNK_SIZE) {
                        tmp1 = table[r_idx];
                        tmp2 = table[l_idx];
                        table[r_idx] += tmp2;
                        table[l_idx] = tmp1;
                    }
                }
                __syncthreads();
            }
            // block exclusive scan done

            // write final result to global memory
            for (idxType k = idx; k < CHUNK_SIZE; k += BLOCK_SIZE) {
                if (table_bak[k] != 0) {
                    csr_cols_D[row_start_D + row_nnz + table[k]] = k + chunk_offset;
                    if constexpr (std::is_same_v<valType, __half>) {
                        csr_vals_D[row_start_D + row_nnz + table[k]] = __float2half(val_table[k]);
                    } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
                        csr_vals_D[row_start_D + row_nnz + table[k]] = __float2bfloat16(val_table[k]);
                    } else if constexpr (std::is_same_v<valType, __half2>) {
                        csr_vals_D[row_start_D + row_nnz + table[k]] =
                            __half2(__float2half(val_table[k].x), __float2half(val_table[k].y));
                    } else if constexpr (std::is_same_v<valType, mcsp_bfloat162>) {
                        csr_vals_D[row_start_D + row_nnz + table[k]] =
                            mcsp_bfloat162(__float2bfloat16(val_table[k].x), __float2bfloat16(val_table[k].y));
                    } else {
                        csr_vals_D[row_start_D + row_nnz + table[k]] = val_table[k];
                    }
                }
            }

            row_nnz += (table[CHUNK_SIZE - 1] + table_bak[CHUNK_SIZE - 1]);

            chunk_start = false;
            __syncthreads();
        }
    }
}

#endif
