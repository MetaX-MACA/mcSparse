#include <vector>

#include "common/mcsp_types.h"
#include "csr_spgemm_device.hpp"
#include "csr_spgemm_host.hpp"
#include "device_radix_sort.hpp"
#include "internal_interface/mcsp_internal_conversion.h"
#include "mcsp_config.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

template <typename idxType>
mcspStatus_t mcspSpgemmExpandSize(mcspHandle_t handle, mcStream_t stream, idxType m, idxType n,
                                  const idxType *csr_rows_A, const idxType *csr_cols_A, const idxType *csr_rows_B,
                                  const idxType *csr_rows_C, idxType *buffer, idxType *ibuffer, bool include_addition,
                                  mcsparseIndexBase_t idx_baseA) {
    int n_elem = 512;
    int g_elem = n_elem / WARP_SIZE;
    int n_block = (m + g_elem - 1) / g_elem;
    mcLaunchKernelGGL(mcspSpgemmExpandSizeKernel, dim3(n_block), dim3(n_elem), n_elem * sizeof(*csr_rows_B), stream, m,
                       n, csr_rows_A, csr_cols_A, csr_rows_B, csr_rows_C, buffer, include_addition, idx_baseA);

    mcspCreateIdentityPermutation(handle, m, ibuffer);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspSpgemmGroupOffset(mcspHandle_t handle, mcStream_t stream, idxType m, idxType *ibuffer,
                                   idxType *obuffer) {
    int n_elem = 512;
    int n_block = (m + 1 + n_elem - 1) / n_elem;
    mcLaunchKernelGGL(mcspSpgemmGroupOffsetKernel, dim3(n_block), dim3(n_elem), 0, stream, m, ibuffer, obuffer);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspSpgemmNnzSizeGroup(mcspHandle_t handle, mcStream_t stream, idxType m, const idxType *csr_rows_D,
                                    idxType *buffer, idxType *ibuffer) {
    int n_elem = 512;
    int n_block = (m + n_elem - 1) / n_elem;
    mcLaunchKernelGGL(mcspSpgemmNnzSizeGroupKernel, dim3(n_block), dim3(n_elem), 0, stream, m, csr_rows_D, buffer);

    mcspCreateIdentityPermutation(handle, m, ibuffer);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCsrgemmBuffersizeTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A,
                                           mcsparseOperation_t trans_B, idxType m, idxType n, idxType k,
                                           mcspMatDescr_t descr_A, idxType nnz_A, const idxType *csr_rows_A,
                                           const idxType *csr_cols_A, mcspMatDescr_t descr_B, idxType nnz_B,
                                           const idxType *csr_rows_B, const idxType *csr_cols_B, mcspMatDescr_t descr_C,
                                           idxType nnz_C, const idxType *csr_rows_C, const idxType *csr_cols_C,
                                           mcspMatInfo_t info_D, size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_C < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (descr_A == nullptr || descr_B == nullptr || descr_C == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE || trans_B != MCSPARSE_OPERATION_NON_TRANSPOSE || nnz_A == 0 ||
        nnz_B == 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_A->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_A->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT || descr_B->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr_B->fill_mode != MCSPARSE_FILL_MODE_FULL || descr_B->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT ||
        descr_C->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_C->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_C->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (m == 0 || n == 0 || k == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    *buffer_size = (5 * m + 9 + nnz_A) * sizeof(idxType);
    idxType temp_buffer_size;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::exclusive_scan(nullptr, temp_buffer_size, (idxType *)nullptr, (idxType *)nullptr, m + 1, (idxType *)nullptr,
                           stream);
    *buffer_size = *buffer_size + temp_buffer_size;
    mcprim::radix_sort_pairs_range8(nullptr, temp_buffer_size, (idxType *)nullptr, (idxType *)nullptr,
                                    (idxType *)nullptr, (idxType *)nullptr, m, stream);
    *buffer_size = *buffer_size + temp_buffer_size;
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCsrgemmNnzTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    idxType m, idxType n, idxType k, mcspMatDescr_t descr_A, idxType nnz_A,
                                    const idxType *csr_rows_A, const idxType *csr_cols_A, mcspMatDescr_t descr_B,
                                    idxType nnz_B, const idxType *csr_rows_B, const idxType *csr_cols_B,
                                    mcspMatDescr_t descr_C, idxType nnz_C, const idxType *csr_rows_C,
                                    const idxType *csr_cols_C, mcspMatDescr_t descr_D, idxType *csr_rows_D,
                                    idxType *nnz_D, mcspMatInfo_t info_D, void *buffer, bool include_addition) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_C < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (csr_rows_A == nullptr || csr_cols_A == nullptr || csr_rows_B == nullptr || csr_cols_B == nullptr ||
        descr_A == nullptr || descr_B == nullptr || descr_C == nullptr || descr_D == nullptr || info_D == nullptr ||
        buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (include_addition && (csr_rows_C == nullptr || csr_cols_C == nullptr)) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE || trans_B != MCSPARSE_OPERATION_NON_TRANSPOSE || nnz_A == 0 ||
        nnz_B == 0) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_A->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_A->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT || descr_B->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr_B->fill_mode != MCSPARSE_FILL_MODE_FULL || descr_B->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT ||
        descr_C->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_C->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_C->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT || descr_D->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr_D->fill_mode != MCSPARSE_FILL_MODE_FULL || descr_D->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (m == 0 || n == 0 || k == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    void *buffer_head = buffer;
    idxType *expand_size_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(expand_size_buffer + m);
    idxType *expand_group_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(expand_group_buffer + m);
    idxType *identity_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(identity_buffer + m);
    idxType *expand_group_buffer2 = (idxType *)buffer_head;
    buffer_head = (void *)(expand_group_buffer2 + m);
    idxType *identity_buffer2 = (idxType *)buffer_head;
    buffer_head = (void *)(identity_buffer2 + m);
    // group rows of D according to expand size
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcspSpgemmExpandSize(handle, stream, m, n, csr_rows_A, csr_cols_A, csr_rows_B, csr_rows_C, expand_size_buffer,
                         identity_buffer, include_addition, descr_A->base);

    // radix sort of expand size group to gather the same group together
    idxType prim_buffer_size;
    mcprim::radix_sort_pairs_range8(nullptr, prim_buffer_size, expand_group_buffer, expand_group_buffer2,
                                    identity_buffer, identity_buffer2, m, stream);
    mcprim::radix_sort_pairs_range8(buffer_head, prim_buffer_size, expand_group_buffer, expand_group_buffer2,
                                    identity_buffer, identity_buffer2, m, stream);
    MACA_ASSERT(mcStreamSynchronize(stream));

    idxType *group_offset_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(group_offset_buffer + 9);

    idxType *multi_front_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(multi_front_buffer + nnz_A);

    MACA_ASSERT(mcMemsetAsync(group_offset_buffer, 0, 9 * sizeof(*group_offset_buffer), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    // get group size and corresponding offset
    mcspSpgemmGroupOffset(handle, stream, m, expand_group_buffer2, group_offset_buffer);

    std::vector<idxType> h_group_offset;
    std::vector<idxType> h_group_size;
    h_group_offset.resize(9);
    h_group_size.resize(8);
    MACA_ASSERT(mcMemcpyAsync(h_group_offset.data(), group_offset_buffer, 9 * sizeof(*group_offset_buffer),
                              mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    for (idxType i = 0; i < 8; i++) {
        h_group_size[i] = h_group_offset[i + 1] - h_group_offset[i];
    }

    // 0 - 32
    if (h_group_size[0] > 0) {
        int n_elem = 512;
        int g_elem = n_elem / WARP_SIZE;
        int n_block = (h_group_size[0] + g_elem - 1) / g_elem;
#define CSRGEMM_HASH_SIZE 32
        mcLaunchKernelGGL((mcspSpgemmNnzWarpRowAKernel<CSRGEMM_HASH_SIZE>), dim3(n_block), dim3(n_elem),
                           g_elem * CSRGEMM_HASH_SIZE * sizeof(idxType), stream, h_group_size[0], h_group_offset[0],
                           csr_rows_A, csr_cols_A, csr_rows_B, csr_cols_B, csr_rows_C, csr_cols_C, csr_rows_D,
                           identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base);
#undef CSRGEMM_HASH_SIZE
    }

    // 33 - 64
    if (h_group_size[1] > 0) {
        int n_elem = 512;
        int g_elem = n_elem / WARP_SIZE;
        int n_block = (h_group_size[1] + g_elem - 1) / g_elem;
#define CSRGEMM_HASH_SIZE 64
        mcLaunchKernelGGL((mcspSpgemmNnzWarpRowAKernel<CSRGEMM_HASH_SIZE>), dim3(n_block), dim3(n_elem),
                           g_elem * CSRGEMM_HASH_SIZE * sizeof(idxType), stream, h_group_size[1], h_group_offset[1],
                           csr_rows_A, csr_cols_A, csr_rows_B, csr_cols_B, csr_rows_C, csr_cols_C, csr_rows_D,
                           identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base);
#undef CSRGEMM_HASH_SIZE
    }

    // 65 - 512
    if (h_group_size[2] > 0) {
#define CSRGEMM_BLOCK_SIZE 256
#define CSRGEMM_HASH_SIZE 512
        int n_block = h_group_size[2];
        mcLaunchKernelGGL((mcspSpgemmNnzBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>), dim3(n_block),
                           dim3(CSRGEMM_BLOCK_SIZE), CSRGEMM_HASH_SIZE * sizeof(idxType), stream, h_group_size[2],
                           h_group_offset[2], csr_rows_A, csr_cols_A, csr_rows_B, csr_cols_B, csr_rows_C, csr_cols_C,
                           csr_rows_D, identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base);
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // 513 - 1024
    if (h_group_size[3] > 0) {
#define CSRGEMM_BLOCK_SIZE 256
#define CSRGEMM_HASH_SIZE 1024
        int n_block = h_group_size[3];
        mcLaunchKernelGGL((mcspSpgemmNnzBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>), dim3(n_block),
                           dim3(CSRGEMM_BLOCK_SIZE), CSRGEMM_HASH_SIZE * sizeof(idxType), stream, h_group_size[3],
                           h_group_offset[3], csr_rows_A, csr_cols_A, csr_rows_B, csr_cols_B, csr_rows_C, csr_cols_C,
                           csr_rows_D, identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base);
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // 1025 - 2048
    if (h_group_size[4] > 0) {
#define CSRGEMM_BLOCK_SIZE 256
#define CSRGEMM_HASH_SIZE 2048
        int n_block = h_group_size[4];
        mcLaunchKernelGGL((mcspSpgemmNnzBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>), dim3(n_block),
                           dim3(CSRGEMM_BLOCK_SIZE), CSRGEMM_HASH_SIZE * sizeof(idxType), stream, h_group_size[4],
                           h_group_offset[4], csr_rows_A, csr_cols_A, csr_rows_B, csr_cols_B, csr_rows_C, csr_cols_C,
                           csr_rows_D, identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base);
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // 2049 - 4096
    if (h_group_size[5] > 0) {
#define CSRGEMM_BLOCK_SIZE 512
#define CSRGEMM_HASH_SIZE 4096
        int n_block = h_group_size[5];
        mcLaunchKernelGGL((mcspSpgemmNnzBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>), dim3(n_block),
                           dim3(CSRGEMM_BLOCK_SIZE), CSRGEMM_HASH_SIZE * sizeof(idxType), stream, h_group_size[5],
                           h_group_offset[5], csr_rows_A, csr_cols_A, csr_rows_B, csr_cols_B, csr_rows_C, csr_cols_C,
                           csr_rows_D, identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base);
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // 4097 - 8192
    if (h_group_size[6] > 0) {
#define CSRGEMM_BLOCK_SIZE 512
#define CSRGEMM_HASH_SIZE 8192
        int n_block = h_group_size[6];
        mcLaunchKernelGGL((mcspSpgemmNnzBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>), dim3(n_block),
                           dim3(CSRGEMM_BLOCK_SIZE), CSRGEMM_HASH_SIZE * sizeof(idxType), stream, h_group_size[6],
                           h_group_offset[6], csr_rows_A, csr_cols_A, csr_rows_B, csr_cols_B, csr_rows_C, csr_cols_C,
                           csr_rows_D, identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base);
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // > 8192
    if (h_group_size[7] > 0) {
#define CSRGEMM_BLOCK_SIZE 512
#define CSRGEMM_CHUNK_SIZE 2048
        int n_block = h_group_size[7];
        mcLaunchKernelGGL((mcspSpgemmNnzMultiRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_CHUNK_SIZE>), dim3(n_block),
                           dim3(CSRGEMM_BLOCK_SIZE), 0, stream, h_group_size[7], h_group_offset[7], k, csr_rows_A,
                           csr_cols_A, csr_rows_B, csr_cols_B, csr_rows_C, csr_cols_C, csr_rows_D, identity_buffer2,
                           multi_front_buffer, include_addition, descr_A->base, descr_B->base, descr_C->base);
#undef CSRGEMM_CHUNK_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }
    MACA_ASSERT(mcStreamSynchronize(stream));

    // exclusive scan row_nnz to get csr_rows_D
    mcprim::exclusive_scan(nullptr, prim_buffer_size, csr_rows_D, csr_rows_D, m + 1, (idxType *)nullptr, stream);
    idxType *exclusive_scan_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(exclusive_scan_buffer + prim_buffer_size / sizeof(idxType));

    mcprim::exclusive_scan(exclusive_scan_buffer, prim_buffer_size, csr_rows_D, csr_rows_D, m + 1, (idxType *)nullptr,
                           stream);
    // the last value of csr_rows_D is nnz_D
    MACA_ASSERT(mcMemcpyAsync(nnz_D, csr_rows_D + m, sizeof(*nnz_D), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    *nnz_D = std::max((idxType)1, *nnz_D);
    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsrgemmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                 idxType m, idxType n, idxType k, const valType *alpha, mcspMatDescr_t descr_A,
                                 idxType nnz_A, const valType *csr_vals_A, const idxType *csr_rows_A,
                                 const idxType *csr_cols_A, mcspMatDescr_t descr_B, idxType nnz_B,
                                 const valType *csr_vals_B, const idxType *csr_rows_B, const idxType *csr_cols_B,
                                 const valType *beta, mcspMatDescr_t descr_C, idxType nnz_C, const valType *csr_vals_C,
                                 const idxType *csr_rows_C, const idxType *csr_cols_C, mcspMatDescr_t descr_D,
                                 valType *csr_vals_D, const idxType *csr_rows_D, idxType *csr_cols_D,
                                 mcspMatInfo_t info_D, void *buffer, bool include_addition) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || k < 0 || nnz_A < 0 || nnz_B < 0 || nnz_C < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if ((alpha == nullptr && beta == nullptr) || csr_rows_D == nullptr || csr_cols_D == nullptr ||
        csr_vals_D == nullptr || descr_A == nullptr || descr_B == nullptr || descr_C == nullptr || descr_D == nullptr ||
        info_D == nullptr || buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (trans_A != MCSPARSE_OPERATION_NON_TRANSPOSE || trans_B != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_A->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_A->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT || descr_B->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr_B->fill_mode != MCSPARSE_FILL_MODE_FULL || descr_B->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT ||
        descr_C->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr_C->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr_C->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT || descr_D->type != MCSPARSE_MATRIX_TYPE_GENERAL ||
        descr_D->fill_mode != MCSPARSE_FILL_MODE_FULL || descr_D->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (m == 0 || n == 0 || k == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    mcStream_t stream = mcspGetStreamInternal(handle);
    if (alpha == nullptr || nnz_A == 0 || nnz_B == 0) {
        if (csr_rows_C == nullptr || csr_cols_C == nullptr || csr_vals_C == nullptr) {
            return MCSP_STATUS_INVALID_POINTER;
        }
        MACA_ASSERT(
            mcMemcpyAsync(csr_cols_D, csr_cols_C, nnz_C * sizeof(*csr_cols_D), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
#define SPGEMMAXPBY_BLOCK_SIZE 512
        int n_block = (nnz_C + SPGEMMAXPBY_BLOCK_SIZE - 1) / SPGEMMAXPBY_BLOCK_SIZE;
        if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
            denseAxpbyLowPrecisionComplex<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, nnz_C, *beta, csr_vals_C,
                                                                  GetTypedValue<valType>(0), csr_vals_D);
        } else {
            denseAxpby<SPGEMMAXPBY_BLOCK_SIZE>(stream, n_block, nnz_C, *beta, csr_vals_C, GetTypedValue<valType>(0),
                                               csr_vals_D);
        }
#undef SPGEMMAXPBY_BLOCK_SIZE
        return MCSP_STATUS_SUCCESS;
    }

    valType h_alpha = getScalarToHost(alpha, handle->ptr_mode);
    valType h_beta;
    if (beta == nullptr) {
        include_addition = false;
        h_beta = GetTypedValue<valType>(0);
    } else {
        h_beta = getScalarToHost(beta, handle->ptr_mode);
    }

    if (csr_rows_A == nullptr || csr_cols_A == nullptr || csr_vals_A == nullptr || csr_rows_B == nullptr ||
        csr_cols_B == nullptr || csr_vals_B == nullptr ||
        (include_addition && (csr_rows_C == nullptr || csr_cols_C == nullptr || csr_vals_C == nullptr))) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    void *buffer_head = buffer;
    idxType *expand_size_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(expand_size_buffer + m);
    idxType *expand_group_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(expand_group_buffer + m);
    idxType *identity_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(identity_buffer + m);
    idxType *expand_group_buffer2 = (idxType *)buffer_head;
    buffer_head = (void *)(expand_group_buffer2 + m);
    idxType *identity_buffer2 = (idxType *)buffer_head;
    buffer_head = (void *)(identity_buffer2 + m);

    // group rows of D according to row_nnz
    mcspSpgemmNnzSizeGroup(handle, stream, m, csr_rows_D, expand_group_buffer, identity_buffer);

    // radix sort of row_nnz group to gather the same group together
    idxType prim_buffer_size;
    mcprim::radix_sort_pairs_range8(nullptr, prim_buffer_size, expand_group_buffer, expand_group_buffer2,
                                    identity_buffer, identity_buffer2, m, stream);
    mcprim::radix_sort_pairs_range8(buffer_head, prim_buffer_size, expand_group_buffer, expand_group_buffer2,
                                    identity_buffer, identity_buffer2, m, stream);
    MACA_ASSERT(mcStreamSynchronize(stream));

    idxType *group_offset_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(group_offset_buffer + 9);
    idxType *multi_front_buffer = (idxType *)buffer_head;
    buffer_head = (void *)(multi_front_buffer + nnz_A);

    // get group size and corresponding offset
    MACA_ASSERT(mcMemsetAsync(group_offset_buffer, 0, 9 * sizeof(*group_offset_buffer), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    mcspSpgemmGroupOffset(handle, stream, m, expand_group_buffer2, group_offset_buffer);

    std::vector<idxType> h_group_offset;
    std::vector<idxType> h_group_size;
    h_group_offset.resize(9);
    h_group_size.resize(8);
    MACA_ASSERT(mcMemcpyAsync(h_group_offset.data(), group_offset_buffer, 9 * sizeof(*group_offset_buffer),
                              mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    for (idxType i = 0; i < 8; i++) {
        h_group_size[i] = h_group_offset[i + 1] - h_group_offset[i];
    }

    idxType nnz_D = 0;
    if (descr_D->base == MCSPARSE_INDEX_BASE_ONE) {
        std::vector<idxType> tmp_D_rows(m + 1, idxType(0));
        MACA_ASSERT(
            mcMemcpyAsync(tmp_D_rows.data(), csr_rows_D, (m + 1) * sizeof(idxType), mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        nnz_D = tmp_D_rows[m]++;
        const int block_size = 512;
        const int n_blocks = (m + block_size) / block_size;
        mcLaunchKernelGGL((selfIncreaseInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream, m + 1,
                           const_cast<idxType *>(csr_rows_D));
    }

    // 0 - 16
    if (h_group_size[0] > 0) {
#define CSRGEMM_BLOCK_SIZE 128
        int n_elem = CSRGEMM_BLOCK_SIZE;
        int g_elem = n_elem / WARP_SIZE;
        int n_block = (h_group_size[0] + g_elem - 1) / g_elem;
#define CSRGEMM_HASH_SIZE 16
        if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcWarpRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType,
                                              mcspComplexFloat>),
                dim3(n_block), dim3(n_elem),
                g_elem * CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(mcspComplexFloat) + 2 * sizeof(short)), stream,
                h_group_size[0], h_group_offset[0], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B,
                csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        } else if constexpr (std::is_same_v<valType, __half> || std::is_same_v<valType, mcsp_bfloat16>) {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcWarpRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType, float>),
                dim3(n_block), dim3(n_elem),
                g_elem * CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(float) + 2 * sizeof(short)), stream,
                h_group_size[0], h_group_offset[0], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B,
                csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        } else {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcWarpRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType>), dim3(n_block),
                dim3(n_elem), g_elem * CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(valType) + 2 * sizeof(short)),
                stream, h_group_size[0], h_group_offset[0], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B,
                csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        }
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }
    // 17 - 32
    if (h_group_size[1] > 0) {
#define CSRGEMM_BLOCK_SIZE 256
        int n_elem = CSRGEMM_BLOCK_SIZE;
        int g_elem = n_elem / WARP_SIZE;
        int n_block = (h_group_size[1] + g_elem - 1) / g_elem;
#define CSRGEMM_HASH_SIZE 32
        if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcWarpRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType,
                                              mcspComplexFloat>),
                dim3(n_block), dim3(n_elem),
                g_elem * CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(mcspComplexFloat) + 2 * sizeof(short)), stream,
                h_group_size[1], h_group_offset[1], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B,
                csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        } else if constexpr (std::is_same_v<valType, __half> || std::is_same_v<valType, mcsp_bfloat16>) {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcWarpRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType, float>),
                dim3(n_block), dim3(n_elem),
                g_elem * CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(float) + 2 * sizeof(short)), stream,
                h_group_size[1], h_group_offset[1], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B,
                csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        } else {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcWarpRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType>), dim3(n_block),
                dim3(n_elem), g_elem * CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(valType) + 2 * sizeof(short)),
                stream, h_group_size[1], h_group_offset[1], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B,
                csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        }
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // 33 - 256
    if (h_group_size[2] > 0) {
#define CSRGEMM_BLOCK_SIZE 128
#define CSRGEMM_HASH_SIZE 256
        int n_block = h_group_size[2];
        if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
            mcLaunchKernelGGL((mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType,
                                                              mcspComplexFloat>),
                               dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                               CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(mcspComplexFloat) + 2 * sizeof(short)),
                               stream, h_group_size[2], h_group_offset[2], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A,
                               csr_vals_B, csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C,
                               csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2, include_addition, descr_A->base,
                               descr_B->base, descr_C->base, descr_D->base);
        } else if constexpr (std::is_same_v<valType, __half> || std::is_same_v<valType, mcsp_bfloat16>) {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType, float>),
                dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(float) + 2 * sizeof(short)), stream, h_group_size[2],
                h_group_offset[2], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B,
                h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2,
                include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        } else {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>), dim3(n_block),
                dim3(CSRGEMM_BLOCK_SIZE), CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(valType) + 2 * sizeof(short)),
                stream, h_group_size[2], h_group_offset[2], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B,
                csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        }
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // 257 - 512
    if (h_group_size[3] > 0) {
#define CSRGEMM_BLOCK_SIZE 256
#define CSRGEMM_HASH_SIZE 512
        int n_block = h_group_size[3];
        if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
            mcLaunchKernelGGL((mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType,
                                                              mcspComplexFloat>),
                               dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                               CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(mcspComplexFloat) + 2 * sizeof(short)),
                               stream, h_group_size[3], h_group_offset[3], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A,
                               csr_vals_B, csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C,
                               csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2, include_addition, descr_A->base,
                               descr_B->base, descr_C->base, descr_D->base);
        } else if constexpr (std::is_same_v<valType, __half> || std::is_same_v<valType, mcsp_bfloat16>) {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType, float>),
                dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(float) + 2 * sizeof(short)), stream, h_group_size[3],
                h_group_offset[3], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B,
                h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2,
                include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        } else {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>), dim3(n_block),
                dim3(CSRGEMM_BLOCK_SIZE), CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(valType) + 2 * sizeof(short)),
                stream, h_group_size[3], h_group_offset[3], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B,
                csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        }
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // 513 - 1024
    if (h_group_size[4] > 0) {
#define CSRGEMM_BLOCK_SIZE 256
#define CSRGEMM_HASH_SIZE 1024
        int n_block = h_group_size[4];
        if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
            mcLaunchKernelGGL((mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType,
                                                              mcspComplexFloat>),
                               dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                               CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(mcspComplexFloat) + 2 * sizeof(short)),
                               stream, h_group_size[4], h_group_offset[4], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A,
                               csr_vals_B, csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C,
                               csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2, include_addition, descr_A->base,
                               descr_B->base, descr_C->base, descr_D->base);
        } else if constexpr (std::is_same_v<valType, __half> || std::is_same_v<valType, mcsp_bfloat16>) {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType, float>),
                dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(float) + 2 * sizeof(short)), stream, h_group_size[4],
                h_group_offset[4], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B,
                h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2,
                include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        } else {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>), dim3(n_block),
                dim3(CSRGEMM_BLOCK_SIZE), CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(valType) + 2 * sizeof(short)),
                stream, h_group_size[4], h_group_offset[4], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B,
                csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        }
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // 1025 - 2048
    if (h_group_size[5] > 0) {
#define CSRGEMM_BLOCK_SIZE 512
#define CSRGEMM_HASH_SIZE 2048
        int n_block = h_group_size[5];
        if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
            mcLaunchKernelGGL((mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType,
                                                              mcspComplexFloat>),
                               dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                               CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(mcspComplexFloat) + 2 * sizeof(short)),
                               stream, h_group_size[5], h_group_offset[5], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A,
                               csr_vals_B, csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C,
                               csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2, include_addition, descr_A->base,
                               descr_B->base, descr_C->base, descr_D->base);
        } else if constexpr (std::is_same_v<valType, __half> || std::is_same_v<valType, mcsp_bfloat16>) {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType, float>),
                dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(float) + 2 * sizeof(short)), stream, h_group_size[5],
                h_group_offset[5], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B,
                h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2,
                include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        } else {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>), dim3(n_block),
                dim3(CSRGEMM_BLOCK_SIZE), CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(valType) + 2 * sizeof(short)),
                stream, h_group_size[5], h_group_offset[5], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B,
                csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                identity_buffer2, include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        }
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }

    // 2049 - 4096
    if (h_group_size[6] > 0) {
        if (std::is_same<valType, mcspComplexDouble>::value) {
#define CSRGEMM_BLOCK_SIZE 512
#define CSRGEMM_CHUNK_SIZE 2048
            int n_block = h_group_size[6];
            mcLaunchKernelGGL((mcspSpgemmCalcMultiRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_CHUNK_SIZE>), dim3(n_block),
                               dim3(CSRGEMM_BLOCK_SIZE), 0, stream, h_group_size[6], h_group_offset[6], k, h_alpha,
                               csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B, h_beta,
                               csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2,
                               multi_front_buffer, include_addition, descr_A->base, descr_B->base, descr_C->base,
                               descr_D->base);
#undef CSRGEMM_CHUNK_SIZE
#undef CSRGEMM_BLOCK_SIZE
        } else {
#define CSRGEMM_BLOCK_SIZE 1024
#define CSRGEMM_HASH_SIZE 4096
#define CSRGEMM_CHUNK_SIZE 2048
            int n_block = h_group_size[6];
#if defined(__MACA__)
            if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
                mcLaunchKernelGGL((mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType,
                                                                  valType, mcspComplexFloat>),
                                   dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                                   CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(mcspComplexFloat) + 2 * sizeof(short)),
                                   stream, h_group_size[6], h_group_offset[6], h_alpha, csr_vals_A, csr_rows_A,
                                   csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C,
                                   csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2, include_addition,
                                   descr_A->base, descr_B->base, descr_C->base, descr_D->base);
            } else if constexpr (std::is_same_v<valType, __half> || std::is_same_v<valType, mcsp_bfloat16>) {
                mcLaunchKernelGGL(
                    (mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE, idxType, valType, float>),
                    dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                    CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(float) + 2 * sizeof(short)), stream, h_group_size[6],
                    h_group_offset[6], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B,
                    h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2,
                    include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
            } else {
                mcLaunchKernelGGL((mcspSpgemmCalcBlockRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_HASH_SIZE>),
                                   dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE),
                                   CSRGEMM_HASH_SIZE * (sizeof(idxType) + sizeof(valType) + 2 * sizeof(short)), stream,
                                   h_group_size[6], h_group_offset[6], h_alpha, csr_vals_A, csr_rows_A, csr_cols_A,
                                   csr_vals_B, csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C, csr_cols_C,
                                   csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2, include_addition,
                                   descr_A->base, descr_B->base, descr_C->base, descr_D->base);
            }
#else
            // Use MultiRow kernel instead
            mcLaunchKernelGGL((mcspSpgemmCalcMultiRowAKernel<512, CSRGEMM_CHUNK_SIZE>), dim3(n_block), dim3(512), 0,
                               stream, h_group_size[6], h_group_offset[6], k, h_alpha, csr_vals_A, csr_rows_A,
                               csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C,
                               csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2, multi_front_buffer,
                               include_addition, descr_A->base, descr_B->base, descr_C->base, descr_D->base);
#endif

#undef CSRGEMM_CHUNK_SIZE
#undef CSRGEMM_HASH_SIZE
#undef CSRGEMM_BLOCK_SIZE
        }
    }

    // > 4096
    if (h_group_size[7] > 0) {
#define CSRGEMM_BLOCK_SIZE 512
#define CSRGEMM_CHUNK_SIZE 2048
        int n_block = h_group_size[7];
        if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
            mcLaunchKernelGGL((mcspSpgemmCalcMultiRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_CHUNK_SIZE, idxType, valType,
                                                              mcspComplexFloat>),
                               dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE), 0, stream, h_group_size[7], h_group_offset[7],
                               k, h_alpha, csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B,
                               h_beta, csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D,
                               identity_buffer2, multi_front_buffer, include_addition, descr_A->base, descr_B->base,
                               descr_C->base, descr_D->base);
        } else if constexpr (std::is_same_v<valType, __half> || std::is_same_v<valType, mcsp_bfloat16>) {
            mcLaunchKernelGGL(
                (mcspSpgemmCalcMultiRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_CHUNK_SIZE, idxType, valType, float>),
                dim3(n_block), dim3(CSRGEMM_BLOCK_SIZE), 0, stream, h_group_size[7], h_group_offset[7], k, h_alpha,
                csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B, h_beta, csr_vals_C, csr_rows_C,
                csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2, multi_front_buffer, include_addition,
                descr_A->base, descr_B->base, descr_C->base, descr_D->base);
        } else {
            mcLaunchKernelGGL((mcspSpgemmCalcMultiRowAKernel<CSRGEMM_BLOCK_SIZE, CSRGEMM_CHUNK_SIZE>), dim3(n_block),
                               dim3(CSRGEMM_BLOCK_SIZE), 0, stream, h_group_size[7], h_group_offset[7], k, h_alpha,
                               csr_vals_A, csr_rows_A, csr_cols_A, csr_vals_B, csr_rows_B, csr_cols_B, h_beta,
                               csr_vals_C, csr_rows_C, csr_cols_C, csr_vals_D, csr_rows_D, csr_cols_D, identity_buffer2,
                               multi_front_buffer, include_addition, descr_A->base, descr_B->base, descr_C->base,
                               descr_D->base);
        }
#undef CSRGEMM_CHUNK_SIZE
#undef CSRGEMM_BLOCK_SIZE
    }
    MACA_ASSERT(mcStreamSynchronize(stream));

    if (descr_D->base == MCSPARSE_INDEX_BASE_ONE) {
        const int block_size = 512;
        const int n_blocks = (nnz_D + block_size - 1) / block_size;
        mcLaunchKernelGGL((selfIncreaseInplaceKernel), dim3(n_blocks), dim3(block_size), 0, stream, nnz_D, csr_cols_D);
    }

    return MCSP_STATUS_SUCCESS;
}

#ifdef __MACA__
template mcspStatus_t mcspCsrgemmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                          mcspInt m, mcspInt n, mcspInt k, const __half *alpha, mcspMatDescr_t descr_A,
                                          mcspInt nnz_A, const __half *csr_vals_A, const mcspInt *csr_rows_A,
                                          const mcspInt *csr_cols_A, mcspMatDescr_t descr_B, mcspInt nnz_B,
                                          const __half *csr_vals_B, const mcspInt *csr_rows_B,
                                          const mcspInt *csr_cols_B, const __half *beta, mcspMatDescr_t descr_C,
                                          mcspInt nnz_C, const __half *csr_vals_C, const mcspInt *csr_rows_C,
                                          const mcspInt *csr_cols_C, mcspMatDescr_t descr_D, __half *csr_vals_D,
                                          const mcspInt *csr_rows_D, mcspInt *csr_cols_D, mcspMatInfo_t info_D,
                                          void *buffer, bool include_addition);

template mcspStatus_t mcspCsrgemmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                          mcspInt m, mcspInt n, mcspInt k, const __half2 *alpha, mcspMatDescr_t descr_A,
                                          mcspInt nnz_A, const __half2 *csr_vals_A, const mcspInt *csr_rows_A,
                                          const mcspInt *csr_cols_A, mcspMatDescr_t descr_B, mcspInt nnz_B,
                                          const __half2 *csr_vals_B, const mcspInt *csr_rows_B,
                                          const mcspInt *csr_cols_B, const __half2 *beta, mcspMatDescr_t descr_C,
                                          mcspInt nnz_C, const __half2 *csr_vals_C, const mcspInt *csr_rows_C,
                                          const mcspInt *csr_cols_C, mcspMatDescr_t descr_D, __half2 *csr_vals_D,
                                          const mcspInt *csr_rows_D, mcspInt *csr_cols_D, mcspMatInfo_t info_D,
                                          void *buffer, bool include_addition);

template mcspStatus_t mcspCsrgemmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                          mcspInt m, mcspInt n, mcspInt k, const mcsp_bfloat16 *alpha,
                                          mcspMatDescr_t descr_A, mcspInt nnz_A, const mcsp_bfloat16 *csr_vals_A,
                                          const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, mcspMatDescr_t descr_B,
                                          mcspInt nnz_B, const mcsp_bfloat16 *csr_vals_B, const mcspInt *csr_rows_B,
                                          const mcspInt *csr_cols_B, const mcsp_bfloat16 *beta, mcspMatDescr_t descr_C,
                                          mcspInt nnz_C, const mcsp_bfloat16 *csr_vals_C, const mcspInt *csr_rows_C,
                                          const mcspInt *csr_cols_C, mcspMatDescr_t descr_D, mcsp_bfloat16 *csr_vals_D,
                                          const mcspInt *csr_rows_D, mcspInt *csr_cols_D, mcspMatInfo_t info_D,
                                          void *buffer, bool include_addition);

template mcspStatus_t mcspCsrgemmTemplate(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                          mcspInt m, mcspInt n, mcspInt k, const mcsp_bfloat162 *alpha,
                                          mcspMatDescr_t descr_A, mcspInt nnz_A, const mcsp_bfloat162 *csr_vals_A,
                                          const mcspInt *csr_rows_A, const mcspInt *csr_cols_A, mcspMatDescr_t descr_B,
                                          mcspInt nnz_B, const mcsp_bfloat162 *csr_vals_B, const mcspInt *csr_rows_B,
                                          const mcspInt *csr_cols_B, const mcsp_bfloat162 *beta, mcspMatDescr_t descr_C,
                                          mcspInt nnz_C, const mcsp_bfloat162 *csr_vals_C, const mcspInt *csr_rows_C,
                                          const mcspInt *csr_cols_C, mcspMatDescr_t descr_D, mcsp_bfloat162 *csr_vals_D,
                                          const mcspInt *csr_rows_D, mcspInt *csr_cols_D, mcspMatInfo_t info_D,
                                          void *buffer, bool include_addition);
#endif

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspScsrgemmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt n, mcspInt k, const float *alpha, mcspMatDescr_t descr_A,
                                    mcspInt nnz_A, const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                                    mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspInt *csr_rows_B,
                                    const mcspInt *csr_cols_B, const float *beta, mcspMatDescr_t descr_C, mcspInt nnz_C,
                                    const mcspInt *csr_rows_C, const mcspInt *csr_cols_C, mcspMatInfo_t info_D,
                                    size_t *buffer_size) {
    return mcspCsrgemmBuffersizeTemplate(handle, trans_A, trans_B, m, n, k, descr_A, nnz_A, csr_rows_A, csr_cols_A,
                                         descr_B, nnz_B, csr_rows_B, csr_cols_B, descr_C, nnz_C, csr_rows_C, csr_cols_C,
                                         info_D, buffer_size);
}

mcspStatus_t mcspDcsrgemmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt n, mcspInt k, const double *alpha, mcspMatDescr_t descr_A,
                                    mcspInt nnz_A, const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                                    mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspInt *csr_rows_B,
                                    const mcspInt *csr_cols_B, const double *beta, mcspMatDescr_t descr_C,
                                    mcspInt nnz_C, const mcspInt *csr_rows_C, const mcspInt *csr_cols_C,
                                    mcspMatInfo_t info_D, size_t *buffer_size) {
    return mcspCsrgemmBuffersizeTemplate(handle, trans_A, trans_B, m, n, k, descr_A, nnz_A, csr_rows_A, csr_cols_A,
                                         descr_B, nnz_B, csr_rows_B, csr_cols_B, descr_C, nnz_C, csr_rows_C, csr_cols_C,
                                         info_D, buffer_size);
}

mcspStatus_t mcspCcsrgemmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt n, mcspInt k, const mcspComplexFloat *alpha,
                                    mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspInt *csr_rows_A,
                                    const mcspInt *csr_cols_A, mcspMatDescr_t descr_B, mcspInt nnz_B,
                                    const mcspInt *csr_rows_B, const mcspInt *csr_cols_B, const mcspComplexFloat *beta,
                                    mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspInt *csr_rows_C,
                                    const mcspInt *csr_cols_C, mcspMatInfo_t info_D, size_t *buffer_size) {
    return mcspCsrgemmBuffersizeTemplate(handle, trans_A, trans_B, m, n, k, descr_A, nnz_A, csr_rows_A, csr_cols_A,
                                         descr_B, nnz_B, csr_rows_B, csr_cols_B, descr_C, nnz_C, csr_rows_C, csr_cols_C,
                                         info_D, buffer_size);
}

mcspStatus_t mcspZcsrgemmBuffersize(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B,
                                    mcspInt m, mcspInt n, mcspInt k, const mcspComplexDouble *alpha,
                                    mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspInt *csr_rows_A,
                                    const mcspInt *csr_cols_A, mcspMatDescr_t descr_B, mcspInt nnz_B,
                                    const mcspInt *csr_rows_B, const mcspInt *csr_cols_B, const mcspComplexDouble *beta,
                                    mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspInt *csr_rows_C,
                                    const mcspInt *csr_cols_C, mcspMatInfo_t info_D, size_t *buffer_size) {
    return mcspCsrgemmBuffersizeTemplate(handle, trans_A, trans_B, m, n, k, descr_A, nnz_A, csr_rows_A, csr_cols_A,
                                         descr_B, nnz_B, csr_rows_B, csr_cols_B, descr_C, nnz_C, csr_rows_C, csr_cols_C,
                                         info_D, buffer_size);
}

mcspStatus_t mcspCsrgemmNnz(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                            mcspInt n, mcspInt k, mcspMatDescr_t descr_A, mcspInt nnz_A, const mcspInt *csr_rows_A,
                            const mcspInt *csr_cols_A, mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspInt *csr_rows_B,
                            const mcspInt *csr_cols_B, mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspInt *csr_rows_C,
                            const mcspInt *csr_cols_C, mcspMatDescr_t descr_D, mcspInt *csr_rows_D, mcspInt *nnz_D,
                            mcspMatInfo_t info_D, void *buffer) {
    return mcspCsrgemmNnzTemplate(handle, trans_A, trans_B, m, n, k, descr_A, nnz_A, csr_rows_A, csr_cols_A, descr_B,
                                  nnz_B, csr_rows_B, csr_cols_B, descr_C, nnz_C, csr_rows_C, csr_cols_C, descr_D,
                                  csr_rows_D, nnz_D, info_D, buffer);
}

mcspStatus_t mcspScsrgemm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, const float *alpha, mcspMatDescr_t descr_A, mcspInt nnz_A,
                          const float *csr_vals_A, const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                          mcspMatDescr_t descr_B, mcspInt nnz_B, const float *csr_vals_B, const mcspInt *csr_rows_B,
                          const mcspInt *csr_cols_B, const float *beta, mcspMatDescr_t descr_C, mcspInt nnz_C,
                          const float *csr_vals_C, const mcspInt *csr_rows_C, const mcspInt *csr_cols_C,
                          mcspMatDescr_t descr_D, float *csr_vals_D, const mcspInt *csr_rows_D, mcspInt *csr_cols_D,
                          mcspMatInfo_t info_D, void *buffer) {
    if (alpha == nullptr || beta == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    return mcspCsrgemmTemplate(handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_vals_A, csr_rows_A,
                               csr_cols_A, descr_B, nnz_B, csr_vals_B, csr_rows_B, csr_cols_B, beta, descr_C, nnz_C,
                               csr_vals_C, csr_rows_C, csr_cols_C, descr_D, csr_vals_D, csr_rows_D, csr_cols_D, info_D,
                               buffer);
}

mcspStatus_t mcspDcsrgemm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, const double *alpha, mcspMatDescr_t descr_A, mcspInt nnz_A,
                          const double *csr_vals_A, const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                          mcspMatDescr_t descr_B, mcspInt nnz_B, const double *csr_vals_B, const mcspInt *csr_rows_B,
                          const mcspInt *csr_cols_B, const double *beta, mcspMatDescr_t descr_C, mcspInt nnz_C,
                          const double *csr_vals_C, const mcspInt *csr_rows_C, const mcspInt *csr_cols_C,
                          mcspMatDescr_t descr_D, double *csr_vals_D, const mcspInt *csr_rows_D, mcspInt *csr_cols_D,
                          mcspMatInfo_t info_D, void *buffer) {
    if (alpha == nullptr || beta == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    return mcspCsrgemmTemplate(handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_vals_A, csr_rows_A,
                               csr_cols_A, descr_B, nnz_B, csr_vals_B, csr_rows_B, csr_cols_B, beta, descr_C, nnz_C,
                               csr_vals_C, csr_rows_C, csr_cols_C, descr_D, csr_vals_D, csr_rows_D, csr_cols_D, info_D,
                               buffer);
}

mcspStatus_t mcspCcsrgemm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, const mcspComplexFloat *alpha, mcspMatDescr_t descr_A, mcspInt nnz_A,
                          const mcspComplexFloat *csr_vals_A, const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                          mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspComplexFloat *csr_vals_B,
                          const mcspInt *csr_rows_B, const mcspInt *csr_cols_B, const mcspComplexFloat *beta,
                          mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspComplexFloat *csr_vals_C,
                          const mcspInt *csr_rows_C, const mcspInt *csr_cols_C, mcspMatDescr_t descr_D,
                          mcspComplexFloat *csr_vals_D, const mcspInt *csr_rows_D, mcspInt *csr_cols_D,
                          mcspMatInfo_t info_D, void *buffer) {
    if (alpha == nullptr || beta == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    return mcspCsrgemmTemplate(handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_vals_A, csr_rows_A,
                               csr_cols_A, descr_B, nnz_B, csr_vals_B, csr_rows_B, csr_cols_B, beta, descr_C, nnz_C,
                               csr_vals_C, csr_rows_C, csr_cols_C, descr_D, csr_vals_D, csr_rows_D, csr_cols_D, info_D,
                               buffer);
}

mcspStatus_t mcspZcsrgemm(mcspHandle_t handle, mcsparseOperation_t trans_A, mcsparseOperation_t trans_B, mcspInt m,
                          mcspInt n, mcspInt k, const mcspComplexDouble *alpha, mcspMatDescr_t descr_A, mcspInt nnz_A,
                          const mcspComplexDouble *csr_vals_A, const mcspInt *csr_rows_A, const mcspInt *csr_cols_A,
                          mcspMatDescr_t descr_B, mcspInt nnz_B, const mcspComplexDouble *csr_vals_B,
                          const mcspInt *csr_rows_B, const mcspInt *csr_cols_B, const mcspComplexDouble *beta,
                          mcspMatDescr_t descr_C, mcspInt nnz_C, const mcspComplexDouble *csr_vals_C,
                          const mcspInt *csr_rows_C, const mcspInt *csr_cols_C, mcspMatDescr_t descr_D,
                          mcspComplexDouble *csr_vals_D, const mcspInt *csr_rows_D, mcspInt *csr_cols_D,
                          mcspMatInfo_t info_D, void *buffer) {
    if (alpha == nullptr || beta == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }
    return mcspCsrgemmTemplate(handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_vals_A, csr_rows_A,
                               csr_cols_A, descr_B, nnz_B, csr_vals_B, csr_rows_B, csr_cols_B, beta, descr_C, nnz_C,
                               csr_vals_C, csr_rows_C, csr_cols_C, descr_D, csr_vals_D, csr_rows_D, csr_cols_D, info_D,
                               buffer);
}

#ifdef __cplusplus
}
#endif
