#include "common/mcsp_types.h"
#include "csr2csr_compress_device.hpp"
#include "device_reduce.hpp"
#include "device_scan.hpp"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"
#include "nnz_compress_device.hpp"

template <typename idxType, typename valType>
mcspStatus_t mcspCsr2csrCompressTemplate(mcspHandle_t handle, idxType m, idxType n, const mcspMatDescr_t mcsp_descr_A,
                                         const valType* csr_val_A, const idxType* csr_row_A, const idxType* csr_col_A,
                                         idxType nnz_A, const idxType* nnz_per_row, valType* csr_val_C,
                                         idxType* csr_row_C, idxType* csr_col_C, valType tol) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || n < 0 || nnz_A < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (mcsp_descr_A->type != MCSPARSE_MATRIX_TYPE_GENERAL) {
        return MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (std::real(tol) < std::real(static_cast<valType>(0))) {
        return MCSP_STATUS_INVALID_VALUE;
    }

    if (m == 0 || n == 0 || nnz_A == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (csr_val_A == nullptr || csr_row_A == nullptr || csr_col_A == nullptr || nnz_per_row == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if ((csr_val_C == nullptr && csr_col_C != nullptr) || (csr_val_C != nullptr && csr_col_C == nullptr)) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    constexpr uint32_t block_size = 512;
    uint32_t grid_size = (m + block_size - 1) / block_size;
    mcStream_t stream = mcspGetStreamInternal(handle);

    idxType* tmp_csr_row_C;
    MACA_ASSERT(mcMalloc(&tmp_csr_row_C, (m + 1) * sizeof(*tmp_csr_row_C)));
    mcLaunchKernelGGL(FillRowDevice<block_size>, dim3(grid_size), dim3(block_size), 0, stream, m, mcsp_descr_A->base,
                       nnz_per_row, tmp_csr_row_C);

    idxType buffer_size;
    void* buffer_device;
    bool use_buffer_pool;

    mcprim::inclusive_scan(nullptr, buffer_size, tmp_csr_row_C, csr_row_C, m + 1, stream);
    use_buffer_pool = handle->mcspUsePoolBuffer(&buffer_device, buffer_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc(&buffer_device, buffer_size));
    }
    mcprim::inclusive_scan(buffer_device, buffer_size, tmp_csr_row_C, csr_row_C, m + 1, stream);

    constexpr uint32_t segment_size = 32;
    constexpr uint32_t segments_per_block = block_size / segment_size;
    grid_size = (m + segments_per_block - 1) / segments_per_block;

    mcLaunchKernelGGL((csr2csrCompressKernel<block_size, segments_per_block, segment_size, WARP_SIZE>),
                       dim3(grid_size), dim3(block_size), 0, stream, m, n, mcsp_descr_A->base, csr_val_A, csr_row_A,
                       csr_col_A, nnz_A, mcsp_descr_A->base, csr_val_C, csr_row_C, csr_col_C, tol);

    MACA_ASSERT(mcFree(tmp_csr_row_C));
    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(buffer_device));
    }
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspScsr2csrCompress(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                  const float* csr_val_A, const mcspInt* csr_row_A, const mcspInt* csr_col_A,
                                  mcspInt nnz_A, const mcspInt* nnz_per_row, float* csr_val_C, mcspInt* csr_row_C,
                                  mcspInt* csr_col_C, float tol) {
    return mcspCsr2csrCompressTemplate(handle, m, n, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, nnz_A, nnz_per_row,
                                       csr_val_C, csr_row_C, csr_col_C, tol);
}

mcspStatus_t mcspDcsr2csrCompress(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                  const double* csr_val_A, const mcspInt* csr_row_A, const mcspInt* csr_col_A,
                                  mcspInt nnz_A, const mcspInt* nnz_per_row, double* csr_val_C, mcspInt* csr_row_C,
                                  mcspInt* csr_col_C, double tol) {
    return mcspCsr2csrCompressTemplate(handle, m, n, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, nnz_A, nnz_per_row,
                                       csr_val_C, csr_row_C, csr_col_C, tol);
}

mcspStatus_t mcspCcsr2csrCompress(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                  const mcspComplexFloat* csr_val_A, const mcspInt* csr_row_A, const mcspInt* csr_col_A,
                                  mcspInt nnz_A, const mcspInt* nnz_per_row, mcspComplexFloat* csr_val_C,
                                  mcspInt* csr_row_C, mcspInt* csr_col_C, mcspComplexFloat tol) {
    return mcspCsr2csrCompressTemplate(handle, m, n, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, nnz_A, nnz_per_row,
                                       csr_val_C, csr_row_C, csr_col_C, tol);
}

mcspStatus_t mcspZcsr2csrCompress(mcspHandle_t handle, mcspInt m, mcspInt n, const mcspMatDescr_t mcsp_descr_A,
                                  const mcspComplexDouble* csr_val_A, const mcspInt* csr_row_A,
                                  const mcspInt* csr_col_A, mcspInt nnz_A, const mcspInt* nnz_per_row,
                                  mcspComplexDouble* csr_val_C, mcspInt* csr_row_C, mcspInt* csr_col_C,
                                  mcspComplexDouble tol) {
    return mcspCsr2csrCompressTemplate(handle, m, n, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, nnz_A, nnz_per_row,
                                       csr_val_C, csr_row_C, csr_col_C, tol);
}

mcspStatus_t mcspCuinScsr2csr_compress(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                                     const float* csr_val_A, const int* csr_col_A, const int* csr_row_A, int nnz_A,
                                     const int* nnz_per_row, float* csr_val_C, int* csr_col_C, int* csr_row_C,
                                     float tol) {
    return mcspCsr2csrCompressTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csr_val_A, (mcspInt*)csr_row_A,
                                       (mcspInt*)csr_col_A, (mcspInt)nnz_A, (mcspInt*)nnz_per_row, csr_val_C,
                                       (mcspInt*)csr_row_C, (mcspInt*)csr_col_C, tol);
}

mcspStatus_t mcspCuinDcsr2csr_compress(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                                     const double* csr_val_A, const int* csr_col_A, const int* csr_row_A, int nnz_A,
                                     const int* nnz_per_row, double* csr_val_C, int* csr_col_C, int* csr_row_C,
                                     double tol) {
    return mcspCsr2csrCompressTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csr_val_A, (mcspInt*)csr_row_A,
                                       (mcspInt*)csr_col_A, (mcspInt)nnz_A, (mcspInt*)nnz_per_row, csr_val_C,
                                       (mcspInt*)csr_row_C, (mcspInt*)csr_col_C, tol);
}

mcspStatus_t mcspCuinCcsr2csr_compress(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                                     const mcspComplexFloat* csr_val_A, const int* csr_col_A, const int* csr_row_A,
                                     int nnz_A, const int* nnz_per_row, mcspComplexFloat* csr_val_C, int* csr_col_C,
                                     int* csr_row_C, mcspComplexFloat tol) {
    return mcspCsr2csrCompressTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csr_val_A, (mcspInt*)csr_row_A,
                                       (mcspInt*)csr_col_A, (mcspInt)nnz_A, (mcspInt*)nnz_per_row, csr_val_C,
                                       (mcspInt*)csr_row_C, (mcspInt*)csr_col_C, tol);
}

mcspStatus_t mcspCuinZcsr2csr_compress(mcspHandle_t handle, int m, int n, const mcspMatDescr_t mcsp_descr_A,
                                     const mcspComplexDouble* csr_val_A, const int* csr_col_A, const int* csr_row_A,
                                     int nnz_A, const int* nnz_per_row, mcspComplexDouble* csr_val_C, int* csr_col_C,
                                     int* csr_row_C, mcspComplexDouble tol) {
    return mcspCsr2csrCompressTemplate(handle, (mcspInt)m, (mcspInt)n, mcsp_descr_A, csr_val_A, (mcspInt*)csr_row_A,
                                       (mcspInt*)csr_col_A, (mcspInt)nnz_A, (mcspInt*)nnz_per_row, csr_val_C,
                                       (mcspInt*)csr_row_C, (mcspInt*)csr_col_C, tol);
}

#ifdef __cplusplus
}
#endif