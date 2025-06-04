#include "csr2csc_device.hpp"
#include "device_radix_sort.hpp"
#include "mcsp_config.h"
#include "mcsp_handle.h"
#include "mcsp_internal_host_utils.hpp"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

template <typename idxType>
mcspStatus_t mcspCsr2CscBufferSizeTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz,
                                           const idxType *csr_rows, const idxType *csr_cols,
                                           mcsparseAction_t csc_action, size_t *buffer_size) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (buffer_size == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (n == 0 || m == 0 || nnz == 0) {
        *buffer_size = MIN_BUFFER_SIZE;
        return MCSP_STATUS_SUCCESS;
    }

    constexpr unsigned int csr2csc_buffer_block = 4;
    idxType tmp_buffersize = 0;
    idxType *tmp_ptr = nullptr;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::radix_sort_pairs(nullptr, tmp_buffersize, csr_cols, tmp_ptr, csr_cols, tmp_ptr, nnz, stream);
    *buffer_size = tmp_buffersize;
    *buffer_size += (csr2csc_buffer_block * nnz * sizeof(*csr_rows));

    *buffer_size = *buffer_size < MIN_BUFFER_SIZE ? MIN_BUFFER_SIZE : *buffer_size;

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsr2CscTemplate(mcspHandle_t handle, idxType m, idxType n, idxType nnz, const valType *csr_val,
                                 const idxType *csr_rows, const idxType *csr_cols, valType *csc_val, idxType *csc_rows,
                                 idxType *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                                 void *temp_buffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (n < 0 || m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (n == 0 || m == 0 || nnz == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    if (csr_val == nullptr || csr_rows == nullptr || csr_cols == nullptr || csc_val == nullptr || csc_rows == nullptr ||
        csc_cols == nullptr || temp_buffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    idxType *buffer_ptr = nullptr;
    idxType tmp_buffersize = 0;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::radix_sort_pairs(nullptr, tmp_buffersize, csc_cols, csc_rows, csc_cols, csc_rows, nnz, stream);

    buffer_ptr = reinterpret_cast<idxType *>(temp_buffer);

    idxType *ibuffer = buffer_ptr;
    buffer_ptr += nnz;

    idxType *sorted_ibuffer = buffer_ptr;
    buffer_ptr += nnz;

    idxType *temp_coo_rows = buffer_ptr;
    buffer_ptr += nnz;

    idxType *temp_csr_cols = buffer_ptr;
    buffer_ptr += nnz;

    mcspInt start_bit = 0;
    mcspInt end_bit = getHighBitLocOneBase(n);

    if (csc_action == MCSPARSE_ACTION_NUMERIC) {
        constexpr unsigned int n_elem = 512;
        int n_block = (nnz + n_elem - 1) / n_elem;
        if constexpr (std::is_same_v<idxType, uint32_t> || std::is_same_v<idxType, int32_t>) {
            mcspCreateIdentityPermutation(handle, nnz, ibuffer);
        } else if constexpr (std::is_same_v<idxType, uint64_t> || std::is_same_v<idxType, int64_t>) {
            mcspCreateIdentityPermutation64(handle, nnz, ibuffer);
        } else {
            return MCSP_STATUS_NOT_IMPLEMENTED;
        }
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, csr_cols, temp_csr_cols, ibuffer, sorted_ibuffer, nnz,
                                 stream, start_bit, end_bit);
        if constexpr (std::is_same_v<idxType, uint32_t> || std::is_same_v<idxType, int32_t>) {
            mcspCoo2Csr(handle, temp_csr_cols, nnz, n, csc_cols, idx_base);
            mcspCsr2Coo(handle, csr_rows, nnz, m, temp_coo_rows, idx_base);
        } else if constexpr (std::is_same_v<idxType, uint64_t> || std::is_same_v<idxType, int64_t>) {
            mcspCoo2Csr64(handle, temp_csr_cols, nnz, n, csc_cols, idx_base);
            mcspCsr2Coo64(handle, csr_rows, nnz, m, temp_coo_rows, idx_base);
        } else {
            return MCSP_STATUS_NOT_IMPLEMENTED;
        }
        mcLaunchKernelGGL((mcspCsr2CscKernel<n_elem>), dim3(n_block), dim3(n_elem), 0, stream, nnz, temp_coo_rows,
                           csr_val, sorted_ibuffer, csc_rows, csc_val);
    } else {
        if constexpr (std::is_same_v<idxType, uint32_t> || std::is_same_v<idxType, int32_t>) {
            mcspCsr2Coo(handle, csr_rows, nnz, m, csc_rows, idx_base);
        } else if constexpr (std::is_same_v<idxType, uint64_t> || std::is_same_v<idxType, int64_t>) {
            mcspCsr2Coo64(handle, csr_rows, nnz, m, csc_rows, idx_base);
        } else {
            return MCSP_STATUS_NOT_IMPLEMENTED;
        }
        mcprim::radix_sort_pairs(buffer_ptr, tmp_buffersize, csr_cols, temp_csr_cols, csc_rows, temp_coo_rows, nnz,
                                 stream, start_bit, end_bit);
        if constexpr (std::is_same_v<idxType, uint32_t> || std::is_same_v<idxType, int32_t>) {
            mcspCoo2Csr(handle, temp_csr_cols, nnz, n, csc_cols, idx_base);
        } else if constexpr (std::is_same_v<idxType, uint64_t> || std::is_same_v<idxType, int64_t>) {
            mcspCoo2Csr64(handle, temp_csr_cols, nnz, n, csc_cols, idx_base);
        } else {
            return MCSP_STATUS_NOT_IMPLEMENTED;
        }
        MACA_ASSERT(mcMemcpyAsync(csc_rows, temp_coo_rows, nnz * sizeof(idxType), mcMemcpyDeviceToDevice, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
    }

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspCsr2CscBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspInt *csr_rows,
                                   const mcspInt *csr_cols, mcsparseAction_t csc_action, size_t *buffer_size) {
    return mcspCsr2CscBufferSizeTemplate<mcspInt>(handle, m, n, nnz, csr_rows, csr_cols, csc_action, buffer_size);
}

mcspStatus_t mcspCsr2CscBufferSize64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const int64_t *csr_rows,
                                     const int64_t *csr_cols, mcsparseAction_t csc_action, size_t *buffer_size) {
    return mcspCsr2CscBufferSizeTemplate<int64_t>(handle, m, n, nnz, csr_rows, csr_cols, csc_action, buffer_size);
}

mcspStatus_t mcspScsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const float *csr_val,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, float *csc_val, mcspInt *csc_rows,
                          mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                          void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspScsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const float *csr_val,
                            const int64_t *csr_rows, const int64_t *csr_cols, float *csc_val, int64_t *csc_rows,
                            int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                            void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspDcsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const double *csr_val,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, double *csc_val, mcspInt *csc_rows,
                          mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                          void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspDcsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const double *csr_val,
                            const int64_t *csr_rows, const int64_t *csr_cols, double *csc_val, int64_t *csc_rows,
                            int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                            void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspCcsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspComplexFloat *csr_val,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspComplexFloat *csc_val,
                          mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                          mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspCcsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const mcspComplexFloat *csr_val,
                            const int64_t *csr_rows, const int64_t *csr_cols, mcspComplexFloat *csc_val,
                            int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                            mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspZcsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspComplexDouble *csr_val,
                          const mcspInt *csr_rows, const mcspInt *csr_cols, mcspComplexDouble *csc_val,
                          mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                          mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspZcsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const mcspComplexDouble *csr_val,
                            const int64_t *csr_rows, const int64_t *csr_cols, mcspComplexDouble *csc_val,
                            int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                            mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

#if defined(__MACA__)
mcspStatus_t mcspR16fCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const __half *csr_val,
                             const mcspInt *csr_rows, const mcspInt *csr_cols, __half *csc_val, mcspInt *csc_rows,
                             mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                             void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspR16fCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const __half *csr_val,
                               const int64_t *csr_rows, const int64_t *csr_cols, __half *csc_val, int64_t *csc_rows,
                               int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                               void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspC16fCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const __half2 *csr_val,
                             const mcspInt *csr_rows, const mcspInt *csr_cols, __half2 *csc_val, mcspInt *csc_rows,
                             mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                             void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspC16fCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const __half2 *csr_val,
                               const int64_t *csr_rows, const int64_t *csr_cols, __half2 *csc_val, int64_t *csc_rows,
                               int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                               void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspR16bfCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcsp_bfloat16 *csr_val,
                              const mcspInt *csr_rows, const mcspInt *csr_cols, mcsp_bfloat16 *csc_val,
                              mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                              mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspR16bfCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const mcsp_bfloat16 *csr_val,
                                const int64_t *csr_rows, const int64_t *csr_cols, mcsp_bfloat16 *csc_val,
                                int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                                mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspC16bfCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcsp_bfloat162 *csr_val,
                              const mcspInt *csr_rows, const mcspInt *csr_cols, mcsp_bfloat162 *csc_val,
                              mcspInt *csc_rows, mcspInt *csc_cols, mcsparseAction_t csc_action,
                              mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspC16bfCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const mcsp_bfloat162 *csr_val,
                                const int64_t *csr_rows, const int64_t *csr_cols, mcsp_bfloat162 *csc_val,
                                int64_t *csc_rows, int64_t *csc_cols, mcsparseAction_t csc_action,
                                mcsparseIndexBase_t idx_base, void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspR8iCsr2Csc(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const int8_t *csr_val,
                            const mcspInt *csr_rows, const mcspInt *csr_cols, int8_t *csc_val, mcspInt *csc_rows,
                            mcspInt *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                            void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}

mcspStatus_t mcspR8iCsr2Csc64(mcspHandle_t handle, int64_t m, int64_t n, int64_t nnz, const int8_t *csr_val,
                              const int64_t *csr_rows, const int64_t *csr_cols, int8_t *csc_val, int64_t *csc_rows,
                              int64_t *csc_cols, mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base,
                              void *temp_buffer) {
    return mcspCsr2CscTemplate(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_rows, csc_cols, csc_action,
                               idx_base, temp_buffer);
}
#endif

mcspStatus_t mcspCuinCsr2cscEx2(mcspHandle_t handle, int m, int n, int nnz, const void *csr_val, const int *csr_rows,
                              const int *csr_cols, void *csc_val, int *csc_cols, int *csc_rows, macaDataType val_type,
                              mcsparseAction_t csc_action, mcsparseIndexBase_t idx_base, mcsparseCsr2CscAlg_t alg,
                              void *temp_buffer) {
    switch (val_type) {
        case MACA_R_32F:
            return mcspCsr2CscTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (float *)csr_val,
                                       (mcspInt *)csr_rows, (mcspInt *)csr_cols, (float *)csc_val, (mcspInt *)csc_rows,
                                       (mcspInt *)csc_cols, csc_action, idx_base, temp_buffer);
        case MACA_R_64F:
            return mcspCsr2CscTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (double *)csr_val,
                                       (mcspInt *)csr_rows, (mcspInt *)csr_cols, (double *)csc_val, (mcspInt *)csc_rows,
                                       (mcspInt *)csc_cols, csc_action, idx_base, temp_buffer);
        case MACA_C_32F:
            return mcspCsr2CscTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (mcspComplexFloat *)csr_val,
                                       (mcspInt *)csr_rows, (mcspInt *)csr_cols, (mcspComplexFloat *)csc_val,
                                       (mcspInt *)csc_rows, (mcspInt *)csc_cols, csc_action, idx_base, temp_buffer);
        case MACA_C_64F:
            return mcspCsr2CscTemplate(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (mcspComplexDouble *)csr_val,
                                       (mcspInt *)csr_rows, (mcspInt *)csr_cols, (mcspComplexDouble *)csc_val,
                                       (mcspInt *)csc_rows, (mcspInt *)csc_cols, csc_action, idx_base, temp_buffer);
        default:
            return MCSP_STATUS_TYPE_MISMATCH;
    }
}

mcspStatus_t mcspCuinCsr2cscEx2_bufferSize(mcspHandle_t handle, int m, int n, int nnz, const void *csr_val,
                                         const int *csr_rows, const int *csr_cols, void *csc_val, int *csc_cols,
                                         int *csc_rows, macaDataType val_type, mcsparseAction_t csc_action,
                                         mcsparseIndexBase_t idx_base, mcsparseCsr2CscAlg_t alg, size_t *buffer_size) {
    return mcspCsr2CscBufferSize(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnz, (mcspInt *)csr_rows, (mcspInt *)csr_cols,
                                 csc_action, buffer_size);
}

#ifdef __cplusplus
}
#endif
