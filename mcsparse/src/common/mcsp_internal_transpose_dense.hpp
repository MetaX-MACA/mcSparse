#ifndef COMMON_MCSP_INTERNAL_TRANSPOSE_DENSE_HPP_
#define COMMON_MCSP_INTERNAL_TRANSPOSE_DENSE_HPP_

#include "common/mcsp_types.h"
#include "mcsp_debug.h"
#include "mcsp_general_utility.h"
#include "mcsp_internal_device_kernels.hpp"

template <typename idxType, typename valType>
static mcspStatus_t mcspTransferDeviceDenseMat(mcspHandle_t handle, mcsparseOperation_t op, idxType m_in, idxType n_in,
                                               idxType ld_in, idxType m_out, idxType n_out, idxType ld_out,
                                               const valType *mat_in, valType *mat_out) {
    if (mat_in == nullptr || mat_out == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (ld_in < m_in || ld_out < m_out) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    const int block_size = 256;
    const int trans_dimX = 32;
    const int trans_dimY = 8;
    dim3 copy_threads(block_size);
    dim3 copy_blocks(CEIL(m_out, block_size), n_out);
    dim3 trans_threads(trans_dimX * trans_dimY);
    dim3 trans_blocks(CEIL(m_in, trans_dimX));
    mcStream_t stream = mcspGetStreamInternal(handle);
    if (op != MCSPARSE_OPERATION_NON_TRANSPOSE) {
        mcLaunchKernelGGL((mcspDenseTransposeKernel<trans_dimX, trans_dimY>), trans_blocks, trans_threads, 0, stream,
                           (idxType)m_in, (idxType)n_in, (valType *)mat_in, (idxType)ld_in, (valType *)mat_out,
                           (idxType)ld_out);
    } else {
        mcLaunchKernelGGL((transferDenseMatrixKernel<idxType, valType>), copy_blocks, copy_threads, 0, stream,
                           (idxType)m_in, (idxType)n_in, (idxType)ld_in, (idxType)m_out, (idxType)n_out,
                           (idxType)ld_out, (valType *)mat_in, (valType *)mat_out);
    }
    return MCSP_STATUS_SUCCESS;
}

#endif
