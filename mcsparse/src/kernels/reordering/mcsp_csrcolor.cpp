#include <limits>
#include <vector>

#include "common/mcsp_types.h"
#include "csrcolor_device.hpp"
#include "device_radix_sort.hpp"
#include "device_reduce.hpp"
#include "device_scan.hpp"
#include "internal_interface/mcsp_internal_conversion.h"
#include "mcsp_debug.h"
#include "mcsp_handle.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"
#include "utils/mcsp_logger.h"

template <typename idxType>
static mcspStatus_t mcspCsrcolorReorderTemplate(mcspHandle_t handle, idxType m, const idxType *coloring,
                                                idxType *reordering, idxType *workspace) {
    idxType *identity = workspace;
    idxType *tmp_identity = identity + m;
    idxType *sorted_colors = tmp_identity + m;

    mcspCreateIdentityPermutation(handle, m, identity);

    idxType *tmp_workspace = sorted_colors + m;
    idxType tmp_buffersize;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::radix_sort_pairs(nullptr, tmp_buffersize, coloring, sorted_colors, identity, reordering, m, stream);
    mcprim::radix_sort_pairs(tmp_workspace, tmp_buffersize, coloring, sorted_colors, identity, reordering, m, stream);

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCsrcolorFullTemplate(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                                      const idxType *csr_rows, const idxType *csr_cols, idxType *ncolors,
                                      idxType *coloring, idxType *reordering, mcspMatInfo_t info, idxType *workspace) {
    idxType buffersize;
    idxType max_color_base = (std::numeric_limits<idxType>::max)();
    idxType max_color = max_color_base;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemsetAsync(coloring, max_color_base, sizeof(idxType) * m, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    idxType *max_color_dev = workspace;
    idxType *buffer = workspace + 1;

    idxType ncolored = 0;
    int n_elem = 512;
    int n_block = (m + n_elem / WARP_SIZE - 1) / (n_elem / WARP_SIZE);

    constexpr uint32_t seed = 5;
    while (max_color == max_color_base) {
        mcLaunchKernelGGL(mcspCsrColorJplWarpKernel, dim3(n_block), dim3(n_elem), 0, stream, m, ncolored, csr_rows,
                           csr_cols, descr->base, coloring, max_color_base, seed);
        MACA_ASSERT(mcStreamSynchronize(stream));
        mcprim::reduce(buffer, buffersize, coloring, max_color_dev, m, mcprim::maximum<idxType>(), stream);
        MACA_ASSERT(mcMemcpyAsync(&max_color, max_color_dev, sizeof(idxType), mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        ncolored += 2;
    }
    *ncolors = max_color + 1;

    if (reordering != nullptr) {
        mcspCsrcolorReorderTemplate(handle, m, coloring, reordering, workspace);
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType>
mcspStatus_t mcspCsrcolorPartialTemplate(mcspHandle_t handle, idxType m, idxType nnz, idxType max_uncolored_row,
                                         const mcspMatDescr_t descr, const idxType *csr_rows, const idxType *csr_cols,
                                         idxType *ncolors, idxType *coloring, idxType *reordering, mcspMatInfo_t info,
                                         idxType *workspace) {
    idxType buffersize1;
    idxType buffersize2;
    idxType max_color_base = (std::numeric_limits<idxType>::max)();
    idxType max_color = max_color_base;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMemsetAsync(coloring, max_color_base, sizeof(idxType) * m, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));

    mcprim::reduce(nullptr, buffersize1, coloring, workspace, m, mcprim::plus<idxType>(), stream);
    mcprim::exclusive_scan(nullptr, buffersize2, coloring, workspace, m, (idxType *)nullptr, stream);
    idxType *color_src_buffer = workspace;
    idxType *tmp_scalar = workspace + m;
    idxType *color_dest_buffer = workspace + m;
    idxType *prim_buffer = workspace + 2 * m;
    idxType num_uncolored = m + 1;

    idxType ncolored = 0;
    int n_elem = 512;
    int n_block = (m + n_elem / WARP_SIZE - 1) / (n_elem / WARP_SIZE);
    while (num_uncolored > max_uncolored_row) {
        mcLaunchKernelGGL(mcspCsrColorJplWarpKernel, dim3(n_block), dim3(n_elem), 0, stream, m, ncolored, csr_rows,
                           csr_cols, descr->base, coloring, max_color_base, 5);
        MACA_ASSERT(mcStreamSynchronize(stream));

        // counting number of uncolored rows
        mcLaunchKernelGGL(mcspCsrTagUncoloredKernel, dim3(n_block), dim3(n_elem), 0, stream, m, coloring,
                           color_src_buffer, max_color_base);
        MACA_ASSERT(mcStreamSynchronize(stream));
        mcprim::reduce(prim_buffer, buffersize1, color_src_buffer, tmp_scalar, m, mcprim::plus<idxType>(), stream);
        MACA_ASSERT(mcMemcpyAsync(&num_uncolored, tmp_scalar, sizeof(idxType), mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));

        // Get max color index
        mcLaunchKernelGGL(mcspCsrClearUncoloredKernel, dim3(n_block), dim3(n_elem), 0, stream, m, coloring,
                           color_src_buffer, max_color_base, std::numeric_limits<idxType>::min());
        MACA_ASSERT(mcStreamSynchronize(stream));
        mcprim::reduce(prim_buffer, buffersize1, color_src_buffer, tmp_scalar, m, mcprim::maximum<idxType>(), stream);
        MACA_ASSERT(mcMemcpyAsync(&max_color, tmp_scalar, sizeof(idxType), mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));

        ncolored += 2;
    }

    // assign uncolored rows
    n_block = (m + n_elem - 1) / (n_elem);
    mcLaunchKernelGGL(mcspCsrTagUncoloredKernel, dim3(n_block), dim3(n_elem), 0, stream, m, coloring, color_src_buffer,
                       max_color_base);
    MACA_ASSERT(mcStreamSynchronize(stream));
    mcprim::exclusive_scan(prim_buffer, buffersize2, color_src_buffer, color_dest_buffer, m, (idxType *)nullptr,
                           stream);
    MACA_ASSERT(mcStreamSynchronize(stream));
    mcLaunchKernelGGL(mcspCsrAssignUncoloredKernel, dim3(n_block), dim3(n_elem), 0, stream, m, coloring,
                       color_dest_buffer, max_color_base, max_color + 1);
    MACA_ASSERT(mcStreamSynchronize(stream));
    *ncolors = max_color + 1 + num_uncolored;

    if (reordering != nullptr) {
        mcspCsrcolorReorderTemplate(handle, m, coloring, reordering, workspace);
    }

    return MCSP_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcspStatus_t mcspCsrColorTemplate(mcspHandle_t handle, idxType m, idxType nnz, const mcspMatDescr_t descr,
                                  const idxType *csr_rows, const idxType *csr_cols, const valType *fraction_to_color,
                                  idxType *ncolors, idxType *coloring, idxType *reordering, mcspMatInfo_t info) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }

    if (m < 0 || nnz < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (fraction_to_color == nullptr || csr_rows == nullptr || csr_cols == nullptr || ncolors == nullptr ||
        coloring == nullptr || reordering == nullptr || reordering == nullptr || info == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (descr->type != MCSPARSE_MATRIX_TYPE_GENERAL || descr->fill_mode != MCSPARSE_FILL_MODE_FULL ||
        descr->diag_type != MCSPARSE_DIAG_TYPE_NON_UNIT) {
        return MCSP_STATUS_NOT_IMPLEMENTED;
    }

    if (nnz == 0 || m == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    valType fraction = (*fraction_to_color > 0.0) ? *fraction_to_color : 0.0;
    idxType max_uncolored_row = (m * fraction < m) ? (idxType)(m - m * fraction) : 0;

    idxType buffersize1;
    idxType buffersize2;
    idxType total_size;
    idxType *workspace;
    mcStream_t stream = mcspGetStreamInternal(handle);
    mcprim::reduce(nullptr, buffersize1, coloring, workspace, m, mcprim::plus<idxType>(), stream);
    mcprim::exclusive_scan(nullptr, buffersize2, coloring, workspace, m, (idxType *)nullptr, stream);

    idxType color_workspace_size = (3 * m) * sizeof(idxType);
    total_size = std::max(buffersize1, buffersize2) + color_workspace_size;

    void *tmp_buffer;
    bool use_buffer_pool = handle->mcspUsePoolBuffer(&tmp_buffer, total_size);
    if (!use_buffer_pool) {
        MACA_ASSERT(mcMalloc(&tmp_buffer, total_size));
    }
    workspace = (idxType *)tmp_buffer;

    mcspStatus_t ret;
    if (max_uncolored_row == 0) {
        ret = mcspCsrcolorFullTemplate(handle, m, nnz, descr, csr_rows, csr_cols, ncolors, coloring, reordering, info,
                                       workspace);
    } else {
        ret = mcspCsrcolorPartialTemplate(handle, m, nnz, max_uncolored_row, descr, csr_rows, csr_cols, ncolors,
                                          coloring, reordering, info, workspace);
    }

    if (use_buffer_pool) {
        MACA_ASSERT(handle->mcspReturnPoolBuffer());
    } else {
        MACA_ASSERT(mcFree(tmp_buffer));
    }

    return ret;
}

mcspStatus_t mcspCsrcolorCheck(mcspHandle_t handle, mcspInt m, mcspInt nnz, mcsparseIndexBase_t csr_base,
                               const mcspInt *csr_rows, const mcspInt *csr_cols, mcspInt *ncolors, mcspInt *coloring,
                               mcspInt *reordering, mcspMatInfo_t info, bool &check) {
    mcspInt flag_host = 1;
    mcspInt *flag_dev;
    mcStream_t stream = mcspGetStreamInternal(handle);
    MACA_ASSERT(mcMalloc(&flag_dev, sizeof(*flag_dev)));
    MACA_ASSERT(mcMemsetAsync(flag_dev, 0, sizeof(*flag_dev), stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    int n_elem = 512;
    int n_block = (m + n_elem / WARP_SIZE - 1) / (n_elem / WARP_SIZE);
    mcLaunchKernelGGL(mcspCsrColorCheckKernel, dim3(n_block), dim3(n_elem), 0, stream, m, csr_rows, csr_cols, csr_base,
                       coloring, flag_dev);
    MACA_ASSERT(mcMemcpyAsync(&flag_host, flag_dev, sizeof(*flag_dev), mcMemcpyDeviceToHost, stream));
    MACA_ASSERT(mcStreamSynchronize(stream));
    MACA_ASSERT(mcFree(flag_dev));

    check = (flag_host == 1) ? false : true;
    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Coloring of the adjacency graph of the matrix A stored in the CSR format
 * ref: A parallel graph coloring heuristic, 1993
 */
mcspStatus_t mcspScsrcolor(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, float *csr_vals,
                           const mcspInt *csr_rows, const mcspInt *csr_cols, const float *fraction_to_color,
                           mcspInt *ncolors, mcspInt *coloring, mcspInt *reordering, mcspMatInfo_t info) {
    return mcspCsrColorTemplate(handle, m, nnz, descr, csr_rows, csr_cols, fraction_to_color, ncolors, coloring,
                                reordering, info);
}

mcspStatus_t mcspDcsrcolor(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, double *csr_vals,
                           const mcspInt *csr_rows, const mcspInt *csr_cols, const double *fraction_to_color,
                           mcspInt *ncolors, mcspInt *coloring, mcspInt *reordering, mcspMatInfo_t info) {
    return mcspCsrColorTemplate(handle, m, nnz, descr, csr_rows, csr_cols, fraction_to_color, ncolors, coloring,
                                reordering, info);
}

mcspStatus_t mcspCcsrcolor(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                           mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                           const float *fraction_to_color, mcspInt *ncolors, mcspInt *coloring, mcspInt *reordering,
                           mcspMatInfo_t info) {
    return mcspCsrColorTemplate(handle, m, nnz, descr, csr_rows, csr_cols, fraction_to_color, ncolors, coloring,
                                reordering, info);
}

mcspStatus_t mcspZcsrcolor(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                           mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                           const double *fraction_to_color, mcspInt *ncolors, mcspInt *coloring, mcspInt *reordering,
                           mcspMatInfo_t info) {
    return mcspCsrColorTemplate(handle, m, nnz, descr, csr_rows, csr_cols, fraction_to_color, ncolors, coloring,
                                reordering, info);
}

mcspStatus_t mcspCuinScsrColor(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, float *csr_vals,
                             const int *csr_rows, const int *csr_cols, const float *fraction_to_color, int *ncolors,
                             int *coloring, int *reordering, mcspColorInfo_t info) {
    return mcspCsrColorTemplate(handle, (mcspInt)m, (mcspInt)nnz, descr, (mcspInt *)csr_rows, (mcspInt *)csr_cols,
                                fraction_to_color, (mcspInt *)ncolors, (mcspInt *)coloring, (mcspInt *)reordering,
                                info->mat_info);
}

mcspStatus_t mcspCuinDcsrColor(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, double *csr_vals,
                             const int *csr_rows, const int *csr_cols, const double *fraction_to_color, int *ncolors,
                             int *coloring, int *reordering, mcspColorInfo_t info) {
    return mcspCsrColorTemplate(handle, (mcspInt)m, (mcspInt)nnz, descr, (mcspInt *)csr_rows, (mcspInt *)csr_cols,
                                fraction_to_color, (mcspInt *)ncolors, (mcspInt *)coloring, (mcspInt *)reordering,
                                info->mat_info);
}

mcspStatus_t mcspCuinCcsrColor(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                             mcspComplexFloat *csr_vals, const int *csr_rows, const int *csr_cols,
                             const float *fraction_to_color, int *ncolors, int *coloring, int *reordering,
                             mcspColorInfo_t info) {
    return mcspCsrColorTemplate(handle, (mcspInt)m, (mcspInt)nnz, descr, (mcspInt *)csr_rows, (mcspInt *)csr_cols,
                                fraction_to_color, (mcspInt *)ncolors, (mcspInt *)coloring, (mcspInt *)reordering,
                                info->mat_info);
}

mcspStatus_t mcspCuinZcsrColor(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                             mcspComplexDouble *csr_vals, const int *csr_rows, const int *csr_cols,
                             const double *fraction_to_color, int *ncolors, int *coloring, int *reordering,
                             mcspColorInfo_t info) {
    return mcspCsrColorTemplate(handle, (mcspInt)m, (mcspInt)nnz, descr, (mcspInt *)csr_rows, (mcspInt *)csr_cols,
                                fraction_to_color, (mcspInt *)ncolors, (mcspInt *)coloring, (mcspInt *)reordering,
                                info->mat_info);
}
#ifdef __cplusplus
}
#endif
