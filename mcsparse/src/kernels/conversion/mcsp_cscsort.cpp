#include "mcsp_handle.h"
#include "mcsp_runtime_wrapper.h"
#include "mcsparse.h"

#ifdef __cplusplus
extern "C" {
#endif
mcspStatus_t mcspCscSortBufferSize(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, const mcspInt *csc_cols,
                                   const mcspInt *csc_rows, size_t *buffer_size) {
    return mcspCsrSortBufferSize(handle, n, m, nnz, csc_cols, csc_rows, buffer_size);
}

mcspStatus_t mcspCscSort(mcspHandle_t handle, mcspInt m, mcspInt n, mcspInt nnz, mcspMatDescr_t mcsp_descr_A,
                         const mcspInt *csc_cols, mcspInt *csc_rows, mcspInt *perm, void *temp_buffer) {
    return mcspCsrSort(handle, n, m, nnz, mcsp_descr_A, csc_cols, csc_rows, perm, temp_buffer);
}

mcspStatus_t mcspCuinXcscsort_bufferSizeExt(mcspHandle_t handle, int m, int n, int nnz, const int *csc_cols,
                                          const int *csc_rows, size_t *buffer_size) {
    return mcspCsrSortBufferSize(handle, (mcspInt)n, (mcspInt)m, (mcspInt)nnz, (mcspInt *)csc_cols, (mcspInt *)csc_rows,
                                 buffer_size);
}

mcspStatus_t mcspCuinXcscsort(mcspHandle_t handle, int m, int n, int nnz, mcspMatDescr_t mcsp_descr_A,
                            const int *csc_cols, int *csc_rows, int *perm, void *temp_buffer) {
    return mcspCsrSort(handle, (mcspInt)n, (mcspInt)m, (mcspInt)nnz, mcsp_descr_A, (mcspInt *)csc_cols,
                       (mcspInt *)csc_rows, (mcspInt *)perm, temp_buffer);
}

#ifdef __cplusplus
}
#endif