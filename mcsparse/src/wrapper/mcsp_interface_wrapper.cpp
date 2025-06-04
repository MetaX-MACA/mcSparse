#include "common/mcsp_types.h"
#include "mcsp_internal_interface.h"
#include "mcsp_internal_types.h"

mcsparseStatus_t mcspToSparseStatus(mcspStatus_t status) {
    switch (status) {
        case MCSP_STATUS_SUCCESS:
            return MCSPARSE_STATUS_SUCCESS;
        case MCSP_STATUS_INVALID_HANDLE:
            return MCSPARSE_STATUS_NOT_INITIALIZED;
        case MCSP_STATUS_NOT_IMPLEMENTED:
            return MCSPARSE_STATUS_NOT_SUPPORTED;
        case MCSP_STATUS_INVALID_POINTER:
            return MCSPARSE_STATUS_INVALID_VALUE;
        case MCSP_STATUS_INVALID_SIZE:
            return MCSPARSE_STATUS_INVALID_VALUE;
        case MCSP_STATUS_MEMORY_ERROR:
            return MCSPARSE_STATUS_ALLOC_FAILED;
        case MCSP_STATUS_INTERNAL_ERROR:
            return MCSPARSE_STATUS_INTERNAL_ERROR;
        case MCSP_STATUS_INVALID_VALUE:
            return MCSPARSE_STATUS_INVALID_VALUE;
        case MCSP_STATUS_ARCH_MISMATCH:
            return MCSPARSE_STATUS_ARCH_MISMATCH;
        case MCSP_STATUS_ZERO_PIVOT:
            return MCSPARSE_STATUS_ZERO_PIVOT;
        case MCSP_STATUS_NOT_INITIALIZED:
            return MCSPARSE_STATUS_NOT_INITIALIZED;
        case MCSP_STATUS_TYPE_MISMATCH:
            return MCSPARSE_STATUS_INVALID_VALUE;
        case MCSP_STATUS_MAPPING_ERROR:
            return MCSPARSE_STATUS_MAPPING_ERROR;
        case MCSP_STATUS_EXECUTION_FAILED:
            return MCSPARSE_STATUS_EXECUTION_FAILED;
        case MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return MCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
        case MCSP_STATUS_INSUFFICIENT_RESOURCES:
            return MCSPARSE_STATUS_INSUFFICIENT_RESOURCES;
        default:
            return MCSPARSE_STATUS_INTERNAL_ERROR;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

const char* mcsparseGetErrorName(mcsparseStatus_t status) {
    switch (status) {
        case MCSPARSE_STATUS_SUCCESS:
            return "MCSPARSE_STATUS_SUCCESS";
        case MCSPARSE_STATUS_NOT_INITIALIZED:
            return "MCSPARSE_STATUS_NOT_INITIALIZED";
        case MCSPARSE_STATUS_ALLOC_FAILED:
            return "MCSPARSE_STATUS_ALLOC_FAILED";
        case MCSPARSE_STATUS_INVALID_VALUE:
            return "MCSPARSE_STATUS_INVALID_VALUE";
        case MCSPARSE_STATUS_ARCH_MISMATCH:
            return "MCSPARSE_STATUS_ARCH_MISMATCH";
        case MCSPARSE_STATUS_MAPPING_ERROR:
            return "MCSPARSE_STATUS_MAPPING_ERROR";
        case MCSPARSE_STATUS_EXECUTION_FAILED:
            return "MCSPARSE_STATUS_EXECUTION_FAILED";
        case MCSPARSE_STATUS_INTERNAL_ERROR:
            return "MCSPARSE_STATUS_INTERNAL_ERROR";
        case MCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "MCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case MCSPARSE_STATUS_ZERO_PIVOT:
            return "MCSPARSE_STATUS_ZERO_PIVOT";
        case MCSPARSE_STATUS_NOT_SUPPORTED:
            return "MCSPARSE_STATUS_NOT_SUPPORTED";
        case MCSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return "MCSPARSE_STATUS_INSUFFICIENT_RESOURCES";
        default:
            return "unrecognized error code";
    }
}

const char* mcsparseGetErrorString(mcsparseStatus_t status) {
    switch (status) {
        case MCSPARSE_STATUS_SUCCESS:
            return "success";
        case MCSPARSE_STATUS_NOT_INITIALIZED:
            return "initialization error";
        case MCSPARSE_STATUS_ALLOC_FAILED:
            return "out of memory";
        case MCSPARSE_STATUS_INVALID_VALUE:
            return "invalid value";
        case MCSPARSE_STATUS_ARCH_MISMATCH:
            return "architecture mismatch";
        case MCSPARSE_STATUS_MAPPING_ERROR:
            return "texture memory mapping error";
        case MCSPARSE_STATUS_EXECUTION_FAILED:
            return "kernel launch failure";
        case MCSPARSE_STATUS_INTERNAL_ERROR:
            return "internal error";
        case MCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "matrix type not supported";
        case MCSPARSE_STATUS_ZERO_PIVOT:
            return "zero pivot";
        case MCSPARSE_STATUS_NOT_SUPPORTED:
            return "operation not supported";
        case MCSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return "insufficient resources";
        default:
            return "unrecognized error code";
    }
}

mcsparseStatus_t mcsparseXcoo2csr(mcsparseHandle_t handle, const int* coo_rows, int nnz, int m, int* csr_rows,
                                  mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinXcoo2csr(handle, coo_rows, nnz, m, csr_rows, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsr2coo(mcsparseHandle_t handle, const int* csr_rows, int nnz, int m, int* coo_rows,
                                  mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinXcsr2coo(handle, csr_rows, nnz, m, coo_rows, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCsr2cscEx2(mcsparseHandle_t handle, int m, int n, int nnz, const void* csr_val,
                                    const int* csr_rows, const int* csr_cols, void* csc_val, int* csc_cols,
                                    int* csc_rows, macaDataType val_type, mcsparseAction_t csc_action,
                                    mcsparseIndexBase_t idx_base, mcsparseCsr2CscAlg_t alg, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinCsr2cscEx2(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_cols, csc_rows,
                                        val_type, csc_action, idx_base, alg, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCsr2cscEx2_bufferSize(mcsparseHandle_t handle, int m, int n, int nnz, const void* csr_val,
                                               const int* csr_rows, const int* csr_cols, void* csc_val, int* csc_cols,
                                               int* csc_rows, macaDataType val_type, mcsparseAction_t csc_action,
                                               mcsparseIndexBase_t idx_base, mcsparseCsr2CscAlg_t alg,
                                               size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinCsr2cscEx2_bufferSize(handle, m, n, nnz, csr_val, csr_rows, csr_cols, csc_val, csc_cols,
                                                   csc_rows, val_type, csc_action, idx_base, alg, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSnnz_compress(mcsparseHandle_t handle, int m, const mcsparseMatDescr_t mcsp_descr_A,
                                       const float* csr_val_A, const int* csr_row_A, int* nnz_per_row, int* nnz_C,
                                       float tol) {
    mcspStatus_t ret = mcspCuinSnnz_compress(handle, m, mcsp_descr_A, csr_val_A, csr_row_A, nnz_per_row, nnz_C, tol);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnnz_compress(mcsparseHandle_t handle, int m, const mcsparseMatDescr_t mcsp_descr_A,
                                       const double* csr_val_A, const int* csr_row_A, int* nnz_per_row, int* nnz_C,
                                       double tol) {
    mcspStatus_t ret = mcspCuinDnnz_compress(handle, m, mcsp_descr_A, csr_val_A, csr_row_A, nnz_per_row, nnz_C, tol);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCnnz_compress(mcsparseHandle_t handle, int m, const mcsparseMatDescr_t mcsp_descr_A,
                                       const mcspComplexFloat* csr_val_A, const int* csr_row_A, int* nnz_per_row,
                                       int* nnz_C, mcspComplexFloat tol) {
    mcspStatus_t ret = mcspCuinCnnz_compress(handle, m, mcsp_descr_A, csr_val_A, csr_row_A, nnz_per_row, nnz_C, tol);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZnnz_compress(mcsparseHandle_t handle, int m, const mcsparseMatDescr_t mcsp_descr_A,
                                       const mcspComplexDouble* csr_val_A, const int* csr_row_A, int* nnz_per_row,
                                       int* nnz_C, mcspComplexDouble tol) {
    mcspStatus_t ret = mcspCuinZnnz_compress(handle, m, mcsp_descr_A, csr_val_A, csr_row_A, nnz_per_row, nnz_C, tol);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsr2csr_compress(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                           const float* csr_val_A, const int* csr_col_A, const int* csr_row_A,
                                           int nnz_A, const int* nnz_per_row, float* csr_val_C, int* csr_col_C,
                                           int* csr_row_C, float tol) {
    mcspStatus_t ret = mcspCuinScsr2csr_compress(handle, m, n, mcsp_descr_A, csr_val_A, csr_col_A, csr_row_A, nnz_A,
                                               nnz_per_row, csr_val_C, csr_col_C, csr_row_C, tol);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsr2csr_compress(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                           const double* csr_val_A, const int* csr_col_A, const int* csr_row_A,
                                           int nnz_A, const int* nnz_per_row, double* csr_val_C, int* csr_col_C,
                                           int* csr_row_C, double tol) {
    mcspStatus_t ret = mcspCuinDcsr2csr_compress(handle, m, n, mcsp_descr_A, csr_val_A, csr_col_A, csr_row_A, nnz_A,
                                               nnz_per_row, csr_val_C, csr_col_C, csr_row_C, tol);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsr2csr_compress(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                           const mcspComplexFloat* csr_val_A, const int* csr_col_A,
                                           const int* csr_row_A, int nnz_A, const int* nnz_per_row,
                                           mcspComplexFloat* csr_val_C, int* csr_col_C, int* csr_row_C,
                                           mcspComplexFloat tol) {
    mcspStatus_t ret = mcspCuinCcsr2csr_compress(handle, m, n, mcsp_descr_A, csr_val_A, csr_col_A, csr_row_A, nnz_A,
                                               nnz_per_row, csr_val_C, csr_col_C, csr_row_C, tol);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsr2csr_compress(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                           const mcspComplexDouble* csr_val_A, const int* csr_col_A,
                                           const int* csr_row_A, int nnz_A, const int* nnz_per_row,
                                           mcspComplexDouble* csr_val_C, int* csr_col_C, int* csr_row_C,
                                           mcspComplexDouble tol) {
    mcspStatus_t ret = mcspCuinZcsr2csr_compress(handle, m, n, mcsp_descr_A, csr_val_A, csr_col_A, csr_row_A, nnz_A,
                                               nnz_per_row, csr_val_C, csr_col_C, csr_row_C, tol);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsrsort_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, const int* csr_rows,
                                                const int* csr_cols, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinXcsrsort_bufferSizeExt(handle, m, n, nnz, csr_rows, csr_cols, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsrsort(mcsparseHandle_t handle, int m, int n, int nnz, mcsparseMatDescr_t mcsp_descr_A,
                                  const int* csr_rows, int* csr_cols, int* perm, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinXcsrsort(handle, m, n, nnz, mcsp_descr_A, csr_rows, csr_cols, perm, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsr2csru(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, float* csr_vals, const int* csr_rows,
                                   int* csr_cols, mcsparseCsru2csrInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinScsr2csru(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsr2csru(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, double* csr_vals, const int* csr_rows,
                                   int* csr_cols, mcsparseCsru2csrInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinDcsr2csru(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsr2csru(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, mcspComplexFloat* csr_vals,
                                   const int* csr_rows, int* csr_cols, mcsparseCsru2csrInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinCcsr2csru(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsr2csru(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, mcspComplexDouble* csr_vals,
                                   const int* csr_rows, int* csr_cols, mcsparseCsru2csrInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinZcsr2csru(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcscsort_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, const int* csc_cols,
                                                const int* csc_rows, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinXcscsort_bufferSizeExt(handle, m, n, nnz, csc_cols, csc_rows, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcscsort(mcsparseHandle_t handle, int m, int n, int nnz, mcsparseMatDescr_t mcsp_descr_A,
                                  const int* csc_cols, int* csc_rows, int* perm, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinXcscsort(handle, m, n, nnz, mcsp_descr_A, csc_cols, csc_rows, perm, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcoosort_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, const int* coo_rows,
                                                const int* coo_cols, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinXcoosort_bufferSizeExt(handle, m, n, nnz, coo_rows, coo_cols, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcoosortByRow(mcsparseHandle_t handle, int m, int n, int nnz, int* coo_rows, int* coo_cols,
                                       int* perm, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinXcoosortByRow(handle, m, n, nnz, coo_rows, coo_cols, perm, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcoosortByColumn(mcsparseHandle_t handle, int m, int n, int nnz, int* coo_rows, int* coo_cols,
                                          int* perm, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinXcoosortByColumn(handle, m, n, nnz, coo_rows, coo_cols, perm, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateIdentityPermutation(mcsparseHandle_t handle, int n, int* p) {
    mcspStatus_t ret = mcspCuinCreateIdentityPermutation(handle, n, p);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSnnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcsparseMatDescr_t mcsp_descr_A, const float* dense_matrix, int lda,
                              int* nnz_per_row_or_column, int* nnz) {
    mcspStatus_t ret = mcspCuinSnnz(handle, dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcsparseMatDescr_t mcsp_descr_A, const double* dense_matrix, int lda,
                              int* nnz_per_row_or_column, int* nnz) {
    mcspStatus_t ret = mcspCuinDnnz(handle, dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCnnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcsparseMatDescr_t mcsp_descr_A, const mcspComplexFloat* dense_matrix, int lda,
                              int* nnz_per_row_or_column, int* nnz) {
    mcspStatus_t ret = mcspCuinCnnz(handle, dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZnnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                              const mcsparseMatDescr_t mcsp_descr_A, const mcspComplexDouble* dense_matrix, int lda,
                              int* nnz_per_row_or_column, int* nnz) {
    mcspStatus_t ret = mcspCuinZnnz(handle, dir, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, nnz);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSdense2csr(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const float* dense_matrix, int lda, int* nnz_per_row_or_column, float* csr_vals,
                                    int* csr_rows, int* csr_cols) {
    mcspStatus_t ret = mcspCuinSdense2csr(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, csr_vals,
                                        csr_rows, csr_cols);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDdense2csr(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const double* dense_matrix, int lda, int* nnz_per_row_or_column, double* csr_vals,
                                    int* csr_rows, int* csr_cols) {
    mcspStatus_t ret = mcspCuinDdense2csr(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, csr_vals,
                                        csr_rows, csr_cols);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCdense2csr(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcspComplexFloat* dense_matrix, int lda, int* nnz_per_row_or_column,
                                    mcspComplexFloat* csr_vals, int* csr_rows, int* csr_cols) {
    mcspStatus_t ret = mcspCuinCdense2csr(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, csr_vals,
                                        csr_rows, csr_cols);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZdense2csr(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcspComplexDouble* dense_matrix, int lda, int* nnz_per_row_or_column,
                                    mcspComplexDouble* csr_vals, int* csr_rows, int* csr_cols) {
    mcspStatus_t ret = mcspCuinZdense2csr(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, csr_vals,
                                        csr_rows, csr_cols);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSdense2csc(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const float* dense_matrix, int lda, int* nnz_per_row_or_column, float* csc_vals,
                                    int* csc_rows, int* csc_cols) {
    mcspStatus_t ret = mcspCuinSdense2csc(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, csc_vals,
                                        csc_rows, csc_cols);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDdense2csc(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const double* dense_matrix, int lda, int* nnz_per_row_or_column, double* csc_vals,
                                    int* csc_rows, int* csc_cols) {
    mcspStatus_t ret = mcspCuinDdense2csc(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, csc_vals,
                                        csc_rows, csc_cols);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCdense2csc(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcspComplexFloat* dense_matrix, int lda, int* nnz_per_row_or_column,
                                    mcspComplexFloat* csc_vals, int* csc_rows, int* csc_cols) {
    mcspStatus_t ret = mcspCuinCdense2csc(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, csc_vals,
                                        csc_rows, csc_cols);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZdense2csc(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcspComplexDouble* dense_matrix, int lda, int* nnz_per_row_or_column,
                                    mcspComplexDouble* csc_vals, int* csc_rows, int* csc_cols) {
    mcspStatus_t ret = mcspCuinZdense2csc(handle, m, n, mcsp_descr_A, dense_matrix, lda, nnz_per_row_or_column, csc_vals,
                                        csc_rows, csc_cols);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsr2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                    float* dense_matrix, int lda) {
    mcspStatus_t ret = mcspCuinScsr2dense(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsr2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                    double* dense_matrix, int lda) {
    mcspStatus_t ret = mcspCuinDcsr2dense(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsr2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                    mcspComplexFloat* dense_matrix, int lda) {
    mcspStatus_t ret = mcspCuinCcsr2dense(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsr2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                    mcspComplexDouble* dense_matrix, int lda) {
    mcspStatus_t ret = mcspCuinZcsr2dense(handle, m, n, mcsp_descr_A, csr_vals, csr_rows, csr_cols, dense_matrix, lda);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsc2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const float* csc_vals, const int* csc_rows, const int* csc_cols,
                                    float* dense_matrix, int lda) {
    mcspStatus_t ret = mcspCuinScsc2dense(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsc2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const double* csc_vals, const int* csc_rows, const int* csc_cols,
                                    double* dense_matrix, int lda) {
    mcspStatus_t ret = mcspCuinDcsc2dense(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsc2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcspComplexFloat* csc_vals, const int* csc_rows, const int* csc_cols,
                                    mcspComplexFloat* dense_matrix, int lda) {
    mcspStatus_t ret = mcspCuinCcsc2dense(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsc2dense(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t mcsp_descr_A,
                                    const mcspComplexDouble* csc_vals, const int* csc_rows, const int* csc_cols,
                                    mcspComplexDouble* dense_matrix, int lda) {
    mcspStatus_t ret = mcspCuinZcsc2dense(handle, m, n, mcsp_descr_A, csc_vals, csc_rows, csc_cols, dense_matrix, lda);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpruneDense2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const float* dense_matrix,
                                                       int lda, const float* threshold,
                                                       const mcsparseMatDescr_t mcsp_descr_A, const float* csr_vals,
                                                       const int* csr_rows, const int* csr_cols, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinSpruneDense2csr_bufferSizeExt(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A,
                                                           csr_vals, csr_rows, csr_cols, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneDense2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n,
                                                       const double* dense_matrix, int lda, const double* threshold,
                                                       const mcsparseMatDescr_t mcsp_descr_A, const double* csr_vals,
                                                       const int* csr_rows, const int* csr_cols, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinDpruneDense2csr_bufferSizeExt(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A,
                                                           csr_vals, csr_rows, csr_cols, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpruneDense2csrNnz(mcsparseHandle_t handle, int m, int n, const float* dense_matrix, int lda,
                                            const float* threshold, const mcsparseMatDescr_t mcsp_descr_A,
                                            int* csr_rows, int* nnz, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinSpruneDense2csrNnz(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_rows, nnz, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneDense2csrNnz(mcsparseHandle_t handle, int m, int n, const double* dense_matrix, int lda,
                                            const double* threshold, const mcsparseMatDescr_t mcsp_descr_A,
                                            int* csr_rows, int* nnz, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinDpruneDense2csrNnz(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_rows, nnz, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpruneDense2csr(mcsparseHandle_t handle, int m, int n, const float* dense_matrix, int lda,
                                         const float* threshold, const mcsparseMatDescr_t mcsp_descr_A, float* csr_vals,
                                         const int* csr_rows, int* csr_cols, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinSpruneDense2csr(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_vals,
                                             csr_rows, csr_cols, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneDense2csr(mcsparseHandle_t handle, int m, int n, const double* dense_matrix, int lda,
                                         const double* threshold, const mcsparseMatDescr_t mcsp_descr_A,
                                         double* csr_vals, const int* csr_rows, int* csr_cols, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinDpruneDense2csr(handle, m, n, dense_matrix, lda, threshold, mcsp_descr_A, csr_vals,
                                             csr_rows, csr_cols, temp_buffer);
    return mcspToSparseStatus(ret);
}
#if defined(__MACA__)
mcsparseStatus_t mcsparseHpruneDense2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const __half* A, int lda,
                                                       const __half* threshold, const mcsparseMatDescr_t descrC,
                                                       const __half* csrSortedValC, const int* csrSortedRowPtrC,
                                                       const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    mcspStatus_t ret =
        mcspHpruneDense2CsrBufferSize(handle, (mcspInt)m, (mcspInt)n, A, (mcspInt)lda, threshold, descrC, csrSortedValC,
                                      (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseHpruneDense2csrNnz(mcsparseHandle_t handle, int m, int n, const __half* A, int lda,
                                            const __half* threshold, const mcsparseMatDescr_t descrC, int* csrRowPtrC,
                                            int* nnzTotalDevHostPtr, void* pBuffer) {
    mcspStatus_t ret = mcspHpruneDense2CsrNnz(handle, (mcspInt)m, (mcspInt)n, A, (mcspInt)lda, threshold, descrC,
                                              (mcspInt*)csrRowPtrC, (mcspInt*)nnzTotalDevHostPtr, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseHpruneDense2csr(mcsparseHandle_t handle, int m, int n, const __half* A, int lda,
                                         const __half* threshold, const mcsparseMatDescr_t descrC,
                                         __half* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC,
                                         void* pBuffer) {
    mcspStatus_t ret =
        mcspHpruneDense2Csr(handle, (mcspInt)m, (mcspInt)n, A, (mcspInt)lda, threshold, descrC, csrSortedValC,
                            (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBuffer);
    return mcspToSparseStatus(ret);
}
#endif
mcsparseStatus_t mcsparseSpruneDense2csrByPercentage_bufferSizeExt(mcsparseHandle_t handle, int m, int n,
                                                                   const float* dense_matrix, int lda, float percentage,
                                                                   const mcsparseMatDescr_t mcsp_descr_A,
                                                                   const float* csr_vals, const int* csr_rows,
                                                                   const int* csr_cols, const mcsparsePruneInfo_t info,
                                                                   size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinSpruneDense2csrByPercentage_bufferSizeExt(
        handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneDense2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, const double* dense_matrix, int lda, double percentage,
    const mcsparseMatDescr_t mcsp_descr_A, const double* csr_vals, const int* csr_rows, const int* csr_cols,
    const mcsparsePruneInfo_t info, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinDpruneDense2csrByPercentage_bufferSizeExt(
        handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpruneDense2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n,
                                                        const float* dense_matrix, int lda, float percentage,
                                                        const mcsparseMatDescr_t mcsp_descr_A, int* csr_rows, int* nnz,
                                                        const mcsparsePruneInfo_t info, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinSpruneDense2csrNnzByPercentage(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                            csr_rows, nnz, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneDense2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n,
                                                        const double* dense_matrix, int lda, double percentage,
                                                        const mcsparseMatDescr_t mcsp_descr_A, int* csr_rows, int* nnz,
                                                        const mcsparsePruneInfo_t info, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinDpruneDense2csrNnzByPercentage(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                            csr_rows, nnz, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpruneDense2csrByPercentage(mcsparseHandle_t handle, int m, int n, const float* dense_matrix,
                                                     int lda, float percentage, const mcsparseMatDescr_t mcsp_descr_A,
                                                     float* csr_vals, const int* csr_rows, int* csr_cols,
                                                     const mcsparsePruneInfo_t info, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinSpruneDense2csrByPercentage(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                         csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneDense2csrByPercentage(mcsparseHandle_t handle, int m, int n, const double* dense_matrix,
                                                     int lda, double percentage, const mcsparseMatDescr_t mcsp_descr_A,
                                                     double* csr_vals, const int* csr_rows, int* csr_cols,
                                                     const mcsparsePruneInfo_t info, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinDpruneDense2csrByPercentage(handle, m, n, dense_matrix, lda, percentage, mcsp_descr_A,
                                                         csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
#if defined(__MACA__)
mcsparseStatus_t mcsparseHpruneDense2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, const __half* A, int lda, float percentage, const mcsparseMatDescr_t descrC,
    const __half* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, mcsparsePruneInfo_t info,
    size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspHpruneDense2CsrByPercentageBufferSize(
        handle, (mcspInt)m, (mcspInt)n, A, (mcspInt)lda, percentage, descrC, csrSortedValC, (mcspInt*)csrSortedRowPtrC,
        (mcspInt*)csrSortedColIndC, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseHpruneDense2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n, const __half* A, int lda,
                                                        float percentage, const mcsparseMatDescr_t descrC,
                                                        int* csrRowPtrC, int* nnzTotalDevHostPtr,
                                                        mcsparsePruneInfo_t info, void* pBuffer) {
    mcspStatus_t ret =
        mcspHpruneDense2CsrNnzByPercentage(handle, (mcspInt)m, (mcspInt)n, A, (mcspInt)lda, percentage, descrC,
                                           (mcspInt*)csrRowPtrC, (mcspInt*)nnzTotalDevHostPtr, info, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseHpruneDense2csrByPercentage(mcsparseHandle_t handle, int m, int n, const __half* A, int lda,
                                                     float percentage, const mcsparseMatDescr_t descrC,
                                                     __half* csrSortedValC, const int* csrSortedRowPtrC,
                                                     int* csrSortedColIndC, mcsparsePruneInfo_t info, void* pBuffer) {
    mcspStatus_t ret = mcspHpruneDense2CsrByPercentage(handle, (mcspInt)m, (mcspInt)n, A, (mcspInt)lda, percentage,
                                                       descrC, csrSortedValC, (mcspInt*)csrSortedRowPtrC,
                                                       (mcspInt*)csrSortedColIndC, info, pBuffer);
    return mcspToSparseStatus(ret);
}
#endif
mcsparseStatus_t mcsparseSpruneCsr2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                     const mcsparseMatDescr_t mcsp_descr_A, const float* csr_val_A,
                                                     const int* csr_row_A, const int* csr_col_A, const float* tol,
                                                     const mcsparseMatDescr_t mcsp_descr_C, const float* csr_val_C,
                                                     const int* csr_row_C, const int* csr_col_C, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinSpruneCsr2csr_bufferSizeExt(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                          mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneCsr2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                     const mcsparseMatDescr_t mcsp_descr_A, const double* csr_val_A,
                                                     const int* csr_row_A, const int* csr_col_A, const double* tol,
                                                     const mcsparseMatDescr_t mcsp_descr_C, const double* csr_val_C,
                                                     const int* csr_row_C, const int* csr_col_C, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinDpruneCsr2csr_bufferSizeExt(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                          mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpruneCsr2csrNnz(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                          const mcsparseMatDescr_t mcsp_descr_A, const float* csr_val_A,
                                          const int* csr_row_A, const int* csr_col_A, const float* tol,
                                          const mcsparseMatDescr_t mcsp_descr_C, int* csr_row_C, int* nnz_C,
                                          void* temp_buffer) {
    mcspStatus_t ret = mcspCuinSpruneCsr2csrNnz(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                              mcsp_descr_C, csr_row_C, nnz_C, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneCsr2csrNnz(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                          const mcsparseMatDescr_t mcsp_descr_A, const double* csr_val_A,
                                          const int* csr_row_A, const int* csr_col_A, const double* tol,
                                          const mcsparseMatDescr_t mcsp_descr_C, int* csr_row_C, int* nnz_C,
                                          void* temp_buffer) {
    mcspStatus_t ret = mcspCuinDpruneCsr2csrNnz(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                              mcsp_descr_C, csr_row_C, nnz_C, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpruneCsr2csr(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                       const mcsparseMatDescr_t mcsp_descr_A, const float* csr_val_A,
                                       const int* csr_row_A, const int* csr_col_A, const float* tol,
                                       const mcsparseMatDescr_t mcsp_descr_C, float* csr_val_C, const int* csr_row_C,
                                       int* csr_col_C, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinSpruneCsr2csr(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                           mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneCsr2csr(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                       const mcsparseMatDescr_t mcsp_descr_A, const double* csr_val_A,
                                       const int* csr_row_A, const int* csr_col_A, const double* tol,
                                       const mcsparseMatDescr_t mcsp_descr_C, double* csr_val_C, const int* csr_row_C,
                                       int* csr_col_C, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinDpruneCsr2csr(handle, m, n, nnz_A, mcsp_descr_A, csr_val_A, csr_row_A, csr_col_A, tol,
                                           mcsp_descr_C, csr_val_C, csr_row_C, csr_col_C, temp_buffer);
    return mcspToSparseStatus(ret);
}
#if defined(__MACA__)
mcsparseStatus_t mcsparseHpruneCsr2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnzA,
                                                     const mcsparseMatDescr_t descrA, const __half* csrSortedValA,
                                                     const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                     const __half* threshold, const mcsparseMatDescr_t descrC,
                                                     const __half* csrSortedValC, const int* csrSortedRowPtrC,
                                                     const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspHpruneCsr2csrBufferSize(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnzA, descrA, csrSortedValA,
                                                   (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, threshold,
                                                   descrC, csrSortedValC, (mcspInt*)csrSortedRowPtrC,
                                                   (mcspInt*)csrSortedColIndC, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseHpruneCsr2csrNnz(mcsparseHandle_t handle, int m, int n, int nnzA,
                                          const mcsparseMatDescr_t descrA, const __half* csrSortedValA,
                                          const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                          const __half* threshold, const mcsparseMatDescr_t descrC,
                                          int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* pBuffer) {
    mcspStatus_t ret = mcspHpruneCsr2csrNnz(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnzA, descrA, csrSortedValA,
                                            (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, threshold, descrC,
                                            (mcspInt*)csrSortedRowPtrC, (mcspInt*)nnzTotalDevHostPtr, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseHpruneCsr2csr(mcsparseHandle_t handle, int m, int n, int nnzA, const mcsparseMatDescr_t descrA,
                                       const __half* csrSortedValA, const int* csrSortedRowPtrA,
                                       const int* csrSortedColIndA, const __half* threshold,
                                       const mcsparseMatDescr_t descrC, __half* csrSortedValC,
                                       const int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) {
    mcspStatus_t ret =
        mcspHpruneCsr2csr(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnzA, descrA, csrSortedValA,
                          (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, threshold, descrC, csrSortedValC,
                          (mcspInt*)csrSortedRowPtrC, (mcspInt*)csrSortedColIndC, pBuffer);
    return mcspToSparseStatus(ret);
}
#endif
mcsparseStatus_t mcsparseSpruneCsr2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, int nnz_A, const mcsparseMatDescr_t mcsp_descr_A, const float* csr_vals_A,
    const int* csr_rows_A, const int* csr_cols_A, float percentage, const mcsparseMatDescr_t mcsp_descr_C,
    const float* csr_vals_C, const int* csr_rows_C, const int* csr_cols_C, const mcsparsePruneInfo_t info,
    size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinSpruneCsr2csrByPercentage_bufferSizeExt(
        handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A, csr_cols_A, percentage, mcsp_descr_C, csr_vals_C,
        csr_rows_C, csr_cols_C, info, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneCsr2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, int nnz_A, const mcsparseMatDescr_t mcsp_descr_A, const double* csr_vals_A,
    const int* csr_rows_A, const int* csr_cols_A, double percentage, const mcsparseMatDescr_t mcsp_descr_C,
    const double* csr_vals_C, const int* csr_rows_C, const int* csr_cols_C, const mcsparsePruneInfo_t info,
    size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinDpruneCsr2csrByPercentage_bufferSizeExt(
        handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A, csr_cols_A, percentage, mcsp_descr_C, csr_vals_C,
        csr_rows_C, csr_cols_C, info, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpruneCsr2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                      const mcsparseMatDescr_t mcsp_descr_A, const float* csr_vals_A,
                                                      const int* csr_rows_A, const int* csr_cols_A, float percentage,
                                                      const mcsparseMatDescr_t mcsp_descr_C, int* csr_rows_C,
                                                      int* nnz_C, const mcsparsePruneInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinSpruneCsr2csrNnzByPercentage(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A, csr_cols_A,
                                           percentage, mcsp_descr_C, csr_rows_C, nnz_C, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneCsr2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                      const mcsparseMatDescr_t mcsp_descr_A, const double* csr_vals_A,
                                                      const int* csr_rows_A, const int* csr_cols_A, double percentage,
                                                      const mcsparseMatDescr_t mcsp_descr_C, int* csr_rows_C,
                                                      int* nnz_C, const mcsparsePruneInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinDpruneCsr2csrNnzByPercentage(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A, csr_cols_A,
                                           percentage, mcsp_descr_C, csr_rows_C, nnz_C, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpruneCsr2csrByPercentage(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                   const mcsparseMatDescr_t mcsp_descr_A, const float* csr_vals_A,
                                                   const int* csr_rows_A, const int* csr_cols_A, float percentage,
                                                   const mcsparseMatDescr_t mcsp_descr_C, float* csr_vals_C,
                                                   const int* csr_rows_C, int* csr_cols_C,
                                                   const mcsparsePruneInfo_t info, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinSpruneCsr2csrByPercentage(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A,
                                                       csr_cols_A, percentage, mcsp_descr_C, csr_vals_C, csr_rows_C,
                                                       csr_cols_C, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDpruneCsr2csrByPercentage(mcsparseHandle_t handle, int m, int n, int nnz_A,
                                                   const mcsparseMatDescr_t mcsp_descr_A, const double* csr_vals_A,
                                                   const int* csr_rows_A, const int* csr_cols_A, double percentage,
                                                   const mcsparseMatDescr_t mcsp_descr_C, double* csr_vals_C,
                                                   const int* csr_rows_C, int* csr_cols_C,
                                                   const mcsparsePruneInfo_t info, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinDpruneCsr2csrByPercentage(handle, m, n, nnz_A, mcsp_descr_A, csr_vals_A, csr_rows_A,
                                                       csr_cols_A, percentage, mcsp_descr_C, csr_vals_C, csr_rows_C,
                                                       csr_cols_C, info, temp_buffer);
    return mcspToSparseStatus(ret);
}

#if defined(__MACA__)
mcsparseStatus_t mcsparseHpruneCsr2csrByPercentage_bufferSizeExt(
    mcsparseHandle_t handle, int m, int n, int nnzA, const mcsparseMatDescr_t descrA, const __half* csrSortedValA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const mcsparseMatDescr_t descrC,
    const __half* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, mcsparsePruneInfo_t info,
    size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspHpruneCsr2csrByPercentageBufferSize(
        handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnzA, descrA, csrSortedValA, (mcspInt*)csrSortedRowPtrA,
        (mcspInt*)csrSortedColIndA, percentage, descrC, csrSortedValC, (mcspInt*)csrSortedRowPtrC,
        (mcspInt*)csrSortedColIndC, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}

mcsparseStatus_t mcsparseHpruneCsr2csrNnzByPercentage(mcsparseHandle_t handle, int m, int n, int nnzA,
                                                      const mcsparseMatDescr_t descrA, const __half* csrSortedValA,
                                                      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                      float percentage, const mcsparseMatDescr_t descrC,
                                                      int* csrSortedRowPtrC, int* nnzTotalDevHostPtr,
                                                      mcsparsePruneInfo_t info, void* pBuffer) {
    mcspStatus_t ret =
        mcspHpruneCsr2csrNnzByPercentage(handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnzA, descrA, csrSortedValA,
                                         (mcspInt*)csrSortedRowPtrA, (mcspInt*)csrSortedColIndA, percentage, descrC,
                                         (mcspInt*)csrSortedRowPtrC, (mcspInt*)nnzTotalDevHostPtr, info, pBuffer);
    return mcspToSparseStatus(ret);
}

mcsparseStatus_t mcsparseHpruneCsr2csrByPercentage(mcsparseHandle_t handle, int m, int n, int nnzA,
                                                   const mcsparseMatDescr_t descrA, const __half* csrSortedValA,
                                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                   float percentage, const mcsparseMatDescr_t descrC,
                                                   __half* csrSortedValC, const int* csrSortedRowPtrC,
                                                   int* csrSortedColIndC, mcsparsePruneInfo_t info, void* pBuffer) {
    mcspStatus_t ret = mcspHpruneCsr2csrByPercentage(
        handle, (mcspInt)m, (mcspInt)n, (mcspInt)nnzA, descrA, csrSortedValA, (mcspInt*)csrSortedRowPtrA,
        (mcspInt*)csrSortedColIndA, percentage, descrC, csrSortedValC, (mcspInt*)csrSortedRowPtrC,
        (mcspInt*)csrSortedColIndC, info, pBuffer);
    return mcspToSparseStatus(ret);
}
#endif

mcsparseStatus_t mcsparseScsru2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, float* csr_vals,
                                                 const int* csr_rows, int* csr_cols, mcsparseCsru2csrInfo_t info,
                                                 size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinScsru2csr_bufferSizeExt(handle, m, n, nnz, csr_vals, csr_rows, csr_cols, info, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsru2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz, double* csr_vals,
                                                 const int* csr_rows, int* csr_cols, mcsparseCsru2csrInfo_t info,
                                                 size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinDcsru2csr_bufferSizeExt(handle, m, n, nnz, csr_vals, csr_rows, csr_cols, info, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsru2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz,
                                                 mcspComplexFloat* csr_vals, const int* csr_rows, int* csr_cols,
                                                 mcsparseCsru2csrInfo_t info, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinCcsru2csr_bufferSizeExt(handle, m, n, nnz, csr_vals, csr_rows, csr_cols, info, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsru2csr_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int nnz,
                                                 mcspComplexDouble* csr_vals, const int* csr_rows, int* csr_cols,
                                                 mcsparseCsru2csrInfo_t info, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinZcsru2csr_bufferSizeExt(handle, m, n, nnz, csr_vals, csr_rows, csr_cols, info, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsru2csr(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, float* csr_vals, const int* csr_rows,
                                   int* csr_cols, mcsparseCsru2csrInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinScsru2csr(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsru2csr(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, double* csr_vals, const int* csr_rows,
                                   int* csr_cols, mcsparseCsru2csrInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinDcsru2csr(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsru2csr(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, mcspComplexFloat* csr_vals,
                                   const int* csr_rows, int* csr_cols, mcsparseCsru2csrInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinCcsru2csr(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsru2csr(mcsparseHandle_t handle, int m, int n, int nnz,
                                   const mcsparseMatDescr_t mcsp_descr_A, mcspComplexDouble* csr_vals,
                                   const int* csr_rows, int* csr_cols, mcsparseCsru2csrInfo_t info, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinZcsru2csr(handle, m, n, nnz, mcsp_descr_A, csr_vals, csr_rows, csr_cols, info, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                               const mcsparseMatDescr_t csr_descr, const float* csr_val,
                                               const int* csr_row, const int* csr_col, int row_block_dim,
                                               int col_block_dim, int* buffer_size) {
    mcspStatus_t ret = mcspCuinScsr2gebsr_bufferSize(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col,
                                                   row_block_dim, col_block_dim, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                               const mcsparseMatDescr_t csr_descr, const double* csr_val,
                                               const int* csr_row, const int* csr_col, int row_block_dim,
                                               int col_block_dim, int* buffer_size) {
    mcspStatus_t ret = mcspCuinDcsr2gebsr_bufferSize(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col,
                                                   row_block_dim, col_block_dim, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                               const mcsparseMatDescr_t csr_descr, const mcspComplexFloat* csr_val,
                                               const int* csr_row, const int* csr_col, int row_block_dim,
                                               int col_block_dim, int* buffer_size) {
    mcspStatus_t ret = mcspCuinCcsr2gebsr_bufferSize(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col,
                                                   row_block_dim, col_block_dim, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsr2gebsr_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                               const mcsparseMatDescr_t csr_descr, const mcspComplexDouble* csr_val,
                                               const int* csr_row, const int* csr_col, int row_block_dim,
                                               int col_block_dim, int* buffer_size) {
    mcspStatus_t ret = mcspCuinZcsr2gebsr_bufferSize(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col,
                                                   row_block_dim, col_block_dim, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                                  const mcsparseMatDescr_t csr_descr, const float* csr_val,
                                                  const int* csr_row, const int* csr_col, int row_block_dim,
                                                  int col_block_dim, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinScsr2gebsr_bufferSizeExt(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col,
                                                      row_block_dim, col_block_dim, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                                  const mcsparseMatDescr_t csr_descr, const double* csr_val,
                                                  const int* csr_row, const int* csr_col, int row_block_dim,
                                                  int col_block_dim, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinDcsr2gebsr_bufferSizeExt(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col,
                                                      row_block_dim, col_block_dim, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                                  const mcsparseMatDescr_t csr_descr, const mcspComplexFloat* csr_val,
                                                  const int* csr_row, const int* csr_col, int row_block_dim,
                                                  int col_block_dim, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinCcsr2gebsr_bufferSizeExt(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col,
                                                      row_block_dim, col_block_dim, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsr2gebsr_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                                  const mcsparseMatDescr_t csr_descr, const mcspComplexDouble* csr_val,
                                                  const int* csr_row, const int* csr_col, int row_block_dim,
                                                  int col_block_dim, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinZcsr2gebsr_bufferSizeExt(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col,
                                                      row_block_dim, col_block_dim, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsr2gebsrNnz(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                       const mcsparseMatDescr_t csr_descr, const int* csr_row, const int* csr_col,
                                       const mcsparseMatDescr_t bsr_descr, int* bsr_row, int row_block_dim,
                                       int col_block_dim, int* nnzb, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinXcsr2gebsrNnz(handle, dir, m, n, csr_descr, csr_row, csr_col, bsr_descr, bsr_row,
                                           row_block_dim, col_block_dim, nnzb, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                    const mcsparseMatDescr_t csr_descr, const float* csr_val, const int* csr_row,
                                    const int* csr_col, const mcsparseMatDescr_t bsr_descr, float* bsr_val,
                                    int* bsr_row, int* bsr_col, int row_block_dim, int col_block_dim,
                                    void* temp_buffer) {
    mcspStatus_t ret = mcspCuinScsr2gebsr(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, bsr_descr, bsr_val,
                                        bsr_row, bsr_col, row_block_dim, col_block_dim, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                    const mcsparseMatDescr_t csr_descr, const double* csr_val, const int* csr_row,
                                    const int* csr_col, const mcsparseMatDescr_t bsr_descr, double* bsr_val,
                                    int* bsr_row, int* bsr_col, int row_block_dim, int col_block_dim,
                                    void* temp_buffer) {
    mcspStatus_t ret = mcspCuinDcsr2gebsr(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, bsr_descr, bsr_val,
                                        bsr_row, bsr_col, row_block_dim, col_block_dim, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                    const mcsparseMatDescr_t csr_descr, const mcspComplexFloat* csr_val,
                                    const int* csr_row, const int* csr_col, const mcsparseMatDescr_t bsr_descr,
                                    mcspComplexFloat* bsr_val, int* bsr_row, int* bsr_col, int row_block_dim,
                                    int col_block_dim, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinCcsr2gebsr(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, bsr_descr, bsr_val,
                                        bsr_row, bsr_col, row_block_dim, col_block_dim, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsr2gebsr(mcsparseHandle_t handle, mcsparseDirection_t dir, int m, int n,
                                    const mcsparseMatDescr_t csr_descr, const mcspComplexDouble* csr_val,
                                    const int* csr_row, const int* csr_col, const mcsparseMatDescr_t bsr_descr,
                                    mcspComplexDouble* bsr_val, int* bsr_row, int* bsr_col, int row_block_dim,
                                    int col_block_dim, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinZcsr2gebsr(handle, dir, m, n, csr_descr, csr_val, csr_row, csr_col, bsr_descr, bsr_val,
                                        bsr_row, bsr_col, row_block_dim, col_block_dim, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsr2bsrNnz(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                     const mcsparseMatDescr_t descrA, const int* csrSortedRowPtrA,
                                     const int* csrSortedColIndA, int blockDim, const mcsparseMatDescr_t descrC,
                                     int* bsrSortedRowPtrC, int* nnzTotalDevHostPtr) {
    mcspStatus_t ret = mcspCuinXcsr2bsrNnz(handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, blockDim,
                                         descrC, bsrSortedRowPtrC, nnzTotalDevHostPtr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsr2bsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                  const mcsparseMatDescr_t descrA, const float* csrSortedValA,
                                  const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC,
                                  int* bsrSortedColIndC) {
    mcspStatus_t ret = mcspCuinScsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                      blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsr2bsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                  const mcsparseMatDescr_t descrA, const double* csrSortedValA,
                                  const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC,
                                  int* bsrSortedColIndC) {
    mcspStatus_t ret = mcspCuinDcsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                      blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsr2bsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                  const mcsparseMatDescr_t descrA, const mcFloatComplex* csrSortedValA,
                                  const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, mcFloatComplex* bsrSortedValC, int* bsrSortedRowPtrC,
                                  int* bsrSortedColIndC) {
    mcspStatus_t ret = mcspCuinCcsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                      blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsr2bsr(mcsparseHandle_t handle, mcsparseDirection_t dirA, int m, int n,
                                  const mcsparseMatDescr_t descrA, const mcDoubleComplex* csrSortedValA,
                                  const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim,
                                  const mcsparseMatDescr_t descrC, mcDoubleComplex* bsrSortedValC,
                                  int* bsrSortedRowPtrC, int* bsrSortedColIndC) {
    mcspStatus_t ret = mcspCuinZcsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                      blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrgeam2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const float* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                                 const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                 const float* beta, const mcsparseMatDescr_t descrB, int nnzB,
                                                 const float* csrSortedValB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                                 const float* csrSortedValC, const int* csrSortedRowPtrC,
                                                 const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinScsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                                                     csrSortedColIndA, beta, descrB, nnzB, csrSortedValB,
                                                     csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                                                     csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrgeam2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const double* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                                 const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                 const double* beta, const mcsparseMatDescr_t descrB, int nnzB,
                                                 const double* csrSortedValB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                                 const double* csrSortedValC, const int* csrSortedRowPtrC,
                                                 const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinDcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                                                     csrSortedColIndA, beta, descrB, nnzB, csrSortedValB,
                                                     csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                                                     csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrgeam2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcspComplexFloat* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA,
                                                 const mcspComplexFloat* csrSortedValA, const int* csrSortedRowPtrA,
                                                 const int* csrSortedColIndA, const mcspComplexFloat* beta,
                                                 const mcsparseMatDescr_t descrB, int nnzB,
                                                 const mcspComplexFloat* csrSortedValB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                                 const mcspComplexFloat* csrSortedValC, const int* csrSortedRowPtrC,
                                                 const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinCcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                                                     csrSortedColIndA, beta, descrB, nnzB, csrSortedValB,
                                                     csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                                                     csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrgeam2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcspComplexDouble* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA,
                                                 const mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                                 const int* csrSortedColIndA, const mcspComplexDouble* beta,
                                                 const mcsparseMatDescr_t descrB, int nnzB,
                                                 const mcspComplexDouble* csrSortedValB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                                 const mcspComplexDouble* csrSortedValC, const int* csrSortedRowPtrC,
                                                 const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinZcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                                                     csrSortedColIndA, beta, descrB, nnzB, csrSortedValB,
                                                     csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                                                     csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsrgeam2Nnz(mcsparseHandle_t handle, int m, int n, const mcsparseMatDescr_t descrA, int nnzA,
                                      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      const mcsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
                                      const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                      int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace) {
    mcspStatus_t ret =
        mcspCuinXcsrgeam2Nnz(handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                           csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrgeam2(mcsparseHandle_t handle, int m, int n, const float* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta,
                                   const mcsparseMatDescr_t descrB, int nnzB, const float* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const mcsparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, void* pBuffer) {
    mcspStatus_t ret =
        mcspCuinScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                        descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                        csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrgeam2(mcsparseHandle_t handle, int m, int n, const double* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta,
                                   const mcsparseMatDescr_t descrB, int nnzB, const double* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const mcsparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, void* pBuffer) {
    mcspStatus_t ret =
        mcspCuinDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                        descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                        csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrgeam2(mcsparseHandle_t handle, int m, int n, const mcspComplexFloat* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const mcspComplexFloat* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcspComplexFloat* beta, const mcsparseMatDescr_t descrB, int nnzB,
                                   const mcspComplexFloat* csrSortedValB, const int* csrSortedRowPtrB,
                                   const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                   mcspComplexFloat* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,
                                   void* pBuffer) {
    mcspStatus_t ret =
        mcspCuinCcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                        descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                        csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrgeam2(mcsparseHandle_t handle, int m, int n, const mcspComplexDouble* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const mcspComplexDouble* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcspComplexDouble* beta, const mcsparseMatDescr_t descrB, int nnzB,
                                   const mcspComplexDouble* csrSortedValB, const int* csrSortedRowPtrB,
                                   const int* csrSortedColIndB, const mcsparseMatDescr_t descrC,
                                   mcspComplexDouble* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,
                                   void* pBuffer) {
    mcspStatus_t ret =
        mcspCuinZcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                        descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                        csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrgemm2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int k, const float* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
                                                 const int* csrSortedColIndA, const mcsparseMatDescr_t descrB, int nnzB,
                                                 const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                                 const float* beta, const mcsparseMatDescr_t descrD, int nnzD,
                                                 const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                                 mcsparseCsrgemm2Info_t info, size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinScsrgemm2_bufferSizeExt(
        handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
        csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrgemm2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int k, const double* alpha,
                                                 const mcsparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA,
                                                 const int* csrSortedColIndA, const mcsparseMatDescr_t descrB, int nnzB,
                                                 const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                                 const double* beta, const mcsparseMatDescr_t descrD, int nnzD,
                                                 const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                                 mcsparseCsrgemm2Info_t info, size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinDcsrgemm2_bufferSizeExt(
        handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
        csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrgemm2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int k,
                                                 const mcspComplexFloat* alpha, const mcsparseMatDescr_t descrA,
                                                 int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                 const mcsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcspComplexFloat* beta,
                                                 const mcsparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD,
                                                 const int* csrSortedColIndD, mcsparseCsrgemm2Info_t info,
                                                 size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinCcsrgemm2_bufferSizeExt(
        handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
        csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrgemm2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, int k,
                                                 const mcspComplexDouble* alpha, const mcsparseMatDescr_t descrA,
                                                 int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                                 const mcsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
                                                 const int* csrSortedColIndB, const mcspComplexDouble* beta,
                                                 const mcsparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD,
                                                 const int* csrSortedColIndD, mcsparseCsrgemm2Info_t info,
                                                 size_t* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinZcsrgemm2_bufferSizeExt(
        handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
        csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsrgemm2Nnz(mcsparseHandle_t handle, int m, int n, int k, const mcsparseMatDescr_t descrA,
                                      int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                      const mcsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
                                      const int* csrSortedColIndB, const mcsparseMatDescr_t descrD, int nnzD,
                                      const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                      const mcsparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr,
                                      const mcsparseCsrgemm2Info_t info, void* pBuffer) {
    mcspStatus_t ret =
        mcspCuinXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                           csrSortedRowPtrB, csrSortedColIndB, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, descrC,
                           csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrgemm2(mcsparseHandle_t handle, int m, int n, int k, const float* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const float* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcsparseMatDescr_t descrB, int nnzB, const float* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB, const float* beta,
                                   const mcsparseMatDescr_t descrD, int nnzD, const float* csrSortedValD,
                                   const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                   const mcsparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, const mcsparseCsrgemm2Info_t info, void* pBuffer) {
    mcspStatus_t ret = mcspCuinScsrgemm2(
        handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
        csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
        csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrgemm2(mcsparseHandle_t handle, int m, int n, int k, const double* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const double* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcsparseMatDescr_t descrB, int nnzB, const double* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB, const double* beta,
                                   const mcsparseMatDescr_t descrD, int nnzD, const double* csrSortedValD,
                                   const int* csrSortedRowPtrD, const int* csrSortedColIndD,
                                   const mcsparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC,
                                   int* csrSortedColIndC, const mcsparseCsrgemm2Info_t info, void* pBuffer) {
    mcspStatus_t ret = mcspCuinDcsrgemm2(
        handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
        csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
        csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrgemm2(mcsparseHandle_t handle, int m, int n, int k, const mcspComplexFloat* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const mcspComplexFloat* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcsparseMatDescr_t descrB, int nnzB, const mcspComplexFloat* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const mcspComplexFloat* beta, const mcsparseMatDescr_t descrD, int nnzD,
                                   const mcspComplexFloat* csrSortedValD, const int* csrSortedRowPtrD,
                                   const int* csrSortedColIndD, const mcsparseMatDescr_t descrC,
                                   mcspComplexFloat* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC,
                                   const mcsparseCsrgemm2Info_t info, void* pBuffer) {
    mcspStatus_t ret = mcspCuinCcsrgemm2(
        handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
        csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
        csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrgemm2(mcsparseHandle_t handle, int m, int n, int k, const mcspComplexDouble* alpha,
                                   const mcsparseMatDescr_t descrA, int nnzA, const mcspComplexDouble* csrSortedValA,
                                   const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   const mcsparseMatDescr_t descrB, int nnzB, const mcspComplexDouble* csrSortedValB,
                                   const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                                   const mcspComplexDouble* beta, const mcsparseMatDescr_t descrD, int nnzD,
                                   const mcspComplexDouble* csrSortedValD, const int* csrSortedRowPtrD,
                                   const int* csrSortedColIndD, const mcsparseMatDescr_t descrC,
                                   mcspComplexDouble* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC,
                                   const mcsparseCsrgemm2Info_t info, void* pBuffer) {
    mcspStatus_t ret = mcspCuinZcsrgemm2(
        handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
        csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
        csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseAxpby(mcsparseHandle_t handle, const void* alpha, mcsparseSpVecDescr_t vecX, const void* beta,
                               mcsparseDnVecDescr_t vecY) {
    mcspStatus_t ret = mcspAxpby(handle, alpha, vecX, beta, vecY);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseGather(mcsparseHandle_t handle, mcsparseDnVecDescr_t vecY, mcsparseSpVecDescr_t vecX) {
    mcspStatus_t ret = mcspGather(handle, vecY, vecX);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScatter(mcsparseHandle_t handle, mcsparseSpVecDescr_t vecX, mcsparseDnVecDescr_t vecY) {
    mcspStatus_t ret = mcspScatter(handle, vecX, vecY);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseRot(mcsparseHandle_t handle, const void* c_coeff, const void* s_coeff,
                             mcsparseSpVecDescr_t vecX, mcsparseDnVecDescr_t vecY) {
    mcspStatus_t ret = mcspRot(handle, c_coeff, s_coeff, vecX, vecY);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSparseToDense_bufferSize(mcsparseHandle_t handle, mcsparseSpMatDescr_t matA,
                                                  mcsparseDnMatDescr_t matB, mcsparseSparseToDenseAlg_t alg,
                                                  size_t* bufferSize) {
    mcspStatus_t ret = mcspSparseToDense_bufferSize(handle, matA, matB, alg, bufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSparseToDense(mcsparseHandle_t handle, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                       mcsparseSparseToDenseAlg_t alg, void* externalBuffer) {
    mcspStatus_t ret = mcspSparseToDense(handle, matA, matB, alg, externalBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDenseToSparse_bufferSize(mcsparseHandle_t handle, mcsparseDnMatDescr_t matA,
                                                  mcsparseSpMatDescr_t matB, mcsparseDenseToSparseAlg_t alg,
                                                  size_t* bufferSize) {
    mcspStatus_t ret = mcspDenseToSparse_bufferSize(handle, matA, matB, alg, bufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDenseToSparse_analysis(mcsparseHandle_t handle, mcsparseDnMatDescr_t matA,
                                                mcsparseSpMatDescr_t matB, mcsparseDenseToSparseAlg_t alg,
                                                void* externalBuffer) {
    mcspStatus_t ret = mcspDenseToSparse_analysis(handle, matA, matB, alg, externalBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDenseToSparse_convert(mcsparseHandle_t handle, mcsparseDnMatDescr_t matA,
                                               mcsparseSpMatDescr_t matB, mcsparseDenseToSparseAlg_t alg,
                                               void* externalBuffer) {
    mcspStatus_t ret = mcspDenseToSparse_convert(handle, matA, matB, alg, externalBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMV(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                              mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX, const void* beta,
                              mcsparseDnVecDescr_t vecY, macaDataType computeType, mcsparseSpMVAlg_t alg,
                              void* externalBuffer) {
    mcspStatus_t ret = mcspSpMV(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMV_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                                         mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX, const void* beta,
                                         mcsparseDnVecDescr_t vecY, macaDataType computeType, mcsparseSpMVAlg_t alg,
                                         size_t* bufferSize) {
    mcspStatus_t ret = mcspSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMM_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                         const void* beta, mcsparseDnMatDescr_t matC, macaDataType computeType,
                                         mcsparseSpMMAlg_t alg, size_t* bufferSize) {
    mcspStatus_t ret =
        mcspSpMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMM_preprocess(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                         const void* beta, mcsparseDnMatDescr_t matC, macaDataType computeType,
                                         mcsparseSpMMAlg_t alg, void* externalBuffer) {
    mcspStatus_t ret =
        mcspSpMM_preprocess(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMM(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                              const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB, const void* beta,
                              mcsparseDnMatDescr_t matC, macaDataType computeType, mcsparseSpMMAlg_t alg,
                              void* externalBuffer) {
    mcspStatus_t ret = mcspSpMM(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpGEMM_createDescr(mcsparseSpGEMMDescr_t* spgemm_descr) {
    mcspStatus_t ret = mcspSpGEMM_createDescr(spgemm_descr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpGEMM_destroyDescr(mcsparseSpGEMMDescr_t spgemm_descr) {
    mcspStatus_t ret = mcspSpGEMM_destroyDescr(spgemm_descr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpGEMM_workEstimation(mcsparseHandle_t handle, mcsparseOperation_t opA,
                                               mcsparseOperation_t opB, const void* alpha, mcsparseSpMatDescr_t matA,
                                               mcsparseSpMatDescr_t matB, const void* beta, mcsparseSpMatDescr_t matC,
                                               macaDataType computeType, mcsparseSpGEMMAlg_t alg,
                                               mcsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1,
                                               void* externalBuffer1) {
    mcspStatus_t ret = mcspSpGEMM_workEstimation(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
                                                 spgemmDescr, bufferSize1, externalBuffer1);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpGEMM_compute(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                        const void* alpha, mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                        const void* beta, mcsparseSpMatDescr_t matC, macaDataType computeType,
                                        mcsparseSpGEMMAlg_t alg, mcsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2,
                                        void* externalBuffer2) {
    mcspStatus_t ret = mcspSpGEMM_compute(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
                                          spgemmDescr, bufferSize2, externalBuffer2);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpGEMM_copy(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                     const void* alpha, mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                     const void* beta, mcsparseSpMatDescr_t matC, macaDataType computeType,
                                     mcsparseSpGEMMAlg_t alg, mcsparseSpGEMMDescr_t spgemmDescr) {
    mcspStatus_t ret = mcspSpGEMM_copy(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpGEMMreuse_workEstimation(mcsparseHandle_t handle, mcsparseOperation_t opA,
                                                    mcsparseOperation_t opB, mcsparseSpMatDescr_t matA,
                                                    mcsparseSpMatDescr_t matB, mcsparseSpMatDescr_t matC,
                                                    mcsparseSpGEMMAlg_t alg, mcsparseSpGEMMDescr_t spgemmDescr,
                                                    size_t* bufferSize1, void* externalBuffer1) {
    mcspStatus_t ret = mcspSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize1,
                                                      externalBuffer1);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpGEMMreuse_nnz(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                         mcsparseSpMatDescr_t matC, mcsparseSpGEMMAlg_t alg,
                                         mcsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2,
                                         size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize4,
                                         void* externalBuffer4) {
    mcspStatus_t ret = mcspSpGEMMreuse_nnz(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize2,
                                           externalBuffer2, bufferSize3, externalBuffer3, bufferSize4, externalBuffer4);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpGEMMreuse_copy(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                          mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                          mcsparseSpMatDescr_t matC, mcsparseSpGEMMAlg_t alg,
                                          mcsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize5,
                                          void* externalBuffer5) {
    mcspStatus_t ret =
        mcspSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize5, externalBuffer5);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpGEMMreuse_compute(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                             const void* alpha, mcsparseSpMatDescr_t matA, mcsparseSpMatDescr_t matB,
                                             const void* beta, mcsparseSpMatDescr_t matC, macaDataType computeType,
                                             mcsparseSpGEMMAlg_t alg, mcsparseSpGEMMDescr_t spgemmDescr) {
    mcspStatus_t ret =
        mcspSpGEMMreuse_compute(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSDDMM_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                          const void* alpha, const mcsparseDnMatDescr_t A, const mcsparseDnMatDescr_t B,
                                          const void* beta, mcsparseSpMatDescr_t C, macaDataType compute_type,
                                          mcsparseSDDMMAlg_t alg, size_t* buffer_size) {
    mcspStatus_t ret = mcspSddmmBufferSize(handle, opA, opB, alpha, A, B, beta, C, compute_type, alg, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSDDMM_preprocess(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                          const void* alpha, const mcsparseDnMatDescr_t A, const mcsparseDnMatDescr_t B,
                                          const void* beta, mcsparseSpMatDescr_t C, macaDataType compute_type,
                                          mcsparseSDDMMAlg_t alg, void* temp_buffer) {
    mcspStatus_t ret = mcspSddmmPreprocess(handle, opA, opB, alpha, A, B, beta, C, compute_type, alg, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSDDMM(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                               const void* alpha, const mcsparseDnMatDescr_t A, const mcsparseDnMatDescr_t B,
                               const void* beta, mcsparseSpMatDescr_t C, macaDataType compute_type,
                               mcsparseSDDMMAlg_t alg, void* temp_buffer) {
    mcspStatus_t ret = mcspSddmm(handle, opA, opB, alpha, A, B, beta, C, compute_type, alg, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpVV_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t op_x, mcsparseSpVecDescr_t vec_x,
                                         mcsparseDnVecDescr_t vec_y, void* result, macaDataType compute_type,
                                         size_t* buffer_size) {
    mcspStatus_t ret = mcspSpVV_bufferSize(handle, op_x, vec_x, vec_y, result, compute_type, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpVV(mcsparseHandle_t handle, mcsparseOperation_t op_x, mcsparseSpVecDescr_t vec_x,
                              mcsparseDnVecDescr_t vec_y, void* result, macaDataType compute_type, void* temp_buffer) {
    mcspStatus_t ret = mcspSpVV(handle, op_x, vec_x, vec_y, result, compute_type, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSV_createDescr(mcsparseSpSVDescr_t* descr) {
    mcspStatus_t ret = mcspSpSV_createDescr(descr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSV_destroyDescr(mcsparseSpSVDescr_t descr) {
    mcspStatus_t ret = mcspSpSV_destroyDescr(descr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSV_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                                         mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX,
                                         mcsparseDnVecDescr_t vecY, macaDataType computeType, mcsparseSpSVAlg_t alg,
                                         mcsparseSpSVDescr_t spsvDescr, size_t* bufferSize) {
    mcspStatus_t ret =
        mcspCuinSpSV_bufferSize(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, bufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSV_analysis(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                                       mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX, mcsparseDnVecDescr_t vecY,
                                       macaDataType computeType, mcsparseSpSVAlg_t alg, mcsparseSpSVDescr_t spsvDescr,
                                       void* externalBuffer) {
    mcspStatus_t ret =
        mcspCuinSpSV_analysis(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, externalBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSV_solve(mcsparseHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                                    mcsparseSpMatDescr_t matA, mcsparseDnVecDescr_t vecX, mcsparseDnVecDescr_t vecY,
                                    macaDataType computeType, mcsparseSpSVAlg_t alg, mcsparseSpSVDescr_t spsvDescr) {
    mcspStatus_t ret = mcspCuinSpSV_solve(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSM_createDescr(mcsparseSpSMDescr_t* descr) {
    mcspStatus_t ret = mcspSpSM_createDescr(descr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSM_destroyDescr(mcsparseSpSMDescr_t descr) {
    mcspStatus_t ret = mcspSpSM_destroyDescr(descr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSM_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                         const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                         mcsparseDnMatDescr_t matC, macaDataType computeType, mcsparseSpSMAlg_t alg,
                                         mcsparseSpSMDescr_t spsmDescr, size_t* bufferSize) {
    mcspStatus_t ret =
        mcspSpSM_bufferSize(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, bufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSM_analysis(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                       const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                       mcsparseDnMatDescr_t matC, macaDataType computeType, mcsparseSpSMAlg_t alg,
                                       mcsparseSpSMDescr_t spsmDescr, void* externalBuffer) {
    mcspStatus_t ret =
        mcspSpSM_analysis(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpSM_solve(mcsparseHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                    const void* alpha, mcsparseSpMatDescr_t matA, mcsparseDnMatDescr_t matB,
                                    mcsparseDnMatDescr_t matC, macaDataType computeType, mcsparseSpSMAlg_t alg,
                                    mcsparseSpSMDescr_t spsmDescr) {
    mcspStatus_t ret = mcspSpSM_solve(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreate(mcsparseHandle_t* handle) {
    mcspStatus_t ret = mcspCreateHandle(handle);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroy(mcsparseHandle_t handle) {
    mcspStatus_t ret = mcspDestroyHandle(handle);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseGetVersion(mcsparseHandle_t handle, int* version) {
    mcspStatus_t ret = mcspGetVersion(handle, version);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseGetProperty(libraryPropertyType type, int* value) {
    mcspStatus_t ret = mcspGetProperty(type, value);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSetStream(mcsparseHandle_t handle, mcStream_t sid) {
    mcspStatus_t ret = mcspSetStream(handle, sid);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseGetStream(const mcsparseHandle_t handle, mcStream_t* sid) {
    mcspStatus_t ret = mcspGetStream(handle, sid);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateMatDescr(mcsparseMatDescr_t* descr) {
    mcspStatus_t ret = mcspCreateMatDescr(descr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyMatDescr(mcsparseMatDescr_t descr) {
    mcspStatus_t ret = mcspDestroyMatDescr(descr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCopyMatDescr(mcsparseMatDescr_t dest, const mcsparseMatDescr_t src) {
    mcspStatus_t ret = mcspCopyMatDescr(dest, src);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSetPointerMode(mcsparseHandle_t handle, mcsparsePointerMode_t pointer_mode) {
    mcspStatus_t ret = mcspSetPointerMode(handle, pointer_mode);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseGetPointerMode(const mcsparseHandle_t handle, mcsparsePointerMode_t* pointer_mode) {
    mcspStatus_t ret = mcspGetPointerMode(handle, pointer_mode);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSetMatIndexBase(mcsparseMatDescr_t descr, mcsparseIndexBase_t index_base) {
    mcspStatus_t ret = mcspSetMatIndexBase(descr, index_base);
    return mcspToSparseStatus(ret);
}
mcsparseIndexBase_t mcsparseGetMatIndexBase(const mcsparseMatDescr_t descr) {
    return mcspGetMatIndexBase(descr);
}
mcsparseStatus_t mcsparseSetMatType(mcsparseMatDescr_t descr, mcsparseMatrixType_t matrix_type) {
    mcspStatus_t ret = mcspSetMatType(descr, matrix_type);
    return mcspToSparseStatus(ret);
}
mcsparseMatrixType_t mcsparseGetMatType(const mcsparseMatDescr_t descr) {
    return mcspGetMatType(descr);
}
mcsparseStatus_t mcsparseSetMatFillMode(mcsparseMatDescr_t descr, mcsparseFillMode_t mode) {
    mcspStatus_t ret = mcspSetMatFillMode(descr, mode);
    return mcspToSparseStatus(ret);
}
mcsparseFillMode_t mcsparseGetMatFillMode(const mcsparseMatDescr_t descr) {
    return mcspGetMatFillMode(descr);
}
mcsparseStatus_t mcsparseSetMatDiagType(mcsparseMatDescr_t descr, mcsparseDiagType_t diag_type) {
    mcspStatus_t ret = mcspSetMatDiagType(descr, diag_type);
    return mcspToSparseStatus(ret);
}
mcsparseDiagType_t mcsparseGetMatDiagType(const mcsparseMatDescr_t descr) {
    return mcspGetMatDiagType(descr);
}
mcsparseStatus_t mcsparseSetStorageMode(mcsparseMatDescr_t descr, mcsparseStorageMode_t storage_mode) {
    mcspStatus_t ret = mcspSetStorageMode(descr, storage_mode);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseGetStorageMode(const mcsparseMatDescr_t descr, mcsparseStorageMode_t* storage_mode) {
    mcspStatus_t ret = mcspGetStorageMode(descr, storage_mode);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateDnVec(mcsparseDnVecDescr_t* dnVecDescr, int64_t size, void* values,
                                     macaDataType valueType) {
    mcspStatus_t ret = mcspCreateDnVec(dnVecDescr, size, values, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyDnVec(mcsparseDnVecDescr_t dnVecDescr) {
    mcspStatus_t ret = mcspDestroyDnVec(dnVecDescr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnVecGet(mcsparseDnVecDescr_t dnVecDescr, int64_t* size, void** values,
                                  macaDataType* valueType) {
    mcspStatus_t ret = mcspDnVecGet(dnVecDescr, size, values, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnVecGetValues(mcsparseDnVecDescr_t dnVecDescr, void** values) {
    mcspStatus_t ret = mcspDnVecGetValues(dnVecDescr, values);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnVecSetValues(mcsparseDnVecDescr_t dnVecDescr, void* values) {
    mcspStatus_t ret = mcspDnVecSetValues(dnVecDescr, values);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCoo(mcsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void* cooRowInd, void* cooColInd, void* cooValues, mcsparseIndexType_t cooIdxType,
                                   mcsparseIndexBase_t idxBase, macaDataType valueType) {
    mcspStatus_t ret =
        mcspCreateCoo(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCooGet(mcsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz,
                                void** cooRowInd, void** cooColInd, void** cooValues, mcsparseIndexType_t* idxType,
                                mcsparseIndexBase_t* idxBase, macaDataType* valueType) {
    mcspStatus_t ret =
        mcspCooGet(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCooSetPointers(mcsparseSpMatDescr_t spMatDescr, void* cooRows, void* cooColumns,
                                        void* cooValues) {
    mcspStatus_t ret = mcspCooSetPointers(spMatDescr, cooRows, cooColumns, cooValues);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCsr(mcsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void* csrRowOffsets, void* csrColInd, void* csrValues,
                                   mcsparseIndexType_t csrRowOffsetsType, mcsparseIndexType_t csrColIndType,
                                   mcsparseIndexBase_t idxBase, macaDataType valueType) {
    mcspStatus_t ret = mcspCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
                                     csrRowOffsetsType, csrColIndType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCsrGet(mcsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz,
                                void** csrRowOffsets, void** csrColInd, void** csrValues,
                                mcsparseIndexType_t* csrRowOffsetsType, mcsparseIndexType_t* csrColIndType,
                                mcsparseIndexBase_t* idxBase, macaDataType* valueType) {
    mcspStatus_t ret = mcspCsrGet(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType,
                                  csrColIndType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCsrSetPointers(mcsparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd,
                                        void* csrValues) {
    mcspStatus_t ret = mcspCsrSetPointers(spMatDescr, csrRowOffsets, csrColInd, csrValues);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCsc(mcsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                   void* cscColOffsets, void* cscRowInd, void* cscValues,
                                   mcsparseIndexType_t cscColOffsetsType, mcsparseIndexType_t cscRowIndType,
                                   mcsparseIndexBase_t idxBase, macaDataType valueType) {
    mcspStatus_t ret = mcspCreateCsc(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues,
                                     cscColOffsetsType, cscRowIndType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCscSetPointers(mcsparseSpMatDescr_t spMatDescr, void* cscColOffsets, void* cscRowInd,
                                        void* cscValues) {
    mcspStatus_t ret = mcspCscSetPointers(spMatDescr, cscColOffsets, cscRowInd, cscValues);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateBlockedEll(mcsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols,
                                          int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue,
                                          mcsparseIndexType_t ellIdxType, mcsparseIndexBase_t idxBase,
                                          macaDataType valueType) {
    mcspStatus_t ret = mcspCreateBlockedEll(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue,
                                            ellIdxType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseBlockedEllGet(mcsparseSpMatDescr_t spMatDescr, int64_t* rowNum, int64_t* colNum,
                                       int64_t* ellBlockSize, int64_t* ellCols, void** ellColInd, void** ellValue,
                                       mcsparseIndexType_t* ellIdxType, mcsparseIndexBase_t* idxBase,
                                       macaDataType* valueType) {
    mcspStatus_t ret = mcspBlockedEllGet(spMatDescr, rowNum, colNum, ellBlockSize, ellCols, ellColInd, ellValue,
                                         ellIdxType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCooAoS(mcsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
                                      void* cooInd, void* cooValues, mcsparseIndexType_t cooIdxType,
                                      mcsparseIndexBase_t idxBase, macaDataType valueType) {
    mcspStatus_t ret = mcspCreateCooAos(spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCooAoSGet(mcsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz,
                                   void** cooInd,     // COO indices
                                   void** cooValues,  // COO values
                                   mcsparseIndexType_t* idxType, mcsparseIndexBase_t* idxBase,
                                   macaDataType* valueType) {
    mcspStatus_t ret = mcspCooAosGet(spMatDescr, rows, cols, nnz, cooInd, cooValues, idxType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroySpMat(mcsparseSpMatDescr_t spMatDescr) {
    mcspStatus_t ret = mcspDestroySpMat(spMatDescr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMatGetSize(mcsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz) {
    mcspStatus_t ret = mcspSpMatGetSize(spMatDescr, rows, cols, nnz);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMatGetFormat(mcsparseSpMatDescr_t spMatDescr, mcsparseFormat_t* format) {
    mcspStatus_t ret = mcspSpMatGetFormat(spMatDescr, format);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMatGetIndexBase(mcsparseSpMatDescr_t spMatDescr, mcsparseIndexBase_t* idxBase) {
    mcspStatus_t ret = mcspSpMatGetIndexBase(spMatDescr, idxBase);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMatGetValues(mcsparseSpMatDescr_t spMatDescr, void** values) {
    mcspStatus_t ret = mcspSpMatGetValues(spMatDescr, values);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMatSetValues(mcsparseSpMatDescr_t spMatDescr, void* values) {
    mcspStatus_t ret = mcspSpMatSetValues(spMatDescr, values);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMatGetAttribute(mcsparseSpMatDescr_t spMatDescr, mcsparseSpMatAttribute_t attribute,
                                           void* data, size_t dataSize) {
    mcspStatus_t ret = mcspSpMatGetAttribute(spMatDescr, attribute, data, dataSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpMatSetAttribute(mcsparseSpMatDescr_t spMatDescr, mcsparseSpMatAttribute_t attribute,
                                           const void* data, size_t dataSize) {
    mcspStatus_t ret = mcspSpMatSetAttribute(spMatDescr, attribute, data, dataSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateDnMat(mcsparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld,
                                     void* values, macaDataType valueType, mcsparseOrder_t order) {
    mcspStatus_t ret = mcspCreateDnMat(dnMatDescr, rows, cols, ld, values, valueType, order);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyDnMat(mcsparseDnMatDescr_t dnMatDescr) {
    mcspStatus_t ret = mcspDestroyDnMat(dnMatDescr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnMatGet(mcsparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld,
                                  void** values, macaDataType* type, mcsparseOrder_t* order) {
    mcspStatus_t ret = mcspDnMatGet(dnMatDescr, rows, cols, ld, values, type, order);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnMatGetValues(mcsparseDnMatDescr_t dnMatDescr, void** values) {
    mcspStatus_t ret = mcspDnMatGetValues(dnMatDescr, values);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnMatSetValues(mcsparseDnMatDescr_t dnMatDescr, void* values) {
    mcspStatus_t ret = mcspDnMatSetValues(dnMatDescr, values);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCsrsv2Info(mcsparseCsrsv2Info_t* info) {
    mcspStatus_t ret = mcspCreateCsrsv2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyCsrsv2Info(mcsparseCsrsv2Info_t info) {
    mcspStatus_t ret = mcspDestroyCsrsv2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCsrsm2Info(mcsparseCsrsm2Info_t* info) {
    mcspStatus_t ret = mcspCreateCsrsm2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyCsrsm2Info(mcsparseCsrsm2Info_t info) {
    mcspStatus_t ret = mcspDestroyCsrsm2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateSpVec(mcsparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices,
                                     void* values, mcsparseIndexType_t idxType, mcsparseIndexBase_t idxBase,
                                     macaDataType valueType) {
    mcspStatus_t ret = mcspCreateSpVec(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroySpVec(mcsparseSpVecDescr_t spVecDescr) {
    mcspStatus_t ret = mcspDestroySpVec(spVecDescr);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpVecGet(mcsparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices,
                                  void** values, mcsparseIndexType_t* idxType, mcsparseIndexBase_t* idxBase,
                                  macaDataType* valueType) {
    mcspStatus_t ret = mcspSpVecGet(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpVecGetIndexBase(mcsparseSpVecDescr_t spVecDescr, mcsparseIndexBase_t* idxBase) {
    mcspStatus_t ret = mcspSpVecGetIndexBase(spVecDescr, idxBase);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpVecGetValues(mcsparseSpVecDescr_t spVecDescr, void** values) {
    mcspStatus_t ret = mcspSpVecGetValues(spVecDescr, values);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSpVecSetValues(mcsparseSpVecDescr_t spVecDescr, void* values) {
    mcspStatus_t ret = mcspSpVecSetValues(spVecDescr, values);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCsrilu02Info(mcsparseCsrilu02Info_t* info) {
    mcspStatus_t ret = mcspCreateCsrilu02Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyCsrilu02Info(mcsparseCsrilu02Info_t info) {
    mcspStatus_t ret = mcspDestroyCsrilu02Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCsric02Info(mcsparseCsric02Info_t* info) {
    mcspStatus_t ret = mcspCreateCsric02Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyCsric02Info(mcsparseCsric02Info_t info) {
    mcspStatus_t ret = mcspDestroyCsric02Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateColorInfo(mcsparseColorInfo_t* info) {
    mcspStatus_t ret = mcspCreateColorInfo(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyColorInfo(mcsparseColorInfo_t info) {
    mcspStatus_t ret = mcspDestroyColorInfo(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSetColorAlgs(mcsparseColorInfo_t info, mcsparseColorAlg_t alg) {
    mcspStatus_t ret = mcspSetColorAlgs(info, alg);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseGetColorAlgs(mcsparseColorInfo_t info, mcsparseColorAlg_t* alg) {
    mcspStatus_t ret = mcspGetColorAlgs(info, alg);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCsrgemm2Info(mcsparseCsrgemm2Info_t* info) {
    mcspStatus_t ret = mcspCreateCsrgemm2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyCsrgemm2Info(mcsparseCsrgemm2Info_t info) {
    mcspStatus_t ret = mcspDestroyCsrgemm2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreatePruneInfo(mcsparsePruneInfo_t* info) {
    mcspStatus_t ret = mcspCreatePruneInfo(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyPruneInfo(mcsparsePruneInfo_t info) {
    mcspStatus_t ret = mcspDestroyPruneInfo(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateCsru2csrInfo(mcsparseCsru2csrInfo_t* info) {
    mcspStatus_t ret = mcspCreateCsru2csrInfo(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyCsru2csrInfo(mcsparseCsru2csrInfo_t info) {
    mcspStatus_t ret = mcspDestroyCsru2csrInfo(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSaxpyi(mcsparseHandle_t handle, int nnz, const float* alpha, const float* x_val,
                                const int* x_ind, float* y, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinSaxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDaxpyi(mcsparseHandle_t handle, int nnz, const double* alpha, const double* x_val,
                                const int* x_ind, double* y, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinDaxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCaxpyi(mcsparseHandle_t handle, int nnz, const mcspComplexFloat* alpha,
                                const mcspComplexFloat* x_val, const int* x_ind, mcspComplexFloat* y,
                                mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinCaxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZaxpyi(mcsparseHandle_t handle, int nnz, const mcspComplexDouble* alpha,
                                const mcspComplexDouble* x_val, const int* x_ind, mcspComplexDouble* y,
                                mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinZaxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgthr(mcsparseHandle_t handle, int nnz, const float* y, float* x_val, const int* x_ind,
                               mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinSgthr(handle, nnz, y, x_val, x_ind, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgthr(mcsparseHandle_t handle, int nnz, const double* y, double* x_val, const int* x_ind,
                               mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinDgthr(handle, nnz, y, x_val, x_ind, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgthr(mcsparseHandle_t handle, int nnz, const mcspComplexFloat* y, mcspComplexFloat* x_val,
                               const int* x_ind, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinCgthr(handle, nnz, y, x_val, x_ind, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgthr(mcsparseHandle_t handle, int nnz, const mcspComplexDouble* y, mcspComplexDouble* x_val,
                               const int* x_ind, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinZgthr(handle, nnz, y, x_val, x_ind, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgthrz(mcsparseHandle_t handle, int nnz, float* y, float* x_val, const int* x_ind,
                                mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinSgthrz(handle, nnz, y, x_val, x_ind, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgthrz(mcsparseHandle_t handle, int nnz, double* y, double* x_val, const int* x_ind,
                                mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinDgthrz(handle, nnz, y, x_val, x_ind, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgthrz(mcsparseHandle_t handle, int nnz, mcspComplexFloat* y, mcspComplexFloat* x_val,
                                const int* x_ind, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinCgthrz(handle, nnz, y, x_val, x_ind, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgthrz(mcsparseHandle_t handle, int nnz, mcspComplexDouble* y, mcspComplexDouble* x_val,
                                const int* x_ind, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinZgthrz(handle, nnz, y, x_val, x_ind, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSsctr(mcsparseHandle_t handle, int nnz, const float* x_val, const int* x_ind, float* y,
                               mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinSsctr(handle, nnz, x_val, x_ind, y, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDsctr(mcsparseHandle_t handle, int nnz, const double* x_val, const int* x_ind, double* y,
                               mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinDsctr(handle, nnz, x_val, x_ind, y, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCsctr(mcsparseHandle_t handle, int nnz, const mcspComplexFloat* x_val, const int* x_ind,
                               mcspComplexFloat* y, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinCsctr(handle, nnz, x_val, x_ind, y, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZsctr(mcsparseHandle_t handle, int nnz, const mcspComplexDouble* x_val, const int* x_ind,
                               mcspComplexDouble* y, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinZsctr(handle, nnz, x_val, x_ind, y, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSroti(mcsparseHandle_t handle, int nnz, float* x_val, const int* x_ind, float* y,
                               const float* c, const float* s, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinSroti(handle, nnz, x_val, x_ind, y, c, s, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDroti(mcsparseHandle_t handle, int nnz, double* x_val, const int* x_ind, double* y,
                               const double* c, const double* s, mcsparseIndexBase_t idx_base) {
    mcspStatus_t ret = mcspCuinDroti(handle, nnz, x_val, x_ind, y, c, s, idx_base);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCsrmvEx_bufferSize(mcsparseHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA,
                                            int m, int n, int nnz, const void* alpha, macaDataType alphatype,
                                            const mcsparseMatDescr_t descrA, const void* csrValA,
                                            macaDataType csrValAtype, const int* csrRowPtrA, const int* csrColIndA,
                                            const void* x, macaDataType xtype, const void* beta, macaDataType betatype,
                                            void* y, macaDataType ytype, macaDataType executiontype,
                                            size_t* bufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinCsrmvEx_bufferSize(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA,
                                                csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype,
                                                executiontype, bufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCsrmvEx(mcsparseHandle_t handle, mcsparseAlgMode_t alg, mcsparseOperation_t transA, int m,
                                 int n, int nnz, const void* alpha, macaDataType alphatype,
                                 const mcsparseMatDescr_t descrA, const void* csrValA, macaDataType csrValAtype,
                                 const int* csrRowPtrA, const int* csrColIndA, const void* x, macaDataType xtype,
                                 const void* beta, macaDataType betatype, void* y, macaDataType ytype,
                                 macaDataType executiontype, void* buffer) {
    mcspStatus_t ret = mcspCuinCsrmvEx(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype,
                                     csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsrsv2_zeroPivot(mcsparseHandle_t handle, mcsparseCsrsv2Info_t info, int* position) {
    mcspStatus_t ret = mcspCuinXcsrsv2_zeroPivot(handle, info, position);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                            const mcsparseMatDescr_t descrA, float* csrSortedValA,
                                            const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                            mcsparseCsrsv2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinScsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                csrSortedColIndA, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                            const mcsparseMatDescr_t descrA, double* csrSortedValA,
                                            const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                            mcsparseCsrsv2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinDcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                csrSortedColIndA, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                            const mcsparseMatDescr_t descrA, mcspComplexFloat* csrSortedValA,
                                            const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                            mcsparseCsrsv2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinCcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                csrSortedColIndA, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                            const mcsparseMatDescr_t descrA, mcspComplexDouble* csrSortedValA,
                                            const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                            mcsparseCsrsv2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinZcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                csrSortedColIndA, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                               const mcsparseMatDescr_t descrA, float* csrSortedValA,
                                               const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                               mcsparseCsrsv2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinScsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                   csrSortedColIndA, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                               const mcsparseMatDescr_t descrA, double* csrSortedValA,
                                               const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                               mcsparseCsrsv2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinDcsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                   csrSortedColIndA, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                               const mcsparseMatDescr_t descrA, mcspComplexFloat* csrSortedValA,
                                               const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                               mcsparseCsrsv2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinCcsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                   csrSortedColIndA, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                               const mcsparseMatDescr_t descrA, mcspComplexDouble* csrSortedValA,
                                               const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                               mcsparseCsrsv2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinZcsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                   csrSortedColIndA, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                          const mcsparseMatDescr_t descrA, const float* csrSortedValA,
                                          const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                          mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinScsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                              csrSortedColIndA, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                          const mcsparseMatDescr_t descrA, const double* csrSortedValA,
                                          const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                          mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinDcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                              csrSortedColIndA, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                          const mcsparseMatDescr_t descrA, const mcspComplexFloat* csrSortedValA,
                                          const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                          mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinCcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                              csrSortedColIndA, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrsv2_analysis(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                          const mcsparseMatDescr_t descrA, const mcspComplexDouble* csrSortedValA,
                                          const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                          mcsparseCsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinZcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                              csrSortedColIndA, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                       const float* alpha, const mcsparseMatDescr_t descrA, const float* csrSortedValA,
                                       const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                       mcsparseCsrsv2Info_t info, const float* f, float* x,
                                       mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinScsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, info, f, x, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                       const double* alpha, const mcsparseMatDescr_t descrA,
                                       const double* csrSortedValA, const int* csrSortedRowPtrA,
                                       const int* csrSortedColIndA, mcsparseCsrsv2Info_t info, const double* f,
                                       double* x, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinDcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, info, f, x, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                       const mcspComplexFloat* alpha, const mcsparseMatDescr_t descrA,
                                       const mcspComplexFloat* csrSortedValA, const int* csrSortedRowPtrA,
                                       const int* csrSortedColIndA, mcsparseCsrsv2Info_t info,
                                       const mcspComplexFloat* f, mcspComplexFloat* x, mcsparseSolvePolicy_t policy,
                                       void* pBuffer) {
    mcspStatus_t ret = mcspCuinCcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, info, f, x, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrsv2_solve(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int nnz,
                                       const mcspComplexDouble* alpha, const mcsparseMatDescr_t descrA,
                                       const mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                       const int* csrSortedColIndA, mcsparseCsrsv2Info_t info,
                                       const mcspComplexDouble* f, mcspComplexDouble* x, mcsparseSolvePolicy_t policy,
                                       void* pBuffer) {
    mcspStatus_t ret = mcspCuinZcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                                           csrSortedColIndA, info, f, x, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgemvi_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                           int* buffer_size) {
    mcspStatus_t ret = mcspCuinSgemvi_bufferSize(handle, trans, m, n, nnz, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgemvi_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                           int* buffer_size) {
    mcspStatus_t ret = mcspCuinDgemvi_bufferSize(handle, trans, m, n, nnz, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgemvi_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                           int* buffer_size) {
    mcspStatus_t ret = mcspCuinCgemvi_bufferSize(handle, trans, m, n, nnz, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgemvi_bufferSize(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, int nnz,
                                           int* buffer_size) {
    mcspStatus_t ret = mcspCuinZgemvi_bufferSize(handle, trans, m, n, nnz, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgemvi(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, const float* alpha,
                                const float* A, int lda, int nnz, const float* x_val, const int* x_ind,
                                const float* beta, float* y, mcsparseIndexBase_t idx_base, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinSgemvi(handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgemvi(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n, const double* alpha,
                                const double* A, int lda, int nnz, const double* x_val, const int* x_ind,
                                const double* beta, double* y, mcsparseIndexBase_t idx_base, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinDgemvi(handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgemvi(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n,
                                const mcspComplexFloat* alpha, const mcspComplexFloat* A, int lda, int nnz,
                                const mcspComplexFloat* x_val, const int* x_ind, const mcspComplexFloat* beta,
                                mcspComplexFloat* y, mcsparseIndexBase_t idx_base, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinCgemvi(handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgemvi(mcsparseHandle_t handle, mcsparseOperation_t trans, int m, int n,
                                const mcspComplexDouble* alpha, const mcspComplexDouble* A, int lda, int nnz,
                                const mcspComplexDouble* x_val, const int* x_ind, const mcspComplexDouble* beta,
                                mcspComplexDouble* y, mcsparseIndexBase_t idx_base, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinZgemvi(handle, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                int nb, int nnzb, const float* alpha, const mcsparseMatDescr_t descrA,
                                const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA,
                                int blockDim, const float* x, const float* beta, float* y) {
    mcspStatus_t ret = mcspCuinSbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                    bsrSortedColIndA, blockDim, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                int nb, int nnzb, const double* alpha, const mcsparseMatDescr_t descrA,
                                const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA,
                                int blockDim, const double* x, const double* beta, double* y) {
    mcspStatus_t ret = mcspCuinDbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                    bsrSortedColIndA, blockDim, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                int nb, int nnzb, const mcspComplexFloat* alpha, const mcsparseMatDescr_t descrA,
                                const mcspComplexFloat* bsrSortedValA, const int* bsrSortedRowPtrA,
                                const int* bsrSortedColIndA, int blockDim, const mcspComplexFloat* x,
                                const mcspComplexFloat* beta, mcspComplexFloat* y) {
    mcspStatus_t ret = mcspCuinCbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                    bsrSortedColIndA, blockDim, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA, int mb,
                                int nb, int nnzb, const mcspComplexDouble* alpha, const mcsparseMatDescr_t descrA,
                                const mcspComplexDouble* bsrSortedValA, const int* bsrSortedRowPtrA,
                                const int* bsrSortedColIndA, int blockDim, const mcspComplexDouble* x,
                                const mcspComplexDouble* beta, mcspComplexDouble* y) {
    mcspStatus_t ret = mcspCuinZbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                    bsrSortedColIndA, blockDim, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrxmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                 int sizeOfMask, int mb, int nb, int nnzb, const float* alpha,
                                 const mcsparseMatDescr_t descrA, const float* bsrSortedValA,
                                 const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA,
                                 const int* bsrSortedColIndA, int blockDim, const float* x, const float* beta,
                                 float* y) {
    mcspStatus_t ret =
        mcspCuinSbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA,
                      bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrxmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                 int sizeOfMask, int mb, int nb, int nnzb, const double* alpha,
                                 const mcsparseMatDescr_t descrA, const double* bsrSortedValA,
                                 const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA,
                                 const int* bsrSortedColIndA, int blockDim, const double* x, const double* beta,
                                 double* y) {
    mcspStatus_t ret =
        mcspCuinDbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA,
                      bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrxmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                 int sizeOfMask, int mb, int nb, int nnzb, const mcFloatComplex* alpha,
                                 const mcsparseMatDescr_t descrA, const mcFloatComplex* bsrSortedValA,
                                 const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA,
                                 const int* bsrSortedColIndA, int blockDim, const mcFloatComplex* x,
                                 const mcFloatComplex* beta, mcFloatComplex* y) {
    mcspStatus_t ret =
        mcspCuinCbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA,
                      bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrxmv(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                 int sizeOfMask, int mb, int nb, int nnzb, const mcDoubleComplex* alpha,
                                 const mcsparseMatDescr_t descrA, const mcDoubleComplex* bsrSortedValA,
                                 const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA,
                                 const int* bsrSortedColIndA, int blockDim, const mcDoubleComplex* x,
                                 const mcDoubleComplex* beta, mcDoubleComplex* y) {
    mcspStatus_t ret =
        mcspCuinZbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA,
                      bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgemmi(mcsparseHandle_t handle, int m, int n, int k, int nnz, const float* alpha,
                                const float* A, int lda, const float* csc_vals, const int* csc_cols,
                                const int* csc_rows, const float* beta, float* C, int ldc) {
    mcspStatus_t ret = mcspCuinSgemmi(handle, m, n, k, nnz, alpha, A, lda, csc_vals, csc_cols, csc_rows, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgemmi(mcsparseHandle_t handle, int m, int n, int k, int nnz, const double* alpha,
                                const double* A, int lda, const double* csc_vals, const int* csc_cols,
                                const int* csc_rows, const double* beta, double* C, int ldc) {
    mcspStatus_t ret = mcspCuinDgemmi(handle, m, n, k, nnz, alpha, A, lda, csc_vals, csc_cols, csc_rows, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgemmi(mcsparseHandle_t handle, int m, int n, int k, int nnz, const mcspComplexFloat* alpha,
                                const mcspComplexFloat* A, int lda, const mcspComplexFloat* csc_vals,
                                const int* csc_cols, const int* csc_rows, const mcspComplexFloat* beta,
                                mcspComplexFloat* C, int ldc) {
    mcspStatus_t ret = mcspCuinCgemmi(handle, m, n, k, nnz, alpha, A, lda, csc_vals, csc_cols, csc_rows, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgemmi(mcsparseHandle_t handle, int m, int n, int k, int nnz, const mcspComplexDouble* alpha,
                                const mcspComplexDouble* A, int lda, const mcspComplexDouble* csc_vals,
                                const int* csc_cols, const int* csc_rows, const mcspComplexDouble* beta,
                                mcspComplexDouble* C, int ldc) {
    mcspStatus_t ret = mcspCuinZgemmi(handle, m, n, k, nnz, alpha, A, lda, csc_vals, csc_cols, csc_rows, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsrsm2_zeroPivot(mcsparseHandle_t handle, mcsparseCsrsm2Info_t info, int* position) {
    mcspStatus_t ret = mcspCuinXcsrsm2_zeroPivot(handle, info, position);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrsm2_bufferSizeExt(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                               mcsparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha,
                                               const mcsparseMatDescr_t descrA, const float* csrSortedValA,
                                               const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B,
                                               int ldb, mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                               size_t* pBufferSize) {
    mcspStatus_t ret =
        mcspCuinScsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrsm2_bufferSizeExt(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                               mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                               const double* alpha, const mcsparseMatDescr_t descrA,
                                               const double* csrSortedValA, const int* csrSortedRowPtrA,
                                               const int* csrSortedColIndA, const double* B, int ldb,
                                               mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                               size_t* pBufferSize) {
    mcspStatus_t ret =
        mcspCuinDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrsm2_bufferSizeExt(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                               mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                               const mcspComplexFloat* alpha, const mcsparseMatDescr_t descrA,
                                               const mcspComplexFloat* csrSortedValA, const int* csrSortedRowPtrA,
                                               const int* csrSortedColIndA, const mcspComplexFloat* B, int ldb,
                                               mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                               size_t* pBufferSize) {
    mcspStatus_t ret =
        mcspCuinCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrsm2_bufferSizeExt(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                               mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                               const mcspComplexDouble* alpha, const mcsparseMatDescr_t descrA,
                                               const mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                               const int* csrSortedColIndA, const mcspComplexDouble* B, int ldb,
                                               mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                               size_t* pBufferSize) {
    mcspStatus_t ret =
        mcspCuinZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrsm2_analysis(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                          mcsparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha,
                                          const mcsparseMatDescr_t descrA, const float* csrSortedValA,
                                          const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B,
                                          int ldb, mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                          void* pBuffer) {
    mcspStatus_t ret = mcspCuinScsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                              csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrsm2_analysis(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                          mcsparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha,
                                          const mcsparseMatDescr_t descrA, const double* csrSortedValA,
                                          const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B,
                                          int ldb, mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy,
                                          void* pBuffer) {
    mcspStatus_t ret = mcspCuinDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                              csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrsm2_analysis(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                          mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                          const mcspComplexFloat* alpha, const mcsparseMatDescr_t descrA,
                                          const mcspComplexFloat* csrSortedValA, const int* csrSortedRowPtrA,
                                          const int* csrSortedColIndA, const mcspComplexFloat* B, int ldb,
                                          mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                              csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrsm2_analysis(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                          mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                          const mcspComplexDouble* alpha, const mcsparseMatDescr_t descrA,
                                          const mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                          const int* csrSortedColIndA, const mcspComplexDouble* B, int ldb,
                                          mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                              csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrsm2_solve(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha,
                                       const mcsparseMatDescr_t descrA, const float* csrSortedValA,
                                       const int* csrSortedRowPtrA, const int* csrSortedColIndA, float* B, int ldb,
                                       mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinScsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrsm2_solve(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha,
                                       const mcsparseMatDescr_t descrA, const double* csrSortedValA,
                                       const int* csrSortedRowPtrA, const int* csrSortedColIndA, double* B, int ldb,
                                       mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrsm2_solve(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                       const mcspComplexFloat* alpha, const mcsparseMatDescr_t descrA,
                                       const mcspComplexFloat* csrSortedValA, const int* csrSortedRowPtrA,
                                       const int* csrSortedColIndA, mcspComplexFloat* B, int ldb,
                                       mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrsm2_solve(mcsparseHandle_t handle, int algo, mcsparseOperation_t transA,
                                       mcsparseOperation_t transB, int m, int nrhs, int nnz,
                                       const mcspComplexDouble* alpha, const mcsparseMatDescr_t descrA,
                                       const mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                       const int* csrSortedColIndA, mcspComplexDouble* B, int ldb,
                                       mcsparseCsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrmm(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const float* alpha,
                                const mcsparseMatDescr_t descrA, const float* bsrSortedValA,
                                const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize,
                                const float* B, const int ldb, const float* beta, float* C, int ldc) {
    mcspStatus_t ret = mcspCuinSbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA,
                                    bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrmm(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                mcsparseOperation_t transB, int mb, int n, int kb, int nnzb, const double* alpha,
                                const mcsparseMatDescr_t descrA, const double* bsrSortedValA,
                                const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize,
                                const double* B, const int ldb, const double* beta, double* C, int ldc) {
    mcspStatus_t ret = mcspCuinDbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA,
                                    bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrmm(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                mcsparseOperation_t transB, int mb, int n, int kb, int nnzb,
                                const mcFloatComplex* alpha, const mcsparseMatDescr_t descrA,
                                const mcFloatComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                const int* bsrSortedColIndA, const int blockSize, const mcFloatComplex* B,
                                const int ldb, const mcFloatComplex* beta, mcFloatComplex* C, int ldc) {
    mcspStatus_t ret = mcspCuinCbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA,
                                    bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrmm(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                mcsparseOperation_t transB, int mb, int n, int kb, int nnzb,
                                const mcDoubleComplex* alpha, const mcsparseMatDescr_t descrA,
                                const mcDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                const int* bsrSortedColIndA, const int blockSize, const mcDoubleComplex* B,
                                const int ldb, const mcDoubleComplex* beta, mcDoubleComplex* C, int ldc) {
    mcspStatus_t ret = mcspCuinZbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA,
                                    bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsrilu02_zeroPivot(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int* position) {
    mcspStatus_t ret = mcspCuinXcsrilu02_zeroPivot(handle, info, position);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrilu02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                              float* csrSortedValA, const int* csrSortedRowPtrA,
                                              const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                              int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinScsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                  csrSortedColIndA, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrilu02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                              double* csrSortedValA, const int* csrSortedRowPtrA,
                                              const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                              int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinDcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                  csrSortedColIndA, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrilu02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                              mcspComplexFloat* csrSortedValA, const int* csrSortedRowPtrA,
                                              const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                              int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinCcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                  csrSortedColIndA, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrilu02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                              mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                              const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                              int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinZcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                  csrSortedColIndA, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrilu02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz,
                                                 const mcsparseMatDescr_t descrA, float* csrSortedVal,
                                                 const int* csrSortedRowPtr, const int* csrSortedColInd,
                                                 mcsparseCsrilu02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinScsrilu02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                                                     csrSortedColInd, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrilu02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz,
                                                 const mcsparseMatDescr_t descrA, double* csrSortedVal,
                                                 const int* csrSortedRowPtr, const int* csrSortedColInd,
                                                 mcsparseCsrilu02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinDcsrilu02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                                                     csrSortedColInd, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrilu02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz,
                                                 const mcsparseMatDescr_t descrA, mcspComplexFloat* csrSortedVal,
                                                 const int* csrSortedRowPtr, const int* csrSortedColInd,
                                                 mcsparseCsrilu02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinCcsrilu02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                                                     csrSortedColInd, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrilu02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz,
                                                 const mcsparseMatDescr_t descrA, mcspComplexDouble* csrSortedVal,
                                                 const int* csrSortedRowPtr, const int* csrSortedColInd,
                                                 mcsparseCsrilu02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinZcsrilu02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr,
                                                     csrSortedColInd, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int enable_boost,
                                                double* tol, float* boost_val) {
    mcspStatus_t ret = mcspCuinScsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int enable_boost,
                                                double* tol, double* boost_val) {
    mcspStatus_t ret = mcspCuinDcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int enable_boost,
                                                double* tol, mcspComplexFloat* boost_val) {
    mcspStatus_t ret = mcspCuinCcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseCsrilu02Info_t info, int enable_boost,
                                                double* tol, mcspComplexDouble* boost_val) {
    mcspStatus_t ret = mcspCuinZcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrilu02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                            const float* csrSortedValA, const int* csrSortedRowPtrA,
                                            const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                            mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinScsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                csrSortedColIndA, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrilu02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                            const double* csrSortedValA, const int* csrSortedRowPtrA,
                                            const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                            mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinDcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                csrSortedColIndA, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrilu02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                            const mcspComplexFloat* csrSortedValA, const int* csrSortedRowPtrA,
                                            const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                            mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinCcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                csrSortedColIndA, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrilu02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                            const mcspComplexDouble* csrSortedValA, const int* csrSortedRowPtrA,
                                            const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                            mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinZcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                                csrSortedColIndA, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrilu02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                   float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   mcsparseCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinScsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA,
                                       info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrilu02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                   double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                                   mcsparseCsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinDcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA,
                                       info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrilu02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                   mcspComplexFloat* csrSortedValA_valM, const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinCcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA,
                                       info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrilu02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descrA,
                                   mcspComplexDouble* csrSortedValA_valM, const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA, mcsparseCsrilu02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinZcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA,
                                       info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsric02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                             const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                             mcsparseCsric02Info_t info, int* buffersize) {
    mcspStatus_t ret = mcspCuinScsric02_bufferSize(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsric02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                             const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                             mcsparseCsric02Info_t info, int* buffersize) {
    mcspStatus_t ret = mcspCuinDcsric02_bufferSize(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsric02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                             const mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                             mcsparseCsric02Info_t info, int* buffersize) {
    mcspStatus_t ret = mcspCuinCcsric02_bufferSize(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsric02_bufferSize(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                             const mcspComplexDouble* csr_vals, const int* csr_rows,
                                             const int* csr_cols, mcsparseCsric02Info_t info, int* buffersize) {
    mcspStatus_t ret = mcspCuinZcsric02_bufferSize(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsric02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                                const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                                mcsparseCsric02Info_t info, size_t* buffersize) {
    mcspStatus_t ret =
        mcspCuinScsric02_bufferSizeExt(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsric02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                                const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                                mcsparseCsric02Info_t info, size_t* buffersize) {
    mcspStatus_t ret =
        mcspCuinDcsric02_bufferSizeExt(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsric02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                                const mcspComplexFloat* csr_vals, const int* csr_rows,
                                                const int* csr_cols, mcsparseCsric02Info_t info, size_t* buffersize) {
    mcspStatus_t ret =
        mcspCuinCcsric02_bufferSizeExt(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsric02_bufferSizeExt(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                                const mcspComplexDouble* csr_vals, const int* csr_rows,
                                                const int* csr_cols, mcsparseCsric02Info_t info, size_t* buffersize) {
    mcspStatus_t ret =
        mcspCuinZcsric02_bufferSizeExt(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, buffersize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsric02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                           const float* csr_vals, const int* csr_rows, const int* csr_cols,
                                           mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy,
                                           void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinScsric02_analysis(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsric02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                           const double* csr_vals, const int* csr_rows, const int* csr_cols,
                                           mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy,
                                           void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinDcsric02_analysis(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsric02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                           const mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                           mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy,
                                           void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinCcsric02_analysis(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsric02_analysis(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                           const mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                           mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy,
                                           void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinZcsric02_analysis(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXcsric02_zeroPivot(mcsparseHandle_t handle, mcsparseCsric02Info_t info, int* position) {
    mcspStatus_t ret = mcspCuinXcsric02_zeroPivot(handle, info, position);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsric02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                  float* csr_vals, const int* csr_rows, const int* csr_cols, mcsparseCsric02Info_t info,
                                  mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinScsric02(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsric02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                  double* csr_vals, const int* csr_rows, const int* csr_cols,
                                  mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinDcsric02(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsric02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                  mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                  mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinCcsric02(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsric02(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                  mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                  mcsparseCsric02Info_t info, mcsparseSolvePolicy_t solve_policy, void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinZcsric02(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, info, solve_policy, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsv2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d,
                                              const float* du, const float* B, int ldb, size_t* bufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinSgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsv2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                                              const double* du, const double* B, int ldb, size_t* bufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinDgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsv2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcspComplexFloat* dl,
                                              const mcspComplexFloat* d, const mcspComplexFloat* du,
                                              const mcspComplexFloat* B, int ldb, size_t* bufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinCgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsv2_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcspComplexDouble* dl,
                                              const mcspComplexDouble* d, const mcspComplexDouble* du,
                                              const mcspComplexDouble* B, int ldb, size_t* bufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinZgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsv2(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du,
                                float* B, int ldb, void* pBuffer) {
    mcspStatus_t ret = mcspCuinSgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsv2(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                                const double* du, double* B, int ldb, void* pBuffer) {
    mcspStatus_t ret = mcspCuinDgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsv2(mcsparseHandle_t handle, int m, int n, const mcspComplexFloat* dl,
                                const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* B, int ldb,
                                void* pBuffer) {
    mcspStatus_t ret = mcspCuinCgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsv2(mcsparseHandle_t handle, int m, int n, const mcspComplexDouble* dl,
                                const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B, int ldb,
                                void* pBuffer) {
    mcspStatus_t ret = mcspCuinZgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsv2_nopivot_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const float* dl,
                                                      const float* d, const float* du, const float* B, int ldb,
                                                      size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinSgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsv2_nopivot_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const double* dl,
                                                      const double* d, const double* du, const double* B, int ldb,
                                                      size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinDgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsv2_nopivot_bufferSizeExt(mcsparseHandle_t handle, int m, int n, const mcspComplexFloat* dl,
                                                      const mcspComplexFloat* d, const mcspComplexFloat* du,
                                                      const mcspComplexFloat* B, int ldb, size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinCgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsv2_nopivot_bufferSizeExt(mcsparseHandle_t handle, int m, int n,
                                                      const mcspComplexDouble* dl, const mcspComplexDouble* d,
                                                      const mcspComplexDouble* du, const mcspComplexDouble* B, int ldb,
                                                      size_t* buffer_size) {
    mcspStatus_t ret = mcspCuinZgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsv2_nopivot(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d,
                                        const float* du, float* B, int ldb, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinSgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsv2_nopivot(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                                        const double* du, double* B, int ldb, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinDgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsv2_nopivot(mcsparseHandle_t handle, int m, int n, const mcspComplexFloat* dl,
                                        const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* B,
                                        int ldb, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinCgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsv2_nopivot(mcsparseHandle_t handle, int m, int n, const mcspComplexDouble* dl,
                                        const mcspComplexDouble* d, const mcspComplexDouble* du, mcspComplexDouble* B,
                                        int ldb, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinZgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsv2StridedBatch_bufferSizeExt(mcsparseHandle_t handle, int m, const float* dl,
                                                          const float* d, const float* du, const float* x,
                                                          int batchCount, int batchStride, size_t* bufferSizeInBytes) {
    mcspStatus_t ret =
        mcspCuinSgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsv2StridedBatch_bufferSizeExt(mcsparseHandle_t handle, int m, const double* dl,
                                                          const double* d, const double* du, const double* x,
                                                          int batchCount, int batchStride, size_t* bufferSizeInBytes) {
    mcspStatus_t ret =
        mcspCuinDgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsv2StridedBatch_bufferSizeExt(mcsparseHandle_t handle, int m, const mcspComplexFloat* dl,
                                                          const mcspComplexFloat* d, const mcspComplexFloat* du,
                                                          const mcspComplexFloat* x, int batchCount, int batchStride,
                                                          size_t* bufferSizeInBytes) {
    mcspStatus_t ret =
        mcspCuinCgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsv2StridedBatch_bufferSizeExt(mcsparseHandle_t handle, int m, const mcspComplexDouble* dl,
                                                          const mcspComplexDouble* d, const mcspComplexDouble* du,
                                                          const mcspComplexDouble* x, int batchCount, int batchStride,
                                                          size_t* bufferSizeInBytes) {
    mcspStatus_t ret =
        mcspCuinZgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsv2StridedBatch(mcsparseHandle_t handle, int m, const float* dl, const float* d,
                                            const float* du, float* x, int batchCount, int batchStride, void* pBuffer) {
    mcspStatus_t ret = mcspCuinSgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsv2StridedBatch(mcsparseHandle_t handle, int m, const double* dl, const double* d,
                                            const double* du, double* x, int batchCount, int batchStride,
                                            void* pBuffer) {
    mcspStatus_t ret = mcspCuinDgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsv2StridedBatch(mcsparseHandle_t handle, int m, const mcspComplexFloat* dl,
                                            const mcspComplexFloat* d, const mcspComplexFloat* du, mcspComplexFloat* x,
                                            int batchCount, int batchStride, void* pBuffer) {
    mcspStatus_t ret = mcspCuinCgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsv2StridedBatch(mcsparseHandle_t handle, int m, const mcspComplexDouble* dl,
                                            const mcspComplexDouble* d, const mcspComplexDouble* du,
                                            mcspComplexDouble* x, int batchCount, int batchStride, void* pBuffer) {
    mcspStatus_t ret = mcspCuinZgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             const float* dl, const float* d, const float* du,
                                                             const float* x, int batch_count, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinSgtsvInterleavedBatch_bufferSizeExt(handle, alg, row_num, dl, d, du, x, batch_count, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             const double* dl, const double* d, const double* du,
                                                             const double* x, int batch_count, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinDgtsvInterleavedBatch_bufferSizeExt(handle, alg, row_num, dl, d, du, x, batch_count, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             const mcspComplexFloat* dl, const mcspComplexFloat* d,
                                                             const mcspComplexFloat* du, const mcspComplexFloat* x,
                                                             int batch_count, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinCgtsvInterleavedBatch_bufferSizeExt(handle, alg, row_num, dl, d, du, x, batch_count, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             const mcspComplexDouble* dl, const mcspComplexDouble* d,
                                                             const mcspComplexDouble* du, const mcspComplexDouble* x,
                                                             int batch_count, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinZgtsvInterleavedBatch_bufferSizeExt(handle, alg, row_num, dl, d, du, x, batch_count, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, float* dl, float* d,
                                               float* du, float* x, int batch_count, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinSgtsvInterleavedBatch(handle, alg, row_num, dl, d, du, x, batch_count, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, double* dl, double* d,
                                               double* du, double* x, int batch_count, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinDgtsvInterleavedBatch(handle, alg, row_num, dl, d, du, x, batch_count, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, mcspComplexFloat* dl,
                                               mcspComplexFloat* d, mcspComplexFloat* du, mcspComplexFloat* x,
                                               int batch_count, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinCgtsvInterleavedBatch(handle, alg, row_num, dl, d, du, x, batch_count, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, mcspComplexDouble* dl,
                                               mcspComplexDouble* d, mcspComplexDouble* du, mcspComplexDouble* x,
                                               int batch_count, void* temp_buffer) {
    mcspStatus_t ret = mcspCuinZgtsvInterleavedBatch(handle, alg, row_num, dl, d, du, x, batch_count, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgpsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num, float* ds,
                                                             float* dl, float* d, float* du, float* dw, float* x,
                                                             int batch_count, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinSgpsvInterleavedBatch_bufferSizeExt(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgpsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num, double* ds,
                                                             double* dl, double* d, double* du, double* dw, double* x,
                                                             int batch_count, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinDgpsvInterleavedBatch_bufferSizeExt(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgpsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             mcspComplexFloat* ds, mcspComplexFloat* dl,
                                                             mcspComplexFloat* d, mcspComplexFloat* du,
                                                             mcspComplexFloat* dw, mcspComplexFloat* x, int batch_count,
                                                             size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinCgpsvInterleavedBatch_bufferSizeExt(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgpsvInterleavedBatch_bufferSizeExt(mcsparseHandle_t handle, int alg, int row_num,
                                                             mcspComplexDouble* ds, mcspComplexDouble* dl,
                                                             mcspComplexDouble* d, mcspComplexDouble* du,
                                                             mcspComplexDouble* dw, mcspComplexDouble* x,
                                                             int batch_count, size_t* buffer_size) {
    mcspStatus_t ret =
        mcspCuinZgpsvInterleavedBatch_bufferSizeExt(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, buffer_size);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgpsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, float* ds, float* dl,
                                               float* d, float* du, float* dw, float* x, int batch_count,
                                               void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinSgpsvInterleavedBatch(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgpsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, double* ds, double* dl,
                                               double* d, double* du, double* dw, double* x, int batch_count,
                                               void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinDgpsvInterleavedBatch(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgpsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, mcspComplexFloat* ds,
                                               mcspComplexFloat* dl, mcspComplexFloat* d, mcspComplexFloat* du,
                                               mcspComplexFloat* dw, mcspComplexFloat* x, int batch_count,
                                               void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinCgpsvInterleavedBatch(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgpsvInterleavedBatch(mcsparseHandle_t handle, int alg, int row_num, mcspComplexDouble* ds,
                                               mcspComplexDouble* dl, mcspComplexDouble* d, mcspComplexDouble* du,
                                               mcspComplexDouble* dw, mcspComplexDouble* x, int batch_count,
                                               void* temp_buffer) {
    mcspStatus_t ret =
        mcspCuinZgpsvInterleavedBatch(handle, alg, row_num, ds, dl, d, du, dw, x, batch_count, temp_buffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrcolor(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                   float* csr_vals, const int* csr_rows, const int* csr_cols,
                                   const float* fraction_to_color, int* ncolors, int* coloring, int* reordering,
                                   mcsparseColorInfo_t info) {
    mcspStatus_t ret = mcspCuinScsrColor(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, fraction_to_color, ncolors,
                                       coloring, reordering, info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrcolor(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                   double* csr_vals, const int* csr_rows, const int* csr_cols,
                                   const double* fraction_to_color, int* ncolors, int* coloring, int* reordering,
                                   mcsparseColorInfo_t info) {
    mcspStatus_t ret = mcspCuinDcsrColor(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, fraction_to_color, ncolors,
                                       coloring, reordering, info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrcolor(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                   mcspComplexFloat* csr_vals, const int* csr_rows, const int* csr_cols,
                                   const float* fraction_to_color, int* ncolors, int* coloring, int* reordering,
                                   mcsparseColorInfo_t info) {
    mcspStatus_t ret = mcspCuinCcsrColor(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, fraction_to_color, ncolors,
                                       coloring, reordering, info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrcolor(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                   mcspComplexDouble* csr_vals, const int* csr_rows, const int* csr_cols,
                                   const double* fraction_to_color, int* ncolors, int* coloring, int* reordering,
                                   mcsparseColorInfo_t info) {
    mcspStatus_t ret = mcspCuinZcsrColor(handle, m, nnz, descr, csr_vals, csr_rows, csr_cols, fraction_to_color, ncolors,
                                       coloring, reordering, info);
    return mcspToSparseStatus(ret);
}

mcsparseStatus_t mcsparseSbsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, float* bsrSortedValA,
                                            const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                            mcsparseBsrsv2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinSbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                                bsrSortedColIndA, blockDim, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, double* bsrSortedValA,
                                            const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                            mcsparseBsrsv2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinDbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                                bsrSortedColIndA, blockDim, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedValA,
                                            const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                            mcsparseBsrsv2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinCbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                                bsrSortedColIndA, blockDim, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrsv2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedValA,
                                            const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                            mcsparseBsrsv2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCuinZbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                                bsrSortedColIndA, blockDim, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, int mb, int nnzb,
                                               const mcsparseMatDescr_t descrA, float* bsrSortedValA,
                                               const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize,
                                               mcsparseBsrsv2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinSbsrsv2_bufferSizeExt(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                                   bsrSortedRowPtrA, bsrSortedColIndA, blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, int mb, int nnzb,
                                               const mcsparseMatDescr_t descrA, double* bsrSortedValA,
                                               const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize,
                                               mcsparseBsrsv2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinDbsrsv2_bufferSizeExt(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                                   bsrSortedRowPtrA, bsrSortedColIndA, blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, int mb, int nnzb,
                                               const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedValA,
                                               const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize,
                                               mcsparseBsrsv2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinCbsrsv2_bufferSizeExt(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                                   bsrSortedRowPtrA, bsrSortedColIndA, blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrsv2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, int mb, int nnzb,
                                               const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedValA,
                                               const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize,
                                               mcsparseBsrsv2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCuinZbsrsv2_bufferSizeExt(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                                   bsrSortedRowPtrA, bsrSortedColIndA, blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrsv2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          int mb, int nnzb, const mcsparseMatDescr_t descrA, const float* bsrSortedValA,
                                          const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
                                          mcsparseBsrsv2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinSbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                              bsrSortedColIndA, blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrsv2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          int mb, int nnzb, const mcsparseMatDescr_t descrA,
                                          const double* bsrSortedValA, const int* bsrSortedRowPtrA,
                                          const int* bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                          mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinDbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                              bsrSortedColIndA, blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrsv2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          int mb, int nnzb, const mcsparseMatDescr_t descrA,
                                          const mcFloatComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                          const int* bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                          mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinCbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                              bsrSortedColIndA, blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrsv2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          int mb, int nnzb, const mcsparseMatDescr_t descrA,
                                          const mcDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                          const int* bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                          mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinZbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                                              bsrSortedColIndA, blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrsv2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       int mb, int nnzb, const float* alpha, const mcsparseMatDescr_t descrA,
                                       const float* bsrSortedValA, const int* bsrSortedRowPtrA,
                                       const int* bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                       const float* f, float* x, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinSbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                                           bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrsv2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       int mb, int nnzb, const double* alpha, const mcsparseMatDescr_t descrA,
                                       const double* bsrSortedValA, const int* bsrSortedRowPtrA,
                                       const int* bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                       const double* f, double* x, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCuinDbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                                           bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrsv2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       int mb, int nnzb, const mcFloatComplex* alpha, const mcsparseMatDescr_t descrA,
                                       const mcFloatComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                       const int* bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                       const mcFloatComplex* f, mcFloatComplex* x, mcsparseSolvePolicy_t policy,
                                       void* pBuffer) {
    mcspStatus_t ret = mcspCuinCbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                                           bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrsv2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       int mb, int nnzb, const mcDoubleComplex* alpha, const mcsparseMatDescr_t descrA,
                                       const mcDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA,
                                       const int* bsrSortedColIndA, int blockDim, mcsparseBsrsv2Info_t info,
                                       const mcDoubleComplex* f, mcDoubleComplex* x, mcsparseSolvePolicy_t policy,
                                       void* pBuffer) {
    mcspStatus_t ret = mcspCuinZbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                                           bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXbsrsv2_zeroPivot(mcsparseHandle_t handle, mcsparseBsrsv2Info_t info, int* position) {
    mcspStatus_t ret = mcspCuinXbsrsv2_zeroPivot(handle, info, position);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateBsrsv2Info(mcsparseBsrsv2Info_t* info) {
    mcspStatus_t ret = mcspCreateBsrsv2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyBsrsv2Info(mcsparseBsrsv2Info_t info) {
    mcspStatus_t ret = mcspDestroyBsrsv2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXbsrsm2_zeroPivot(mcsparseHandle_t handle, mcsparseBsrsm2Info_t info, int* position) {
    mcspStatus_t ret = mcspXbsrsm2_zeroPivot(handle, info, position);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrsm2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, mcsparseOperation_t transXY, int mb, int n,
                                            int nnzb, const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                            mcsparseBsrsm2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspSbsrsm2_bufferSize(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                              descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                                              (mcspInt*)bsrSortedColInd, (mcspInt)blockSize, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrsm2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, mcsparseOperation_t transXY, int mb, int n,
                                            int nnzb, const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                            mcsparseBsrsm2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspDbsrsm2_bufferSize(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                              descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                                              (mcspInt*)bsrSortedColInd, (mcspInt)blockSize, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrsm2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, mcsparseOperation_t transXY, int mb, int n,
                                            int nnzb, const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                            mcsparseBsrsm2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCbsrsm2_bufferSize(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                              descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                                              (mcspInt*)bsrSortedColInd, (mcspInt)blockSize, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrsm2_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                            mcsparseOperation_t transA, mcsparseOperation_t transXY, int mb, int n,
                                            int nnzb, const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                            mcsparseBsrsm2Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspZbsrsm2_bufferSize(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                              descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                                              (mcspInt*)bsrSortedColInd, (mcspInt)blockSize, info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrsm2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, mcsparseOperation_t transB, int mb, int n,
                                               int nnzb, const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                               const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                               mcsparseBsrsm2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspSbsrsm2_bufferSizeExt(handle, dirA, transA, transB, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                                 descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                                                 (mcspInt*)bsrSortedColInd, (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrsm2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, mcsparseOperation_t transB, int mb, int n,
                                               int nnzb, const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                               const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                               mcsparseBsrsm2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspDbsrsm2_bufferSizeExt(handle, dirA, transA, transB, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                                 descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                                                 (mcspInt*)bsrSortedColInd, (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrsm2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, mcsparseOperation_t transB, int mb, int n,
                                               int nnzb, const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                               const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                               mcsparseBsrsm2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCbsrsm2_bufferSizeExt(handle, dirA, transA, transB, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                                 descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                                                 (mcspInt*)bsrSortedColInd, (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrsm2_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA,
                                               mcsparseOperation_t transA, mcsparseOperation_t transB, int mb, int n,
                                               int nnzb, const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                               const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                               mcsparseBsrsm2Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspZbsrsm2_bufferSizeExt(handle, dirA, transA, transB, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                                 descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                                                 (mcspInt*)bsrSortedColInd, (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrsm2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                          const mcsparseMatDescr_t descrA, const float* bsrSortedVal,
                                          const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                          mcsparseBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspSbsrsm2_analysis(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                            descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                            (mcspInt)blockSize, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrsm2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                          const mcsparseMatDescr_t descrA, const double* bsrSortedVal,
                                          const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                          mcsparseBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspDbsrsm2_analysis(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                            descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                            (mcspInt)blockSize, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrsm2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                          const mcsparseMatDescr_t descrA, const mcFloatComplex* bsrSortedVal,
                                          const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                          mcsparseBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCbsrsm2_analysis(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                            descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                            (mcspInt)blockSize, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrsm2_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                          mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                          const mcsparseMatDescr_t descrA, const mcDoubleComplex* bsrSortedVal,
                                          const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                          mcsparseBsrsm2Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspZbsrsm2_analysis(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb,
                                            descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                            (mcspInt)blockSize, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrsm2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transXY, int mb, int n, int nnzb, const float* alpha,
                                       const mcsparseMatDescr_t descrA, const float* bsrSortedVal,
                                       const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                       mcsparseBsrsm2Info_t info, const float* B, int ldb, float* X, int ldx,
                                       mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspSbsrsm2_solve(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb, alpha,
                                         descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                         (mcspInt)blockSize, info, B, (mcspInt)ldb, X, (mcspInt)ldx, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrsm2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transXY, int mb, int n, int nnzb, const double* alpha,
                                       const mcsparseMatDescr_t descrA, const double* bsrSortedVal,
                                       const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                       mcsparseBsrsm2Info_t info, const double* B, int ldb, double* X, int ldx,
                                       mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspDbsrsm2_solve(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb, alpha,
                                         descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                         (mcspInt)blockSize, info, B, (mcspInt)ldb, X, (mcspInt)ldx, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrsm2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                       const mcFloatComplex* alpha, const mcsparseMatDescr_t descrA,
                                       const mcFloatComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                       const int* bsrSortedColInd, int blockSize, mcsparseBsrsm2Info_t info,
                                       const mcFloatComplex* B, int ldb, mcFloatComplex* X, int ldx,
                                       mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCbsrsm2_solve(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb, alpha,
                                         descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                         (mcspInt)blockSize, info, B, (mcspInt)ldb, X, (mcspInt)ldx, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrsm2_solve(mcsparseHandle_t handle, mcsparseDirection_t dirA, mcsparseOperation_t transA,
                                       mcsparseOperation_t transXY, int mb, int n, int nnzb,
                                       const mcDoubleComplex* alpha, const mcsparseMatDescr_t descrA,
                                       const mcDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                       const int* bsrSortedColInd, int blockSize, mcsparseBsrsm2Info_t info,
                                       const mcDoubleComplex* B, int ldb, mcDoubleComplex* X, int ldx,
                                       mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspZbsrsm2_solve(handle, dirA, transA, transXY, (mcspInt)mb, (mcspInt)n, (mcspInt)nnzb, alpha,
                                         descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                         (mcspInt)blockSize, info, B, (mcspInt)ldb, X, (mcspInt)ldx, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateBsrsm2Info(mcsparseBsrsm2Info_t* info) {
    mcspStatus_t ret = mcspCreateBsrsm2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyBsrsm2Info(mcsparseBsrsm2Info_t info) {
    mcspStatus_t ret = mcspDestroyBsrsm2Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgebsr2gebsc_bufferSize(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                 const float* bsrSortedVal, const int* bsrSortedRowPtr,
                                                 const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                 int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspSgebsr2gebsc_bufferSize(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal,
                                                   (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                   (mcspInt)rowBlockDim, (mcspInt)colBlockDim, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgebsr2gebsc_bufferSize(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                 const double* bsrSortedVal, const int* bsrSortedRowPtr,
                                                 const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                 int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspDgebsr2gebsc_bufferSize(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal,
                                                   (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                   (mcspInt)rowBlockDim, (mcspInt)colBlockDim, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgebsr2gebsc_bufferSize(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                 const mcFloatComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                                 const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                 int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCgebsr2gebsc_bufferSize(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal,
                                                   (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                   (mcspInt)rowBlockDim, (mcspInt)colBlockDim, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgebsr2gebsc_bufferSize(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                 const mcDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                                 const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                 int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspZgebsr2gebsc_bufferSize(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal,
                                                   (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                   (mcspInt)rowBlockDim, (mcspInt)colBlockDim, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgebsr2gebsc_bufferSizeExt(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                    const float* bsrSortedVal, const int* bsrSortedRowPtr,
                                                    const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                    size_t* pBufferSize) {
    mcspStatus_t ret = mcspSgebsr2gebsc_bufferSizeExt(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal,
                                                      (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                      (mcspInt)rowBlockDim, (mcspInt)colBlockDim, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgebsr2gebsc_bufferSizeExt(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                    const double* bsrSortedVal, const int* bsrSortedRowPtr,
                                                    const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                    size_t* pBufferSize) {
    mcspStatus_t ret = mcspDgebsr2gebsc_bufferSizeExt(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal,
                                                      (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                      (mcspInt)rowBlockDim, (mcspInt)colBlockDim, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgebsr2gebsc_bufferSizeExt(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                    const mcFloatComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                                    const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                    size_t* pBufferSize) {
    mcspStatus_t ret = mcspCgebsr2gebsc_bufferSizeExt(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal,
                                                      (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                      (mcspInt)rowBlockDim, (mcspInt)colBlockDim, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgebsr2gebsc_bufferSizeExt(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                                    const mcDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                                    const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                                    size_t* pBufferSize) {
    mcspStatus_t ret = mcspZgebsr2gebsc_bufferSizeExt(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal,
                                                      (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                      (mcspInt)rowBlockDim, (mcspInt)colBlockDim, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgebsr2gebsc(mcsparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal,
                                      const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim,
                                      int colBlockDim, float* bscVal, int* bscRowInd, int* bscColPtr,
                                      mcsparseAction_t copyValues, mcsparseIndexBase_t idxBase, void* pBuffer) {
    mcspStatus_t ret =
        mcspSgebsr2gebsc(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                         (mcspInt*)bsrSortedColInd, (mcspInt)rowBlockDim, (mcspInt)colBlockDim, bscVal,
                         (mcspInt*)bscRowInd, (mcspInt*)bscColPtr, copyValues, idxBase, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgebsr2gebsc(mcsparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal,
                                      const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim,
                                      int colBlockDim, double* bscVal, int* bscRowInd, int* bscColPtr,
                                      mcsparseAction_t copyValues, mcsparseIndexBase_t idxBase, void* pBuffer) {
    mcspStatus_t ret =
        mcspDgebsr2gebsc(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                         (mcspInt*)bsrSortedColInd, (mcspInt)rowBlockDim, (mcspInt)colBlockDim, bscVal,
                         (mcspInt*)bscRowInd, (mcspInt*)bscColPtr, copyValues, idxBase, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgebsr2gebsc(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                      const mcFloatComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                      const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                      mcFloatComplex* bscVal, int* bscRowInd, int* bscColPtr,
                                      mcsparseAction_t copyValues, mcsparseIndexBase_t idxBase, void* pBuffer) {
    mcspStatus_t ret =
        mcspCgebsr2gebsc(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                         (mcspInt*)bsrSortedColInd, (mcspInt)rowBlockDim, (mcspInt)colBlockDim, bscVal,
                         (mcspInt*)bscRowInd, (mcspInt*)bscColPtr, copyValues, idxBase, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgebsr2gebsc(mcsparseHandle_t handle, int mb, int nb, int nnzb,
                                      const mcDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr,
                                      const int* bsrSortedColInd, int rowBlockDim, int colBlockDim,
                                      mcDoubleComplex* bscVal, int* bscRowInd, int* bscColPtr,
                                      mcsparseAction_t copyValues, mcsparseIndexBase_t idxBase, void* pBuffer) {
    mcspStatus_t ret =
        mcspZgebsr2gebsc(handle, (mcspInt)mb, (mcspInt)nb, (mcspInt)nnzb, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                         (mcspInt*)bsrSortedColInd, (mcspInt)rowBlockDim, (mcspInt)colBlockDim, bscVal,
                         (mcspInt*)bscRowInd, (mcspInt*)bscColPtr, copyValues, idxBase, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateBsrilu02Info(mcsparseBsrilu02Info_t* info) {
    mcspStatus_t ret = mcspCreateBsrilu02Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyBsrilu02Info(mcsparseBsrilu02Info_t info) {
    mcspStatus_t ret = mcspDestroyBsrilu02Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int enable_boost,
                                                double* tol, float* boost_val) {
    mcspStatus_t ret = mcspSbsrilu02_numericBoost(handle, info, (mcspInt)enable_boost, tol, boost_val);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int enable_boost,
                                                double* tol, double* boost_val) {
    mcspStatus_t ret = mcspDbsrilu02_numericBoost(handle, info, (mcspInt)enable_boost, tol, boost_val);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int enable_boost,
                                                double* tol, mcFloatComplex* boost_val) {
    mcspStatus_t ret = mcspCbsrilu02_numericBoost(handle, info, (mcspInt)enable_boost, tol, boost_val);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrilu02_numericBoost(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int enable_boost,
                                                double* tol, mcDoubleComplex* boost_val) {
    mcspStatus_t ret = mcspZbsrilu02_numericBoost(handle, info, (mcspInt)enable_boost, tol, boost_val);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXbsrilu02_zeroPivot(mcsparseHandle_t handle, mcsparseBsrilu02Info_t info, int* position) {
    mcspStatus_t ret = mcspXbsrilu02_zeroPivot(handle, info, position);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrilu02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                              const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                              const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                              mcsparseBsrilu02Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspSbsrilu02_bufferSize(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                                info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrilu02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                              const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                              const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                              mcsparseBsrilu02Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspDbsrilu02_bufferSize(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                                info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrilu02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                              const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                              const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                              mcsparseBsrilu02Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCbsrilu02_bufferSize(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                                info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrilu02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                              const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                              const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                              mcsparseBsrilu02Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspZbsrilu02_bufferSize(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                                info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrilu02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                 const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                                 const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                 mcsparseBsrilu02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspSbsrilu02_bufferSizeExt(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                   (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                   (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrilu02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                 const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                                 const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                 mcsparseBsrilu02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspDbsrilu02_bufferSizeExt(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                   (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                   (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrilu02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                 const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                                 const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                 mcsparseBsrilu02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCbsrilu02_bufferSizeExt(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                   (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                   (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrilu02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                 const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                                 const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                 mcsparseBsrilu02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspZbsrilu02_bufferSizeExt(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                   (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                   (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrilu02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                            mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspSbsrilu02_analysis(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                              (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                              info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrilu02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                            mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspDbsrilu02_analysis(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                              (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                              info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrilu02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                            mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspCbsrilu02_analysis(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                              (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                              info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrilu02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                            const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                            const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                            mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret = mcspZbsrilu02_analysis(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                              (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                              info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsrilu02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                   const mcsparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr,
                                   const int* bsrSortedColInd, int blockDim, mcsparseBsrilu02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret =
        mcspSbsrilu02(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                      (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsrilu02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                   const mcsparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr,
                                   const int* bsrSortedColInd, int blockDim, mcsparseBsrilu02Info_t info,
                                   mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret =
        mcspDbsrilu02(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                      (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsrilu02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                   const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                   const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                   mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret =
        mcspCbsrilu02(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                      (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsrilu02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                   const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                   const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                   mcsparseBsrilu02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret =
        mcspZbsrilu02(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                      (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCreateBsric02Info(mcsparseBsric02Info_t* info) {
    mcspStatus_t ret = mcspCreateBsric02Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDestroyBsric02Info(mcsparseBsric02Info_t info) {
    mcspStatus_t ret = mcspDestroyBsric02Info(info);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseXbsric02_zeroPivot(mcsparseHandle_t handle, mcsparseBsric02Info_t info, int* position) {
    mcspStatus_t ret = mcspXbsric02_zeroPivot(handle, info, position);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsric02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                             const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                             const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                             mcsparseBsric02Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspSbsric02_bufferSize(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                               (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                               info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsric02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                             const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                             const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                             mcsparseBsric02Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspDbsric02_bufferSize(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                               (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                               info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsric02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                             const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                             const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                             mcsparseBsric02Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspCbsric02_bufferSize(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                               (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                               info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsric02_bufferSize(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                             const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                             const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                             mcsparseBsric02Info_t info, int* pBufferSizeInBytes) {
    mcspStatus_t ret = mcspZbsric02_bufferSize(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                               (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd, (mcspInt)blockDim,
                                               info, pBufferSizeInBytes);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsric02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                const mcsparseMatDescr_t descrA, float* bsrSortedVal,
                                                const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                mcsparseBsric02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspSbsric02_bufferSizeExt(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                  (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                  (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsric02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                const mcsparseMatDescr_t descrA, double* bsrSortedVal,
                                                const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                mcsparseBsric02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspDbsric02_bufferSizeExt(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                  (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                  (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsric02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                                const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                mcsparseBsric02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspCbsric02_bufferSizeExt(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                  (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                  (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsric02_bufferSizeExt(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                                const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                                const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize,
                                                mcsparseBsric02Info_t info, size_t* pBufferSize) {
    mcspStatus_t ret = mcspZbsric02_bufferSizeExt(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal,
                                                  (mcspInt*)bsrSortedRowPtr, (mcspInt*)bsrSortedColInd,
                                                  (mcspInt)blockSize, info, pBufferSize);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsric02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                           const mcsparseMatDescr_t descrA, const float* bsrSortedVal,
                                           const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                           mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy,
                                           void* pInputBuffer) {
    mcspStatus_t ret =
        mcspSbsric02_analysis(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                              (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pInputBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsric02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                           const mcsparseMatDescr_t descrA, const double* bsrSortedVal,
                                           const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                           mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy,
                                           void* pInputBuffer) {
    mcspStatus_t ret =
        mcspDbsric02_analysis(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                              (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pInputBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsric02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                           const mcsparseMatDescr_t descrA, const mcFloatComplex* bsrSortedVal,
                                           const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                           mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy,
                                           void* pInputBuffer) {
    mcspStatus_t ret =
        mcspCbsric02_analysis(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                              (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pInputBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsric02_analysis(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                           const mcsparseMatDescr_t descrA, const mcDoubleComplex* bsrSortedVal,
                                           const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                           mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy,
                                           void* pInputBuffer) {
    mcspStatus_t ret =
        mcspZbsric02_analysis(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                              (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pInputBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSbsric02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                  const mcsparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr,
                                  const int* bsrSortedColInd, int blockDim, mcsparseBsric02Info_t info,
                                  mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret =
        mcspSbsric02(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                     (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDbsric02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                  const mcsparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr,
                                  const int* bsrSortedColInd, int blockDim, mcsparseBsric02Info_t info,
                                  mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret =
        mcspDbsric02(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                     (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCbsric02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                  const mcsparseMatDescr_t descrA, mcFloatComplex* bsrSortedVal,
                                  const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                  mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret =
        mcspCbsric02(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                     (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZbsric02(mcsparseHandle_t handle, mcsparseDirection_t dirA, int mb, int nnzb,
                                  const mcsparseMatDescr_t descrA, mcDoubleComplex* bsrSortedVal,
                                  const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
                                  mcsparseBsric02Info_t info, mcsparseSolvePolicy_t policy, void* pBuffer) {
    mcspStatus_t ret =
        mcspZbsric02(handle, dirA, (mcspInt)mb, (mcspInt)nnzb, descrA, bsrSortedVal, (mcspInt*)bsrSortedRowPtr,
                     (mcspInt*)bsrSortedColInd, (mcspInt)blockDim, info, policy, pBuffer);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCooSetStridedBatch(mcsparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride) {
    mcspStatus_t ret = mcspCooSetStridedBatch(spMatDescr, batchCount, batchStride);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCsrSetStridedBatch(mcsparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride,
                                            int64_t columnsValuesBatchStride) {
    mcspStatus_t ret = mcspCsrSetStridedBatch(spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnMatSetStridedBatch(mcsparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride) {
    mcspStatus_t ret = mcspDnMatSetStridedBatch(dnMatDescr, batchCount, batchStride);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDnMatGetStridedBatch(mcsparseDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride) {
    mcspStatus_t ret = mcspDnMatGetStridedBatch(dnMatDescr, (mcspInt*)batchCount, batchStride);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrmm2(mcsparseHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m,
                                 int n, int k, int nnz, const float* alpha, const mcsparseMatDescr_t descrA,
                                 const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const float* B,
                                 int ldb, const float* beta, float* C, int ldc) {
    mcspStatus_t ret = mcspCuinScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
                                     csrColIndA, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrmm2(mcsparseHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m,
                                 int n, int k, int nnz, const double* alpha, const mcsparseMatDescr_t descrA,
                                 const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const double* B,
                                 int ldb, const double* beta, double* C, int ldc) {
    mcspStatus_t ret = mcspCuinDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
                                     csrColIndA, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrmm2(mcsparseHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m,
                                 int n, int k, int nnz, const mcspComplexFloat* alpha, const mcsparseMatDescr_t descrA,
                                 const mcspComplexFloat* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                 const mcspComplexFloat* B, int ldb, const mcspComplexFloat* beta, mcspComplexFloat* C,
                                 int ldc) {
    mcspStatus_t ret = mcspCuinCcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
                                     csrColIndA, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrmm2(mcsparseHandle_t handle, mcsparseOperation_t transA, mcsparseOperation_t transB, int m,
                                 int n, int k, int nnz, const mcspComplexDouble* alpha, const mcsparseMatDescr_t descrA,
                                 const mcspComplexDouble* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                 const mcspComplexDouble* B, int ldb, const mcspComplexDouble* beta,
                                 mcspComplexDouble* C, int ldc) {
    mcspStatus_t ret = mcspCuinZcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
                                     csrColIndA, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrmm(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int k, int nnz,
                                const float* alpha, const mcsparseMatDescr_t descrA, const float* csrValA,
                                const int* csrRowPtrA, const int* csrColIndA, const float* B, int ldb,
                                const float* beta, float* C, int ldc) {
    mcspStatus_t ret = mcspCuinScsrmm2(handle, transA, MCSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, nnz, alpha, descrA,
                                     csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrmm(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int k, int nnz,
                                const double* alpha, const mcsparseMatDescr_t descrA, const double* csrValA,
                                const int* csrRowPtrA, const int* csrColIndA, const double* B, int ldb,
                                const double* beta, double* C, int ldc) {
    mcspStatus_t ret = mcspCuinDcsrmm2(handle, transA, MCSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, nnz, alpha, descrA,
                                     csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrmm(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int k, int nnz,
                                const mcspComplexFloat* alpha, const mcsparseMatDescr_t descrA,
                                const mcspComplexFloat* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                const mcspComplexFloat* B, int ldb, const mcspComplexFloat* beta, mcspComplexFloat* C,
                                int ldc) {
    mcspStatus_t ret = mcspCuinCcsrmm2(handle, transA, MCSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, nnz, alpha, descrA,
                                     csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrmm(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int k, int nnz,
                                const mcspComplexDouble* alpha, const mcsparseMatDescr_t descrA,
                                const mcspComplexDouble* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                const mcspComplexDouble* B, int ldb, const mcspComplexDouble* beta,
                                mcspComplexDouble* C, int ldc) {
    mcspStatus_t ret = mcspCuinZcsrmm2(handle, transA, MCSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, nnz, alpha, descrA,
                                     csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseScsrmv(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                                const float* alpha, const mcsparseMatDescr_t descrA, const float* csrValA,
                                const int* csrRowPtrA, const int* csrColIndA, const float* x, const float* beta,
                                float* y) {
    mcspStatus_t ret =
        mcspCuinScsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDcsrmv(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                                const double* alpha, const mcsparseMatDescr_t descrA, const double* csrValA,
                                const int* csrRowPtrA, const int* csrColIndA, const double* x, const double* beta,
                                double* y) {
    mcspStatus_t ret =
        mcspCuinDcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCcsrmv(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                                const mcspComplexFloat* alpha, const mcsparseMatDescr_t descrA,
                                const mcspComplexFloat* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                const mcspComplexFloat* x, const mcspComplexFloat* beta, mcspComplexFloat* y) {
    mcspStatus_t ret =
        mcspCuinCcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZcsrmv(mcsparseHandle_t handle, mcsparseOperation_t transA, int m, int n, int nnz,
                                const mcspComplexDouble* alpha, const mcsparseMatDescr_t descrA,
                                const mcspComplexDouble* csrValA, const int* csrRowPtrA, const int* csrColIndA,
                                const mcspComplexDouble* x, const mcspComplexDouble* beta, mcspComplexDouble* y) {
    mcspStatus_t ret =
        mcspCuinZcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, beta, y);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsv(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du,
                               float* B, int ldb) {
    mcspStatus_t ret = mcspSgtsv10x(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsv(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                               const double* du, double* B, int ldb) {
    mcspStatus_t ret = mcspDgtsv10x(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsv(mcsparseHandle_t handle, int m, int n, const mcComplex* dl, const mcComplex* d,
                               const mcComplex* du, mcComplex* B, int ldb) {
    mcspStatus_t ret = mcspCgtsv10x(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsv(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* dl,
                               const mcDoubleComplex* d, const mcDoubleComplex* du, mcDoubleComplex* B, int ldb) {
    mcspStatus_t ret = mcspZgtsv10x(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsv_nopivot(mcsparseHandle_t handle, int m, int n, const float* dl, const float* d,
                                       const float* du, float* B, int ldb) {
    mcspStatus_t ret = mcspSgtsv_nopivot10x(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsv_nopivot(mcsparseHandle_t handle, int m, int n, const double* dl, const double* d,
                                       const double* du, double* B, int ldb) {
    mcspStatus_t ret = mcspDgtsv_nopivot10x(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsv_nopivot(mcsparseHandle_t handle, int m, int n, const mcComplex* dl, const mcComplex* d,
                                       const mcComplex* du, mcComplex* B, int ldb) {
    mcspStatus_t ret = mcspCgtsv_nopivot10x(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsv_nopivot(mcsparseHandle_t handle, int m, int n, const mcDoubleComplex* dl,
                                       const mcDoubleComplex* d, const mcDoubleComplex* du, mcDoubleComplex* B,
                                       int ldb) {
    mcspStatus_t ret = mcspZgtsv_nopivot10x(handle, (mcspInt)m, (mcspInt)n, dl, d, du, B, (mcspInt)ldb);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseSgtsvStridedBatch(mcsparseHandle_t handle, int m, const float* dl, const float* d,
                                           const float* du, float* x, int batch_count, int batch_stride) {
    mcspStatus_t ret =
        mcspSgtsvStridedBatch10x(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batch_count, (mcspInt)batch_stride);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseDgtsvStridedBatch(mcsparseHandle_t handle, int m, const double* dl, const double* d,
                                           const double* du, double* x, int batch_count, int batch_stride) {
    mcspStatus_t ret =
        mcspDgtsvStridedBatch10x(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batch_count, (mcspInt)batch_stride);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseCgtsvStridedBatch(mcsparseHandle_t handle, int m, const mcComplex* dl, const mcComplex* d,
                                           const mcComplex* du, mcComplex* x, int batch_count, int batch_stride) {
    mcspStatus_t ret =
        mcspCgtsvStridedBatch10x(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batch_count, (mcspInt)batch_stride);
    return mcspToSparseStatus(ret);
}
mcsparseStatus_t mcsparseZgtsvStridedBatch(mcsparseHandle_t handle, int m, const mcDoubleComplex* dl,
                                           const mcDoubleComplex* d, const mcDoubleComplex* du, mcDoubleComplex* x,
                                           int batch_count, int batch_stride) {
    mcspStatus_t ret =
        mcspZgtsvStridedBatch10x(handle, (mcspInt)m, dl, d, du, x, (mcspInt)batch_count, (mcspInt)batch_stride);
    return mcspToSparseStatus(ret);
}
#ifdef __cplusplus
}
#endif
