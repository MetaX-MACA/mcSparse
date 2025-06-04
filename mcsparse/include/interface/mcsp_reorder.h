#ifndef INTERFACE_MCSPARSE_REORDER_H_
#define INTERFACE_MCSPARSE_REORDER_H_

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// csrColor
mcsparseStatus_t mcsparseScsrcolor(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                   float *csr_vals, const int *csr_rows, const int *csr_cols,
                                   const float *fraction_to_color, int *ncolors, int *coloring, int *reordering,
                                   mcsparseColorInfo_t info);
mcsparseStatus_t mcsparseDcsrcolor(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                   double *csr_vals, const int *csr_rows, const int *csr_cols,
                                   const double *fraction_to_color, int *ncolors, int *coloring, int *reordering,
                                   mcsparseColorInfo_t info);
mcsparseStatus_t mcsparseCcsrcolor(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                   mcComplex *csr_vals, const int *csr_rows, const int *csr_cols,
                                   const float *fraction_to_color, int *ncolors, int *coloring, int *reordering,
                                   mcsparseColorInfo_t info);
mcsparseStatus_t mcsparseZcsrcolor(mcsparseHandle_t handle, int m, int nnz, const mcsparseMatDescr_t descr,
                                   mcDoubleComplex *csr_vals, const int *csr_rows, const int *csr_cols,
                                   const double *fraction_to_color, int *ncolors, int *coloring, int *reordering,
                                   mcsparseColorInfo_t info);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSPARSE_REORDER_H_
