#ifndef INTERFACE_MCSP_INTERNAL_REORDER_H_
#define INTERFACE_MCSP_INTERNAL_REORDER_H_

#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief   Coloring of the adjacency graph of the matrix A stored in the CSR format
 *          ref: A parallel graph coloring heuristic, 1993
 *
 * @param handle             [in]        handle of mcsp library
 * @param m                  [in]        number of rows of A
 * @param nnz                [in]        number of nonzeros of A
 * @param descr              [in]        descriptor of the sparse matrix A
 * @param csr_vals           [in]        pointer to the values of nonzeros in CSR matrix A
 * @param csr_rows           [in]        pointer to the row offset in CSR matrix A
 * @param csr_cols           [in]        pointer to the column indexes of nonzeros in CSR matrix A
 * @param fraction_to_color  [in]        fraction of nodes to be colored
 * @param ncolors            [out]       resulting number of distinct colors
 * @param coloring           [out]       resulting mapping of colors
 * @param reordering         [out]       optional resulting reordering permutation
 * @param info               [in/out]    meta data for CSR matrix A
 * @return mcspStatus_t
 */
mcspStatus_t mcspScsrcolor(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, float *csr_vals,
                           const mcspInt *csr_rows, const mcspInt *csr_cols, const float *fraction_to_color,
                           mcspInt *ncolors, mcspInt *coloring, mcspInt *reordering, mcspMatInfo_t info);
mcspStatus_t mcspDcsrcolor(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr, double *csr_vals,
                           const mcspInt *csr_rows, const mcspInt *csr_cols, const double *fraction_to_color,
                           mcspInt *ncolors, mcspInt *coloring, mcspInt *reordering, mcspMatInfo_t info);
mcspStatus_t mcspCcsrcolor(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                           mcspComplexFloat *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                           const float *fraction_to_color, mcspInt *ncolors, mcspInt *coloring, mcspInt *reordering,
                           mcspMatInfo_t info);
mcspStatus_t mcspZcsrcolor(mcspHandle_t handle, mcspInt m, mcspInt nnz, const mcspMatDescr_t descr,
                           mcspComplexDouble *csr_vals, const mcspInt *csr_rows, const mcspInt *csr_cols,
                           const double *fraction_to_color, mcspInt *ncolors, mcspInt *coloring, mcspInt *reordering,
                           mcspMatInfo_t info);

mcspStatus_t mcspCuinScsrColor(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, float *csr_vals,
                             const int *csr_rows, const int *csr_cols, const float *fraction_to_color, int *ncolors,
                             int *coloring, int *reordering, mcspColorInfo_t info);
mcspStatus_t mcspCuinDcsrColor(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr, double *csr_vals,
                             const int *csr_rows, const int *csr_cols, const double *fraction_to_color, int *ncolors,
                             int *coloring, int *reordering, mcspColorInfo_t info);
mcspStatus_t mcspCuinCcsrColor(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                             mcspComplexFloat *csr_vals, const int *csr_rows, const int *csr_cols,
                             const float *fraction_to_color, int *ncolors, int *coloring, int *reordering,
                             mcspColorInfo_t info);
mcspStatus_t mcspCuinZcsrColor(mcspHandle_t handle, int m, int nnz, const mcspMatDescr_t descr,
                             mcspComplexDouble *csr_vals, const int *csr_rows, const int *csr_cols,
                             const double *fraction_to_color, int *ncolors, int *coloring, int *reordering,
                             mcspColorInfo_t info);

#ifdef __cplusplus
}
#endif

#endif  // INTERFACE_MCSP_REORDER_H_
