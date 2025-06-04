#ifndef INTERFACE_MCSPARSE_LEVEL1_H_
#define INTERFACE_MCSPARSE_LEVEL1_H_

#include "common/mcsp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// axpyi
mcsparseStatus_t mcsparseSaxpyi(mcsparseHandle_t handle, int nnz, const float* alpha, const float* x_val,
                                const int* x_ind, float* y, mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseDaxpyi(mcsparseHandle_t handle, int nnz, const double* alpha, const double* x_val,
                                const int* x_ind, double* y, mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseCaxpyi(mcsparseHandle_t handle, int nnz, const mcComplex* alpha, const mcComplex* x_val,
                                const int* x_ind, mcComplex* y, mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseZaxpyi(mcsparseHandle_t handle, int nnz, const mcDoubleComplex* alpha,
                                const mcDoubleComplex* x_val, const int* x_ind, mcDoubleComplex* y,
                                mcsparseIndexBase_t idx_base);

// gthr
mcsparseStatus_t mcsparseSgthr(mcsparseHandle_t handle, int nnz, const float* y, float* x_val, const int* x_ind,
                               mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseDgthr(mcsparseHandle_t handle, int nnz, const double* y, double* x_val, const int* x_ind,
                               mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseCgthr(mcsparseHandle_t handle, int nnz, const mcComplex* y, mcComplex* x_val, const int* x_ind,
                               mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseZgthr(mcsparseHandle_t handle, int nnz, const mcDoubleComplex* y, mcDoubleComplex* x_val,
                               const int* x_ind, mcsparseIndexBase_t idx_base);

// gthrz
mcsparseStatus_t mcsparseSgthrz(mcsparseHandle_t handle, int nnz, float* y, float* x_val, const int* x_ind,
                                mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseDgthrz(mcsparseHandle_t handle, int nnz, double* y, double* x_val, const int* x_ind,
                                mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseCgthrz(mcsparseHandle_t handle, int nnz, mcComplex* y, mcComplex* x_val, const int* x_ind,
                                mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseZgthrz(mcsparseHandle_t handle, int nnz, mcDoubleComplex* y, mcDoubleComplex* x_val,
                                const int* x_ind, mcsparseIndexBase_t idx_base);

// sctr
mcsparseStatus_t mcsparseSsctr(mcsparseHandle_t handle, int nnz, const float* x_val, const int* x_ind, float* y,
                               mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseDsctr(mcsparseHandle_t handle, int nnz, const double* x_val, const int* x_ind, double* y,
                               mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseCsctr(mcsparseHandle_t handle, int nnz, const mcComplex* x_val, const int* x_ind, mcComplex* y,
                               mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseZsctr(mcsparseHandle_t handle, int nnz, const mcDoubleComplex* x_val, const int* x_ind,
                               mcDoubleComplex* y, mcsparseIndexBase_t idx_base);

// roti
mcsparseStatus_t mcsparseSroti(mcsparseHandle_t handle, int nnz, float* x_val, const int* x_ind, float* y,
                               const float* c, const float* s, mcsparseIndexBase_t idx_base);

mcsparseStatus_t mcsparseDroti(mcsparseHandle_t handle, int nnz, double* x_val, const int* x_ind, double* y,
                               const double* c, const double* s, mcsparseIndexBase_t idx_base);

#ifdef __cplusplus
}
#endif

#endif  // end of INTERFACE_MCSPARSE_LEVEL1_H_
