#ifndef INTERFACE_MCSP_INTERNAL_LEVEL1_H_
#define INTERFACE_MCSP_INTERNAL_LEVEL1_H_

#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ******************************************************************************
 *   level 1 SPARSE
 * ******************************************************************************
 */

/**
 * @brief   Scale a sparse vector and add it to a dense vector
 *          y = alpha * x + y
 *
 * @param handle        [in]        handle of mcsp library
 * @param nnz           [in]        number of nonzeros
 * @param alpha         [in]        scale alpha
 * @param x_val         [in]        pointer to the array of nnz elements contains the values of x
 * @param x_ind         [in]        pointer to the array of nnz elements contains the index of the non-zero values of x
 * @param y             [inout]     pointer to the dense vector y
 * @param idx_base      [in]        MCSPARSE_INDEX_BASE_ZERO or MCSPARSE_INDEX_BASE_ONE
 * @return mcspStatus_t
 */
mcspStatus_t mcspSaxpyi(mcspHandle_t handle, mcspInt nnz, const float* alpha, const float* x_val, const mcspInt* x_ind,
                        float* y, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspDaxpyi(mcspHandle_t handle, mcspInt nnz, const double* alpha, const double* x_val,
                        const mcspInt* x_ind, double* y, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCaxpyi(mcspHandle_t handle, mcspInt nnz, const mcspComplexFloat* alpha, const mcspComplexFloat* x_val,
                        const mcspInt* x_ind, mcspComplexFloat* y, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspZaxpyi(mcspHandle_t handle, mcspInt nnz, const mcspComplexDouble* alpha,
                        const mcspComplexDouble* x_val, const mcspInt* x_ind, mcspComplexDouble* y,
                        mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinSaxpyi(mcspHandle_t handle, int nnz, const float* alpha, const float* x_val, const int* x_ind,
                          float* y, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinDaxpyi(mcspHandle_t handle, int nnz, const double* alpha, const double* x_val, const int* x_ind,
                          double* y, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinCaxpyi(mcspHandle_t handle, int nnz, const mcspComplexFloat* alpha, const mcspComplexFloat* x_val,
                          const int* x_ind, mcspComplexFloat* y, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinZaxpyi(mcspHandle_t handle, int nnz, const mcspComplexDouble* alpha, const mcspComplexDouble* x_val,
                          const int* x_ind, mcspComplexDouble* y, mcsparseIndexBase_t idx_base);

/**
 * @brief   Compute the dot product of a sparse vector with a dense vector
 *          result = $y^T \dot x$
 *
 * @param handle        [in]        handle of mcsp library
 * @param nnz           [in]        number of nonzeros
 * @param x_val         [in]        pointer to the array of nnz elements contains the values of x
 * @param x_ind         [in]        pointer to the array of nnz elements contains the index of the non-zero values of x
 * @param y             [in]        pointer to the dense vector y
 * @param result        [out]       pointer to the result
 * @param idx_base      [in]        MCSPARSE_INDEX_BASE_ZERO or MCSPARSE_INDEX_BASE_ONE
 * @return mcspStatus_t
 */

mcspStatus_t mcspSdoti(mcspHandle_t handle, mcspInt nnz, const float* x_val, const mcspInt* x_ind, const float* y,
                       float* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspDdoti(mcspHandle_t handle, mcspInt nnz, const double* x_val, const mcspInt* x_ind, const double* y,
                       double* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCdoti(mcspHandle_t handle, mcspInt nnz, const mcspComplexFloat* x_val, const mcspInt* x_ind,
                       const mcspComplexFloat* y, mcspComplexFloat* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspZdoti(mcspHandle_t handle, mcspInt nnz, const mcspComplexDouble* x_val, const mcspInt* x_ind,
                       const mcspComplexDouble* y, mcspComplexDouble* result, mcsparseIndexBase_t idx_base);

#if defined(__MACA__)
mcspStatus_t mcspR16fR32fDoti(mcspHandle_t handle, mcspInt nnz, const __half* x_val, const mcspInt* x_ind,
                              const __half* y, float* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspR16bfR32fDoti(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat16* x_val, const mcspInt* x_ind,
                               const mcsp_bfloat16* y, float* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspR8iR32fDoti(mcspHandle_t handle, mcspInt nnz, const int8_t* x_val, const mcspInt* x_ind,
                             const int8_t* y, float* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspR8iR32iDoti(mcspHandle_t handle, mcspInt nnz, const int8_t* x_val, const mcspInt* x_ind,
                             const int8_t* y, int32_t* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspC16fC32fDoti(mcspHandle_t handle, mcspInt nnz, const __half2* x_val, const mcspInt* x_ind,
                              const __half2* y, mcspComplexFloat* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspC16bfC32fDoti(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat162* x_val, const mcspInt* x_ind,
                               const mcsp_bfloat162* y, mcspComplexFloat* result, mcsparseIndexBase_t idx_base);

#endif

/**
 * @brief   Compute the dot product of the conjugate vector of a complex sparse vector with a dense vector
 *          result = $\hat{x}^H \dot y$
 *
 * @param handle        [in]        handle of mcsp library
 * @param nnz           [in]        number of nonzeros
 * @param x_val         [in]        pointer to the array of nnz elements contains the values of x
 * @param x_ind         [in]        pointer to the array of nnz elements contains the index of the non-zero values of x
 * @param y             [in]        pointer to the dense vector y
 * @param result        [out]       pointer to the result
 * @param idx_base      [in]        MCSPARSE_INDEX_BASE_ZERO or MCSPARSE_INDEX_BASE_ONE
 * @return mcspStatus_t
 */
mcspStatus_t mcspCdotci(mcspHandle_t handle, mcspInt nnz, const mcspComplexFloat* x_val, const mcspInt* x_ind,
                        const mcspComplexFloat* y, mcspComplexFloat* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspZdotci(mcspHandle_t handle, mcspInt nnz, const mcspComplexDouble* x_val, const mcspInt* x_ind,
                        const mcspComplexDouble* y, mcspComplexDouble* result, mcsparseIndexBase_t idx_base);

#if defined(__MACA__)
mcspStatus_t mcspC16fC32fDotci(mcspHandle_t handle, mcspInt nnz, const __half2* x_val, const mcspInt* x_ind,
                               const __half2* y, mcspComplexFloat* result, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspC16bfC32fDotci(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat162* x_val, const mcspInt* x_ind,
                                const mcsp_bfloat162* y, mcspComplexFloat* result, mcsparseIndexBase_t idx_base);
#endif

/**
 * @brief   Gather the elements of the vector y listed in the index array x_ind into the data array x_val.
 *
 * @param handle        [in]        handle of mcsp library
 * @param nnz           [in]        number of nonzeros
 * @param y             [in]        pointer to the dense vector y
 * @param x_val         [out]       pointer to the vector with nnz nonzero values that were gathered from vector y
 * @param x_ind         [in]        pointer to the integer vector with nnz indices of the nonzero values of vector x_val
 * @param idx_base      [in]        MCSPARSE_INDEX_BASE_ZERO or MCSPARSE_INDEX_BASE_ONE
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgthr(mcspHandle_t handle, mcspInt nnz, const float* y, float* x_val, const mcspInt* x_ind,
                       mcsparseIndexBase_t idx_base);

mcspStatus_t mcspDgthr(mcspHandle_t handle, mcspInt nnz, const double* y, double* x_val, const mcspInt* x_ind,
                       mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCgthr(mcspHandle_t handle, mcspInt nnz, const mcspComplexFloat* y, mcspComplexFloat* x_val,
                       const mcspInt* x_ind, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspZgthr(mcspHandle_t handle, mcspInt nnz, const mcspComplexDouble* y, mcspComplexDouble* x_val,
                       const mcspInt* x_ind, mcsparseIndexBase_t idx_base);

#if defined(__MACA__)
mcspStatus_t mcspR16Fgthr(mcspHandle_t handle, mcspInt nnz, const __half* y, __half* x_val, const mcspInt* x_ind,
                          mcsparseIndexBase_t idx_base);
#endif

#ifdef __MACA__
mcspStatus_t mcspC16Fgthr(mcspHandle_t handle, mcspInt nnz, const __half2* y, __half2* x_val, const mcspInt* x_ind,
                          mcsparseIndexBase_t idx_base);

mcspStatus_t mcspR16BFgthr(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat16* y, mcsp_bfloat16* x_val,
                           const mcspInt* x_ind, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspC16BFgthr(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat162* y, mcsp_bfloat162* x_val,
                           const mcspInt* x_ind, mcsparseIndexBase_t idx_base);

#endif

mcspStatus_t mcspCuinSgthr(mcspHandle_t handle, int nnz, const float* y, float* x_val, const int* x_ind,
                         mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinDgthr(mcspHandle_t handle, int nnz, const double* y, double* x_val, const int* x_ind,
                         mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinCgthr(mcspHandle_t handle, int nnz, const mcspComplexFloat* y, mcspComplexFloat* x_val,
                         const int* x_ind, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinZgthr(mcspHandle_t handle, int nnz, const mcspComplexDouble* y, mcspComplexDouble* x_val,
                         const int* x_ind, mcsparseIndexBase_t idx_base);

/**
 * @brief   Gather the elements of the vector y listed in the index array x_ind into the data array x_val and zeros out
 * the gathered elements in the vector y.
 *
 * @param handle        [in]        handle of mcsp library
 * @param nnz           [in]        number of nonzeros
 * @param y             [in/out]    pointer to the dense vector y with elements indexed by x_ind set to zero
 * @param x_val         [out]       pointer to the vector with nnz nonzero values that were gathered from vector y
 * @param x_ind         [in]        pointer to the integer vector with nnz indices of the nonzero values of vector x_val
 * @param idx_base      [in]        MCSPARSE_INDEX_BASE_ZERO or MCSPARSE_INDEX_BASE_ONE
 * @return mcspStatus_t
 */
mcspStatus_t mcspSgthrz(mcspHandle_t handle, mcspInt nnz, float* y, float* x_val, const mcspInt* x_ind,
                        mcsparseIndexBase_t idx_base);

mcspStatus_t mcspDgthrz(mcspHandle_t handle, mcspInt nnz, double* y, double* x_val, const mcspInt* x_ind,
                        mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCgthrz(mcspHandle_t handle, mcspInt nnz, mcspComplexFloat* y, mcspComplexFloat* x_val,
                        const mcspInt* x_ind, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspZgthrz(mcspHandle_t handle, mcspInt nnz, mcspComplexDouble* y, mcspComplexDouble* x_val,
                        const mcspInt* x_ind, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinSgthrz(mcspHandle_t handle, int nnz, float* y, float* x_val, const int* x_ind,
                          mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinDgthrz(mcspHandle_t handle, int nnz, double* y, double* x_val, const int* x_ind,
                          mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinCgthrz(mcspHandle_t handle, int nnz, mcspComplexFloat* y, mcspComplexFloat* x_val, const int* x_ind,
                          mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinZgthrz(mcspHandle_t handle, int nnz, mcspComplexDouble* y, mcspComplexDouble* x_val,
                          const int* x_ind, mcsparseIndexBase_t idx_base);

/**
 * @brief   Scatter the elements of the sparse vector x into the dense vector y.
 *
 * @param handle        [in]        handle of mcsp library
 * @param nnz           [in]        number of nonzeros
 * @param x_val         [in]        pointer to the vector with nnz nonzero values that were gathered from vector y
 * @param x_ind         [in]        pointer to the integer vector with nnz indices of the nonzero values of vector x_val
 * @param y             [out]       pointer to the dense vector y
 * @param idx_base      [in]        MCSPARSE_INDEX_BASE_ZERO or MCSPARSE_INDEX_BASE_ONE
 * @return mcspStatus_t
 */
mcspStatus_t mcspSsctr(mcspHandle_t handle, mcspInt nnz, const float* x_val, const mcspInt* x_ind, float* y,
                       mcsparseIndexBase_t idx_base);

mcspStatus_t mcspDsctr(mcspHandle_t handle, mcspInt nnz, const double* x_val, const mcspInt* x_ind, double* y,
                       mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCsctr(mcspHandle_t handle, mcspInt nnz, const mcspComplexFloat* x_val, const mcspInt* x_ind,
                       mcspComplexFloat* y, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspZsctr(mcspHandle_t handle, mcspInt nnz, const mcspComplexDouble* x_val, const mcspInt* x_ind,
                       mcspComplexDouble* y, mcsparseIndexBase_t idx_base);

#if defined(__MACA__)
mcspStatus_t mcspR16Fsctr(mcspHandle_t handle, mcspInt nnz, const __half* x_val, const mcspInt* x_ind, __half* y,
                          mcsparseIndexBase_t idx_base);
#endif

#ifdef __MACA__
mcspStatus_t mcspC16Fsctr(mcspHandle_t handle, mcspInt nnz, const __half2* x_val, const mcspInt* x_ind, __half2* y,
                          mcsparseIndexBase_t idx_base);

mcspStatus_t mcspR16BFsctr(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat16* x_val, const mcspInt* x_ind,
                           mcsp_bfloat16* y, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspC16BFsctr(mcspHandle_t handle, mcspInt nnz, const mcsp_bfloat162* x_val, const mcspInt* x_ind,
                           mcsp_bfloat162* y, mcsparseIndexBase_t idx_base);
#endif

mcspStatus_t mcspCuinSsctr(mcspHandle_t handle, int nnz, const float* x_val, const int* x_ind, float* y,
                         mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinDsctr(mcspHandle_t handle, int nnz, const double* x_val, const int* x_ind, double* y,
                         mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinCsctr(mcspHandle_t handle, int nnz, const mcspComplexFloat* x_val, const int* x_ind,
                         mcspComplexFloat* y, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinZsctr(mcspHandle_t handle, int nnz, const mcspComplexDouble* x_val, const int* x_ind,
                         mcspComplexDouble* y, mcsparseIndexBase_t idx_base);

/**
 * @brief   Compute the Givens rotation matrix to a sparse vecX and a dense vector vecY.
 *
 * @param handle        [in]        handle of mcsp library
 * @param nnz           [in]        number of nonzeros
 * @param x_val         [in/out]    pointer to the vector with nnz nonzero values that were gathered from vector y
 * @param x_ind         [in]        pointer to the integer vector with nnz indices of the nonzero values of vector x_val
 * @param c             [in]        pointer to the cosine value of the rotation
 * @param s             [in]        pointer to the sine value of the rotation
 * @param y             [out]       pointer to the dense vector y
 * @param idx_base      [in]        MCSPARSE_INDEX_BASE_ZERO or MCSPARSE_INDEX_BASE_ONE
 * @return mcspStatus_t
 */
mcspStatus_t mcspSroti(mcspHandle_t handle, mcspInt nnz, float* x_val, const mcspInt* x_ind, float* y, const float* c,
                       const float* s, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspDroti(mcspHandle_t handle, mcspInt nnz, double* x_val, const mcspInt* x_ind, double* y,
                       const double* c, const double* s, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCroti(mcspHandle_t handle, mcspInt nnz, mcspComplexFloat* x_val, const mcspInt* x_ind,
                       mcspComplexFloat* y, const mcspComplexFloat* c, const mcspComplexFloat* s,
                       mcsparseIndexBase_t idx_base);

mcspStatus_t mcspZroti(mcspHandle_t handle, mcspInt nnz, mcspComplexDouble* x_val, const mcspInt* x_ind,
                       mcspComplexDouble* y, const mcspComplexDouble* c, const mcspComplexDouble* s,
                       mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinSroti(mcspHandle_t handle, int nnz, float* x_val, const int* x_ind, float* y, const float* c,
                         const float* s, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspCuinDroti(mcspHandle_t handle, int nnz, double* x_val, const int* x_ind, double* y, const double* c,
                         const double* s, mcsparseIndexBase_t idx_base);

#if defined(__MACA__)
mcspStatus_t mcspR16fRoti(mcspHandle_t handle, mcspInt nnz, __half* x_val, const mcspInt* x_ind, __half* y,
                          const float* c, const float* s, mcsparseIndexBase_t idx_base);
#endif

#if defined(__MACA__)
mcspStatus_t mcspR16bfRoti(mcspHandle_t handle, mcspInt nnz, mcsp_bfloat16* x_val, const mcspInt* x_ind,
                           mcsp_bfloat16* y, const float* c, const float* s, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspC16fRoti(mcspHandle_t handle, mcspInt nnz, __half2* x_val, const mcspInt* x_ind, __half2* y,
                          const mcspComplexFloat* c, const mcspComplexFloat* s, mcsparseIndexBase_t idx_base);

mcspStatus_t mcspC16bfRoti(mcspHandle_t handle, mcspInt nnz, mcsp_bfloat162* x_val, const mcspInt* x_ind,
                           mcsp_bfloat162* y, const mcspComplexFloat* c, const mcspComplexFloat* s,
                           mcsparseIndexBase_t idx_base);
#endif

#ifdef __cplusplus
}
#endif

#endif  // end of INTERFACE_MCSP_LEVEL1_H_
