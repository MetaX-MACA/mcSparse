#ifndef INTERFACE_MCSP_INTERNAL_GENERIC_H_
#define INTERFACE_MCSP_INTERNAL_GENERIC_H_

#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief   Scale a sparse vector and add it to a dense vector
 *          y = alpha * x + beta * y
 *
 * @param handle        [in]        handle of mcsp library
 * @param nnz           [in]        number of nonzeros
 * @param alpha         [in]        scale alpha
 * @param x_val         [in]        pointer to the array of nnz elements contains the values of x
 * @param x_ind         [in]        pointer to the array of nnz elements contains the index of the non-zero values of x
 * @param beta          [in]        scale beta
 * @param y             [inout]     pointer to the dense vector y
 * @param idx_base      [in]        MCSPARSE_INDEX_BASE_ZERO or MCSPARSE_INDEX_BASE_ONE
 * @return mcspStatus_t
 */
mcspStatus_t mcspAxpby(mcspHandle_t handle, const void* alpha, mcspSpVecDescr_t vecX, const void* beta,
                       mcspDnVecDescr_t vecY);

mcspStatus_t mcspGather(mcspHandle_t handle, mcspDnVecDescr_t vecY, mcspSpVecDescr_t vecX);

mcspStatus_t mcspScatter(mcspHandle_t handle, mcspSpVecDescr_t vecX, mcspDnVecDescr_t vecY);

mcspStatus_t mcspRot(mcspHandle_t handle, const void* c_coeff, const void* s_coeff, mcspSpVecDescr_t vecX,
                     mcspDnVecDescr_t vecY);

mcspStatus_t mcspSparseToDense_bufferSize(mcspHandle_t handle, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB,
                                          mcsparseSparseToDenseAlg_t alg, size_t* bufferSize);
mcspStatus_t mcspSparseToDense(mcspHandle_t handle, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB,
                               mcsparseSparseToDenseAlg_t alg, void* externalBuffer);

mcspStatus_t mcspDenseToSparse_bufferSize(mcspHandle_t handle, mcspDnMatDescr_t matA, mcspSpMatDescr_t matB,
                                          mcsparseDenseToSparseAlg_t alg, size_t* bufferSize);
mcspStatus_t mcspDenseToSparse_analysis(mcspHandle_t handle, mcspDnMatDescr_t matA, mcspSpMatDescr_t matB,
                                        mcsparseDenseToSparseAlg_t alg, void* externalBuffer);
mcspStatus_t mcspDenseToSparse_convert(mcspHandle_t handle, mcspDnMatDescr_t matA, mcspSpMatDescr_t matB,
                                       mcsparseDenseToSparseAlg_t alg, void* externalBuffer);

// #############################################################################
// # SPARSE MATRIX - DENSE VECTOR MULTIPLICATION (SpMV)
// #############################################################################

mcspStatus_t mcspSpMV(mcspHandle_t handle, mcsparseOperation_t opA, const void* alpha, mcspSpMatDescr_t matA,
                      mcspDnVecDescr_t vecX, const void* beta, mcspDnVecDescr_t vecY, macaDataType computeType,
                      mcsparseSpMVAlg_t alg, void* externalBuffer);

mcspStatus_t mcspSpMV_bufferSize(mcspHandle_t handle, mcsparseOperation_t opA, const void* alpha, mcspSpMatDescr_t matA,
                                 mcspDnVecDescr_t vecX, const void* beta, mcspDnVecDescr_t vecY,
                                 macaDataType computeType, mcsparseSpMVAlg_t alg, size_t* bufferSize);

// #############################################################################
// # SPARSE MATRIX - DENSE MATRIX MULTIPLICATION (SpGEMM)
// #############################################################################

mcspStatus_t mcspSpMM_bufferSize(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void* alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, const void* beta,
                                 mcspDnMatDescr_t matC, macaDataType computeType, mcsparseSpMMAlg_t alg,
                                 size_t* bufferSize);

mcspStatus_t mcspSpMM_preprocess(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void* alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, const void* beta,
                                 mcspDnMatDescr_t matC, macaDataType computeType, mcsparseSpMMAlg_t alg,
                                 void* externalBuffer);

mcspStatus_t mcspSpMM(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void* alpha,
                      mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, const void* beta, mcspDnMatDescr_t matC,
                      macaDataType computeType, mcsparseSpMMAlg_t alg, void* externalBuffer);

// #############################################################################
// # SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM)
// #############################################################################

mcspStatus_t mcspSpGEMM_createDescr(mcspSpGEMMDescr_t* spgemm_descr);
mcspStatus_t mcspSpGEMM_destroyDescr(mcspSpGEMMDescr_t spgemm_descr);
mcspStatus_t mcspSpGEMM_workEstimation(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                       const void* alpha, mcspSpMatDescr_t matA, mcspSpMatDescr_t matB,
                                       const void* beta, mcspSpMatDescr_t matC, macaDataType computeType,
                                       mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemmDescr, size_t* bufferSize1,
                                       void* externalBuffer1);

mcspStatus_t mcspSpGEMM_compute(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                const void* alpha, mcspSpMatDescr_t matA, mcspSpMatDescr_t matB, const void* beta,
                                mcspSpMatDescr_t matC, macaDataType computeType, mcsparseSpGEMMAlg_t alg,
                                mcspSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2);

mcspStatus_t mcspSpGEMM_copy(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void* alpha,
                             mcspSpMatDescr_t matA, mcspSpMatDescr_t matB, const void* beta, mcspSpMatDescr_t matC,
                             macaDataType computeType, mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemmDescr);

mcspStatus_t mcspSpGEMMreuse_workEstimation(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                            mcspSpMatDescr_t matA, mcspSpMatDescr_t matB, mcspSpMatDescr_t matC,
                                            mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemmDescr, size_t* bufferSize1,
                                            void* externalBuffer1);

mcspStatus_t mcspSpGEMMreuse_nnz(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 mcspSpMatDescr_t matA, mcspSpMatDescr_t matB, mcspSpMatDescr_t matC,
                                 mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemmDescr, size_t* bufferSize2,
                                 void* externalBuffer2, size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize4,
                                 void* externalBuffer4);

mcspStatus_t mcspSpGEMMreuse_copy(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                  mcspSpMatDescr_t matA, mcspSpMatDescr_t matB, mcspSpMatDescr_t matC,
                                  mcsparseSpGEMMAlg_t alg, mcspSpGEMMDescr_t spgemmDescr, size_t* bufferSize5,
                                  void* externalBuffer5);

mcspStatus_t mcspSpGEMMreuse_compute(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                     const void* alpha, mcspSpMatDescr_t matA, mcspSpMatDescr_t matB, const void* beta,
                                     mcspSpMatDescr_t matC, macaDataType computeType, mcsparseSpGEMMAlg_t alg,
                                     mcspSpGEMMDescr_t spgemmDescr);
/**
 * @brief Sampled Dense-Dense Matrix Multiplication: calculate buffer size
 *
 * @param handle        [in]        handle of mcsp library
 * @param opA           [in]        matrix operation type of dense matrix A
 * @param opB           [in]        matrix operation type of dense matrix B
 * @param alpha         [in]        scalar alpha
 * @param A             [in]        descriptor of the dense matrix A
 * @param B             [in]        descriptor of the dense matrix B
 * @param beta          [in]        scalar beta
 * @param C             [in]        descriptor of the sparse matrix C
 * @param compute_type  [in]        floating point precision for the SDDMM computation
 * @param alg           [in]        specification of the algorithm to use
 * @param buffer_size   [out]       buffer size estimated by sddmm algorithm
 * @return mcspStatus_t
 */
mcspStatus_t mcspSddmmBufferSize(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void* alpha, const mcspDnMatDescr_t A, const mcspDnMatDescr_t B,
                                 const void* beta, mcspSpMatDescr_t C, macaDataType compute_type,
                                 mcsparseSDDMMAlg_t alg, size_t* buffer_size);

/**
 * @brief Sampled Dense-Dense Matrix Multiplication: preprocess step
 *
 * @param handle        [in]        handle of mcsp library
 * @param opA           [in]        matrix operation type of dense matrix A
 * @param opB           [in]        matrix operation type of dense matrix B
 * @param alpha         [in]        scalar alpha
 * @param A             [in]        descriptor of the dense matrix A
 * @param B             [in]        descriptor of the dense matrix B
 * @param beta          [in]        scalar beta
 * @param C             [in/out]    descriptor of the sparse matrix C
 * @param compute_type  [in]        floating point precision for the SDDMM computation
 * @param alg           [in]        specification of the algorithm to use
 * @param temp_buffer   [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSddmmPreprocess(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void* alpha, const mcspDnMatDescr_t A, const mcspDnMatDescr_t B,
                                 const void* beta, mcspSpMatDescr_t C, macaDataType compute_type,
                                 mcsparseSDDMMAlg_t alg, void* temp_buffer);

/**
 * @brief Sampled Dense-Dense Matrix Multiplication: computation step
 *
 * @param handle        [in]        handle of mcsp library
 * @param opA           [in]        matrix operation type of dense matrix A
 * @param opB           [in]        matrix operation type of dense matrix B
 * @param alpha         [in]        scalar alpha
 * @param A             [in]        descriptor of the dense matrix A
 * @param B             [in]        descriptor of the dense matrix B
 * @param beta          [in]        scalar beta
 * @param C             [in/out]    descriptor of the sparse matrix C
 * @param compute_type  [in]        floating point precision for the SDDMM computation
 * @param alg           [in]        specification of the algorithm to use
 * @param temp_buffer   [in]        temporary storage buffer allocated by the user
 * @return mcspStatus_t
 */
mcspStatus_t mcspSddmm(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void* alpha,
                       const mcspDnMatDescr_t A, const mcspDnMatDescr_t B, const void* beta, mcspSpMatDescr_t C,
                       macaDataType compute_type, mcsparseSDDMMAlg_t alg, void* temp_buffer);

// #############################################################################
// # SPARSE VECTOR - DENSE VECTOR MULTIPLICATION (SpVV)
// #############################################################################

mcspStatus_t mcspSpVV_bufferSize(mcspHandle_t handle, mcsparseOperation_t op_x, mcspSpVecDescr_t vec_x,
                                 mcspDnVecDescr_t vec_y, void* result, macaDataType compute_type, size_t* buffer_size);

mcspStatus_t mcspSpVV(mcspHandle_t handle, mcsparseOperation_t op_x, mcspSpVecDescr_t vec_x, mcspDnVecDescr_t vec_y,
                      void* result, macaDataType compute_type, void* temp_buffer);

// #############################################################################
// # SPARSE TRIANGULAR VECTOR SOLVE
// #############################################################################

mcspStatus_t mcspSpSV_createDescr(mcspSpSVDescr_t* descr);

mcspStatus_t mcspSpSV_destroyDescr(mcspSpSVDescr_t descr);

mcspStatus_t mcspCuinSpSV_bufferSize(mcspHandle_t handle, mcsparseOperation_t opA, const void* alpha,
                                   mcspSpMatDescr_t matA, mcspDnVecDescr_t vecX, mcspDnVecDescr_t vecY,
                                   macaDataType computeType, mcsparseSpSVAlg_t alg, mcspSpSVDescr_t spsvDescr,
                                   size_t* bufferSize);

mcspStatus_t mcspCuinSpSV_analysis(mcspHandle_t handle, mcsparseOperation_t opA, const void* alpha, mcspSpMatDescr_t matA,
                                 mcspDnVecDescr_t vecX, mcspDnVecDescr_t vecY, macaDataType computeType,
                                 mcsparseSpSVAlg_t alg, mcspSpSVDescr_t spsvDescr, void* externalBuffer);

mcspStatus_t mcspCuinSpSV_solve(mcspHandle_t handle, mcsparseOperation_t opA, const void* alpha, mcspSpMatDescr_t matA,
                              mcspDnVecDescr_t vecX, mcspDnVecDescr_t vecY, macaDataType computeType,
                              mcsparseSpSVAlg_t alg, mcspSpSVDescr_t spsvDescr);

// #############################################################################
// # SPARSE TRIANGULAR MATRIX SOLVE
// #############################################################################

mcspStatus_t mcspSpSM_createDescr(mcspSpSMDescr_t* descr);

mcspStatus_t mcspSpSM_destroyDescr(mcspSpSMDescr_t descr);

mcspStatus_t mcspSpSM_bufferSize(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB,
                                 const void* alpha, mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, mcspDnMatDescr_t matC,
                                 macaDataType computeType, mcsparseSpSMAlg_t alg, mcspSpSMDescr_t spsmDescr,
                                 size_t* bufferSize);

mcspStatus_t mcspSpSM_analysis(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void* alpha,
                               mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, mcspDnMatDescr_t matC,
                               macaDataType computeType, mcsparseSpSMAlg_t alg, mcspSpSMDescr_t spsmDescr,
                               void* externalBuffer);

mcspStatus_t mcspSpSM_solve(mcspHandle_t handle, mcsparseOperation_t opA, mcsparseOperation_t opB, const void* alpha,
                            mcspSpMatDescr_t matA, mcspDnMatDescr_t matB, mcspDnMatDescr_t matC,
                            macaDataType computeType, mcsparseSpSMAlg_t alg, mcspSpSMDescr_t spsmDescr);

mcspStatus_t mcspCooSetStridedBatch(mcspSpMatDescr_t spMatDescr, mcspInt batchCount, int64_t batchStride);

mcspStatus_t mcspCsrSetStridedBatch(mcspSpMatDescr_t spMatDescr, mcspInt batchCount, int64_t offsetsBatchStride,
                                    int64_t columnsValuesBatchStride);

mcspStatus_t mcspDnMatSetStridedBatch(mcspDnMatDescr_t dnMatDescr, mcspInt batchCount, int64_t batchStride);

mcspStatus_t mcspDnMatGetStridedBatch(mcspDnMatDescr_t dnMatDescr, mcspInt* batchCount, int64_t* batchStride);

#ifdef __cplusplus
}
#endif

#endif  // end of INTERFACE_MCSP_GENERIC_H_