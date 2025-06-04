#ifndef SRC_MCSP_INTERNAL_TYPES_H_
#define SRC_MCSP_INTERNAL_TYPES_H_

typedef uint32_t mcspInt;

typedef mcDoubleComplex mcspComplexDouble;
typedef mcFloatComplex mcspComplexFloat;

// internal types typedef
typedef struct mcsparseContext* mcspHandle_t;

// internal status enum
typedef enum {
    MCSP_STATUS_SUCCESS = 0,
    MCSP_STATUS_INVALID_HANDLE = 1,
    MCSP_STATUS_NOT_IMPLEMENTED = 2,
    MCSP_STATUS_INVALID_POINTER = 3,
    MCSP_STATUS_INVALID_SIZE = 4,
    MCSP_STATUS_MEMORY_ERROR = 5,
    MCSP_STATUS_INTERNAL_ERROR = 6,
    MCSP_STATUS_INVALID_VALUE = 7,
    MCSP_STATUS_ARCH_MISMATCH = 8,
    MCSP_STATUS_ZERO_PIVOT = 9,
    MCSP_STATUS_NOT_INITIALIZED = 10,
    MCSP_STATUS_TYPE_MISMATCH = 11,
    MCSP_STATUS_MAPPING_ERROR = 12,
    MCSP_STATUS_EXECUTION_FAILED = 13,
    MCSP_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 14,
    MCSP_STATUS_INSUFFICIENT_RESOURCES = 15
} mcspStatus_t;

// internal status enum
typedef enum {
    MCSP_MV_ALG_DEFAULT = 0,
    MCSP_COOMV_ALG = 1,
    MCSP_CSRMV_ALG1 = 2,
    MCSP_CSRMV_ALG2 = 3,
    MCSP_SPMV_ALG_DEFAULT = 0,
    MCSP_SPMV_CSR_ALG1 = 2,
    MCSP_SPMV_CSR_ALG2 = 3,
    MCSP_SPMV_COO_ALG1 = 1,
    MCSP_SPMV_COO_ALG2 = 4,
} mcspSpMVAlg_t;

struct mcspTrmInfo {
    int64_t max_row_nnz = 0;
    void* row_map = nullptr;
    void* trm_diag_ind = nullptr;

    void* zero_pivot_array = nullptr;
    void* zero_pivot_lead = nullptr;
};
typedef struct mcspTrmInfo* mcspTrmInfo_t;

struct mcspMatInfo {
    mcspTrmInfo_t csrilu0_info = nullptr;
    mcspTrmInfo_t csric0_info = nullptr;
    mcspTrmInfo_t csr_spsv_lower_info = nullptr;
    mcspTrmInfo_t csr_spsv_upper_info = nullptr;
    mcspTrmInfo_t csr_spsm_lower_info = nullptr;
    mcspTrmInfo_t csr_spsm_upper_info = nullptr;
    mcspTrmInfo_t bsrsv_lower_info = nullptr;
    mcspTrmInfo_t bsrsv_upper_info = nullptr;
    mcspTrmInfo_t bsrsm_lower_info = nullptr;
    mcspTrmInfo_t bsrsm_upper_info = nullptr;
    mcspTrmInfo_t bsrilu0_info = nullptr;
    mcspTrmInfo_t bsric0_info = nullptr;

    // extra device buffer used in APIs which need to-csc conversion
    size_t fixed_length_buffer_size = 0;
    size_t assist_index_buffer_size = 0;
    void* to_csc_rows = nullptr;
    void* to_csc_cols = nullptr;
    void* to_csc_vals = nullptr;

    // extra device buffer used in APIs which need transpose bsr
    void* bsrt_rows = nullptr;
    void* bsrt_cols = nullptr;
    void* bsrt_vals = nullptr;
    mcsparseDirection_t bsrt_dir = MCSPARSE_DIRECTION_ROW;

    int zero_pivot_lead = -1;
};
typedef struct mcspMatInfo* mcspMatInfo_t;

struct mcsparseMatDescr {
    mcsparseMatrixType_t type = MCSPARSE_MATRIX_TYPE_GENERAL;
    mcsparseFillMode_t fill_mode = MCSPARSE_FILL_MODE_FULL;
    mcsparseDiagType_t diag_type = MCSPARSE_DIAG_TYPE_NON_UNIT;
    mcsparseStorageMode_t storage_mode = MCSPARSE_STORAGE_MODE_SORTED;
    mcsparseIndexBase_t base = MCSPARSE_INDEX_BASE_ZERO;
};
typedef struct mcsparseMatDescr mcspMatDescr;
typedef struct mcsparseMatDescr* mcspMatDescr_t;

struct mcsparseCsrgemm2Info {
    mcspMatInfo_t mat_info = nullptr;
    bool alpha_null = true;
    bool beta_null = true;
};
typedef struct mcsparseCsrgemm2Info mcspCsrgemm2Info;
typedef struct mcsparseCsrgemm2Info* mcspCsrgemm2Info_t;

struct mcsparseColorInfo {
    mcspMatInfo_t mat_info = nullptr;
    mcsparseColorAlg_t algo = MCSPARSE_COLOR_ALG0;
};
typedef struct mcsparseColorInfo mcspColorInfo;
typedef struct mcsparseColorInfo* mcspColorInfo_t;

struct mcsparsePruneInfo {};
typedef struct mcsparsePruneInfo mcspPruneInfo;
typedef struct mcsparsePruneInfo* mcspPruneInfo_t;

struct mcsparseSpVecDescr {
    int64_t size = 0;
    int64_t nnz = 0;
    void* indices = nullptr;
    void* values = nullptr;
    mcsparseIndexType_t idxType = MCSPARSE_INDEX_32I;
    mcsparseIndexBase_t idxBase = MCSPARSE_INDEX_BASE_ZERO;
    macaDataType valueType = MACA_R_32F;
};
typedef struct mcsparseSpVecDescr mcspSpVecDescr;
typedef struct mcsparseSpVecDescr* mcspSpVecDescr_t;

struct mcsparseDnVecDescr {
    macaDataType valueType = MACA_R_32F;
    int64_t size = 0;
    void* values = nullptr;
};
typedef struct mcsparseDnVecDescr mcspDnVecDescr;
typedef struct mcsparseDnVecDescr* mcspDnVecDescr_t;

struct mcsparseDnMatDescr {
    mcspMatDescr_t mat_descr = nullptr;
    macaDataType valueType = MACA_R_32F;
    mcsparseOrder_t order = MCSPARSE_ORDER_COL;
    int64_t row_num = 0;
    int64_t col_num = 0;
    int64_t ld = 0;
    void* values = nullptr;
    int64_t batchCount = 1;
    int64_t batchStride = 0;
};
typedef struct mcsparseDnMatDescr mcspDnMatDescr;
typedef struct mcsparseDnMatDescr* mcspDnMatDescr_t;

mcspStatus_t mcspCreateTrmInfo(mcspTrmInfo_t* info);
mcspStatus_t mcspDestroyTrmInfo(mcspTrmInfo_t info);

struct mcspIlu0Config {
    // numeric boost for ilu0
    int boost_enable = 0;
    const void* boost_tol = nullptr;
    const void* boost_val = nullptr;
};

struct mcsparseCsrilu02Info {
    mcspMatInfo_t csrilu0_mat = nullptr;
};
typedef struct mcsparseCsrilu02Info mcspCsrilu02Info;
typedef struct mcsparseCsrilu02Info* mcspCsrilu02Info_t;

struct mcsparseCsric02Info {
    mcspMatInfo_t csric0_mat = nullptr;
};
typedef struct mcsparseCsric02Info mcspCsric02Info;
typedef struct mcsparseCsric02Info* mcspCsric02Info_t;

struct mcsparseBsrilu02Info {
    mcspMatInfo_t bsrilu0_mat = nullptr;
    mcspIlu0Config config;
};
typedef struct mcsparseBsrilu02Info mcspBsrilu02Info;
typedef struct mcsparseBsrilu02Info* mcspBsrilu02Info_t;

struct mcsparseBsric02Info {
    mcspMatInfo_t bsric0_mat = nullptr;
};
typedef struct mcsparseBsric02Info mcspBsric02Info;
typedef struct mcsparseBsric02Info* mcspBsric02Info_t;

struct mcsparseSpMatDescr {
    mcspMatDescr_t mat_descr = nullptr;
    mcsparseFormat_t format = MCSPARSE_FORMAT_CSR;
    mcsparseIndexType_t rowIdxType = MCSPARSE_INDEX_32I;
    mcsparseIndexType_t colIdxType = MCSPARSE_INDEX_32I;
    mcsparseIndexBase_t idxBase = MCSPARSE_INDEX_BASE_ZERO;
    macaDataType valueType = MACA_R_32F;
    int64_t row_num = 0;
    int64_t col_num = 0;
    int64_t nnz = 0;
    int64_t batchCount = 1;
    int64_t batchStride = 0;
    int64_t offsetsBatchStride = 0;
    // extra device buffer size used in APIs which need to-csc and to-csr conversion
    int is_buffersize_called = 0;
    size_t fixed_length_buffer_size = 0;
    size_t assist_index_buffer_size = 0;
    void* nnz_array = nullptr;
    void* rows = nullptr;
    void* cols = nullptr;
    void* vals = nullptr;

    // Variables for BlockedEll format
    int64_t ellBlockSize = 0;
    int64_t ellCols = 0;
    void* ellColInd = nullptr;
    void* ellValue = nullptr;

    // Variables for Aos Coo format
    void* coo_aos_ind = nullptr;

    // extra device pointers used in APIs which need to-csr conversion
    void* to_csr_rows = nullptr;
    void* to_csr_cols = nullptr;
    void* to_csr_vals = nullptr;
    // extra device pointers used in APIs which need to-csc conversion
    void* to_csc_rows = nullptr;
    void* to_csc_cols = nullptr;
    void* to_csc_vals = nullptr;
};
typedef struct mcsparseSpMatDescr mcspSpMatDescr;
typedef struct mcsparseSpMatDescr* mcspSpMatDescr_t;

struct mcsparseSpGEMMDescr {
    mcspMatInfo_t mat_info = nullptr;
    mcspMatDescr_t mat_descr = nullptr;
    macaDataType compute_type = MACA_R_32F;
    int64_t row_num = 0;
    int64_t col_num = 0;
    int64_t nnz = 0;
    bool is_reused_spgemm = false;
    void* rows = nullptr;
    void* cols = nullptr;
    void* vals = nullptr;
    void* buff = nullptr;
};
typedef struct mcsparseSpGEMMDescr mcspSpGEMMDescr;
typedef struct mcsparseSpGEMMDescr* mcspSpGEMMDescr_t;

struct mcsparseCsrsv2Info {
    mcspMatInfo_t mat_info = nullptr;
};
typedef struct mcsparseCsrsv2Info mcspCsrsv2Info;
typedef struct mcsparseCsrsv2Info* mcspCsrsv2Info_t;

struct mcsparseBsrsv2Info {
    mcspMatInfo_t mat_info = nullptr;
};
typedef struct mcsparseBsrsv2Info mcspBsrsv2Info;
typedef struct mcsparseBsrsv2Info* mcspBsrsv2Info_t;

struct mcsparseBsrsm2Info {
    mcspMatInfo_t mat_info = nullptr;
};
typedef struct mcsparseBsrsm2Info mcspBsrsm2Info;
typedef struct mcsparseBsrsm2Info* mcspBsrsm2Info_t;

struct mcsparseSpSVDescr {
    mcspMatInfo_t mat_info = nullptr;
    void* external_buffer = nullptr;
};
typedef struct mcsparseSpSVDescr mcspSpSVDescr;
typedef struct mcsparseSpSVDescr* mcspSpSVDescr_t;

struct mcsparseCsrsm2Info {
    mcspMatInfo_t mat_info = nullptr;
};
typedef struct mcsparseCsrsm2Info mcspCsrsm2Info;
typedef struct mcsparseCsrsm2Info* mcspCsrsm2Info_t;

struct mcsparseSpSMDescr {
    mcspMatInfo_t mat_info = nullptr;
    void* external_buffer = nullptr;
};
typedef struct mcsparseSpSMDescr mcspSpSMDescr;
typedef struct mcsparseSpSMDescr* mcspSpSMDescr_t;

struct mcsparseCsru2csrInfo {
    void* perm = nullptr;
};
typedef struct mcsparseCsru2csrInfo mcspCsru2csrInfo;
typedef struct mcsparseCsru2csrInfo* mcspCsru2csrInfo_t;

// TODO

struct mcsparseSpMMOpPlan;
typedef struct mcsparseSpMMOpPlan mcspSpMMOpPlan;
typedef struct mcsparseSpMMOpPlan* mcspSpMMOpPlan_t;

#endif
