// for mcsp wrapper
#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*mcsparseLoggerCallback_t)(int logLevel, const char *functionName, const char *message);

mcsparseStatus_t mcsparseLoggerSetCallback(mcsparseLoggerCallback_t callback) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}
mcsparseStatus_t mcsparseLoggerSetFile(FILE *file) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}
mcsparseStatus_t mcsparseLoggerOpenFile(const char *logFile) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}
mcsparseStatus_t mcsparseLoggerSetLevel(int level) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}
mcsparseStatus_t mcsparseLoggerSetMask(int mask) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}
mcsparseStatus_t mcsparseLoggerForceDisable(void) {
    return MCSPARSE_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif
