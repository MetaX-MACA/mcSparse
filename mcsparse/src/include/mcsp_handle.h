#ifndef COMMON_MCSP_HANDLE_H_
#define COMMON_MCSP_HANDLE_H_

#include <atomic>
#include <complex>

#include "common/mcsp_types.h"
#include "mcsp_internal_types.h"
#include "mcsp_runtime_wrapper.h"

struct mcsparseContext {
    mcsparseContext();
    ~mcsparseContext();

    mcspStatus_t mcspMallocPoolBuffer();
    bool mcspUsePoolBuffer(void **buffer, size_t size);
    mcspStatus_t mcspReturnPoolBuffer();

    int device;
    mcDeviceProp_t properties;
    mcStream_t stream = nullptr;
    mcsparsePointerMode_t ptr_mode = MCSPARSE_POINTER_MODE_HOST;

    const size_t poolbuffer_size;
    void *pool_buffer;
    std::atomic<int> poolbuffer_inuse = 0;
};
typedef struct mcsparseContext mcspHandle;

#endif
