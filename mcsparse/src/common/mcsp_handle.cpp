#include "mcsp_handle.h"

#include "mcsp_debug.h"

mcsparseContext::mcsparseContext() : poolbuffer_size(1024 * 1024 * 512) {
    MACA_ASSERT(mcGetDevice(&device));
    MACA_ASSERT(mcGetDeviceProperties(&properties, device));
}

mcsparseContext::~mcsparseContext() {
    MACA_ASSERT(mcFree(pool_buffer));
}

mcspStatus_t mcsparseContext::mcspMallocPoolBuffer() {
    auto ret = mcMalloc(&pool_buffer, poolbuffer_size);
    if (ret == mcSuccess) {
        return MCSP_STATUS_SUCCESS;
    } else {
        MACA_ASSERT(mcFree(pool_buffer));
        return MCSP_STATUS_MEMORY_ERROR;
    }
}

bool mcsparseContext::mcspUsePoolBuffer(void **buffer, size_t size) {
    if (poolbuffer_size < size) {
        return false;
    }
    if (buffer == nullptr) {
        return false;
    }
    int non_use_flag = 0;
    int in_use_flag = 1;
    bool use_buffer_pool = poolbuffer_inuse.compare_exchange_strong(non_use_flag, in_use_flag);

    if (use_buffer_pool) {
        *buffer = pool_buffer;
    }
    return use_buffer_pool;
}

mcspStatus_t mcsparseContext::mcspReturnPoolBuffer() {
    int non_use_flag = 0;
    int in_use_flag = 1;
    bool cas_status = poolbuffer_inuse.compare_exchange_strong(in_use_flag, non_use_flag);
    if (cas_status) {
        return MCSP_STATUS_SUCCESS;
    } else {
        return MCSP_STATUS_INTERNAL_ERROR;
    }
}
