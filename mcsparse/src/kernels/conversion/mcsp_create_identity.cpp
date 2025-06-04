#include "common/mcsp_types.h"
#include "create_identity_device.hpp"
#include "mcsp_debug.h"
#include "mcsp_internal_types.h"

template <typename idxType>
mcspStatus_t mcspCreateIdentitlyTemplate(mcspHandle_t handle, idxType m, idxType* ibuffer) {
    if (handle == nullptr) {
        return MCSP_STATUS_INVALID_HANDLE;
    }
    if (m < 0) {
        return MCSP_STATUS_INVALID_SIZE;
    }

    if (ibuffer == nullptr) {
        return MCSP_STATUS_INVALID_POINTER;
    }

    if (m == 0) {
        return MCSP_STATUS_SUCCESS;
    }

    constexpr unsigned int n_elem = 512;
    int n_block = (m + n_elem - 1) / n_elem;
    mcStream_t stream = mcspGetStreamInternal(handle);

    mcLaunchKernelGGL(mcspCreateIdentityKernel, dim3(n_block), dim3(n_elem), 0, stream, m, ibuffer);

    return MCSP_STATUS_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

mcspStatus_t mcspCreateIdentityPermutation(mcspHandle_t handle, mcspInt n, mcspInt* p) {
    return mcspCreateIdentitlyTemplate(handle, n, p);
}

mcspStatus_t mcspCreateIdentityPermutation64(mcspHandle_t handle, int64_t n, int64_t* p) {
    return mcspCreateIdentitlyTemplate(handle, n, p);
}

mcspStatus_t mcspCuinCreateIdentityPermutation(mcspHandle_t handle, int n, int* p) {
    return mcspCreateIdentitlyTemplate(handle, (mcspInt)n, (mcspInt*)p);
}

#ifdef __cplusplus
}
#endif
