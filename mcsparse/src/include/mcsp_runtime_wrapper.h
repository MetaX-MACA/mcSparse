#ifndef COMMON_MCSP_RUNTIME_WRAPPER_H_
#define COMMON_MCSP_RUNTIME_WRAPPER_H_

#include <mcr/mc_runtime.h>
#include "mcsp_internal_interface.h"


// API redefinition
#define mcLaunchKernelGGLInternal(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...)   \
    do {                                                                                           \
        kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(__VA_ARGS__);         \
    } while (0)

#define mcLaunchKernelGGL(kernelName, ...) mcLaunchKernelGGLInternal((kernelName), __VA_ARGS__)

#endif