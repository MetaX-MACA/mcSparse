#ifndef MCPRIM_PRIM_TYPES_H_
#define MCPRIM_PRIM_TYPES_H_

#include "mcsp_runtime_wrapper.h"

#ifndef NDEBUG
#define MACA_ASSERT(status) assert(status == MCSP_STATUS_SUCCESS)
#else
#define MACA_ASSERT(status) ((void)status)
#endif

namespace mcprim {

#define MCPRIM_DEVICE __device__
#define MCPRIM_KERNEL __global__
#define MCPRIM_INLINE __inline__
#define MCPRIM_FORCE_INLINE __forceinline__

typedef enum {
    MCPRIM_STATUS_SUCCESS = 0,
    MCPRIM_STATUS_INVALID_HANDLE = 1,
    MCPRIM_STATUS_NOT_IMPLEMENTED = 2,
    MCPRIM_STATUS_INVALID_POINTER = 3,
    MCPRIM_STATUS_INVALID_SIZE = 4,
    MCPRIM_STATUS_MEMORY_ERROR = 5,
    MCPRIM_STATUS_INTERNAL_ERROR = 6,
    MCPRIM_STATUS_INVALID_VALUE = 7,
    MCPRIM_STATUS_ARCH_MISMATCH = 8,
    MCPRIM_STATUS_ZERO_PIVOT = 9,
    MCPRIM_STATUS_NOT_INITIALIZED = 10,
    MCPRIM_STATUS_TYPE_MISMATCH = 11
} mcprimStatus_t;

template <class T>
struct plus {
    __host__ __device__ inline constexpr T operator()(const T& a, const T& b) const { return (a + b); }
};

template <class T>
struct maximum {
    __host__ __device__ inline constexpr T operator()(const T& a, const T& b) const { return (a < b ? b : a); }
};

template <class T>
struct minimum {
    __host__ __device__ inline constexpr T operator()(const T& a, const T& b) const { return (a < b ? a : b); }
};

}  // namespace mcprim

#endif
