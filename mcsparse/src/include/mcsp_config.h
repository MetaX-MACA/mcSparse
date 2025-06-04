#ifndef COMMON_MCSP_CONFIG_H
#define COMMON_MCSP_CONFIG_H

#if defined(__MACA__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

#define UINT64_BIT_MASK 0xffffffffffffffff
#define UINT32_BIT_MASK 0xffffffff

#define ALIGNED_SIZE 128

#define MIN_BUFFER_SIZE ALIGNED_SIZE

#endif  // COMMON_MCSP_CONFIG_H
