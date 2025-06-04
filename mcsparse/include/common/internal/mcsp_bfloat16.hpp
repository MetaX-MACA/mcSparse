
#ifndef COMMON_MCSP_INTERNAL_BFLOAT16_HPP_
#define COMMON_MCSP_INTERNAL_BFLOAT16_HPP_

#include "stdint.h"

#if defined(__MACA__)
#include "common/maca_bfloat16.h"
using mcsp_bfloat16 = __maca_bfloat16;
using mcsp_bfloat162 = __maca_bfloat162;
#else
// pseudo bfloat for mc compiling
struct __mc_bf16 {
    uint16_t x;
};
struct __mc_bf162 {
    uint32_t x;
};
using mcsp_bfloat16 = __mc_bf16;
using mcsp_bfloat162 = __mc_bf162;
#endif

#endif  // COMMON_MCSP_INTERNAL_BFLOAT16_HPP_
