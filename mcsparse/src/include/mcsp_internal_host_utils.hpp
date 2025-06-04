#ifndef COMMON_MCSP_INTERNAL_HOST_UTILS_HPP_
#define COMMON_MCSP_INTERNAL_HOST_UTILS_HPP_

#include "common/internal/mcsp_bfloat16.hpp"
#include "common/mcsp_types.h"
#include "mcsp_debug.h"

template <typename valType>
static bool IsZero(valType in) {
#if defined(__MACA__)
    if constexpr (std::is_same_v<valType, __half>) {
        return (__half2float(in) == 0.f);
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
        return (__bfloat162float(in) == 0.f);
    } else if constexpr (std::is_same_v<valType, __half2> || std::is_same_v<valType, mcsp_bfloat162>) {
        return (__low2float(in) == 0.f && __high2float(in) == 0.f);
    } else {
        return (in == static_cast<valType>(0));
    }
#else
    return (in == static_cast<valType>(0));
#endif
}

static inline mcspInt getHighBitLocOneBase(mcspInt m) {
    mcspInt loc = 0;
    while (m) {
        loc++;
        m >>= 1;
    }
    return loc;
}

template <typename valType>
static valType getScalarToHost(const valType *ptr, mcsparsePointerMode_t ptr_mode) {
    valType value;
    if (ptr_mode == MCSPARSE_POINTER_MODE_HOST) {
#if defined(__MACA__)
        if constexpr (std::is_same_v<valType, __half>) {
            float float_value = __half2float(*ptr);
            value = __float2half(float_value);
        } else if constexpr (std::is_same_v<valType, __half2>) {
            float x_value = __low2float(*ptr);
            float y_value = __high2float(*ptr);
            value = __half2(__float2half(x_value), __float2half(y_value));
        } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
            float float_value = __bfloat162float(*ptr);
            value = __float2bfloat16(float_value);
        } else if constexpr (std::is_same_v<valType, mcsp_bfloat162>) {
            float x_value = __low2float(*ptr);
            float y_value = __high2float(*ptr);
            value = mcsp_bfloat162(__float2bfloat16(x_value), __float2bfloat16(y_value));
        } else {
            value = *ptr;
        }
#else
        value = *ptr;
#endif
    } else {
        MACA_ASSERT(mcMemcpy(&value, ptr, sizeof(*ptr), mcMemcpyDeviceToHost));
    }
    return value;
}

static inline size_t GetMacaDataTypeSize(macaDataType compute_type) {
    switch (compute_type) {
        case MACA_R_32F:
            return sizeof(float);
        case MACA_R_64F:
            return sizeof(double);
        case MACA_C_32F:
            return sizeof(mcspComplexFloat);
        case MACA_C_64F:
            return sizeof(mcspComplexDouble);
#if defined(__MACA__)
        case MACA_R_16F:
            return sizeof(__half);
        case MACA_C_16F:
            return sizeof(__half2);
        case MACA_R_16BF:
            return sizeof(mcsp_bfloat16);
        case MACA_C_16BF:
            return sizeof(mcsp_bfloat162);
        case MACA_R_32I:
            return sizeof(int32_t);
        case MACA_R_8I:
            return sizeof(int8_t);
#endif
        default:
            return 0;
    }
}

template <typename valType>
static inline macaDataType GetMacaDataTypeFromTypename() {
    if constexpr (std::is_same_v<valType, float>) {
        return MACA_R_32F;
    } else if constexpr (std::is_same_v<valType, double>) {
        return MACA_R_64F;
    } else if constexpr (std::is_same_v<valType, mcspComplexFloat>) {
        return MACA_C_32F;
    } else if constexpr (std::is_same_v<valType, mcspComplexDouble>) {
        return MACA_C_64F;
    } else if constexpr (std::is_same_v<valType, __half>) {
        return MACA_R_16F;
    } else if constexpr (std::is_same_v<valType, __half2>) {
        return MACA_C_16F;
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat16>) {
        return MACA_R_16BF;
    } else if constexpr (std::is_same_v<valType, mcsp_bfloat162>) {
        return MACA_C_16BF;
    } else if constexpr (std::is_same_v<valType, int8_t>) {
        return MACA_R_8I;
    } else if constexpr (std::is_same_v<valType, int32_t>) {
        return MACA_R_32I;
    }
}

static constexpr uint64_t GetMixedDataType(macaDataType data_left, macaDataType data_right) {
    uint64_t lett_bit_map = 0x100000000 << uint64_t(data_left);
    uint64_t right_bit_map = 0x1 << uint64_t(data_right);
    uint64_t mixed_bit_map = lett_bit_map | right_bit_map;
    return mixed_bit_map;
}

#endif
