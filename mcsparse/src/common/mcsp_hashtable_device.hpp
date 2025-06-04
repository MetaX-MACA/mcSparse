#ifndef COMMON_MCSP_HASHTABLE_DEVICE_HPP_
#define COMMON_MCSP_HASHTABLE_DEVICE_HPP_

#include "common/mcsp_types.h"
#include "mcsp_runtime_wrapper.h"

#define HASH_MAGIC_NULL_VAL (0xFFFFFFFF)

static __device__ double atomicAdd_(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

template <typename valType>
static __device__ valType atomicAdd_(valType *address, valType val) {
    return atomicAdd(address, val);
}

static __device__ mcspComplexFloat complexAtomicAddByPart_(mcspComplexFloat *address, mcspComplexFloat val) {
    float x = atomicAdd_(&address->x, val.x);
    float y = atomicAdd_(&address->y, val.y);

    return mcspComplexFloat{x, y};
}

static __device__ mcspComplexDouble complexAtomicAddByPart_(mcspComplexDouble *address, mcspComplexDouble val) {
    double x = atomicAdd_(&address->x, val.x);
    double y = atomicAdd_(&address->y, val.y);

    return mcspComplexDouble{x, y};
}

#ifdef __MACA__
static __device__ __half2 complexAtomicAddByPart_(__half2 *address, __half2 val) {
    __half x = atomicAdd_(&address->x, val.x);
    __half y = atomicAdd_(&address->y, val.y);

    return __half2(x, y);
}

static __device__ mcsp_bfloat162 complexAtomicAddByPart_(mcsp_bfloat162 *address, mcsp_bfloat162 val) {
    mcsp_bfloat16 x = atomicAdd_(&address->x, val.x);
    mcsp_bfloat16 y = atomicAdd_(&address->y, val.y);

    return mcsp_bfloat162(x, y);
}
#endif

template <unsigned int HASH_SIZE, unsigned int HASH_MUL, typename idxType>
static __device__ __forceinline__ idxType mcspInsertHashTableKey(idxType key, idxType *table) {
    idxType hash = (key * HASH_MUL) & (HASH_SIZE - 1);

    for (unsigned int i = 0; i < 2 * HASH_SIZE; i++) {
        idxType old = atomicCAS(&(table[hash]), (idxType)HASH_MAGIC_NULL_VAL, key);
        if (old == key) {
            return 0;
        } else if (old == (idxType)HASH_MAGIC_NULL_VAL) {
            return 1;
        } else {
            hash = (hash + 1) & (HASH_SIZE - 1);
        }
    }
    return 0;
}

template <unsigned int HASH_SIZE, unsigned int HASH_MUL, typename idxType, typename valType>
__device__ void mcspInsertHashTablePairs(idxType key, valType val, idxType *key_table, valType *val_table) {
    idxType hash = (key * HASH_MUL) & (HASH_SIZE - 1);

    for (unsigned int i = 0; i < 2 * HASH_SIZE; i++) {
        idxType old = atomicCAS(&(key_table[hash]), (idxType)HASH_MAGIC_NULL_VAL, key);
        if (old == key || old == (idxType)HASH_MAGIC_NULL_VAL) {
            if constexpr (std::is_same_v<valType, mcspComplexDouble> || std::is_same_v<valType, mcspComplexFloat>) {
                complexAtomicAddByPart_(&(val_table[hash]), val);
            } else {
                atomicAdd_(&(val_table[hash]), val);
            }
            return;
        } else {
            hash = (hash + 1) & (HASH_SIZE - 1);
        }
    }
    return;
}

template <unsigned int HASH_SIZE, unsigned int HASH_MUL, typename idxType, typename valType>
__device__ void mcspReadHashTablePairs(idxType key, const idxType *key_table, const valType *val_table, valType *val,
                                       int *found) {
    idxType hash = (key * HASH_MUL) & (HASH_SIZE - 1);

    for (unsigned int i = 0; i < HASH_SIZE; i++) {
        idxType key_read = key_table[hash];
        if (key_read == (idxType)HASH_MAGIC_NULL_VAL) {
            *found = 0;
            return;
        } else if (key_read == key) {
            *found = 1;
            *val = val_table[hash];
            return;
        } else {
            hash = (hash + 1) & (HASH_SIZE - 1);
        }
    }
    *found = 0;
    return;
}

#endif
