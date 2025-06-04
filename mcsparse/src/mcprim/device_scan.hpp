#ifndef MCPRIM_DEVICE_SCAN_HPP_
#define MCPRIM_DEVICE_SCAN_HPP_

#include <vector>

#include "block_scan.hpp"
#include "prim_types.h"

#if defined(__MACA__)
#define MCPRIM_BLOCK_SIZE 512
#define MCPRIM_PART_SIZE 2048
#else
#define MCPRIM_BLOCK_SIZE 256
#define MCPRIM_PART_SIZE 1024
#endif

namespace mcprim {

template <typename idxType, typename valType>
mcprimStatus_t exclusive_scan(void *buffer, idxType &buffer_size, valType *input, valType *output, idxType data_size,
                              const valType *h_init_val = nullptr, mcStream_t stream = nullptr) {
    constexpr unsigned int block_size = MCPRIM_BLOCK_SIZE;
    constexpr unsigned int part_size = MCPRIM_PART_SIZE;

    if (buffer == nullptr) {
        unsigned int n_storage = 0;
        for (unsigned int n_block = (data_size + part_size - 1) / part_size; n_block > 1;
             n_block = (n_block + part_size - 1) / part_size) {
            n_storage += n_block;
        }
        n_storage += 1;
        n_storage = n_storage < 4 ? 4 : n_storage;
        buffer_size = n_storage * sizeof(valType);
        return MCPRIM_STATUS_SUCCESS;
    }

    std::vector<valType *> buffer_ptr_vec;
    std::vector<idxType> data_size_vec;

    valType *buffer_ptr = (valType *)buffer;

    buffer_ptr_vec.push_back(buffer_ptr);
    data_size_vec.push_back(data_size);

    unsigned int n_iter = 0;
    unsigned int n_block = 0;

    n_block = (data_size + part_size - 1) / part_size;
    while (n_block > 1) {
        data_size_vec.push_back(n_block);

        buffer_ptr += n_block;
        buffer_ptr_vec.push_back(buffer_ptr);

        n_block = (n_block + part_size - 1) / part_size;
        n_iter++;
    }

    unsigned int block_num = (data_size + part_size - 1) / part_size;
    unsigned int shm_size = (part_size + (part_size + NUM_BANKS - 1) / NUM_BANKS) * sizeof(valType);
    mcLaunchKernelGGL((block_exclusive_scan_kernel<MCPRIM_BLOCK_SIZE, MCPRIM_PART_SIZE>), block_num, block_size,
                       shm_size, stream, input, buffer_ptr_vec[0], output, data_size);

    for (unsigned int i = 0; i < n_iter; i++) {
        unsigned int block_num = (data_size_vec[i + 1] + part_size - 1) / part_size;
        mcLaunchKernelGGL((block_exclusive_scan_kernel<MCPRIM_BLOCK_SIZE, MCPRIM_PART_SIZE>), block_num, block_size,
                           shm_size, stream, buffer_ptr_vec[i], buffer_ptr_vec[i + 1], buffer_ptr_vec[i],
                           data_size_vec[i + 1]);
    }

    for (int i = static_cast<int>(n_iter) - 1; i >= 1; i--) {
        unsigned int block_num = (data_size_vec[i] + part_size - 1) / part_size;
        mcLaunchKernelGGL((block_exclusive_scan_plus_kernel<MCPRIM_BLOCK_SIZE, MCPRIM_PART_SIZE>), block_num,
                           block_size, 0, stream, buffer_ptr_vec[i], buffer_ptr_vec[i - 1], data_size_vec[i]);
    }

    mcLaunchKernelGGL((block_exclusive_scan_plus_kernel<MCPRIM_BLOCK_SIZE, MCPRIM_PART_SIZE>), block_num, block_size,
                       0, stream, buffer_ptr_vec[0], output, data_size);

    if (h_init_val != nullptr) {
        unsigned int block_num = (data_size + block_size - 1) / block_size;
        mcLaunchKernelGGL((add_initial_value_kernel<MCPRIM_BLOCK_SIZE>), block_num, block_size, 0, stream, output,
                           data_size, *h_init_val);
    }

    return MCPRIM_STATUS_SUCCESS;
}

template <typename idxType, typename valType>
mcprimStatus_t inclusive_scan(void *buffer, idxType &buffer_size, valType *input, valType *output, idxType data_size,
                              mcStream_t stream = nullptr) {
    constexpr unsigned block_size = MCPRIM_BLOCK_SIZE;
    constexpr unsigned part_size = MCPRIM_PART_SIZE;

    if (buffer == nullptr) {
        unsigned int n_storage = 0;
        for (unsigned int n_block = (data_size + part_size - 1) / part_size; n_block > 1;
             n_block = (n_block + part_size - 1) / part_size) {
            n_storage += n_block;
        }
        n_storage += 1;
        n_storage = n_storage < 4 ? 4 : n_storage;
        buffer_size = n_storage * sizeof(valType);

        return MCPRIM_STATUS_SUCCESS;
    }

    std::vector<valType *> buffer_ptr_vec;
    std::vector<idxType> data_size_vec;

    valType *buffer_ptr = (valType *)buffer;

    buffer_ptr_vec.push_back(buffer_ptr);
    data_size_vec.push_back(data_size);

    unsigned int n_iter = 0;
    unsigned int n_block = 0;

    n_block = (data_size + part_size - 1) / part_size;
    while (n_block > 1) {
        data_size_vec.push_back(n_block);

        buffer_ptr += n_block;
        buffer_ptr_vec.push_back(buffer_ptr);

        n_block = (n_block + part_size - 1) / part_size;
        n_iter++;
    }
    unsigned int block_num = (data_size + part_size - 1) / part_size;
    unsigned int shm_size = (part_size + (part_size + NUM_BANKS - 1) / NUM_BANKS) * sizeof(valType);
    mcLaunchKernelGGL((block_inclusive_scan_kernel<MCPRIM_BLOCK_SIZE, MCPRIM_PART_SIZE>), block_num, block_size,
                       shm_size, stream, input, buffer_ptr_vec[0], output, data_size);

    for (unsigned int i = 0; i < n_iter; i++) {
        unsigned int block_num = (data_size_vec[i + 1] + part_size - 1) / part_size;
        mcLaunchKernelGGL((block_inclusive_scan_kernel<MCPRIM_BLOCK_SIZE, MCPRIM_PART_SIZE>), block_num, block_size,
                           shm_size, stream, buffer_ptr_vec[i], buffer_ptr_vec[i + 1], buffer_ptr_vec[i],
                           data_size_vec[i + 1]);
    }

    for (int i = static_cast<int>(n_iter) - 1; i >= 1; i--) {
        unsigned int block_num = (data_size_vec[i] + part_size - 1) / part_size;
        mcLaunchKernelGGL((block_inclusive_scan_plus_kernel<MCPRIM_BLOCK_SIZE, MCPRIM_PART_SIZE>), block_num,
                           block_size, 0, stream, buffer_ptr_vec[i], buffer_ptr_vec[i - 1], data_size_vec[i]);
    }

    mcLaunchKernelGGL((block_inclusive_scan_plus_kernel<MCPRIM_BLOCK_SIZE, MCPRIM_PART_SIZE>), block_num, block_size,
                       0, stream, buffer_ptr_vec[0], output, data_size);

    return MCPRIM_STATUS_SUCCESS;
}

}  // namespace mcprim

#endif
