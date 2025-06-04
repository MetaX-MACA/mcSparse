#ifndef MCPRIM_DEVICE_SEGMENTED_SCAN_HPP_
#define MCPRIM_DEVICE_SEGMENTED_SCAN_HPP_

#include <vector>

#include "block_segmented_scan.hpp"
#include "mcsp_debug.h"
#include "prim_types.h"

namespace mcprim {

template <typename idxType, typename flagType, typename valType>
mcprimStatus_t segmented_exclusive_scan(void *temp_buffer, idxType &buffer_size, valType *input, valType *output,
                                        flagType *flags, idxType data_size, mcStream_t stream = nullptr) {
    if (temp_buffer == nullptr) {
        unsigned int n_elem = 1024;
#ifdef __MACA__
        n_elem = 512;
#endif
        unsigned int n_iter = 0;
        unsigned int n_block = 0;
        unsigned int n_remain = data_size;
        unsigned int n_storage = 0;
        while (n_remain > n_elem) {
            n_block = (n_remain + n_elem - 1) / n_elem;
            n_storage += (n_block + n_elem - 1) / n_elem * n_elem;
            n_iter++;
            n_remain = n_block;
        }
        n_storage = max(n_storage + 1, 4);
        buffer_size = n_storage * (sizeof(*input) + sizeof(*flags)) + data_size * sizeof(*flags);
    } else {
        std::vector<valType *> buffer_ptr_vec;
        std::vector<flagType *> flag_buffer_ptr_vec;
        std::vector<idxType> data_size_vec;
        unsigned int offset = 0;
        valType *buffer_ptr;
        flagType *flag_buffer_ptr;

        buffer_ptr_vec.push_back(output);
        flag_buffer_ptr = (flagType *)temp_buffer;
        flag_buffer_ptr_vec.push_back(flag_buffer_ptr);
        buffer_ptr = (valType *)(flag_buffer_ptr + data_size);
        buffer_ptr_vec.push_back(buffer_ptr);
        data_size_vec.push_back(data_size);

        unsigned int n_elem = 1024;
#ifdef __MACA__
        n_elem = 512;
#endif
        unsigned int n_iter = 0;
        unsigned int n_block = 0;
        unsigned int n_remain = data_size;
        while (n_remain > n_elem) {
            n_block = (n_remain + n_elem - 1) / n_elem;
            offset += (n_block + n_elem - 1) / n_elem * n_elem;
            data_size_vec.push_back(n_block);
            flag_buffer_ptr = (flagType *)(buffer_ptr + n_block);
            flag_buffer_ptr_vec.push_back(flag_buffer_ptr);
            buffer_ptr = (valType *)(flag_buffer_ptr + n_block);
            buffer_ptr_vec.push_back(buffer_ptr);
            n_iter++;
            n_remain = n_block;
        }
        n_block = 1;
        data_size_vec.push_back(n_block);
        flag_buffer_ptr = (flagType *)(buffer_ptr + n_block);
        flag_buffer_ptr_vec.push_back(flag_buffer_ptr);
        buffer_ptr = (valType *)(flag_buffer_ptr + n_block);
        buffer_ptr_vec.push_back(buffer_ptr);
        n_iter++;

        mcprimStatus_t status;
        size_t lda = 1;
        status = block_segmented_exclusive_scan_upsweep(buffer_ptr_vec[1], flag_buffer_ptr_vec[1], buffer_size, input,
                                                        output, flags, flag_buffer_ptr_vec[0], data_size, stream);
        std::vector<valType> debug1;
        std::vector<flagType> debug2;
        debug1.resize(data_size_vec[1]);
        debug2.resize(data_size_vec[1]);
        MACA_ASSERT(mcMemcpyAsync(debug1.data(), buffer_ptr_vec[1], data_size_vec[1] * sizeof(valType),
                                  mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcMemcpyAsync(debug2.data(), flag_buffer_ptr_vec[1], data_size_vec[1] * sizeof(flagType),
                                  mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        for (int i = 1; i < (int)n_iter; i++) {
            status = block_segmented_exclusive_scan_upsweep(
                buffer_ptr_vec[i + 1], flag_buffer_ptr_vec[i + 1], buffer_size, buffer_ptr_vec[i], buffer_ptr_vec[i],
                flag_buffer_ptr_vec[i], flag_buffer_ptr_vec[i], data_size_vec[i], stream);
            lda *= n_elem;
        }
        bool is_root = true;
        for (int i = (int)n_iter - 1; i >= 1; i--) {
            status = block_segmented_exclusive_scan_downsweep(buffer_ptr_vec[i + 1], flag_buffer_ptr_vec[i + 1],
                                                              buffer_ptr_vec[i], buffer_ptr_vec[i],
                                                              flag_buffer_ptr_vec[i], flag_buffer_ptr_vec[i], flags,
                                                              data_size_vec[i], data_size, lda, false, is_root, stream);
            is_root = false;
            lda /= n_elem;
        }
        MACA_ASSERT(mcMemcpyAsync(debug1.data(), buffer_ptr_vec[1], data_size_vec[1] * sizeof(valType),
                                  mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcMemcpyAsync(debug2.data(), flag_buffer_ptr_vec[1], data_size_vec[1] * sizeof(flagType),
                                  mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        status = block_segmented_exclusive_scan_downsweep(buffer_ptr_vec[1], flag_buffer_ptr_vec[1], output, output,
                                                          flag_buffer_ptr_vec[0], flag_buffer_ptr_vec[0], flags,
                                                          data_size, data_size, (idxType)1, true, is_root, stream);
    }

    return MCPRIM_STATUS_SUCCESS;
}

template <typename idxType, typename flagType, typename valType>
mcprimStatus_t segmented_inclusive_scan(void *temp_buffer, idxType &buffer_size, valType *input, valType *output,
                                        flagType *flags, idxType data_size, mcStream_t stream = nullptr) {
    if (temp_buffer == nullptr) {
        unsigned int n_elem = 1024;
#ifdef __MACA__
        n_elem = 512;
#endif
        unsigned int n_iter = 0;
        unsigned int n_block = 0;
        unsigned int n_remain = data_size;
        unsigned int n_storage = 0;
        while (n_remain > n_elem) {
            n_block = (n_remain + n_elem - 1) / n_elem;
            n_storage += (n_block + n_elem - 1) / n_elem * n_elem;
            n_iter++;
            n_remain = n_block;
        }
        n_storage = max(n_storage + 1, 4);
        buffer_size = n_storage * (sizeof(*input) + sizeof(*flags)) + data_size * sizeof(*flags);
    } else {
        std::vector<valType *> buffer_ptr_vec;
        std::vector<flagType *> flag_buffer_ptr_vec;
        std::vector<idxType> data_size_vec;
        unsigned int offset = 0;
        valType *buffer_ptr;
        flagType *flag_buffer_ptr;

        buffer_ptr_vec.push_back(output);
        flag_buffer_ptr = (flagType *)temp_buffer;
        flag_buffer_ptr_vec.push_back(flag_buffer_ptr);
        buffer_ptr = (valType *)(flag_buffer_ptr + data_size);
        buffer_ptr_vec.push_back(buffer_ptr);
        data_size_vec.push_back(data_size);

        unsigned int n_elem = 1024;
#ifdef __MACA__
        n_elem = 512;
#endif
        unsigned int n_iter = 0;
        unsigned int n_block = 0;
        unsigned int n_remain = data_size;
        while (n_remain > n_elem) {
            n_block = (n_remain + n_elem - 1) / n_elem;
            offset += (n_block + n_elem - 1) / n_elem * n_elem;
            data_size_vec.push_back(n_block);
            flag_buffer_ptr = (flagType *)(buffer_ptr + n_block);
            flag_buffer_ptr_vec.push_back(flag_buffer_ptr);
            buffer_ptr = (valType *)(flag_buffer_ptr + n_block);
            buffer_ptr_vec.push_back(buffer_ptr);
            n_iter++;
            n_remain = n_block;
        }
        n_block = 1;
        data_size_vec.push_back(n_block);
        flag_buffer_ptr = (flagType *)(buffer_ptr + n_block);
        flag_buffer_ptr_vec.push_back(flag_buffer_ptr);
        buffer_ptr = (valType *)(flag_buffer_ptr + n_block);
        buffer_ptr_vec.push_back(buffer_ptr);
        n_iter++;

        mcprimStatus_t status;
        size_t lda = 1;
        status = block_segmented_exclusive_scan_upsweep(buffer_ptr_vec[1], flag_buffer_ptr_vec[1], buffer_size, input,
                                                        output, flags, flag_buffer_ptr_vec[0], data_size, stream);
        std::vector<valType> debug1;
        std::vector<flagType> debug2;
        debug1.resize(data_size_vec[1]);
        debug2.resize(data_size_vec[1]);
        MACA_ASSERT(mcMemcpyAsync(debug1.data(), buffer_ptr_vec[1], data_size_vec[1] * sizeof(valType),
                                  mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcMemcpyAsync(debug2.data(), flag_buffer_ptr_vec[1], data_size_vec[1] * sizeof(flagType),
                                  mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        for (int i = 1; i < (int)n_iter; i++) {
            status = block_segmented_exclusive_scan_upsweep(
                buffer_ptr_vec[i + 1], flag_buffer_ptr_vec[i + 1], buffer_size, buffer_ptr_vec[i], buffer_ptr_vec[i],
                flag_buffer_ptr_vec[i], flag_buffer_ptr_vec[i], data_size_vec[i], stream);
            lda *= n_elem;
        }
        bool is_root = true;
        for (int i = (int)n_iter - 1; i >= 1; i--) {
            status = block_segmented_exclusive_scan_downsweep(buffer_ptr_vec[i + 1], flag_buffer_ptr_vec[i + 1],
                                                              buffer_ptr_vec[i], buffer_ptr_vec[i],
                                                              flag_buffer_ptr_vec[i], flag_buffer_ptr_vec[i], flags,
                                                              data_size_vec[i], data_size, lda, false, is_root, stream);
            is_root = false;
            lda /= n_elem;
        }
        MACA_ASSERT(mcMemcpyAsync(debug1.data(), buffer_ptr_vec[1], data_size_vec[1] * sizeof(valType),
                                  mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcMemcpyAsync(debug2.data(), flag_buffer_ptr_vec[1], data_size_vec[1] * sizeof(flagType),
                                  mcMemcpyDeviceToHost, stream));
        MACA_ASSERT(mcStreamSynchronize(stream));
        status = block_segmented_inclusive_scan_downsweep(buffer_ptr_vec[1], flag_buffer_ptr_vec[1], output, output,
                                                          flag_buffer_ptr_vec[0], flag_buffer_ptr_vec[0], input, flags,
                                                          data_size, data_size, (idxType)1, true, is_root, stream);
    }

    return MCPRIM_STATUS_SUCCESS;
}

}  // namespace mcprim

#endif
