#ifndef MCPRIM_DEVICE_REDUCE_HPP_
#define MCPRIM_DEVICE_REDUCE_HPP_

#include <vector>

#include "block_reduce.hpp"
#include "prim_types.h"

namespace mcprim {

template <typename idxType, typename valType, class binaryOp>
mcprimStatus_t reduce(void *temp_buffer, idxType &buffer_size, valType *input, valType *output, idxType data_size,
                      binaryOp op, mcStream_t stream = nullptr) {
    constexpr unsigned int n_repeat = 1024;
    unsigned int n_elem = 1024 * n_repeat;
#ifdef __MACA__
    n_elem = 512 * n_repeat;
#endif
    if (temp_buffer == nullptr) {
        buffer_size = (data_size + n_elem - 1) / n_elem;
        buffer_size = (buffer_size < 4) ? 4 : buffer_size;
        buffer_size *= sizeof(valType);
    } else {
        valType *buffer = (valType *)temp_buffer;
        mcprimStatus_t status;
        idxType ndata = data_size;
        if (ndata <= n_elem) {
            status = block_reduce(input, output, ndata, op, stream);
        } else {
            status = block_reduce(input, buffer, ndata, op, stream);
            ndata = (ndata + n_elem - 1) / n_elem;
            while (ndata > n_elem) {
                status = block_reduce(buffer, buffer, ndata, op, stream);
                ndata = (ndata + n_elem - 1) / n_elem;
            }
            status = block_reduce(buffer, output, ndata, op, stream);
        }
    }

    return MCPRIM_STATUS_SUCCESS;
}

}  // namespace mcprim

#endif
