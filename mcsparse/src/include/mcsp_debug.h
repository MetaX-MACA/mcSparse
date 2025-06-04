#ifndef COMMON_MCSP_DEBUG_H_
#define COMMON_MCSP_DEBUG_H_

#include <assert.h>

#include "common/mcsp_types.h"

#ifndef NDEBUG
#define MACA_ASSERT(status) assert(status == MCSP_STATUS_SUCCESS)
#else
#define MACA_ASSERT(status) ((void)status)
#endif

#endif
