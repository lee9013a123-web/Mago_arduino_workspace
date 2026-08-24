#ifndef CAMPP_PROFILL_DEQUANT_SCALAR_FASTPATH_H
#define CAMPP_PROFILL_DEQUANT_SCALAR_FASTPATH_H

#include <stdint.h>
#include <string.h>

#include "campp_runtime/tensor_descriptor.h"

/* These helpers stay inline because the scalar candidate calls them once per
 * element and the profiling build does not enable cross-TU LTO. */
static inline int32_t campp_dequant_read_direct(
    const uint8_t *data, uint8_t dtype)
{
    if (dtype == CAMPP_DTYPE_UINT8) return *data;
    {
        int8_t value;
        memcpy(&value, data, sizeof(value));
        return value;
    }
}

static inline float campp_dequant_scalar_value(
    int32_t value, float scale, int32_t zero_point)
{
    return (float)(value - zero_point) * scale;
}

static inline void campp_dequant_store_direct(float *output, float value)
{
    memcpy(output, &value, sizeof(value));
}

#endif /* CAMPP_PROFILL_DEQUANT_SCALAR_FASTPATH_H */
