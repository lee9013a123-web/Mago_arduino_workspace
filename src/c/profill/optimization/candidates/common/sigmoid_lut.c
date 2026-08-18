#include "sigmoid_lut.h"

#include <math.h>
#include <stdint.h>

#include "campp_runtime/tensor_descriptor.h"

void campp_sigmoid_lut_build(
    uint8_t input_dtype, float scale, int32_t zero_point, float table[256])
{
    uint32_t raw;
    for (raw = 0u; raw < 256u; ++raw) {
        const int32_t quantized = input_dtype == CAMPP_DTYPE_UINT8
            ? (int32_t)raw : (int32_t)(int8_t)(uint8_t)raw;
        const float value = (float)(quantized - zero_point) * scale;
        if (value >= 0.0f) {
            table[raw] = 1.0f / (1.0f + expf(-value));
        } else {
            const float exponential = expf(value);
            table[raw] = exponential / (1.0f + exponential);
        }
    }
}
