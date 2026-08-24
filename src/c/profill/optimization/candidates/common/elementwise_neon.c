#include "elementwise_neon.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "campp_runtime/tensor_descriptor.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

void campp_elementwise_add_f32(
    const float *left, const float *right, float *output, uint32_t count)
{
    uint32_t index = 0u;
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (; index + 4u <= count; index += 4u) {
        vst1q_f32(
            output + index,
            vaddq_f32(vld1q_f32(left + index), vld1q_f32(right + index)));
    }
#endif
    for (; index < count; ++index) output[index] = left[index] + right[index];
}

void campp_elementwise_relu_f32(
    const float *input, float *output, uint32_t count)
{
    uint32_t index = 0u;
#if defined(__aarch64__) && defined(__ARM_NEON)
    const uint32x4_t sign_bit = vdupq_n_u32(UINT32_C(0x80000000));
    const uint32x4_t magnitude_mask = vdupq_n_u32(UINT32_C(0x7fffffff));
    const uint32x4_t infinity = vdupq_n_u32(UINT32_C(0x7f800000));
    const uint32x4_t zero = vdupq_n_u32(0u);
    for (; index + 4u <= count; index += 4u) {
        const uint32x4_t bits = vreinterpretq_u32_f32(
            vld1q_f32(input + index));
        const uint32x4_t magnitude = vandq_u32(bits, magnitude_mask);
        const uint32x4_t negative = vceqq_u32(
            vandq_u32(bits, sign_bit), sign_bit);
        const uint32x4_t ordered = vcleq_u32(magnitude, infinity);
        const uint32x4_t nonzero = vcgtq_u32(magnitude, zero);
        const uint32x4_t replace = vandq_u32(
            negative, vandq_u32(ordered, nonzero));
        vst1q_f32(
            output + index,
            vreinterpretq_f32_u32(vbicq_u32(bits, replace)));
    }
#endif
    for (; index < count; ++index) {
        uint32_t bits;
        uint32_t magnitude;
        memcpy(&bits, &input[index], sizeof(bits));
        magnitude = bits & UINT32_C(0x7fffffff);
        if ((bits & UINT32_C(0x80000000)) != 0u && magnitude != 0u &&
            magnitude <= UINT32_C(0x7f800000)) {
            bits = 0u;
        }
        memcpy(&output[index], &bits, sizeof(bits));
    }
}

static int32_t campp_quantize_scalar(
    float value, float scale, int32_t zero_point, uint8_t output_dtype)
{
    const int32_t minimum = output_dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
    const int32_t maximum = output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    long rounded = lrintf(value / scale) + zero_point;
    if (rounded < minimum) rounded = minimum;
    if (rounded > maximum) rounded = maximum;
    return (int32_t)rounded;
}

void campp_elementwise_quantize_f32(
    const float *input, uint8_t *output, uint32_t count, float scale,
    int32_t zero_point, uint8_t output_dtype)
{
    uint32_t index = 0u;
#if defined(__aarch64__) && defined(__ARM_NEON)
    const int32_t minimum = output_dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
    const int32_t maximum = output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    const float32x4_t divisor = vdupq_n_f32(scale);
    const float32x4_t lower = vdupq_n_f32((float)(minimum - zero_point));
    const float32x4_t upper = vdupq_n_f32((float)(maximum - zero_point));
    const int32x4_t zero = vdupq_n_s32(zero_point);
    for (; index + 4u <= count; index += 4u) {
        int32_t lanes[4];
        float32x4_t scaled = vdivq_f32(vld1q_f32(input + index), divisor);
        scaled = vmaxq_f32(lower, vminq_f32(upper, scaled));
        vst1q_s32(lanes, vaddq_s32(vcvtnq_s32_f32(scaled), zero));
        output[index] = (uint8_t)lanes[0];
        output[index + 1u] = (uint8_t)lanes[1];
        output[index + 2u] = (uint8_t)lanes[2];
        output[index + 3u] = (uint8_t)lanes[3];
    }
#endif
    for (; index < count; ++index) {
        const int32_t value = campp_quantize_scalar(
            input[index], scale, zero_point, output_dtype);
        output[index] = output_dtype == CAMPP_DTYPE_UINT8
            ? (uint8_t)value : (uint8_t)(int8_t)value;
    }
}

void campp_elementwise_sigmoid_lut_mul_f32(
    const uint8_t *quantized, const float *other, float *output,
    uint32_t count, const float table[256])
{
    uint32_t index = 0u;
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (; index + 4u <= count; index += 4u) {
        const float gates[4] = {
            table[quantized[index]], table[quantized[index + 1u]],
            table[quantized[index + 2u]], table[quantized[index + 3u]]
        };
        vst1q_f32(
            output + index,
            vmulq_f32(vld1q_f32(other + index), vld1q_f32(gates)));
    }
#endif
    for (; index < count; ++index) {
        output[index] = other[index] * table[quantized[index]];
    }
}
