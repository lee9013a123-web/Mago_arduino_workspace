#include "bn_quant_neon.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>

#include "campp_runtime/tensor_descriptor.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static int32_t campp_bn_clamp_output(int64_t value, uint8_t dtype)
{
    if (dtype == CAMPP_DTYPE_UINT8) {
        return (int32_t)(value < 0 ? 0 : value > 255 ? 255 : value);
    }
    return (int32_t)(value < -128 ? -128 : value > 127 ? 127 : value);
}

CamppStatus campp_bn_quantize_scalar(
    float input, float multiplier, float additive, float quant_scale,
    int32_t quant_zero, uint8_t output_dtype, int32_t *out_value)
{
    float value;
    float scaled;
    int64_t rounded;
    if (out_value == NULL || !(quant_scale > 0.0f) ||
        (output_dtype != CAMPP_DTYPE_UINT8 &&
         output_dtype != CAMPP_DTYPE_INT8)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    value = input * multiplier + additive;
    if (value < 0.0f) value = 0.0f;
    if (!isfinite(value)) return CAMPP_STATUS_KERNEL_FAILED;
    scaled = value / quant_scale;
    if (scaled <= -2147483648.0f) {
        rounded = INT32_MIN;
    } else if (scaled >= 2147483520.0f) {
        rounded = INT32_MAX;
    } else {
        rounded = (int64_t)nearbyintf(scaled);
    }
    *out_value = campp_bn_clamp_output(rounded + quant_zero, output_dtype);
    return CAMPP_STATUS_OK;
}

CamppStatus campp_bn_quantize_neon4(
    const float input[4], const float multiplier[4],
    const float additive[4], float quant_scale, int32_t quant_zero,
    uint8_t output_dtype, int32_t output[4])
{
    uint32_t lane;
    if (input == NULL || multiplier == NULL || additive == NULL ||
        output == NULL || !(quant_scale > 0.0f) ||
        (output_dtype != CAMPP_DTYPE_UINT8 &&
         output_dtype != CAMPP_DTYPE_INT8)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
#if defined(__aarch64__)
    {
        const float32x4_t zero = vdupq_n_f32(0.0f);
        const float32x4_t lower = vdupq_n_f32(-2147483648.0f);
        const float32x4_t upper = vdupq_n_f32(2147483520.0f);
        float32x4_t value = vmulq_f32(
            vld1q_f32(input), vld1q_f32(multiplier));
        float values[4];
        int32_t rounded[4];
        value = vaddq_f32(value, vld1q_f32(additive));
        value = vmaxq_f32(value, zero);
        vst1q_f32(values, value);
        for (lane = 0u; lane < 4u; ++lane) {
            if (!isfinite(values[lane])) return CAMPP_STATUS_KERNEL_FAILED;
        }
        value = vdivq_f32(value, vdupq_n_f32(quant_scale));
        value = vmaxq_f32(value, lower);
        value = vminq_f32(value, upper);
        vst1q_s32(rounded, vcvtnq_s32_f32(value));
        for (lane = 0u; lane < 4u; ++lane) {
            output[lane] = campp_bn_clamp_output(
                (int64_t)rounded[lane] + quant_zero, output_dtype);
        }
    }
#else
    for (lane = 0u; lane < 4u; ++lane) {
        const CamppStatus status = campp_bn_quantize_scalar(
            input[lane], multiplier[lane], additive[lane], quant_scale,
            quant_zero, output_dtype, &output[lane]);
        if (status != CAMPP_STATUS_OK) return status;
    }
#endif
    return CAMPP_STATUS_OK;
}
