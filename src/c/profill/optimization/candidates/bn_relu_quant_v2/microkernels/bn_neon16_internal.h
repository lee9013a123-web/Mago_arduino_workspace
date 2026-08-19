#ifndef CAMPP_PROFILL_BN_NEON16_INTERNAL_H
#define CAMPP_PROFILL_BN_NEON16_INTERNAL_H

#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "bn_quant_neon.h"
#include "campp_runtime/tensor_descriptor.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

static inline int campp_bn_v2_finite4(float32x4_t value)
{
    const uint32x4_t magnitude = vandq_u32(
        vreinterpretq_u32_f32(value), vdupq_n_u32(UINT32_C(0x7fffffff)));
    const uint32x4_t invalid = vcgeq_u32(
        magnitude, vdupq_n_u32(UINT32_C(0x7f800000)));
    return vmaxvq_u32(invalid) == 0u;
}

static inline CamppStatus campp_bn_v2_exact4_coefficients(
    const float *input, float32x4_t multiplier, float32x4_t additive,
    float32x4_t divisor, float32x4_t upper, int32x4_t quant_zero,
    int32x4_t *out_value)
{
    float32x4_t value = vmulq_f32(vld1q_f32(input), multiplier);
    value = vaddq_f32(value, additive);
    value = vmaxq_f32(value, vdupq_n_f32(0.0f));
    if (!campp_bn_v2_finite4(value)) return CAMPP_STATUS_KERNEL_FAILED;
    value = vdivq_f32(value, divisor);
    value = vminq_f32(value, upper);
    *out_value = vaddq_s32(vcvtnq_s32_f32(value), quant_zero);
    return CAMPP_STATUS_OK;
}

static inline CamppStatus campp_bn_v2_exact4(
    const float *input, const float *multiplier, const float *additive,
    float32x4_t divisor, float32x4_t upper, int32x4_t quant_zero,
    int32x4_t *out_value)
{
    return campp_bn_v2_exact4_coefficients(
        input, vld1q_f32(multiplier), vld1q_f32(additive), divisor,
        upper, quant_zero, out_value);
}

static inline CamppStatus campp_bn_v2_prescaled4_coefficients(
    const float *input, float32x4_t multiplier, float32x4_t additive,
    float32x4_t upper, int32x4_t quant_zero, int32x4_t *out_value)
{
    float32x4_t value = vmulq_f32(vld1q_f32(input), multiplier);
    value = vaddq_f32(value, additive);
    value = vmaxq_f32(value, vdupq_n_f32(0.0f));
    if (!campp_bn_v2_finite4(value)) return CAMPP_STATUS_KERNEL_FAILED;
    value = vminq_f32(value, upper);
    *out_value = vaddq_s32(vcvtnq_s32_f32(value), quant_zero);
    return CAMPP_STATUS_OK;
}

static inline CamppStatus campp_bn_v2_prescaled4(
    const float *input, const float *multiplier, const float *additive,
    float32x4_t upper, int32x4_t quant_zero, int32x4_t *out_value)
{
    return campp_bn_v2_prescaled4_coefficients(
        input, vld1q_f32(multiplier), vld1q_f32(additive), upper,
        quant_zero, out_value);
}

static inline void campp_bn_v2_pack16(
    const int32x4_t values[4], uint8_t output_dtype, uint8_t *output)
{
    if (output_dtype == CAMPP_DTYPE_UINT8) {
        const uint16x8_t low = vcombine_u16(
            vqmovun_s32(values[0]), vqmovun_s32(values[1]));
        const uint16x8_t high = vcombine_u16(
            vqmovun_s32(values[2]), vqmovun_s32(values[3]));
        vst1q_u8(
            output, vcombine_u8(vqmovn_u16(low), vqmovn_u16(high)));
    } else {
        const int16x8_t low = vcombine_s16(
            vqmovn_s32(values[0]), vqmovn_s32(values[1]));
        const int16x8_t high = vcombine_s16(
            vqmovn_s32(values[2]), vqmovn_s32(values[3]));
        vst1q_s8(
            (int8_t *)output,
            vcombine_s8(vqmovn_s16(low), vqmovn_s16(high)));
    }
}
#endif

static inline uint8_t campp_bn_v2_store_byte(
    int32_t value, uint8_t output_dtype)
{
    return output_dtype == CAMPP_DTYPE_UINT8
        ? (uint8_t)value : (uint8_t)(int8_t)value;
}

static inline CamppStatus campp_bn_v2_prescaled_scalar(
    float input, float multiplier, float additive, int32_t quant_zero,
    uint8_t output_dtype, uint8_t *out_value)
{
    const int32_t maximum =
        output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    float value = input * multiplier + additive;
    int64_t rounded;

    if (value < 0.0f) value = 0.0f;
    if (!isfinite(value)) return CAMPP_STATUS_KERNEL_FAILED;
    if (value > (float)(maximum - quant_zero)) {
        value = (float)(maximum - quant_zero);
    }
    rounded = (int64_t)nearbyintf(value) + quant_zero;
    *out_value = campp_bn_v2_store_byte((int32_t)rounded, output_dtype);
    return CAMPP_STATUS_OK;
}

#endif /* CAMPP_PROFILL_BN_NEON16_INTERNAL_H */
