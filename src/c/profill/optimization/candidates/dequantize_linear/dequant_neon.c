#include "dequant_neon.h"

#include "campp_runtime/tensor_descriptor.h"
#include "dequant_scalar_fastpath.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

static void campp_dequant_store_quarter(
    int32x4_t values, int32x4_t zero_points, float32x4_t scales,
    float *output)
{
    const int32x4_t centered = vsubq_s32(values, zero_points);
    const float32x4_t converted = vcvtq_f32_s32(centered);
    vst1q_f32(output, vmulq_f32(converted, scales));
}

static void campp_dequant_neon_values16(
    const uint8_t *input, uint8_t input_dtype,
    const int32x4_t zero_points[4], const float32x4_t scales[4],
    float *output)
{
    if (input_dtype == CAMPP_DTYPE_UINT8) {
        const uint8x16_t bytes = vld1q_u8(input);
        const uint16x8_t low = vmovl_u8(vget_low_u8(bytes));
        const uint16x8_t high = vmovl_u8(vget_high_u8(bytes));
        campp_dequant_store_quarter(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(low))),
            zero_points[0], scales[0], output);
        campp_dequant_store_quarter(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(low))),
            zero_points[1], scales[1], output + 4u);
        campp_dequant_store_quarter(
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(high))),
            zero_points[2], scales[2], output + 8u);
        campp_dequant_store_quarter(
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(high))),
            zero_points[3], scales[3], output + 12u);
    } else {
        const int8x16_t bytes = vld1q_s8((const int8_t *)input);
        const int16x8_t low = vmovl_s8(vget_low_s8(bytes));
        const int16x8_t high = vmovl_s8(vget_high_s8(bytes));
        campp_dequant_store_quarter(
            vmovl_s16(vget_low_s16(low)), zero_points[0], scales[0], output);
        campp_dequant_store_quarter(
            vmovl_s16(vget_high_s16(low)), zero_points[1], scales[1],
            output + 4u);
        campp_dequant_store_quarter(
            vmovl_s16(vget_low_s16(high)), zero_points[2], scales[2],
            output + 8u);
        campp_dequant_store_quarter(
            vmovl_s16(vget_high_s16(high)), zero_points[3], scales[3],
            output + 12u);
    }
}
#endif

void campp_dequant_neon_scalar16(
    const uint8_t *input, uint8_t input_dtype, float scale,
    int32_t zero_point, float *output)
{
#if defined(__aarch64__) && defined(__ARM_NEON)
    const int32x4_t zero_points[4] = {
        vdupq_n_s32(zero_point), vdupq_n_s32(zero_point),
        vdupq_n_s32(zero_point), vdupq_n_s32(zero_point)
    };
    const float32x4_t scales[4] = {
        vdupq_n_f32(scale), vdupq_n_f32(scale),
        vdupq_n_f32(scale), vdupq_n_f32(scale)
    };
    campp_dequant_neon_values16(
        input, input_dtype, zero_points, scales, output);
#else
    uint32_t lane;
    for (lane = 0u; lane < CAMPP_DEQUANT_NEON_LANES; ++lane) {
        output[lane] = campp_dequant_scalar_value(
            campp_dequant_read_direct(input + lane, input_dtype),
            scale, zero_point);
    }
#endif
}

void campp_dequant_neon_per_axis16(
    const uint8_t *input, uint8_t input_dtype,
    const float scales[CAMPP_DEQUANT_NEON_LANES],
    const int32_t zero_points[CAMPP_DEQUANT_NEON_LANES], float *output)
{
#if defined(__aarch64__) && defined(__ARM_NEON)
    const int32x4_t zero_vectors[4] = {
        vld1q_s32(zero_points), vld1q_s32(zero_points + 4u),
        vld1q_s32(zero_points + 8u), vld1q_s32(zero_points + 12u)
    };
    const float32x4_t scale_vectors[4] = {
        vld1q_f32(scales), vld1q_f32(scales + 4u),
        vld1q_f32(scales + 8u), vld1q_f32(scales + 12u)
    };
    campp_dequant_neon_values16(
        input, input_dtype, zero_vectors, scale_vectors, output);
#else
    uint32_t lane;
    for (lane = 0u; lane < CAMPP_DEQUANT_NEON_LANES; ++lane) {
        output[lane] = campp_dequant_scalar_value(
            campp_dequant_read_direct(input + lane, input_dtype),
            scales[lane], zero_points[lane]);
    }
#endif
}
