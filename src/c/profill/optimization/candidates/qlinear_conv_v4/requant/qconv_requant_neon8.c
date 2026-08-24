#include "qconv_requant_neon8.h"

#include <limits.h>
#include <math.h>

#include "campp_runtime/tensor_descriptor.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if !defined(__aarch64__) || !defined(__ARM_NEON)
static int campp_qconv_v4_requantize_scalar(
    int32_t accumulator, float multiplier, uint8_t output_dtype,
    int32_t output_zero, uint8_t *out_byte)
{
    const int64_t output_min =
        output_dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
    const int64_t output_max =
        output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    const float scaled = (float)accumulator * multiplier;
    int64_t rounded;

    if (out_byte == NULL || !isfinite(multiplier)) return 1;
    if (scaled <= -2147483648.0f) {
        rounded = INT32_MIN;
    } else if (scaled >= 2147483520.0f) {
        rounded = INT32_MAX;
    } else {
        rounded = (int64_t)nearbyintf(scaled);
    }
    rounded += output_zero;
    if (rounded < output_min) rounded = output_min;
    if (rounded > output_max) rounded = output_max;
    *out_byte = output_dtype == CAMPP_DTYPE_UINT8
        ? (uint8_t)rounded : (uint8_t)(int8_t)rounded;
    return 0;
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
static int32x4_t campp_qconv_v4_requantize4(
    const int32_t *accumulators, const float *multipliers,
    int32_t output_min, int32_t output_max, int32_t output_zero)
{
    float32x4_t scaled = vmulq_f32(
        vcvtq_f32_s32(vld1q_s32(accumulators)),
        vld1q_f32(multipliers));
    scaled = vmaxq_f32(
        scaled, vdupq_n_f32((float)(output_min - output_zero)));
    scaled = vminq_f32(
        scaled, vdupq_n_f32((float)(output_max - output_zero)));
    return vaddq_s32(
        vcvtnq_s32_f32(scaled), vdupq_n_s32(output_zero));
}
#endif

void campp_qconv_v4_requantize8_validated(
    const int32_t accumulators[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const float multipliers[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint8_t output_dtype, int32_t output_zero,
    uint8_t output_bytes[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
#if defined(__aarch64__) && defined(__ARM_NEON)
    {
        const int32_t output_min =
            output_dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
        const int32_t output_max =
            output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
        const int32x4_t low = campp_qconv_v4_requantize4(
            accumulators, multipliers, output_min, output_max, output_zero);
        const int32x4_t high = campp_qconv_v4_requantize4(
            accumulators + 4u, multipliers + 4u,
            output_min, output_max, output_zero);
        if (output_dtype == CAMPP_DTYPE_UINT8) {
            const uint8x8_t packed = vqmovn_u16(vcombine_u16(
                vqmovun_s32(low), vqmovun_s32(high)));
            vst1_u8(output_bytes, packed);
        } else {
            const int8x8_t packed = vqmovn_s16(vcombine_s16(
                vqmovn_s32(low), vqmovn_s32(high)));
            vst1_s8((int8_t *)output_bytes, packed);
        }
        return;
    }
#else
    uint32_t lane;
    for (lane = 0u; lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE; ++lane) {
        (void)campp_qconv_v4_requantize_scalar(
            accumulators[lane], multipliers[lane], output_dtype,
            output_zero, &output_bytes[lane]);
    }
#endif
}

int campp_qconv_v4_requantize8(
    const int32_t accumulators[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const float multipliers[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint8_t output_dtype, int32_t output_zero,
    uint8_t output_bytes[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    uint32_t lane;
    if (accumulators == NULL || multipliers == NULL || output_bytes == NULL ||
        (output_dtype != CAMPP_DTYPE_UINT8 &&
         output_dtype != CAMPP_DTYPE_INT8)) {
        return 1;
    }
    for (lane = 0u; lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE; ++lane) {
        if (!isfinite(multipliers[lane])) return 1;
    }
    campp_qconv_v4_requantize8_validated(
        accumulators, multipliers, output_dtype, output_zero, output_bytes);
    return 0;
}
