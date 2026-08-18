#include "reduction_neon.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static float campp_load_f32(const uint8_t *data)
{
    float value;
    memcpy(&value, data, sizeof(value));
    return value;
}

static void campp_store_f32(uint8_t *data, float value)
{
    memcpy(data, &value, sizeof(value));
}

void campp_reduction_mean_channels_f32(
    const uint8_t *input, uint32_t frame_stride, uint32_t channels,
    uint32_t frames, uint32_t divisor, uint8_t *output)
{
    uint32_t channel = 0u;
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (; channel + 4u <= channels; channel += 4u) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        float lanes[4];
        uint32_t frame;
        for (frame = 0u; frame < frames; ++frame) {
            sum = vaddq_f32(
                sum,
                vld1q_f32((const float *)(input +
                    (uint64_t)frame * frame_stride +
                    (uint64_t)channel * sizeof(float))));
        }
        vst1q_f32(lanes, sum);
        for (frame = 0u; frame < 4u; ++frame) {
            campp_store_f32(
                output + (uint64_t)(channel + frame) * sizeof(float),
                lanes[frame] / (float)divisor);
        }
    }
#endif
    for (; channel < channels; ++channel) {
        float sum = 0.0f;
        uint32_t frame;
        for (frame = 0u; frame < frames; ++frame) {
            sum += campp_load_f32(
                input + (uint64_t)frame * frame_stride +
                (uint64_t)channel * sizeof(float));
        }
        campp_store_f32(
            output + (uint64_t)channel * sizeof(float),
            sum / (float)divisor);
    }
}

void campp_reduction_statistics_channels_f32(
    const uint8_t *input, uint32_t frame_stride, uint32_t channels,
    uint32_t frames, float variance_multiplier, float variance_divisor,
    uint8_t *mean_output, uint8_t *std_output)
{
    uint32_t channel = 0u;
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (; channel + 4u <= channels; channel += 4u) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        float32x4_t variance_sum = vdupq_n_f32(0.0f);
        float means[4];
        float variances[4];
        uint32_t frame;
        for (frame = 0u; frame < frames; ++frame) {
            sum = vaddq_f32(
                sum,
                vld1q_f32((const float *)(input +
                    (uint64_t)frame * frame_stride +
                    (uint64_t)channel * sizeof(float))));
        }
        vst1q_f32(means, sum);
        for (frame = 0u; frame < 4u; ++frame) {
            means[frame] /= (float)frames;
        }
        {
            const float32x4_t mean = vld1q_f32(means);
            for (frame = 0u; frame < frames; ++frame) {
                const float32x4_t value = vld1q_f32(
                    (const float *)(input +
                        (uint64_t)frame * frame_stride +
                        (uint64_t)channel * sizeof(float)));
                const float32x4_t centered = vsubq_f32(value, mean);
                /* fmla를 쓰면 곱셈 결과가 반올림되지 않아 reference와
                   bitwise가 어긋난다. 곱셈과 덧셈을 분리해 유지한다. */
                variance_sum = vaddq_f32(
                    variance_sum, vmulq_f32(centered, centered));
            }
        }
        vst1q_f32(variances, variance_sum);
        for (frame = 0u; frame < 4u; ++frame) {
            float variance = variances[frame] / (float)frames;
            variance = variance * variance_multiplier;
            variance = variance / variance_divisor;
            campp_store_f32(
                mean_output +
                    (uint64_t)(channel + frame) * sizeof(float),
                means[frame]);
            campp_store_f32(
                std_output +
                    (uint64_t)(channel + frame) * sizeof(float),
                sqrtf(variance));
        }
    }
#endif
    for (; channel < channels; ++channel) {
        float sum = 0.0f;
        float variance_sum = 0.0f;
        float mean;
        float variance;
        uint32_t frame;
        for (frame = 0u; frame < frames; ++frame) {
            sum += campp_load_f32(
                input + (uint64_t)frame * frame_stride +
                (uint64_t)channel * sizeof(float));
        }
        mean = sum / (float)frames;
        for (frame = 0u; frame < frames; ++frame) {
            const float centered = campp_load_f32(
                input + (uint64_t)frame * frame_stride +
                (uint64_t)channel * sizeof(float)) - mean;
            variance_sum += centered * centered;
        }
        variance = variance_sum / (float)frames;
        variance = variance * variance_multiplier;
        variance = variance / variance_divisor;
        campp_store_f32(
            mean_output + (uint64_t)channel * sizeof(float), mean);
        campp_store_f32(
            std_output + (uint64_t)channel * sizeof(float), sqrtf(variance));
    }
}
