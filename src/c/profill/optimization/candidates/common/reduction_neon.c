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
    /*
     * Statistics pooling is an exact-output boundary.  The former four-channel
     * NEON loop let the compiler select a different contraction sequence for
     * the variance pass and changed low float bits on real E7 activations.
     * Keep the reference channel/frame order here.  Pointer stepping still
     * removes the generic tensor-coordinate work that this helper replaced.
     */
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
