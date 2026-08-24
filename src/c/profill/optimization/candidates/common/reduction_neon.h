#ifndef CAMPP_PROFILL_REDUCTION_NEON_H
#define CAMPP_PROFILL_REDUCTION_NEON_H

#include <stdint.h>

void campp_reduction_mean_channels_f32(
    const uint8_t *input, uint32_t frame_stride, uint32_t channels,
    uint32_t frames, uint32_t divisor, uint8_t *output);

void campp_reduction_statistics_channels_f32(
    const uint8_t *input, uint32_t frame_stride, uint32_t channels,
    uint32_t frames, float variance_multiplier, float variance_divisor,
    uint8_t *mean_output, uint8_t *std_output);

#endif /* CAMPP_PROFILL_REDUCTION_NEON_H */
