#ifndef CAMPP_PROFILL_DEQUANT_NEON_H
#define CAMPP_PROFILL_DEQUANT_NEON_H

#include <stdint.h>

#define CAMPP_DEQUANT_NEON_LANES 16u

void campp_dequant_neon_scalar16(
    const uint8_t *input, uint8_t input_dtype, float scale,
    int32_t zero_point, float *output);

void campp_dequant_neon_per_axis16(
    const uint8_t *input, uint8_t input_dtype,
    const float scales[CAMPP_DEQUANT_NEON_LANES],
    const int32_t zero_points[CAMPP_DEQUANT_NEON_LANES], float *output);

#endif /* CAMPP_PROFILL_DEQUANT_NEON_H */
