#ifndef CAMPP_PROFILL_ELEMENTWISE_NEON_H
#define CAMPP_PROFILL_ELEMENTWISE_NEON_H

#include <stdint.h>

void campp_elementwise_add_f32(
    const float *left, const float *right, float *output, uint32_t count);

void campp_elementwise_relu_f32(
    const float *input, float *output, uint32_t count);

void campp_elementwise_quantize_f32(
    const float *input, uint8_t *output, uint32_t count, float scale,
    int32_t zero_point, uint8_t output_dtype);

void campp_elementwise_sigmoid_lut_mul_f32(
    const uint8_t *quantized, const float *other, float *output,
    uint32_t count, const float table[256]);

#endif /* CAMPP_PROFILL_ELEMENTWISE_NEON_H */
