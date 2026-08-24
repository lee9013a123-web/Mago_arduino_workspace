#ifndef CAMPP_PROFILL_BN_QUANT_NEON_H
#define CAMPP_PROFILL_BN_QUANT_NEON_H

#include <stdint.h>

#include "campp_runtime/status_code.h"

CamppStatus campp_bn_quantize_scalar(
    float input, float multiplier, float additive, float quant_scale,
    int32_t quant_zero, uint8_t output_dtype, int32_t *out_value);

CamppStatus campp_bn_quantize_neon4(
    const float input[4], const float multiplier[4],
    const float additive[4], float quant_scale, int32_t quant_zero,
    uint8_t output_dtype, int32_t output[4]);

#endif /* CAMPP_PROFILL_BN_QUANT_NEON_H */
