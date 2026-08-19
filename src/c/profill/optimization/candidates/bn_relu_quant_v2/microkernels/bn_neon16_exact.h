#ifndef CAMPP_PROFILL_BN_NEON16_EXACT_H
#define CAMPP_PROFILL_BN_NEON16_EXACT_H

#include <stdint.h>

#include "campp_runtime/status_code.h"

CamppStatus campp_bn_v2_exact16_span(
    const float *input, uint8_t *output, uint32_t channels,
    const float *multiplier, const float *additive, float quant_scale,
    int32_t quant_zero, uint8_t output_dtype);

#endif /* CAMPP_PROFILL_BN_NEON16_EXACT_H */
