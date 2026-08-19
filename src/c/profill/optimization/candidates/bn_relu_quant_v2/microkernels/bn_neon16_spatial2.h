#ifndef CAMPP_PROFILL_BN_NEON16_SPATIAL2_H
#define CAMPP_PROFILL_BN_NEON16_SPATIAL2_H

#include <stdint.h>

#include "campp_runtime/status_code.h"

CamppStatus campp_bn_v2_exact16_spatial2(
    const float *input0, const float *input1,
    uint8_t *output0, uint8_t *output1, uint32_t channels,
    const float *multiplier, const float *additive, float quant_scale,
    int32_t quant_zero, uint8_t output_dtype);

#endif /* CAMPP_PROFILL_BN_NEON16_SPATIAL2_H */
