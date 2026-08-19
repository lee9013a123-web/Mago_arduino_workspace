#ifndef CAMPP_PROFILL_BN_NEON16_PRESCALED_H
#define CAMPP_PROFILL_BN_NEON16_PRESCALED_H

#include <stdint.h>

#include "campp_runtime/status_code.h"

CamppStatus campp_bn_v2_prescaled16_span(
    const float *input, uint8_t *output, uint32_t channels,
    const float *multiplier, const float *additive,
    int32_t quant_zero, uint8_t output_dtype);

CamppStatus campp_bn_v2_prescaled16_spatial2(
    const float *input0, const float *input1,
    uint8_t *output0, uint8_t *output1, uint32_t channels,
    const float *multiplier, const float *additive,
    int32_t quant_zero, uint8_t output_dtype);

#endif /* CAMPP_PROFILL_BN_NEON16_PRESCALED_H */
