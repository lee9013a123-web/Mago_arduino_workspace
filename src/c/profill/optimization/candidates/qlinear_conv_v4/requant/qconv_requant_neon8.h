#ifndef CAMPP_PROFILL_QCONV_V4_REQUANT_NEON8_H
#define CAMPP_PROFILL_QCONV_V4_REQUANT_NEON8_H

#include <stdint.h>

#include "qconv_mac_neon.h"

int campp_qconv_v4_requantize8(
    const int32_t accumulators[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const float multipliers[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint8_t output_dtype, int32_t output_zero,
    uint8_t output_bytes[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE]);

/* Parameter and dtype validation has already been hoisted by the v4 plan. */
void campp_qconv_v4_requantize8_validated(
    const int32_t accumulators[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const float multipliers[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint8_t output_dtype, int32_t output_zero,
    uint8_t output_bytes[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE]);

#endif /* CAMPP_PROFILL_QCONV_V4_REQUANT_NEON8_H */
