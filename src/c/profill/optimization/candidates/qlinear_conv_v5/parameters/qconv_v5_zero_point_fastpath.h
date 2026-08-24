#ifndef CAMPP_PROFILL_QCONV_V5_ZERO_POINT_FASTPATH_H
#define CAMPP_PROFILL_QCONV_V5_ZERO_POINT_FASTPATH_H

#include <stdbool.h>
#include <stdint.h>

#include "qconv_mac_neon.h"

bool campp_qconv_v5_weight_zero_all_zero(
    const int32_t *weight_zero, uint32_t valid_outputs);

int32_t campp_qconv_v5_correct_bias_for_input_zero(
    int32_t bias, int32_t input_zero, int32_t weight_sum);

bool campp_qconv_v5_weight_sums_o4i4(
    const uint8_t *const packed_weights[2], uint32_t kernel_elements,
    uint32_t input_channels,
    int32_t weight_sums[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE]);

bool campp_qconv_v5_correct_bias_block(
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    int32_t input_zero,
    const int32_t weight_sums[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    int32_t corrected_bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE]);

#endif /* CAMPP_PROFILL_QCONV_V5_ZERO_POINT_FASTPATH_H */
