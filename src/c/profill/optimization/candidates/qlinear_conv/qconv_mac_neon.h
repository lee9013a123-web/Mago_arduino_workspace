#ifndef CAMPP_PROFILL_QCONV_MAC_NEON_H
#define CAMPP_PROFILL_QCONV_MAC_NEON_H

#include <stdint.h>

#include "qconv_address_fastpath.h"

#define CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK 4u
#define CAMPP_QCONV_CANDIDATE_INPUT_BLOCK 4u

void campp_qconv_mac_scalar_tile(
    const uint8_t *const input_points[CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t tile_count, const uint8_t *packed_weight,
    uint32_t input_channels, uint8_t input_dtype, uint8_t weight_dtype,
    int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK],
    int32_t contribution[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK]);

void campp_qconv_mac_neon_tile(
    const uint8_t *const input_points[CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t tile_count, const uint8_t *packed_weight,
    uint32_t input_channels, uint8_t input_dtype, uint8_t weight_dtype,
    int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK],
    int32_t contribution[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK]);

#endif /* CAMPP_PROFILL_QCONV_MAC_NEON_H */
