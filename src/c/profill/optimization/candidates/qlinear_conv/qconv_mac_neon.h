#ifndef CAMPP_PROFILL_QCONV_MAC_NEON_H
#define CAMPP_PROFILL_QCONV_MAC_NEON_H

#include <stdint.h>

#include "qconv_address_fastpath.h"

#define CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK 4u
#define CAMPP_QCONV_CANDIDATE_OUTPUT_TILE 8u
#define CAMPP_QCONV_CANDIDATE_INPUT_BLOCK 4u
#define CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS 9u

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

/*
 * Two O4I4 output blocks share each input load. All kernel positions are
 * accumulated before the result is written back, so the hot loop does not
 * round-trip partial contributions through memory.
 */
int campp_qconv_mac_neon_tile_v2(
    const uint8_t *const input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t kernel_elements, uint32_t tile_count,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    uint8_t input_dtype, uint8_t weight_dtype, int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE]);

/*
 * Requantize one spatial tile and write directly to the channel-packed
 * output. The AArch64 path rounds, clamps and packs four channels at once.
 */
int campp_qconv_requantize_store_neon_tile(
    CamppTensorView *output, uint32_t batch, uint32_t output_channel,
    uint8_t spatial_rank,
    const uint32_t output_coordinates[CAMPP_QCONV_CANDIDATE_TILE][2],
    uint32_t tile_count, uint32_t valid_outputs,
    const int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const float multipliers[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    int32_t output_zero);

#endif /* CAMPP_PROFILL_QCONV_MAC_NEON_H */
