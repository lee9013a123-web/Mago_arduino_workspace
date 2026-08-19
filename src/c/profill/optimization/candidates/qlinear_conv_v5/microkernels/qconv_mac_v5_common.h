#ifndef CAMPP_PROFILL_QCONV_MAC_V5_COMMON_H
#define CAMPP_PROFILL_QCONV_MAC_V5_COMMON_H

#include <stdbool.h>

#include "qconv_mac_4x8.h"
#include "qconv_v4_tile_plan.h"

#define CAMPP_QCONV_V5_OUTPUT_TILE CAMPP_QCONV_CANDIDATE_OUTPUT_TILE
#define CAMPP_QCONV_V5_SPATIAL_TILE CAMPP_QCONV_CANDIDATE_TILE

CamppQconvMac4x8Result campp_qconv_v5_mac_8x8_raw(
    const uint8_t *const input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t kernel_elements,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    int32_t input_zero, bool zero_point_fastpath,
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE]);

CamppQconvMac4x8Result campp_qconv_v5_mac_3x3_sliding_raw(
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    int32_t input_zero, bool zero_point_fastpath,
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE]);

#endif /* CAMPP_PROFILL_QCONV_MAC_V5_COMMON_H */
