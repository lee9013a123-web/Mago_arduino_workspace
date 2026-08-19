#ifndef CAMPP_PROFILL_QCONV_V4_MAC_3X3_INTERIOR_8X8_H
#define CAMPP_PROFILL_QCONV_V4_MAC_3X3_INTERIOR_8X8_H

#include "qconv_mac_4x8.h"
#include "qconv_v4_tile_plan.h"

CamppQconvMac4x8Result campp_qconv_v4_mac_3x3_interior_8x8(
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    uint8_t input_dtype, uint8_t weight_dtype, int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE]);

#endif /* CAMPP_PROFILL_QCONV_V4_MAC_3X3_INTERIOR_8X8_H */
