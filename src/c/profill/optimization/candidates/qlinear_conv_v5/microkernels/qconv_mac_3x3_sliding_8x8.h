#ifndef CAMPP_PROFILL_QCONV_MAC_3X3_SLIDING_8X8_H
#define CAMPP_PROFILL_QCONV_MAC_3X3_SLIDING_8X8_H

#include "qconv_mac_v5_common.h"

CamppQconvMac4x8Result campp_qconv_v5_mac_3x3_sliding_8x8(
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    bool sliding_eligible,
    int32_t input_zero, bool zero_point_fastpath,
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE]);

#endif /* CAMPP_PROFILL_QCONV_MAC_3X3_SLIDING_8X8_H */
