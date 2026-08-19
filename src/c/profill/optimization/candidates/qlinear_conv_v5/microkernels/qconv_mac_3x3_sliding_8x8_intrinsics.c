#include "qconv_mac_3x3_sliding_8x8.h"

#include "microkernels/qconv_mac_3x3_interior_8x8.h"

CamppQconvMac4x8Result campp_qconv_v5_mac_3x3_sliding_8x8(
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE])
{
    return campp_qconv_v4_mac_3x3_interior_8x8_validated(
        tile_plan, packed_weights, input_channels, input_zero,
        weight_zero, bias, accumulators);
}
