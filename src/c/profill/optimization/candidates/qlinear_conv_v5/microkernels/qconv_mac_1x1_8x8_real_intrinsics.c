#include "qconv_mac_1x1_8x8_real.h"

CamppQconvMac4x8Result campp_qconv_v5_mac_1x1_8x8_real(
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    int32_t input_zero, bool zero_point_fastpath,
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE])
{
    if (tile_plan == NULL || tile_plan->path != CAMPP_QCONV_V4_PATH_1X1) {
        return CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
    }
    return campp_qconv_v5_mac_8x8_raw(
        tile_plan->input_points, 1u, packed_weights, input_channels,
        input_zero, zero_point_fastpath, weight_zero, bias, accumulators);
}
