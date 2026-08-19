#include "qconv_mac_3x3_interior_8x8.h"

CamppQconvMac4x8Result campp_qconv_v4_mac_3x3_interior_8x8(
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    uint8_t input_dtype, uint8_t weight_dtype, int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    if (tile_plan == NULL ||
        tile_plan->path != CAMPP_QCONV_V4_PATH_3X3_INTERIOR) {
        return CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
    }
    return campp_qconv_mac_4x8_try_tile(
        tile_plan->input_points, 9u, tile_plan->tile_count,
        packed_weights, input_channels, input_dtype, weight_dtype,
        input_zero, weight_zero, bias, valid_outputs,
        CAMPP_QCONV_MAC_4X8_INTRINSICS, accumulators);
}
