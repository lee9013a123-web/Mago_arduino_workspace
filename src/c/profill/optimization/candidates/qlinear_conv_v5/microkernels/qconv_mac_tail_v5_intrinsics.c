#include "qconv_mac_tail_v5.h"

#include "microkernels/qconv_mac_tail.h"

int campp_qconv_v5_mac_tail(
    const CamppQconvV4ExecutionPlan *plan,
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2],
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    uint32_t valid_outputs,
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE])
{
    return campp_qconv_v4_mac_tail(
        plan, tile_plan, packed_weights, weight_zero, bias,
        valid_outputs, accumulators);
}
