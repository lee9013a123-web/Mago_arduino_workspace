#include "qconv_mac_tail.h"

int campp_qconv_v4_mac_tail(
    const CamppQconvV4ExecutionPlan *plan,
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2],
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    if (plan == NULL || tile_plan == NULL) return 1;
    return campp_qconv_mac_neon_tile_v2(
        tile_plan->input_points, plan->kernel_elements,
        tile_plan->tile_count, packed_weights, plan->inputs_per_group,
        plan->input->dtype, plan->weight->dtype, plan->input_zero,
        weight_zero, bias, valid_outputs, accumulators);
}
