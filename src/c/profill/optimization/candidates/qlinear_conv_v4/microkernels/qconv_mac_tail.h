#ifndef CAMPP_PROFILL_QCONV_V4_MAC_TAIL_H
#define CAMPP_PROFILL_QCONV_V4_MAC_TAIL_H

#include "qconv_v4_execution_plan.h"
#include "qconv_v4_tile_plan.h"

int campp_qconv_v4_mac_tail(
    const CamppQconvV4ExecutionPlan *plan,
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2],
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE]);

#endif /* CAMPP_PROFILL_QCONV_V4_MAC_TAIL_H */
