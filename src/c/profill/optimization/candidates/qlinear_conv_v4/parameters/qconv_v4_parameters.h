#ifndef CAMPP_PROFILL_QCONV_V4_PARAMETERS_H
#define CAMPP_PROFILL_QCONV_V4_PARAMETERS_H

#include <stdint.h>

#include "qconv_v4_execution_plan.h"
#include "qconv_mac_neon.h"

typedef struct CamppQconvV4ParameterBlock {
    int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
    int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
    float multiplier[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
    uint32_t valid_outputs;
} CamppQconvV4ParameterBlock;

CamppStatus campp_qconv_v4_parameters_load(
    const CamppQconvV4ExecutionPlan *plan,
    const CamppTensorView *inputs, uint8_t input_count,
    uint32_t group_index, uint32_t first_within,
    CamppQconvV4ParameterBlock *out_parameters);

#endif /* CAMPP_PROFILL_QCONV_V4_PARAMETERS_H */
