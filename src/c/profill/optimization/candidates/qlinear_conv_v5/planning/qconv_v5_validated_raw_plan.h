#ifndef CAMPP_PROFILL_QCONV_V5_VALIDATED_RAW_PLAN_H
#define CAMPP_PROFILL_QCONV_V5_VALIDATED_RAW_PLAN_H

#include <stdbool.h>
#include <stdint.h>

#include "planning/qconv_v4_execution_plan.h"
#include "parameters/qconv_v4_parameters.h"

typedef struct CamppQconvV5ValidatedRawPlan {
    bool enabled;
    bool weight_zero_is_all_zero;
    uint32_t input_channels;
    uint32_t valid_outputs;
} CamppQconvV5ValidatedRawPlan;

CamppQconvV5ValidatedRawPlan campp_qconv_v5_validated_raw_plan_create(
    const CamppQconvV4ExecutionPlan *plan,
    const CamppQconvV4ParameterBlock *parameters);

#endif /* CAMPP_PROFILL_QCONV_V5_VALIDATED_RAW_PLAN_H */
