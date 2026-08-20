#include "qconv_v5_validated_raw_plan.h"

#include "parameters/qconv_v5_zero_point_fastpath.h"

CamppQconvV5ValidatedRawPlan campp_qconv_v5_validated_raw_plan_create(
    const CamppQconvV4ExecutionPlan *plan,
    const CamppQconvV4ParameterBlock *parameters)
{
    CamppQconvV5ValidatedRawPlan result;

    result.enabled = false;
    result.weight_zero_is_all_zero = false;
    result.sliding_3x3_eligible = false;
    result.input_channels = 0u;
    result.valid_outputs = 0u;
    if (plan == NULL || parameters == NULL) return result;

    result.input_channels = plan->inputs_per_group;
    result.valid_outputs = parameters->valid_outputs;
    result.weight_zero_is_all_zero =
        campp_qconv_v5_weight_zero_all_zero(
            parameters->weight_zero, parameters->valid_outputs);
    result.enabled =
        plan->fixed_mac_plan_eligible &&
        parameters->fixed_mac_block_eligible &&
        parameters->valid_outputs == 8u;
    /* The sliding body indexes one input row as base + column * stride
     * and reads 10 columns for 8 outputs.  Both hold only when the
     * kernel column step equals the output column step, i.e. stride 1. */
    result.sliding_3x3_eligible =
        plan->preferred_path == CAMPP_QCONV_V4_PATH_3X3_INTERIOR &&
        plan->spatial_rank == 2u &&
        plan->strides[1] == 1 && plan->dilations[1] == 1;
    return result;
}
