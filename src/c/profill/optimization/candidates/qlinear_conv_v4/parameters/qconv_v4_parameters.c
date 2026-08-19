#include "qconv_v4_parameters.h"

#include <math.h>
#include <string.h>

#include "backends/cpu_reference/reference_kernel_utils.h"

CamppStatus campp_qconv_v4_parameters_load(
    const CamppQconvV4ExecutionPlan *plan,
    const CamppTensorView *inputs, uint8_t input_count,
    uint32_t group_index, uint32_t first_within,
    CamppQconvV4ParameterBlock *out_parameters)
{
    uint32_t remaining;
    uint32_t lane;

    if (plan == NULL || inputs == NULL || out_parameters == NULL ||
        input_count < 8u || input_count > 9u || group_index >= plan->group ||
        first_within >= plan->outputs_per_group) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    memset(out_parameters, 0, sizeof(*out_parameters));
    remaining = plan->outputs_per_group - first_within;
    out_parameters->valid_outputs =
        remaining < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE
        ? remaining : CAMPP_QCONV_CANDIDATE_OUTPUT_TILE;

    for (lane = 0u; lane < out_parameters->valid_outputs; ++lane) {
        const uint32_t channel = group_index * plan->outputs_per_group
            + first_within + lane;
        const uint64_t scale_index =
            plan->scalar_weight_scale ? 0u : channel;
        const uint64_t zero_index =
            plan->scalar_weight_zero ? 0u : channel;
        float weight_scale;
        CamppStatus status = campp_reference_read_f32(
            &inputs[4], scale_index, &weight_scale);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_read_quantized(
            &inputs[5], zero_index, &out_parameters->weight_zero[lane]);
        if (status != CAMPP_STATUS_OK) return status;
        if (!(weight_scale > 0.0f) || !isfinite(weight_scale)) {
            return CAMPP_STATUS_KERNEL_FAILED;
        }
        out_parameters->multiplier[lane] =
            plan->input_scale * weight_scale / plan->output_scale;
        if (!isfinite(out_parameters->multiplier[lane])) {
            return CAMPP_STATUS_KERNEL_FAILED;
        }
        if (plan->has_bias) {
            status = campp_reference_read_quantized(
                &inputs[8], channel, &out_parameters->bias[lane]);
            if (status != CAMPP_STATUS_OK) return status;
        }
    }
    return CAMPP_STATUS_OK;
}
