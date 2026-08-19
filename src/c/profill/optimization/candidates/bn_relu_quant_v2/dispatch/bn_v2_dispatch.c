#include "bn_v2_dispatch.h"

#include "bn_neon16_exact.h"
#include "bn_neon16_prescaled.h"
#include "bn_neon16_spatial2.h"

static CamppStatus campp_bn_v2_run_one(
    CamppBnV2Mode mode, const CamppBnV2ExecutionPlan *plan,
    const CamppBnV2Parameters *parameters, uint32_t batch,
    uint32_t height, uint32_t width)
{
    const float *input = campp_bn_v2_input_pointer(
        plan, batch, height, width);
    uint8_t *output = campp_bn_v2_output_pointer(
        plan, batch, height, width);

    if (mode == CAMPP_BN_V2_PRESCALED) {
        return campp_bn_v2_prescaled16_span(
            input, output, plan->channels, parameters->multiplier,
            parameters->additive, plan->quant_zero, plan->output_dtype);
    }
    return campp_bn_v2_exact16_span(
        input, output, plan->channels, parameters->multiplier,
        parameters->additive, plan->quant_scale, plan->quant_zero,
        plan->output_dtype);
}

CamppStatus campp_bn_v2_dispatch(
    CamppBnV2Mode mode, const CamppBnV2ExecutionPlan *plan,
    const CamppBnV2Parameters *parameters)
{
    uint32_t batch;

    if (plan == NULL || parameters == NULL ||
        parameters->channels != plan->channels ||
        parameters->prescaled != (mode == CAMPP_BN_V2_PRESCALED)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    for (batch = 0u; batch < plan->batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < plan->height; ++height) {
            uint32_t width = 0u;
            if (mode != CAMPP_BN_V2_EXACT16) {
                for (; width + 1u < plan->width; width += 2u) {
                    const float *input0 = campp_bn_v2_input_pointer(
                        plan, batch, height, width);
                    const float *input1 = campp_bn_v2_input_pointer(
                        plan, batch, height, width + 1u);
                    uint8_t *output0 = campp_bn_v2_output_pointer(
                        plan, batch, height, width);
                    uint8_t *output1 = campp_bn_v2_output_pointer(
                        plan, batch, height, width + 1u);
                    CamppStatus status;
                    if (mode == CAMPP_BN_V2_PRESCALED) {
                        status = campp_bn_v2_prescaled16_spatial2(
                            input0, input1, output0, output1,
                            plan->channels, parameters->multiplier,
                            parameters->additive, plan->quant_zero,
                            plan->output_dtype);
                    } else {
                        status = campp_bn_v2_exact16_spatial2(
                            input0, input1, output0, output1,
                            plan->channels, parameters->multiplier,
                            parameters->additive, plan->quant_scale,
                            plan->quant_zero, plan->output_dtype);
                    }
                    if (status != CAMPP_STATUS_OK) return status;
                }
            }
            for (; width < plan->width; ++width) {
                CamppStatus status = campp_bn_v2_run_one(
                    mode, plan, parameters, batch, height, width);
                if (status != CAMPP_STATUS_OK) return status;
            }
        }
    }
    return CAMPP_STATUS_OK;
}
