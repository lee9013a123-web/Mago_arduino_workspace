#include "bn_v2_parameter_block.h"

#include <math.h>
#include <stddef.h>

#include "bn_affine_fastpath.h"

CamppStatus campp_bn_v2_parameters_prepare(
    const CamppBnV2ExecutionPlan *plan, bool prescaled,
    CamppBnV2Parameters *out_parameters)
{
    uint32_t channel;

    if (plan == NULL || out_parameters == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (plan->channels > CAMPP_BN_V2_MAX_CHANNELS) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (channel = 0u; channel < plan->channels; ++channel) {
        CamppBnAffine affine;
        CamppStatus status = campp_bn_affine_channel(
            plan->scale, plan->bias, plan->mean, plan->variance,
            channel, plan->epsilon, &affine);
        if (status != CAMPP_STATUS_OK) return status;
        if (prescaled) {
            out_parameters->multiplier[channel] =
                affine.multiplier / plan->quant_scale;
            out_parameters->additive[channel] =
                affine.additive / plan->quant_scale;
            if (!isfinite(out_parameters->multiplier[channel]) ||
                !isfinite(out_parameters->additive[channel])) {
                return CAMPP_STATUS_KERNEL_FAILED;
            }
        } else {
            out_parameters->multiplier[channel] = affine.multiplier;
            out_parameters->additive[channel] = affine.additive;
        }
    }
    out_parameters->channels = plan->channels;
    out_parameters->prescaled = prescaled;
    return CAMPP_STATUS_OK;
}
