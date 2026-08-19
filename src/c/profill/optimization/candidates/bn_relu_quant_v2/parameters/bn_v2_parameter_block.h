#ifndef CAMPP_PROFILL_BN_V2_PARAMETER_BLOCK_H
#define CAMPP_PROFILL_BN_V2_PARAMETER_BLOCK_H

#include <stdbool.h>
#include <stdint.h>

#include "bn_v2_execution_plan.h"

#define CAMPP_BN_V2_LANES 16u
#define CAMPP_BN_V2_MAX_CHANNELS 4096u

typedef struct CamppBnV2Parameters {
    _Alignas(16) float multiplier[CAMPP_BN_V2_MAX_CHANNELS];
    _Alignas(16) float additive[CAMPP_BN_V2_MAX_CHANNELS];
    uint32_t channels;
    bool prescaled;
} CamppBnV2Parameters;

CamppStatus campp_bn_v2_parameters_prepare(
    const CamppBnV2ExecutionPlan *plan, bool prescaled,
    CamppBnV2Parameters *out_parameters);

#endif /* CAMPP_PROFILL_BN_V2_PARAMETER_BLOCK_H */
