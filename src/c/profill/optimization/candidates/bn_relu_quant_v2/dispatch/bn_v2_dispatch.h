#ifndef CAMPP_PROFILL_BN_V2_DISPATCH_H
#define CAMPP_PROFILL_BN_V2_DISPATCH_H

#include "bn_v2_parameter_block.h"

typedef enum CamppBnV2Mode {
    CAMPP_BN_V2_EXACT16 = 0,
    CAMPP_BN_V2_SPATIAL2 = 1,
    CAMPP_BN_V2_PRESCALED = 2
} CamppBnV2Mode;

CamppStatus campp_bn_v2_dispatch(
    CamppBnV2Mode mode, const CamppBnV2ExecutionPlan *plan,
    const CamppBnV2Parameters *parameters);

#endif /* CAMPP_PROFILL_BN_V2_DISPATCH_H */
