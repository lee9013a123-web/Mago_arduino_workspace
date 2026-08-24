#ifndef CAMPP_PROFILL_QCONV_V5_EXECUTION_PLAN_H
#define CAMPP_PROFILL_QCONV_V5_EXECUTION_PLAN_H

#include "address/qconv_v5_address.h"
#include "planning/qconv_v4_execution_plan.h"

typedef struct CamppQconvV5ExecutionPlan {
    CamppQconvV4ExecutionPlan v4;
    CamppQconvV5AddressPlan address;
} CamppQconvV5ExecutionPlan;

CamppStatus campp_qconv_v5_address_plan_from_v4(
    const CamppQconvV4ExecutionPlan *v4_plan,
    CamppQconvV5AddressPlan *out_address);

CamppStatus campp_qconv_v5_execution_plan_create(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppQconvV5ExecutionPlan *out_plan);

#endif /* CAMPP_PROFILL_QCONV_V5_EXECUTION_PLAN_H */
