#ifndef CAMPP_PROFILL_BN_V2_EXECUTION_PLAN_H
#define CAMPP_PROFILL_BN_V2_EXECUTION_PLAN_H

#include <stdint.h>

#include "internal/runtime_model.h"

typedef struct CamppBnV2ExecutionPlan {
    const CamppTensorView *input;
    CamppTensorView *output;
    const CamppTensorView *scale;
    const CamppTensorView *bias;
    const CamppTensorView *mean;
    const CamppTensorView *variance;
    uint32_t batches;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t input_strides[4];
    uint32_t output_strides[4];
    float epsilon;
    float quant_scale;
    int32_t quant_zero;
    uint8_t output_dtype;
} CamppBnV2ExecutionPlan;

CamppStatus campp_bn_v2_execution_plan_create(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppBnV2ExecutionPlan *out_plan);

const float *campp_bn_v2_input_pointer(
    const CamppBnV2ExecutionPlan *plan, uint32_t batch,
    uint32_t height, uint32_t width);

uint8_t *campp_bn_v2_output_pointer(
    const CamppBnV2ExecutionPlan *plan, uint32_t batch,
    uint32_t height, uint32_t width);

#endif /* CAMPP_PROFILL_BN_V2_EXECUTION_PLAN_H */
