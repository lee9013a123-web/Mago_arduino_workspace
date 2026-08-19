#ifndef CAMPP_PROFILL_QCONV_V4_EXECUTION_PLAN_H
#define CAMPP_PROFILL_QCONV_V4_EXECUTION_PLAN_H

#include <stdbool.h>
#include <stdint.h>

#include "internal/kernel_registry.h"
#include "internal/runtime_model.h"
#include "qconv_address_fastpath.h"

typedef enum CamppQconvV4Path {
    CAMPP_QCONV_V4_PATH_GENERIC = 0,
    CAMPP_QCONV_V4_PATH_1X1 = 1,
    CAMPP_QCONV_V4_PATH_3X3_INTERIOR = 2
} CamppQconvV4Path;

typedef struct CamppQconvV4ExecutionPlan {
    const CamppTensorView *input;
    const CamppTensorView *weight;
    CamppTensorView *output;
    CamppQconvAddressPlan address;
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[2];
    int64_t dilations[2];
    uint8_t spatial_rank;
    uint32_t group;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t inputs_per_group;
    uint32_t outputs_per_group;
    uint32_t input_blocks;
    uint32_t output_blocks;
    uint32_t kernel_elements;
    uint32_t output_spatial;
    float input_scale;
    float output_scale;
    int32_t input_zero;
    int32_t output_zero;
    CamppQconvV4Path preferred_path;
    bool direct_channel_store;
    bool scalar_weight_scale;
    bool scalar_weight_zero;
    bool has_bias;
} CamppQconvV4ExecutionPlan;

CamppStatus campp_qconv_v4_execution_plan_create(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppQconvV4ExecutionPlan *out_plan);

#endif /* CAMPP_PROFILL_QCONV_V4_EXECUTION_PLAN_H */
