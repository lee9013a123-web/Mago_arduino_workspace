#ifndef CAMPP_PROFILL_BN_ITERATION_FASTPATH_H
#define CAMPP_PROFILL_BN_ITERATION_FASTPATH_H

#include <stdbool.h>
#include <stdint.h>

#include "internal/tensor_view.h"

typedef struct CamppBnIterationPlan {
    uint32_t batches;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t input_strides[4];
    uint32_t output_strides[4];
    bool channel_contiguous;
} CamppBnIterationPlan;

bool campp_bn_iteration_plan_create(
    const CamppTensorView *input, const CamppTensorView *output,
    CamppBnIterationPlan *out_plan);

uint64_t campp_bn_input_offset(
    const CamppBnIterationPlan *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width);

uint64_t campp_bn_output_offset(
    const CamppBnIterationPlan *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width);

#endif /* CAMPP_PROFILL_BN_ITERATION_FASTPATH_H */
