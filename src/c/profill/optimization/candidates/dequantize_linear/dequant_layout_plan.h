#ifndef CAMPP_PROFILL_DEQUANT_LAYOUT_PLAN_H
#define CAMPP_PROFILL_DEQUANT_LAYOUT_PLAN_H

#include <stdbool.h>
#include <stdint.h>

#include "internal/tensor_view.h"

typedef struct CamppDequantLayoutPlan {
    uint32_t batches;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t input_strides[4];
    uint32_t output_strides[4];
    bool channel_contiguous;
} CamppDequantLayoutPlan;

bool campp_dequant_layout_plan_create(
    const CamppTensorView *input, const CamppTensorView *output,
    CamppDequantLayoutPlan *out_plan);

uint64_t campp_dequant_input_offset(
    const CamppDequantLayoutPlan *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width);

uint64_t campp_dequant_output_offset(
    const CamppDequantLayoutPlan *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width);

#endif /* CAMPP_PROFILL_DEQUANT_LAYOUT_PLAN_H */
