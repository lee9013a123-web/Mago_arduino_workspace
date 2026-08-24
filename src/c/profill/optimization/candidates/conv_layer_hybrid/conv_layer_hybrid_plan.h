#ifndef CAMPP_PROFILL_CONV_LAYER_HYBRID_PLAN_H
#define CAMPP_PROFILL_CONV_LAYER_HYBRID_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "fused_quant_qconv_candidate.h"
#include "internal/runtime_model.h"
#include "qconv_candidate.h"

int campp_conv_layer_hybrid_has_bucket_plan(uint32_t bucket_frames);
size_t campp_conv_layer_hybrid_bucket_plan_count(void);
uint32_t campp_conv_layer_hybrid_bucket_plan_at(size_t index);

CamppQconvCandidateMode campp_conv_layer_hybrid_select_qconv(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op);

CamppFusedQconvCandidateMode campp_conv_layer_hybrid_select_fused_qconv(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op);

#endif /* CAMPP_PROFILL_CONV_LAYER_HYBRID_PLAN_H */
