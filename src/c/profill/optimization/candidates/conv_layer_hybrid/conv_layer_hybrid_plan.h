#ifndef CAMPP_PROFILL_CONV_LAYER_HYBRID_PLAN_H
#define CAMPP_PROFILL_CONV_LAYER_HYBRID_PLAN_H

#include "fused_quant_qconv_candidate.h"
#include "internal/runtime_model.h"
#include "qconv_candidate.h"

CamppQconvCandidateMode campp_conv_layer_hybrid_select_qconv(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op);

CamppFusedQconvCandidateMode campp_conv_layer_hybrid_select_fused_qconv(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op);

#endif /* CAMPP_PROFILL_CONV_LAYER_HYBRID_PLAN_H */
