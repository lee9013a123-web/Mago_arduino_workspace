#ifndef CAMPP_PROFILL_FUSED_DEQUANT_RELU_QUANT_CANDIDATE_H
#define CAMPP_PROFILL_FUSED_DEQUANT_RELU_QUANT_CANDIDATE_H

#include "internal/kernel_registry.h"

typedef enum CamppFusedDqRqCandidateMode {
    CAMPP_FUSED_DQRQ_CANDIDATE_BASELINE = 0,
    CAMPP_FUSED_DQRQ_CANDIDATE_SCALAR = 1,
    CAMPP_FUSED_DQRQ_CANDIDATE_NEON = 2
} CamppFusedDqRqCandidateMode;

const char *campp_fused_dqrq_candidate_mode_name(
    CamppFusedDqRqCandidateMode mode);

int campp_fused_dqrq_candidate_mode_parse(
    const char *text, CamppFusedDqRqCandidateMode *out_mode);

const CamppKernelEntry *campp_fused_dqrq_candidate_entry(
    CamppFusedDqRqCandidateMode mode);

#endif /* CAMPP_PROFILL_FUSED_DEQUANT_RELU_QUANT_CANDIDATE_H */
