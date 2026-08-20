#ifndef CAMPP_PROFILL_FUSED_QUANT_QCONV_CANDIDATE_H
#define CAMPP_PROFILL_FUSED_QUANT_QCONV_CANDIDATE_H

#include "internal/kernel_registry.h"

typedef enum CamppFusedQconvCandidateMode {
    CAMPP_FUSED_QCONV_CANDIDATE_BASELINE = 0,
    CAMPP_FUSED_QCONV_CANDIDATE_MAC = 1,
    CAMPP_FUSED_QCONV_CANDIDATE_COMBINED = 2,
    CAMPP_FUSED_QCONV_CANDIDATE_MAC_FIXED = 3,
    CAMPP_FUSED_QCONV_CANDIDATE_QUANT_NEON = 4,
    CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED = 5,
    CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_V4 = 6,
    CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID = 7,
    CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_V5 = 8
} CamppFusedQconvCandidateMode;

const char *campp_fused_qconv_candidate_mode_name(
    CamppFusedQconvCandidateMode mode);

int campp_fused_qconv_candidate_mode_parse(
    const char *text, CamppFusedQconvCandidateMode *out_mode);

const CamppKernelEntry *campp_fused_qconv_candidate_entry(
    CamppFusedQconvCandidateMode mode);

#endif /* CAMPP_PROFILL_FUSED_QUANT_QCONV_CANDIDATE_H */
