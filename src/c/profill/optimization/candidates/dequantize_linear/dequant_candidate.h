#ifndef CAMPP_PROFILL_DEQUANT_CANDIDATE_H
#define CAMPP_PROFILL_DEQUANT_CANDIDATE_H

#include "internal/kernel_registry.h"

typedef enum CamppDequantCandidateMode {
    CAMPP_DEQUANT_CANDIDATE_BASELINE = 0,
    CAMPP_DEQUANT_CANDIDATE_ADDRESS = 1,
    CAMPP_DEQUANT_CANDIDATE_PARAMETER = 2,
    CAMPP_DEQUANT_CANDIDATE_SCALAR_COMBINED = 3,
    CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED = 4
} CamppDequantCandidateMode;

const char *campp_dequant_candidate_mode_name(
    CamppDequantCandidateMode mode);

int campp_dequant_candidate_mode_parse(
    const char *text, CamppDequantCandidateMode *out_mode);

const CamppKernelEntry *campp_dequant_candidate_entry(
    CamppDequantCandidateMode mode);

#endif /* CAMPP_PROFILL_DEQUANT_CANDIDATE_H */
