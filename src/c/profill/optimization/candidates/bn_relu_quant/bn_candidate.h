#ifndef CAMPP_PROFILL_BN_CANDIDATE_H
#define CAMPP_PROFILL_BN_CANDIDATE_H

#include "internal/kernel_registry.h"

typedef enum CamppBnCandidateMode {
    CAMPP_BN_CANDIDATE_BASELINE = 0,
    CAMPP_BN_CANDIDATE_ADDRESS = 1,
    CAMPP_BN_CANDIDATE_AFFINE = 2,
    CAMPP_BN_CANDIDATE_QUANT = 3,
    CAMPP_BN_CANDIDATE_COMBINED = 4,
    CAMPP_BN_CANDIDATE_V2_EXACT16 = 5,
    CAMPP_BN_CANDIDATE_V2_SPATIAL2 = 6,
    CAMPP_BN_CANDIDATE_V2_PRESCALED = 7
} CamppBnCandidateMode;

const char *campp_bn_candidate_mode_name(CamppBnCandidateMode mode);

int campp_bn_candidate_mode_parse(
    const char *text, CamppBnCandidateMode *out_mode);

const CamppKernelEntry *campp_bn_candidate_entry(CamppBnCandidateMode mode);

#endif /* CAMPP_PROFILL_BN_CANDIDATE_H */
