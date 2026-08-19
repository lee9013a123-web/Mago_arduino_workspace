#ifndef CAMPP_PROFILL_QCONV_CANDIDATE_H
#define CAMPP_PROFILL_QCONV_CANDIDATE_H

#include "internal/kernel_registry.h"

typedef enum CamppQconvCandidateMode {
    CAMPP_QCONV_CANDIDATE_BASELINE = 0,
    CAMPP_QCONV_CANDIDATE_ADDRESS = 1,
    CAMPP_QCONV_CANDIDATE_MAC = 2,
    CAMPP_QCONV_CANDIDATE_COMBINED = 3,
    CAMPP_QCONV_CANDIDATE_MAC_FIXED = 4,
    CAMPP_QCONV_CANDIDATE_MAC_ASM = 5,
    CAMPP_QCONV_CANDIDATE_V4 = 6
} CamppQconvCandidateMode;

const char *campp_qconv_candidate_mode_name(CamppQconvCandidateMode mode);

int campp_qconv_candidate_mode_parse(
    const char *text, CamppQconvCandidateMode *out_mode);

const CamppKernelEntry *campp_qconv_candidate_entry(
    CamppQconvCandidateMode mode);

#endif /* CAMPP_PROFILL_QCONV_CANDIDATE_H */
