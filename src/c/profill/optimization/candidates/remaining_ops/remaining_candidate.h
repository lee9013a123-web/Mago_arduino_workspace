#ifndef CAMPP_PROFILL_REMAINING_CANDIDATE_H
#define CAMPP_PROFILL_REMAINING_CANDIDATE_H

#include <stdint.h>

#include "internal/kernel_registry.h"

typedef enum CamppRemainingCandidateMode {
    CAMPP_REMAINING_CANDIDATE_BASELINE = 0,
    CAMPP_REMAINING_CANDIDATE_OPTIMIZED = 1
} CamppRemainingCandidateMode;

const char *campp_remaining_candidate_mode_name(
    CamppRemainingCandidateMode mode);

int campp_remaining_candidate_mode_parse(
    const char *text, CamppRemainingCandidateMode *out_mode);

const CamppKernelEntry *campp_remaining_candidate_entry(
    CamppRemainingCandidateMode mode, uint16_t opcode, uint16_t kernel_id);

#endif /* CAMPP_PROFILL_REMAINING_CANDIDATE_H */
