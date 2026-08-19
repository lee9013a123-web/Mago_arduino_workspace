#ifndef CAMPP_PROFILL_FINAL_CANDIDATE_SUITE_H
#define CAMPP_PROFILL_FINAL_CANDIDATE_SUITE_H

#include <stdint.h>

#include "internal/kernel_registry.h"

#define CAMPP_FINAL_CANDIDATE_MAX_KERNELS 64u

typedef struct CamppFinalCandidateSuiteStats {
    uint32_t qconv_entries;
    uint32_t fused_qconv_entries;
    uint32_t bn_entries;
    uint32_t dequant_entries;
    uint32_t remaining_entries;
    uint32_t total_entries;
} CamppFinalCandidateSuiteStats;

/*
 * Production AArch64 registry를 복사하고 검증이 끝난 최종 후보 entry만
 * 교체한다. registry와 entries는 context보다 오래 살아 있어야 하므로 suite를
 * 호출자 소유 구조체로 둔다.
 */
typedef struct CamppFinalCandidateSuite {
    CamppKernelRegistry registry;
    CamppKernelEntry entries[CAMPP_FINAL_CANDIDATE_MAX_KERNELS];
    CamppFinalCandidateSuiteStats stats;
} CamppFinalCandidateSuite;

CamppStatus campp_final_candidate_suite_create(
    const CamppKernelRegistry *base_registry,
    CamppFinalCandidateSuite *out_suite);

const char *campp_final_candidate_suite_name(void);

#endif /* CAMPP_PROFILL_FINAL_CANDIDATE_SUITE_H */
