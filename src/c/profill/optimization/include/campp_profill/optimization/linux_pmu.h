#ifndef CAMPP_PROFILL_OPTIMIZATION_LINUX_PMU_H
#define CAMPP_PROFILL_OPTIMIZATION_LINUX_PMU_H

#include <stdbool.h>
#include <stdint.h>

typedef enum CamppOptimizationPmuEvent {
    CAMPP_OPT_PMU_CPU_CYCLES = 0,
    CAMPP_OPT_PMU_INSTRUCTIONS = 1,
    CAMPP_OPT_PMU_CACHE_REFERENCES = 2,
    CAMPP_OPT_PMU_CACHE_MISSES = 3,
    CAMPP_OPT_PMU_BRANCHES = 4,
    CAMPP_OPT_PMU_BRANCH_MISSES = 5,
    CAMPP_OPT_PMU_EVENT_COUNT = 6
} CamppOptimizationPmuEvent;

typedef struct CamppOptimizationPmu {
    int leader_fd;
    int file_descriptors[CAMPP_OPT_PMU_EVENT_COUNT];
    uint32_t sample_capacity;
    bool available;
    int unavailable_errno;
    uint64_t *samples;
} CamppOptimizationPmu;

int campp_optimization_pmu_create(
    uint32_t sample_capacity, CamppOptimizationPmu *pmu);

void campp_optimization_pmu_release(CamppOptimizationPmu *pmu);

int campp_optimization_pmu_begin(CamppOptimizationPmu *pmu);

int campp_optimization_pmu_end(
    CamppOptimizationPmu *pmu, uint32_t iteration);

const char *campp_optimization_pmu_event_name(
    CamppOptimizationPmuEvent event);

const uint64_t *campp_optimization_pmu_samples(
    const CamppOptimizationPmu *pmu, CamppOptimizationPmuEvent event);

#endif /* CAMPP_PROFILL_OPTIMIZATION_LINUX_PMU_H */
