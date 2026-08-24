#ifndef CAMPP_PROFILL_OPTIMIZATION_STAGE_PROBE_H
#define CAMPP_PROFILL_OPTIMIZATION_STAGE_PROBE_H

#include <stdbool.h>
#include <stdint.h>

#include "campp_runtime/status_code.h"

typedef enum CamppOptimizationStage {
    CAMPP_OPT_STAGE_QCONV_SETUP = 0,
    CAMPP_OPT_STAGE_QCONV_MAC_ADDRESS = 1,
    CAMPP_OPT_STAGE_QCONV_REQUANT_WRITE = 2,
    CAMPP_OPT_STAGE_FUSED_INPUT_QUANTIZE = 3,
    CAMPP_OPT_STAGE_FUSED_QCONV = 4,
    CAMPP_OPT_STAGE_BN_SETUP = 5,
    CAMPP_OPT_STAGE_BN_ELEMENTWISE = 6,
    CAMPP_OPT_STAGE_DEQUANT_SETUP = 7,
    CAMPP_OPT_STAGE_DEQUANT_ELEMENTWISE = 8,
    CAMPP_OPT_STAGE_COUNT = 9
} CamppOptimizationStage;

typedef struct CamppOptimizationProbe {
    uint32_t sample_capacity;
    uint32_t current_iteration;
    bool enabled;
    bool iteration_active;
    uint64_t current_ns[CAMPP_OPT_STAGE_COUNT];
    uint64_t *samples_ns;
} CamppOptimizationProbe;

CamppStatus campp_optimization_probe_create(
    uint32_t sample_capacity, CamppOptimizationProbe *probe);

void campp_optimization_probe_release(CamppOptimizationProbe *probe);

void campp_optimization_probe_set_active(CamppOptimizationProbe *probe);

CamppStatus campp_optimization_probe_begin_iteration(
    CamppOptimizationProbe *probe, uint32_t iteration);

CamppStatus campp_optimization_probe_finish_iteration(
    CamppOptimizationProbe *probe);

uint64_t campp_optimization_stage_begin(void);

void campp_optimization_stage_end(
    CamppOptimizationStage stage, uint64_t started_ns);

const char *campp_optimization_stage_name(CamppOptimizationStage stage);

const uint64_t *campp_optimization_stage_samples(
    const CamppOptimizationProbe *probe, CamppOptimizationStage stage);

#define CAMPP_OPTIMIZATION_STAGE_BEGIN(variable) \
    const uint64_t variable = campp_optimization_stage_begin()

#define CAMPP_OPTIMIZATION_STAGE_END(stage, variable) \
    campp_optimization_stage_end((stage), (variable))

#endif /* CAMPP_PROFILL_OPTIMIZATION_STAGE_PROBE_H */
