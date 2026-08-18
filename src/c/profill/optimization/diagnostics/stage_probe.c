#include "campp_profill/optimization/stage_probe.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "campp_profill/operator_profiler.h"

static _Thread_local CamppOptimizationProbe *campp_active_probe = NULL;

CamppStatus campp_optimization_probe_create(
    uint32_t sample_capacity, CamppOptimizationProbe *probe)
{
    size_t sample_count;

    if (probe == NULL || sample_capacity == 0u ||
        sample_capacity > SIZE_MAX / CAMPP_OPT_STAGE_COUNT) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    memset(probe, 0, sizeof(*probe));
    sample_count = (size_t)sample_capacity * CAMPP_OPT_STAGE_COUNT;
    if (sample_count > SIZE_MAX / sizeof(*probe->samples_ns)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    probe->samples_ns = (uint64_t *)calloc(
        sample_count, sizeof(*probe->samples_ns));
    if (probe->samples_ns == NULL) {
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    probe->sample_capacity = sample_capacity;
    return CAMPP_STATUS_OK;
}

void campp_optimization_probe_release(CamppOptimizationProbe *probe)
{
    if (probe == NULL) {
        return;
    }
    if (campp_active_probe == probe) {
        campp_active_probe = NULL;
    }
    free(probe->samples_ns);
    memset(probe, 0, sizeof(*probe));
}

void campp_optimization_probe_set_active(CamppOptimizationProbe *probe)
{
    campp_active_probe = probe;
}

CamppStatus campp_optimization_probe_begin_iteration(
    CamppOptimizationProbe *probe, uint32_t iteration)
{
    if (probe == NULL || probe->samples_ns == NULL ||
        iteration >= probe->sample_capacity || probe->iteration_active) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    memset(probe->current_ns, 0, sizeof(probe->current_ns));
    probe->current_iteration = iteration;
    probe->iteration_active = true;
    probe->enabled = true;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_optimization_probe_finish_iteration(
    CamppOptimizationProbe *probe)
{
    uint32_t stage;

    if (probe == NULL || !probe->iteration_active ||
        probe->current_iteration >= probe->sample_capacity) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    probe->enabled = false;
    for (stage = 0u; stage < CAMPP_OPT_STAGE_COUNT; ++stage) {
        probe->samples_ns[
            (size_t)stage * probe->sample_capacity +
            probe->current_iteration] = probe->current_ns[stage];
    }
    probe->iteration_active = false;
    return CAMPP_STATUS_OK;
}

uint64_t campp_optimization_stage_begin(void)
{
    uint64_t now_ns = 0u;

    if (campp_active_probe == NULL || !campp_active_probe->enabled ||
        !campp_active_probe->iteration_active) {
        return 0u;
    }
    if (campp_operator_profiler_clock_now_ns(&now_ns) != CAMPP_STATUS_OK) {
        return 0u;
    }
    return now_ns;
}

void campp_optimization_stage_end(
    CamppOptimizationStage stage, uint64_t started_ns)
{
    uint64_t finished_ns = 0u;

    if (campp_active_probe == NULL || !campp_active_probe->enabled ||
        !campp_active_probe->iteration_active || started_ns == 0u ||
        stage < 0 || stage >= CAMPP_OPT_STAGE_COUNT ||
        campp_operator_profiler_clock_now_ns(&finished_ns) != CAMPP_STATUS_OK ||
        finished_ns < started_ns) {
        return;
    }
    campp_active_probe->current_ns[stage] += finished_ns - started_ns;
}

const char *campp_optimization_stage_name(CamppOptimizationStage stage)
{
    static const char *const names[CAMPP_OPT_STAGE_COUNT] = {
        "qconv_setup",
        "qconv_mac_address",
        "qconv_requant_write",
        "fused_input_quantize",
        "fused_qconv",
        "bn_setup",
        "bn_elementwise",
        "dequant_setup",
        "dequant_elementwise"
    };

    return stage >= 0 && stage < CAMPP_OPT_STAGE_COUNT
        ? names[stage] : "invalid";
}

const uint64_t *campp_optimization_stage_samples(
    const CamppOptimizationProbe *probe, CamppOptimizationStage stage)
{
    if (probe == NULL || probe->samples_ns == NULL ||
        stage < 0 || stage >= CAMPP_OPT_STAGE_COUNT) {
        return NULL;
    }
    return probe->samples_ns + (size_t)stage * probe->sample_capacity;
}
