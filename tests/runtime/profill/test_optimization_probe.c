#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "campp_profill/optimization/stage_probe.h"
#include "campp_profill/optimization/perf_sample_window.h"

int main(void)
{
    CamppOptimizationProbe probe;
    CamppPerfSampleWindow perf_window;
    const uint64_t *samples;
    volatile uint64_t accumulator = 0u;
    uint64_t started_ns;
    uint32_t index;

    memset(&probe, 0, sizeof(probe));
    campp_perf_sample_window_initialize(&perf_window, false);
    if (perf_window.requested ||
        campp_perf_sample_window_disable(&perf_window) != 0 ||
        campp_perf_sample_window_enable(&perf_window) != 0) {
        return 1;
    }
    if (campp_optimization_probe_create(2u, &probe) != CAMPP_STATUS_OK) {
        return 2;
    }
    campp_optimization_probe_set_active(&probe);
    if (campp_optimization_probe_begin_iteration(&probe, 0u) !=
        CAMPP_STATUS_OK) {
        return 3;
    }
    started_ns = campp_optimization_stage_begin();
    for (index = 0u; index < 100000u; ++index) accumulator += index;
    campp_optimization_stage_end(
        CAMPP_OPT_STAGE_QCONV_MAC_ADDRESS, started_ns);
    if (campp_optimization_probe_finish_iteration(&probe) !=
        CAMPP_STATUS_OK) {
        return 4;
    }
    samples = campp_optimization_stage_samples(
        &probe, CAMPP_OPT_STAGE_QCONV_MAC_ADDRESS);
    if (samples == NULL || samples[0] == 0u || samples[1] != 0u ||
        accumulator == 0u) {
        return 5;
    }
    if (strcmp(
            campp_optimization_stage_name(CAMPP_OPT_STAGE_BN_ELEMENTWISE),
            "bn_elementwise") != 0) {
        return 6;
    }
    campp_optimization_probe_set_active(NULL);
    campp_optimization_probe_release(&probe);
    puts("optimization probe test passed");
    return 0;
}
