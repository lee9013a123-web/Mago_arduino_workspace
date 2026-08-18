#ifndef CAMPP_PROFILL_OPTIMIZATION_PERF_SAMPLE_WINDOW_H
#define CAMPP_PROFILL_OPTIMIZATION_PERF_SAMPLE_WINDOW_H

#include <stdbool.h>

typedef struct CamppPerfSampleWindow {
    bool requested;
    bool supported;
    bool enabled;
    int error_number;
} CamppPerfSampleWindow;

void campp_perf_sample_window_initialize(
    CamppPerfSampleWindow *window, bool requested);

int campp_perf_sample_window_disable(CamppPerfSampleWindow *window);

int campp_perf_sample_window_enable(CamppPerfSampleWindow *window);

#endif /* CAMPP_PROFILL_OPTIMIZATION_PERF_SAMPLE_WINDOW_H */
