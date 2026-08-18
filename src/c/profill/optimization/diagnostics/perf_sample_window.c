#include "campp_profill/optimization/perf_sample_window.h"

#include <errno.h>
#include <string.h>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

void campp_perf_sample_window_initialize(
    CamppPerfSampleWindow *window, bool requested)
{
    if (window == NULL) {
        return;
    }
    memset(window, 0, sizeof(*window));
    window->requested = requested;
#if defined(__linux__)
    window->supported = true;
#else
    window->supported = false;
    if (requested) window->error_number = ENOSYS;
#endif
}

int campp_perf_sample_window_disable(CamppPerfSampleWindow *window)
{
    if (window == NULL) return 1;
    if (!window->requested) return 0;
    if (!window->supported) return 1;
#if defined(__linux__)
    if (prctl(PR_TASK_PERF_EVENTS_DISABLE, 0, 0, 0, 0) != 0) {
        window->error_number = errno;
        window->supported = false;
        window->enabled = false;
        return 1;
    }
    window->enabled = false;
    return 0;
#else
    return 1;
#endif
}

int campp_perf_sample_window_enable(CamppPerfSampleWindow *window)
{
    if (window == NULL) return 1;
    if (!window->requested) return 0;
    if (!window->supported) return 1;
#if defined(__linux__)
    if (prctl(PR_TASK_PERF_EVENTS_ENABLE, 0, 0, 0, 0) != 0) {
        window->error_number = errno;
        window->supported = false;
        window->enabled = false;
        return 1;
    }
    window->enabled = true;
    return 0;
#else
    return 1;
#endif
}
