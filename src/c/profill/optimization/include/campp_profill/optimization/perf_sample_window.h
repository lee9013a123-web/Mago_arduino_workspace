#ifndef CAMPP_PROFILL_OPTIMIZATION_PERF_SAMPLE_WINDOW_H
#define CAMPP_PROFILL_OPTIMIZATION_PERF_SAMPLE_WINDOW_H

#include <stdbool.h>

/*
 * Target-only perf sampling window.
 *
 * prctl(PR_TASK_PERF_EVENTS_ENABLE/DISABLE) only affects perf events owned by
 * the calling task. Events created by an external `perf record` belong to the
 * perf process, so prctl cannot gate them and the prelude leaks into the
 * profile. The window therefore drives perf's own control FIFO protocol
 * (`perf record -D -1 --control=fifo:<ctl>,<ack>`): perf starts with events
 * disabled and only enables them when we send "enable" and it answers "ack".
 *
 * prctl is kept as a best-effort extra for events the process owns itself, but
 * it is never the mechanism the isolation relies on.
 */
typedef struct CamppPerfSampleWindow {
    bool requested;
    bool supported;
    bool enabled;
    bool control_active; /* perf control FIFO negotiated successfully */
    int error_number;
    int control_fd; /* write side of perf's ctl FIFO, -1 when unused */
    int ack_fd;     /* read side of perf's ack FIFO, -1 when unused */
} CamppPerfSampleWindow;

void campp_perf_sample_window_initialize(
    CamppPerfSampleWindow *window, bool requested);

/*
 * Attach to perf's control FIFOs. Both paths must be FIFOs created by the
 * caller before `perf record` starts. Returns 0 on success.
 */
int campp_perf_sample_window_attach_control(
    CamppPerfSampleWindow *window, const char *control_path,
    const char *ack_path);

int campp_perf_sample_window_disable(CamppPerfSampleWindow *window);

int campp_perf_sample_window_enable(CamppPerfSampleWindow *window);

void campp_perf_sample_window_release(CamppPerfSampleWindow *window);

#endif /* CAMPP_PROFILL_OPTIMIZATION_PERF_SAMPLE_WINDOW_H */
