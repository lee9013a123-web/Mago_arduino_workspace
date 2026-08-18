#include "campp_profill/optimization/perf_sample_window.h"

#include <errno.h>
#include <string.h>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/prctl.h>
#include <unistd.h>
#endif

#if defined(__linux__)
/* perf answers every accepted command with "ack\n" on the ack FIFO. */
static int campp_perf_write_all(int fd, const char *text)
{
    size_t remaining = strlen(text);
    const char *cursor = text;
    while (remaining > 0u) {
        ssize_t written = write(fd, cursor, remaining);
        if (written < 0) {
            if (errno == EINTR) continue;
            return 1;
        }
        cursor += (size_t)written;
        remaining -= (size_t)written;
    }
    return 0;
}

static int campp_perf_wait_ack(int fd)
{
    char buffer[16];
    for (;;) {
        ssize_t got = read(fd, buffer, sizeof(buffer));
        if (got > 0) return 0;
        if (got == 0) return 1; /* perf closed the FIFO */
        if (errno == EINTR) continue;
        return 1;
    }
}

static int campp_perf_command(
    CamppPerfSampleWindow *window, const char *command)
{
    if (campp_perf_write_all(window->control_fd, command) != 0 ||
        campp_perf_wait_ack(window->ack_fd) != 0) {
        window->error_number = errno;
        window->control_active = false;
        window->supported = false;
        return 1;
    }
    return 0;
}
#endif

void campp_perf_sample_window_initialize(
    CamppPerfSampleWindow *window, bool requested)
{
    if (window == NULL) {
        return;
    }
    memset(window, 0, sizeof(*window));
    window->requested = requested;
    window->control_fd = -1;
    window->ack_fd = -1;
#if defined(__linux__)
    window->supported = true;
#else
    window->supported = false;
    if (requested) window->error_number = ENOSYS;
#endif
}

int campp_perf_sample_window_attach_control(
    CamppPerfSampleWindow *window, const char *control_path,
    const char *ack_path)
{
    if (window == NULL || control_path == NULL || ack_path == NULL) return 1;
    if (!window->requested) return 0;
#if defined(__linux__)
    /*
     * Open the ack FIFO first and non-blocking: perf opens its read end of the
     * ctl FIFO at startup, and opening ours for write would otherwise block
     * until perf is ready.
     */
    window->ack_fd = open(ack_path, O_RDONLY | O_NONBLOCK);
    if (window->ack_fd < 0) {
        window->error_number = errno;
        window->supported = false;
        return 1;
    }
    window->control_fd = open(control_path, O_WRONLY);
    if (window->control_fd < 0) {
        window->error_number = errno;
        window->supported = false;
        (void)close(window->ack_fd);
        window->ack_fd = -1;
        return 1;
    }
    /* Block on the ack side once it is connected so acknowledgements wait. */
    {
        int flags = fcntl(window->ack_fd, F_GETFL, 0);
        if (flags >= 0) {
            (void)fcntl(window->ack_fd, F_SETFL, flags & ~O_NONBLOCK);
        }
    }
    window->control_active = true;
    return 0;
#else
    (void)control_path;
    (void)ack_path;
    window->supported = false;
    window->error_number = ENOSYS;
    return 1;
#endif
}

int campp_perf_sample_window_disable(CamppPerfSampleWindow *window)
{
    if (window == NULL) return 1;
    if (!window->requested) return 0;
    if (!window->supported) return 1;
#if defined(__linux__)
    if (window->control_active) {
        if (campp_perf_command(window, "disable\n") != 0) {
            window->enabled = false;
            return 1;
        }
        window->enabled = false;
        return 0;
    }
    /* Best effort for self-owned events only; never the isolation mechanism. */
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
    if (window->control_active) {
        if (campp_perf_command(window, "enable\n") != 0) {
            window->enabled = false;
            return 1;
        }
        window->enabled = true;
        return 0;
    }
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

void campp_perf_sample_window_release(CamppPerfSampleWindow *window)
{
    if (window == NULL) return;
#if defined(__linux__)
    if (window->control_fd >= 0) (void)close(window->control_fd);
    if (window->ack_fd >= 0) (void)close(window->ack_fd);
#endif
    window->control_fd = -1;
    window->ack_fd = -1;
    window->control_active = false;
}
