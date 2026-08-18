#define _POSIX_C_SOURCE 200809L

#include "campp_profill/operator_profiler.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int campp_profile_sample_count_overflows(
    uint32_t operator_count, uint32_t capacity)
{
    const size_t operators = (size_t)operator_count;
    const size_t samples = (size_t)capacity;

    return operators != 0u &&
           (samples > SIZE_MAX / operators ||
            operators * samples > SIZE_MAX / sizeof(uint64_t));
}

CamppStatus campp_operator_profiler_create(
    uint32_t operator_count, uint32_t sample_capacity_per_operator,
    CamppOperatorProfiler *profiler)
{
    size_t sample_count;

    if (profiler == NULL || operator_count == 0u ||
        sample_capacity_per_operator == 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (campp_profile_sample_count_overflows(
            operator_count, sample_capacity_per_operator)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }

    memset(profiler, 0, sizeof(*profiler));
    profiler->call_counts = (uint32_t *)calloc(
        (size_t)operator_count, sizeof(*profiler->call_counts));
    if (profiler->call_counts == NULL) {
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    sample_count =
        (size_t)operator_count * sample_capacity_per_operator;
    profiler->samples_ns = (uint64_t *)calloc(
        sample_count, sizeof(*profiler->samples_ns));
    if (profiler->samples_ns == NULL) {
        free(profiler->call_counts);
        memset(profiler, 0, sizeof(*profiler));
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    profiler->operator_count = operator_count;
    profiler->sample_capacity_per_operator =
        sample_capacity_per_operator;
    profiler->enabled = false;
    return CAMPP_STATUS_OK;
}

void campp_operator_profiler_release(CamppOperatorProfiler *profiler)
{
    if (profiler == NULL) {
        return;
    }
    free(profiler->samples_ns);
    free(profiler->call_counts);
    memset(profiler, 0, sizeof(*profiler));
}

void campp_operator_profiler_set_enabled(
    CamppOperatorProfiler *profiler, bool enabled)
{
    if (profiler != NULL) {
        profiler->enabled = enabled;
    }
}

bool campp_operator_profiler_is_enabled(
    const CamppOperatorProfiler *profiler)
{
    return profiler != NULL && profiler->enabled;
}

CamppStatus campp_operator_profiler_clock_now_ns(uint64_t *out_time_ns)
{
    struct timespec value;

    if (out_time_ns == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
#ifdef _WIN32
    if (timespec_get(&value, TIME_UTC) != TIME_UTC) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
#else
#ifdef CLOCK_MONOTONIC_RAW
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
#else
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
#endif
#endif
    *out_time_ns = (uint64_t)value.tv_sec * UINT64_C(1000000000) +
                   (uint64_t)value.tv_nsec;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_operator_profiler_begin(
    const CamppOperatorProfiler *profiler, uint64_t *out_start_ns)
{
    if (!campp_operator_profiler_is_enabled(profiler) ||
        out_start_ns == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    return campp_operator_profiler_clock_now_ns(out_start_ns);
}

CamppStatus campp_operator_profiler_end(
    CamppOperatorProfiler *profiler, uint32_t operator_id,
    uint64_t start_ns)
{
    uint64_t end_ns;
    uint32_t call_index;
    size_t sample_index;
    CamppStatus status;

    if (!campp_operator_profiler_is_enabled(profiler) ||
        operator_id >= profiler->operator_count ||
        profiler->call_counts == NULL || profiler->samples_ns == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    call_index = profiler->call_counts[operator_id];
    if (call_index >= profiler->sample_capacity_per_operator) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    status = campp_operator_profiler_clock_now_ns(&end_ns);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (end_ns < start_ns) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    sample_index =
        (size_t)operator_id * profiler->sample_capacity_per_operator +
        call_index;
    profiler->samples_ns[sample_index] = end_ns - start_ns;
    profiler->call_counts[operator_id] = call_index + 1u;
    return CAMPP_STATUS_OK;
}

uint32_t campp_operator_profiler_call_count(
    const CamppOperatorProfiler *profiler, uint32_t operator_id)
{
    if (profiler == NULL || profiler->call_counts == NULL ||
        operator_id >= profiler->operator_count) {
        return 0u;
    }
    return profiler->call_counts[operator_id];
}

const uint64_t *campp_operator_profiler_samples(
    const CamppOperatorProfiler *profiler, uint32_t operator_id)
{
    if (profiler == NULL || profiler->samples_ns == NULL ||
        operator_id >= profiler->operator_count) {
        return NULL;
    }
    return profiler->samples_ns +
           (size_t)operator_id * profiler->sample_capacity_per_operator;
}

const char *campp_operator_profiler_clock_name(void)
{
#ifdef _WIN32
    return "TIME_UTC";
#else
#ifdef CLOCK_MONOTONIC_RAW
    return "CLOCK_MONOTONIC_RAW";
#else
    return "CLOCK_MONOTONIC";
#endif
#endif
}
