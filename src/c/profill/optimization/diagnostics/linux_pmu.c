#define _GNU_SOURCE

#include "campp_profill/optimization/linux_pmu.h"

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

static void campp_optimization_pmu_initialize_fds(CamppOptimizationPmu *pmu)
{
    uint32_t event;
    pmu->leader_fd = -1;
    for (event = 0u; event < CAMPP_OPT_PMU_EVENT_COUNT; ++event) {
        pmu->file_descriptors[event] = -1;
    }
}

static void campp_optimization_pmu_mark_unavailable(
    CamppOptimizationPmu *pmu, int failure_errno)
{
    uint32_t event;

#if defined(__linux__)
    for (event = 0u; event < CAMPP_OPT_PMU_EVENT_COUNT; ++event) {
        if (pmu->file_descriptors[event] >= 0) {
            close(pmu->file_descriptors[event]);
            pmu->file_descriptors[event] = -1;
        }
    }
#else
    (void)event;
#endif
    pmu->leader_fd = -1;
    pmu->available = false;
    pmu->unavailable_errno = failure_errno;
}

#if defined(__linux__)
static int campp_perf_event_open(
    struct perf_event_attr *attributes, int group_fd)
{
    return (int)syscall(
        __NR_perf_event_open, attributes, 0, -1, group_fd, 0);
}
#endif

int campp_optimization_pmu_create(
    uint32_t sample_capacity, CamppOptimizationPmu *pmu)
{
    size_t sample_count;

    if (pmu == NULL || sample_capacity == 0u ||
        sample_capacity > SIZE_MAX / CAMPP_OPT_PMU_EVENT_COUNT) {
        return 1;
    }
    memset(pmu, 0, sizeof(*pmu));
    campp_optimization_pmu_initialize_fds(pmu);
    sample_count = (size_t)sample_capacity * CAMPP_OPT_PMU_EVENT_COUNT;
    if (sample_count > SIZE_MAX / sizeof(*pmu->samples)) {
        return 1;
    }
    pmu->samples = (uint64_t *)calloc(sample_count, sizeof(*pmu->samples));
    if (pmu->samples == NULL) {
        return 1;
    }
    pmu->sample_capacity = sample_capacity;

#if defined(__linux__)
    {
        static const uint64_t configurations[CAMPP_OPT_PMU_EVENT_COUNT] = {
            PERF_COUNT_HW_CPU_CYCLES,
            PERF_COUNT_HW_INSTRUCTIONS,
            PERF_COUNT_HW_CACHE_REFERENCES,
            PERF_COUNT_HW_CACHE_MISSES,
            PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
            PERF_COUNT_HW_BRANCH_MISSES
        };
        uint32_t event;

        for (event = 0u; event < CAMPP_OPT_PMU_EVENT_COUNT; ++event) {
            struct perf_event_attr attributes;
            int file_descriptor;
            memset(&attributes, 0, sizeof(attributes));
            attributes.type = PERF_TYPE_HARDWARE;
            attributes.size = sizeof(attributes);
            attributes.config = configurations[event];
            attributes.disabled = 1u;
            attributes.exclude_kernel = 1u;
            attributes.exclude_hv = 1u;
            attributes.read_format = PERF_FORMAT_GROUP;
            file_descriptor = campp_perf_event_open(
                &attributes, event == 0u ? -1 : pmu->leader_fd);
            if (file_descriptor < 0) {
                const int failure_errno = errno;
                campp_optimization_pmu_release(pmu);
                pmu->sample_capacity = sample_capacity;
                pmu->samples = (uint64_t *)calloc(
                    sample_count, sizeof(*pmu->samples));
                if (pmu->samples == NULL) return 1;
                pmu->unavailable_errno = failure_errno;
                return 0;
            }
            pmu->file_descriptors[event] = file_descriptor;
            if (event == 0u) pmu->leader_fd = file_descriptor;
        }
        pmu->available = true;
    }
#else
    pmu->unavailable_errno = ENOSYS;
#endif
    return 0;
}

void campp_optimization_pmu_release(CamppOptimizationPmu *pmu)
{
    uint32_t event;

    if (pmu == NULL) {
        return;
    }
    if (pmu->samples == NULL && pmu->sample_capacity == 0u &&
        !pmu->available) {
        memset(pmu, 0, sizeof(*pmu));
        campp_optimization_pmu_initialize_fds(pmu);
        return;
    }
#if defined(__linux__)
    for (event = 0u; event < CAMPP_OPT_PMU_EVENT_COUNT; ++event) {
        if (pmu->file_descriptors[event] >= 0) {
            close(pmu->file_descriptors[event]);
        }
    }
#else
    (void)event;
#endif
    free(pmu->samples);
    memset(pmu, 0, sizeof(*pmu));
    campp_optimization_pmu_initialize_fds(pmu);
}

int campp_optimization_pmu_begin(CamppOptimizationPmu *pmu)
{
    if (pmu == NULL) return 1;
    if (!pmu->available) return 0;
#if defined(__linux__)
    if (ioctl(pmu->leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0 ||
        ioctl(pmu->leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        campp_optimization_pmu_mark_unavailable(pmu, errno);
        return 0;
    }
#endif
    return 0;
}

int campp_optimization_pmu_end(
    CamppOptimizationPmu *pmu, uint32_t iteration)
{
    if (pmu == NULL || iteration >= pmu->sample_capacity) return 1;
    if (!pmu->available) return 0;
#if defined(__linux__)
    {
        struct CamppPmuGroupRead {
            uint64_t count;
            uint64_t values[CAMPP_OPT_PMU_EVENT_COUNT];
        } result;
        uint32_t event;
        ssize_t bytes;

        if (ioctl(
                pmu->leader_fd, PERF_EVENT_IOC_DISABLE,
                PERF_IOC_FLAG_GROUP) != 0) {
            campp_optimization_pmu_mark_unavailable(pmu, errno);
            return 0;
        }
        memset(&result, 0, sizeof(result));
        bytes = read(pmu->leader_fd, &result, sizeof(result));
        if (bytes != (ssize_t)sizeof(result) ||
            result.count != CAMPP_OPT_PMU_EVENT_COUNT) {
            campp_optimization_pmu_mark_unavailable(
                pmu, bytes < 0 ? errno : EIO);
            return 0;
        }
        for (event = 0u; event < CAMPP_OPT_PMU_EVENT_COUNT; ++event) {
            pmu->samples[(size_t)event * pmu->sample_capacity + iteration] =
                result.values[event];
        }
    }
#endif
    return 0;
}

const char *campp_optimization_pmu_event_name(
    CamppOptimizationPmuEvent event)
{
    static const char *const names[CAMPP_OPT_PMU_EVENT_COUNT] = {
        "cpu_cycles",
        "instructions",
        "cache_references",
        "cache_misses",
        "branches",
        "branch_misses"
    };
    return event >= 0 && event < CAMPP_OPT_PMU_EVENT_COUNT
        ? names[event] : "invalid";
}

const uint64_t *campp_optimization_pmu_samples(
    const CamppOptimizationPmu *pmu, CamppOptimizationPmuEvent event)
{
    if (pmu == NULL || pmu->samples == NULL || event < 0 ||
        event >= CAMPP_OPT_PMU_EVENT_COUNT) {
        return NULL;
    }
    return pmu->samples + (size_t)event * pmu->sample_capacity;
}
