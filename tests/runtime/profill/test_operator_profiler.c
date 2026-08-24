#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "campp_profill/operator_profiler.h"

#define CHECK_TRUE(condition)                                                   \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__,       \
                    #condition);                                                \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define CHECK_STATUS(expression, expected)                                     \
    do {                                                                        \
        CamppStatus actual_status = (expression);                               \
        if (actual_status != (expected)) {                                      \
            fprintf(stderr, "STATUS failed at line %d: got %s\n",            \
                    __LINE__, campp_status_name(actual_status));                \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main(void)
{
    CamppOperatorProfiler profiler;
    uint64_t started_ns;
    const uint64_t *operator_zero_samples;

    memset(&profiler, 0, sizeof(profiler));
    CHECK_STATUS(
        campp_operator_profiler_create(2u, 2u, &profiler),
        CAMPP_STATUS_OK);
    CHECK_TRUE(!campp_operator_profiler_is_enabled(&profiler));
    CHECK_STATUS(
        campp_operator_profiler_begin(&profiler, &started_ns),
        CAMPP_STATUS_INVALID_ARGUMENT);

    campp_operator_profiler_set_enabled(&profiler, true);
    CHECK_STATUS(
        campp_operator_profiler_begin(&profiler, &started_ns),
        CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_operator_profiler_end(&profiler, 0u, started_ns),
        CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_operator_profiler_begin(&profiler, &started_ns),
        CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_operator_profiler_end(&profiler, 0u, started_ns),
        CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_operator_profiler_begin(&profiler, &started_ns),
        CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_operator_profiler_end(&profiler, 1u, started_ns),
        CAMPP_STATUS_OK);

    CHECK_TRUE(campp_operator_profiler_call_count(&profiler, 0u) == 2u);
    CHECK_TRUE(campp_operator_profiler_call_count(&profiler, 1u) == 1u);
    operator_zero_samples = campp_operator_profiler_samples(&profiler, 0u);
    CHECK_TRUE(operator_zero_samples != NULL);

    CHECK_STATUS(
        campp_operator_profiler_begin(&profiler, &started_ns),
        CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_operator_profiler_end(&profiler, 0u, started_ns),
        CAMPP_STATUS_BUFFER_OVERFLOW);

    campp_operator_profiler_release(&profiler);
    CHECK_TRUE(profiler.samples_ns == NULL);
    CHECK_TRUE(profiler.call_counts == NULL);
    puts("operator profiler tests: PASS");
    return 0;
}
