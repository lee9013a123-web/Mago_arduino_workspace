#ifndef CAMPP_PROFILL_OPERATOR_PROFILER_H
#define CAMPP_PROFILL_OPERATOR_PROFILER_H

/*
 * Operator kernel의 exclusive wall time만 수집한다.
 *
 * 모델 메타데이터는 execution plan에 이미 있으므로 이 모듈은 Tensor나
 * Operator descriptor를 복제하지 않는다. 샘플은 operator-major 순서로
 * 미리 할당하며 측정 구간에서는 allocation과 I/O를 수행하지 않는다.
 */

#include <stdbool.h>
#include <stdint.h>

#include "campp_runtime/status_code.h"

typedef struct CamppOperatorProfiler {
    uint32_t operator_count;
    uint32_t sample_capacity_per_operator;
    uint32_t *call_counts;
    uint64_t *samples_ns;
    bool enabled;
} CamppOperatorProfiler;

CamppStatus campp_operator_profiler_create(
    uint32_t operator_count, uint32_t sample_capacity_per_operator,
    CamppOperatorProfiler *profiler);

void campp_operator_profiler_release(CamppOperatorProfiler *profiler);

void campp_operator_profiler_set_enabled(
    CamppOperatorProfiler *profiler, bool enabled);

bool campp_operator_profiler_is_enabled(
    const CamppOperatorProfiler *profiler);

CamppStatus campp_operator_profiler_clock_now_ns(uint64_t *out_time_ns);

CamppStatus campp_operator_profiler_begin(
    const CamppOperatorProfiler *profiler, uint64_t *out_start_ns);

CamppStatus campp_operator_profiler_end(
    CamppOperatorProfiler *profiler, uint32_t operator_id,
    uint64_t start_ns);

uint32_t campp_operator_profiler_call_count(
    const CamppOperatorProfiler *profiler, uint32_t operator_id);

const uint64_t *campp_operator_profiler_samples(
    const CamppOperatorProfiler *profiler, uint32_t operator_id);

const char *campp_operator_profiler_clock_name(void);

#endif /* CAMPP_PROFILL_OPERATOR_PROFILER_H */
