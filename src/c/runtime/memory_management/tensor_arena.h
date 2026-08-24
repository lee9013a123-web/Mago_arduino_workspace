#ifndef CAMPP_RUNTIME_MEMORY_MANAGEMENT_TENSOR_ARENA_H
#define CAMPP_RUNTIME_MEMORY_MANAGEMENT_TENSOR_ARENA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "campp_runtime/status_code.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"

#define CAMPP_TENSOR_ARENA_ALIGNMENT 64u
#define CAMPP_TENSOR_ARENA_GUARD_BYTES 64u
#define CAMPP_TENSOR_ARENA_GUARD_PATTERN 0xA5u

/* ACTIVATION/OUTPUT descriptor가 Arena 형식인지 판별하고 혼합 plan을 거부한다. */
CamppStatus campp_tensor_arena_model_uses_arena(
    const CamppRuntimeModel *model, bool *out_uses_arena);

/* offset, 범위, 정렬, lifetime 충돌을 검사하고 필요한 slab 크기를 계산한다. */
CamppStatus campp_tensor_arena_validate_layout(
    const CamppRuntimeModel *model, size_t *out_arena_size);

/* 검증된 data_offset대로 하나의 slab을 할당하고 Tensor ID별 pointer를 연결한다. */
CamppStatus campp_tensor_arena_create(
    const CamppRuntimeModel *model, CamppActivationStorage *storage);

void campp_tensor_arena_release(CamppActivationStorage *storage);

CamppStatus campp_tensor_arena_buffer(
    const CamppActivationStorage *storage, uint32_t tensor_id,
    void **out_buffer, size_t *out_size);

CamppStatus campp_tensor_arena_check_guards(
    const CamppActivationStorage *storage);

#endif /* CAMPP_RUNTIME_MEMORY_MANAGEMENT_TENSOR_ARENA_H */
