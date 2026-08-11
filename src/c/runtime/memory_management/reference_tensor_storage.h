#ifndef CAMPP_RUNTIME_MEMORY_MANAGEMENT_REFERENCE_TENSOR_STORAGE_H
#define CAMPP_RUNTIME_MEMORY_MANAGEMENT_REFERENCE_TENSOR_STORAGE_H

#include <stddef.h>
#include <stdint.h>

#include "campp_runtime/status_code.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"

/*
 * Reference buffer 하나의 앞뒤에 붙이는 canary 영역이다. payload 시작 주소는
 * malloc이 돌려준 주소에서 64바이트 뒤이므로 malloc의 기본 정렬을 유지한다.
 */
#define CAMPP_REFERENCE_GUARD_BYTES 64u
#define CAMPP_REFERENCE_GUARD_PATTERN 0xA5u

/* ACTIVATION과 OUTPUT마다 독립 buffer를 한 번 할당한다. */
CamppStatus campp_reference_tensor_storage_create(
    const CamppRuntimeModel *model, CamppActivationStorage *storage);

/* create가 소유한 payload와 bookkeeping 배열을 모두 해제한다. */
void campp_reference_tensor_storage_release(CamppActivationStorage *storage);

/* tensor_id에 대응하는 소유 buffer를 조회한다. INPUT/CONSTANT면 NOT_BOUND다. */
CamppStatus campp_reference_tensor_storage_buffer(
    const CamppActivationStorage *storage, uint32_t tensor_id,
    void **out_buffer, size_t *out_size);

/* Tensor 하나 또는 storage 전체의 앞뒤 guard를 검사한다. */
CamppStatus campp_reference_tensor_storage_check_guard(
    const CamppActivationStorage *storage, uint32_t tensor_id);

CamppStatus campp_reference_tensor_storage_check_all_guards(
    const CamppActivationStorage *storage);

#endif /* CAMPP_RUNTIME_MEMORY_MANAGEMENT_REFERENCE_TENSOR_STORAGE_H */
