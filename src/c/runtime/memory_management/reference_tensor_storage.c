/*
 * Phase 3의 정확도 기준 메모리 방식.
 *
 * INPUT은 호출자가 연결하고 CONSTANT는 weights 메모리를 직접 가리킨다.
 * ACTIVATION과 OUTPUT만 Tensor별 독립 buffer를 context 생성 때 할당한다.
 * buffer 앞뒤에는 고정 canary를 두며 inference 중에는 이 파일의 어떤 함수도
 * malloc/free하지 않는다.
 */

#include "reference_tensor_storage.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "execution/tensor_registry.h"
#include "internal/kernel_registry.h"
#include "memory_management/tensor_arena.h"

static int campp_reference_storage_owns(uint8_t storage_type)
{
    return storage_type == CAMPP_TENSOR_STORAGE_ACTIVATION ||
           storage_type == CAMPP_TENSOR_STORAGE_OUTPUT;
}

static CamppStatus campp_reference_allocate_buffer(
    uint64_t span_bytes, void **out_payload, size_t *out_size)
{
    uint8_t *allocation;
    uint8_t *payload;
    size_t payload_size;
    size_t allocation_size;

    if (out_payload == NULL || out_size == NULL || span_bytes == 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (span_bytes > (uint64_t)SIZE_MAX) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    payload_size = (size_t)span_bytes;
    if (payload_size > SIZE_MAX - (2u * CAMPP_REFERENCE_GUARD_BYTES)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    allocation_size =
        payload_size + (2u * CAMPP_REFERENCE_GUARD_BYTES);
    allocation = (uint8_t *)malloc(allocation_size);
    if (allocation == NULL) {
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }

    payload = allocation + CAMPP_REFERENCE_GUARD_BYTES;
    memset(allocation, CAMPP_REFERENCE_GUARD_PATTERN,
           CAMPP_REFERENCE_GUARD_BYTES);
    memset(payload, 0, payload_size);
    memset(payload + payload_size, CAMPP_REFERENCE_GUARD_PATTERN,
           CAMPP_REFERENCE_GUARD_BYTES);

    *out_payload = payload;
    *out_size = payload_size;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_tensor_storage_create(
    const CamppRuntimeModel *model, CamppActivationStorage *storage)
{
    CamppActivationStorage staging;
    uint32_t tensor_id;
    CamppStatus status;

    if (model == NULL || storage == NULL || model->tensors == NULL ||
        model->tensor_count == 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    memset(&staging, 0, sizeof(staging));
    staging.mode = CAMPP_ACTIVATION_STORAGE_REFERENCE;
    if ((size_t)model->tensor_count > SIZE_MAX / sizeof(void *) ||
        (size_t)model->tensor_count > SIZE_MAX / sizeof(size_t)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    staging.buffers =
        (void **)calloc((size_t)model->tensor_count, sizeof(void *));
    staging.buffer_sizes =
        (size_t *)calloc((size_t)model->tensor_count, sizeof(size_t));
    if (staging.buffers == NULL || staging.buffer_sizes == NULL) {
        campp_reference_tensor_storage_release(&staging);
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    /* buffer_count는 소유 buffer 수가 아니라 buffers 배열의 길이다. */
    staging.buffer_count = model->tensor_count;

    for (tensor_id = 0u; tensor_id < model->tensor_count; ++tensor_id) {
        const CamppTensorDescriptor *descriptor = &model->tensors[tensor_id];
        size_t payload_size;

        if (descriptor->tensor_id != tensor_id) {
            campp_reference_tensor_storage_release(&staging);
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        if (!campp_reference_storage_owns(descriptor->storage_type)) {
            continue;
        }

        status = campp_reference_allocate_buffer(
            descriptor->storage_span_bytes, &staging.buffers[tensor_id],
            &payload_size);
        if (status != CAMPP_STATUS_OK) {
            campp_reference_tensor_storage_release(&staging);
            return status;
        }
        staging.buffer_sizes[tensor_id] = payload_size;
        if (payload_size > SIZE_MAX - staging.total_bytes) {
            campp_reference_tensor_storage_release(&staging);
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
        /* guard를 제외한 Tensor payload 합계만 센다. */
        staging.total_bytes += payload_size;
    }

    *storage = staging;
    return CAMPP_STATUS_OK;
}

void campp_reference_tensor_storage_release(CamppActivationStorage *storage)
{
    uint32_t tensor_id;

    if (storage == NULL) {
        return;
    }
    if (storage->mode == CAMPP_ACTIVATION_STORAGE_REFERENCE &&
        storage->buffers != NULL) {
        for (tensor_id = 0u; tensor_id < storage->buffer_count; ++tensor_id) {
            if (storage->buffers[tensor_id] != NULL) {
                uint8_t *payload =
                    (uint8_t *)storage->buffers[tensor_id];
                free(payload - CAMPP_REFERENCE_GUARD_BYTES);
            }
        }
    }
    free(storage->buffers);
    free(storage->buffer_sizes);
    memset(storage, 0, sizeof(*storage));
}

CamppStatus campp_reference_tensor_storage_buffer(
    const CamppActivationStorage *storage, uint32_t tensor_id,
    void **out_buffer, size_t *out_size)
{
    if (storage == NULL || out_buffer == NULL || out_size == NULL ||
        storage->mode != CAMPP_ACTIVATION_STORAGE_REFERENCE ||
        storage->buffers == NULL || storage->buffer_sizes == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (tensor_id >= storage->buffer_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (storage->buffers[tensor_id] == NULL ||
        storage->buffer_sizes[tensor_id] == 0u) {
        return CAMPP_STATUS_TENSOR_NOT_BOUND;
    }
    *out_buffer = storage->buffers[tensor_id];
    *out_size = storage->buffer_sizes[tensor_id];
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_tensor_storage_check_guard(
    const CamppActivationStorage *storage, uint32_t tensor_id)
{
    const uint8_t *payload;
    const uint8_t *prefix;
    const uint8_t *suffix;
    size_t index;

    if (storage == NULL || storage->buffers == NULL ||
        storage->mode != CAMPP_ACTIVATION_STORAGE_REFERENCE ||
        storage->buffer_sizes == NULL || tensor_id >= storage->buffer_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (storage->buffers[tensor_id] == NULL) {
        /* INPUT과 CONSTANT는 Reference storage의 guard 대상이 아니다. */
        return CAMPP_STATUS_OK;
    }

    payload = (const uint8_t *)storage->buffers[tensor_id];
    prefix = payload - CAMPP_REFERENCE_GUARD_BYTES;
    suffix = payload + storage->buffer_sizes[tensor_id];
    for (index = 0u; index < CAMPP_REFERENCE_GUARD_BYTES; ++index) {
        if (prefix[index] != CAMPP_REFERENCE_GUARD_PATTERN ||
            suffix[index] != CAMPP_REFERENCE_GUARD_PATTERN) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_tensor_storage_check_all_guards(
    const CamppActivationStorage *storage)
{
    uint32_t tensor_id;
    CamppStatus status;

    if (storage == NULL || storage->buffers == NULL ||
        storage->buffer_sizes == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    for (tensor_id = 0u; tensor_id < storage->buffer_count; ++tensor_id) {
        status = campp_reference_tensor_storage_check_guard(storage, tensor_id);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    return CAMPP_STATUS_OK;
}

/* ------------------------------------------------------------------------- */
/* RuntimeContext 생명주기                                                    */
/* ------------------------------------------------------------------------- */

CamppStatus campp_runtime_context_create(
    const CamppRuntimeModel *model, const CamppKernelRegistry *registry,
    CamppRuntimeContext *context)
{
    CamppRuntimeContext staging;
    CamppStatus status;
    uint32_t failing_operator_id;
    uint16_t missing_opcode;
    uint32_t operator_id;
    size_t maximum_scratch = 0u;
    bool uses_arena;

    if (model == NULL || registry == NULL || context == NULL ||
        model->tensors == NULL || model->tensor_count == 0u ||
        (model->operator_count != 0u && model->operators == NULL)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    memset(&staging, 0, sizeof(staging));
    staging.model = model;
    staging.registry = registry;
    staging.backend_id = registry->backend_id;
    staging.diagnostics.current_operator_id = CAMPP_INVALID_OPERATOR_INDEX;
    staging.last_status = CAMPP_STATUS_OK;

    status = campp_kernel_registry_validate(registry);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_kernel_registry_covers_model(
        registry, model, &failing_operator_id, &missing_opcode);
    if (status != CAMPP_STATUS_OK) {
        (void)failing_operator_id;
        (void)missing_opcode;
        return status;
    }

    status = campp_tensor_arena_model_uses_arena(model, &uses_arena);
    if (status != CAMPP_STATUS_OK) {
        goto failed;
    }
    if (uses_arena) {
        status = campp_tensor_arena_create(model, &staging.activations);
    } else {
        status = campp_reference_tensor_storage_create(
            model, &staging.activations);
    }
    if (status != CAMPP_STATUS_OK) {
        goto failed;
    }
    status = campp_tensor_registry_create(
        model, &staging.activations, &staging.tensors,
        &staging.tensor_count);
    if (status != CAMPP_STATUS_OK) {
        goto failed;
    }

    if (model->operator_count != 0u) {
        if ((size_t)model->operator_count >
            SIZE_MAX / sizeof(*staging.resolved_kernels)) {
            status = CAMPP_STATUS_BUFFER_OVERFLOW;
            goto failed;
        }
        staging.resolved_kernels = (const CamppKernelEntry **)calloc(
            (size_t)model->operator_count,
            sizeof(*staging.resolved_kernels));
        if (staging.resolved_kernels == NULL) {
            status = CAMPP_STATUS_OUT_OF_MEMORY;
            goto failed;
        }
        staging.resolved_kernel_count = model->operator_count;
    }

    for (operator_id = 0u; operator_id < model->operator_count; ++operator_id) {
        const CamppOperatorDescriptor *operator_descriptor =
            &model->operators[operator_id];
        const CamppKernelEntry *entry;
        size_t required_scratch = 0u;

        status = campp_kernel_registry_lookup(
            registry, operator_descriptor->opcode,
            operator_descriptor->kernel_id, &entry);
        if (status != CAMPP_STATUS_OK) {
            goto failed;
        }
        staging.resolved_kernels[operator_id] = entry;
        if (entry->scratch_bytes != NULL) {
            status = entry->scratch_bytes(
                model, operator_descriptor, &required_scratch);
            if (status != CAMPP_STATUS_OK) {
                goto failed;
            }
            if (required_scratch > maximum_scratch) {
                maximum_scratch = required_scratch;
            }
        }
    }

    if (maximum_scratch != 0u) {
        staging.scratch = malloc(maximum_scratch);
        if (staging.scratch == NULL) {
            status = CAMPP_STATUS_OUT_OF_MEMORY;
            goto failed;
        }
        memset(staging.scratch, 0, maximum_scratch);
        staging.scratch_size = maximum_scratch;
    }

    *context = staging;
    return CAMPP_STATUS_OK;

failed:
    campp_runtime_context_release(&staging);
    return status;
}

void campp_runtime_context_release(CamppRuntimeContext *context)
{
    if (context == NULL) {
        return;
    }
    free(context->scratch);
    free(context->resolved_kernels);
    campp_tensor_registry_release(&context->tensors, &context->tensor_count);
    if (context->activations.mode == CAMPP_ACTIVATION_STORAGE_ARENA) {
        campp_tensor_arena_release(&context->activations);
    } else {
        campp_reference_tensor_storage_release(&context->activations);
    }
    memset(context, 0, sizeof(*context));
}

CamppStatus campp_runtime_context_reset(CamppRuntimeContext *context)
{
    if (context == NULL || context->model == NULL || context->tensors == NULL ||
        context->tensor_count != context->model->tensor_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    /* 버퍼와 입력 binding은 유지한다. 추론별 관찰 상태만 되돌린다. */
    context->diagnostics.current_operator_id = CAMPP_INVALID_OPERATOR_INDEX;
    context->diagnostics.executed_operator_count = 0u;
    context->last_status = CAMPP_STATUS_OK;
    return CAMPP_STATUS_OK;
}
