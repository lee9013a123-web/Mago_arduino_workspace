/* Python exporter가 plan에 기록한 정적 offset을 실제 RAM 주소로 연결한다. */

#include "tensor_arena.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int campp_arena_owns(uint8_t storage_type)
{
    return storage_type == CAMPP_TENSOR_STORAGE_ACTIVATION ||
           storage_type == CAMPP_TENSOR_STORAGE_OUTPUT;
}

static int campp_u64_add_overflows(uint64_t left, uint64_t right)
{
    return left > UINT64_MAX - right;
}

static int campp_ranges_overlap(
    uint64_t left_begin, uint64_t left_end,
    uint64_t right_begin, uint64_t right_end)
{
    return left_begin < right_end && right_begin < left_end;
}

static int campp_lifetimes_overlap(
    uint32_t left_first, uint32_t left_last,
    uint32_t right_first, uint32_t right_last)
{
    return left_first <= right_last && right_first <= left_last;
}

CamppStatus campp_tensor_arena_model_uses_arena(
    const CamppRuntimeModel *model, bool *out_uses_arena)
{
    uint32_t tensor_id;
    bool found_arena = false;
    bool found_reference = false;

    if (model == NULL || out_uses_arena == NULL || model->tensors == NULL ||
        model->tensor_count == 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    for (tensor_id = 0u; tensor_id < model->tensor_count; ++tensor_id) {
        const CamppTensorDescriptor *descriptor = &model->tensors[tensor_id];
        bool dense;

        if (!campp_arena_owns(descriptor->storage_type)) {
            continue;
        }
        dense = (descriptor->flags & CAMPP_TENSOR_FLAG_DENSE_SLAB) != 0u;
        if (dense && descriptor->data_offset != CAMPP_INVALID_DATA_OFFSET) {
            found_arena = true;
        } else if (!dense &&
                   descriptor->data_offset == CAMPP_INVALID_DATA_OFFSET) {
            found_reference = true;
        } else {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }

    if (!found_arena && !found_reference) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (found_arena && found_reference) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    *out_uses_arena = found_arena;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_tensor_arena_validate_layout(
    const CamppRuntimeModel *model, size_t *out_arena_size)
{
    uint64_t maximum_end = 0u;
    uint32_t left_id;
    bool uses_arena;
    CamppStatus status;

    if (out_arena_size == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_tensor_arena_model_uses_arena(model, &uses_arena);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (!uses_arena) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    for (left_id = 0u; left_id < model->tensor_count; ++left_id) {
        const CamppTensorDescriptor *left = &model->tensors[left_id];
        uint64_t left_end;
        uint32_t right_id;

        if (!campp_arena_owns(left->storage_type)) {
            if ((left->flags & CAMPP_TENSOR_FLAG_DENSE_SLAB) != 0u) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            continue;
        }
        if (left->tensor_id != left_id || left->storage_span_bytes == 0u ||
            left->data_offset == CAMPP_INVALID_DATA_OFFSET ||
            (left->flags & CAMPP_TENSOR_FLAG_DENSE_SLAB) == 0u ||
            (left->data_offset & (CAMPP_TENSOR_ARENA_ALIGNMENT - 1u)) != 0u ||
            left->first_use == CAMPP_INVALID_OPERATOR_INDEX ||
            left->last_use == CAMPP_INVALID_OPERATOR_INDEX ||
            left->first_use > left->last_use ||
            left->last_use >= model->operator_count ||
            campp_u64_add_overflows(
                left->data_offset, left->storage_span_bytes)) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        left_end = left->data_offset + left->storage_span_bytes;
        if (left_end > maximum_end) {
            maximum_end = left_end;
        }

        for (right_id = left_id + 1u; right_id < model->tensor_count;
             ++right_id) {
            const CamppTensorDescriptor *right = &model->tensors[right_id];
            uint64_t right_end;

            if (!campp_arena_owns(right->storage_type)) {
                continue;
            }
            if (right->data_offset == CAMPP_INVALID_DATA_OFFSET ||
                campp_u64_add_overflows(
                    right->data_offset, right->storage_span_bytes)) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            right_end = right->data_offset + right->storage_span_bytes;
            if (campp_ranges_overlap(
                    left->data_offset, left_end,
                    right->data_offset, right_end) &&
                campp_lifetimes_overlap(
                    left->first_use, left->last_use,
                    right->first_use, right->last_use)) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
        }
    }

    if (maximum_end == 0u) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (maximum_end >
        UINT64_MAX - (uint64_t)(CAMPP_TENSOR_ARENA_ALIGNMENT - 1u)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    maximum_end =
        (maximum_end + CAMPP_TENSOR_ARENA_ALIGNMENT - 1u) &
        ~(uint64_t)(CAMPP_TENSOR_ARENA_ALIGNMENT - 1u);
    if (maximum_end > (uint64_t)SIZE_MAX) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    *out_arena_size = (size_t)maximum_end;
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_tensor_arena_allocate(
    size_t arena_size, void **out_allocation, uint8_t **out_arena)
{
    const size_t overhead =
        2u * CAMPP_TENSOR_ARENA_GUARD_BYTES +
        CAMPP_TENSOR_ARENA_ALIGNMENT - 1u;
    uint8_t *allocation;
    uintptr_t candidate;
    uintptr_t aligned;
    uint8_t *arena;

    if (out_allocation == NULL || out_arena == NULL || arena_size == 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (arena_size > SIZE_MAX - overhead) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    allocation = (uint8_t *)malloc(arena_size + overhead);
    if (allocation == NULL) {
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }

    candidate = (uintptr_t)(allocation + CAMPP_TENSOR_ARENA_GUARD_BYTES);
    aligned = (candidate + CAMPP_TENSOR_ARENA_ALIGNMENT - 1u) &
              ~(uintptr_t)(CAMPP_TENSOR_ARENA_ALIGNMENT - 1u);
    arena = (uint8_t *)aligned;
    memset(
        arena - CAMPP_TENSOR_ARENA_GUARD_BYTES,
        CAMPP_TENSOR_ARENA_GUARD_PATTERN,
        CAMPP_TENSOR_ARENA_GUARD_BYTES);
    memset(arena, 0, arena_size);
    memset(
        arena + arena_size, CAMPP_TENSOR_ARENA_GUARD_PATTERN,
        CAMPP_TENSOR_ARENA_GUARD_BYTES);

    *out_allocation = allocation;
    *out_arena = arena;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_tensor_arena_create(
    const CamppRuntimeModel *model, CamppActivationStorage *storage)
{
    CamppActivationStorage staging;
    size_t arena_size;
    uint32_t tensor_id;
    CamppStatus status;

    if (model == NULL || storage == NULL || model->tensors == NULL ||
        model->tensor_count == 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    memset(&staging, 0, sizeof(staging));
    staging.mode = CAMPP_ACTIVATION_STORAGE_ARENA;

    status = campp_tensor_arena_validate_layout(model, &arena_size);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if ((size_t)model->tensor_count > SIZE_MAX / sizeof(void *) ||
        (size_t)model->tensor_count > SIZE_MAX / sizeof(size_t)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    staging.buffers =
        (void **)calloc((size_t)model->tensor_count, sizeof(void *));
    staging.buffer_sizes =
        (size_t *)calloc((size_t)model->tensor_count, sizeof(size_t));
    if (staging.buffers == NULL || staging.buffer_sizes == NULL) {
        campp_tensor_arena_release(&staging);
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    staging.buffer_count = model->tensor_count;

    status = campp_tensor_arena_allocate(
        arena_size, &staging.arena_allocation, &staging.arena_base);
    if (status != CAMPP_STATUS_OK) {
        campp_tensor_arena_release(&staging);
        return status;
    }
    staging.arena_size = arena_size;
    staging.total_bytes = arena_size;

    for (tensor_id = 0u; tensor_id < model->tensor_count; ++tensor_id) {
        const CamppTensorDescriptor *descriptor = &model->tensors[tensor_id];
        if (!campp_arena_owns(descriptor->storage_type)) {
            continue;
        }
        staging.buffers[tensor_id] =
            staging.arena_base + (size_t)descriptor->data_offset;
        staging.buffer_sizes[tensor_id] =
            (size_t)descriptor->storage_span_bytes;
    }

    *storage = staging;
    return CAMPP_STATUS_OK;
}

void campp_tensor_arena_release(CamppActivationStorage *storage)
{
    if (storage == NULL) {
        return;
    }
    free(storage->arena_allocation);
    free(storage->buffers);
    free(storage->buffer_sizes);
    memset(storage, 0, sizeof(*storage));
}

CamppStatus campp_tensor_arena_buffer(
    const CamppActivationStorage *storage, uint32_t tensor_id,
    void **out_buffer, size_t *out_size)
{
    if (storage == NULL || out_buffer == NULL || out_size == NULL ||
        storage->mode != CAMPP_ACTIVATION_STORAGE_ARENA ||
        storage->arena_base == NULL || storage->buffers == NULL ||
        storage->buffer_sizes == NULL || tensor_id >= storage->buffer_count) {
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

CamppStatus campp_tensor_arena_check_guards(
    const CamppActivationStorage *storage)
{
    const uint8_t *prefix;
    const uint8_t *suffix;
    size_t index;

    if (storage == NULL ||
        storage->mode != CAMPP_ACTIVATION_STORAGE_ARENA ||
        storage->arena_base == NULL || storage->arena_size == 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    prefix = storage->arena_base - CAMPP_TENSOR_ARENA_GUARD_BYTES;
    suffix = storage->arena_base + storage->arena_size;
    for (index = 0u; index < CAMPP_TENSOR_ARENA_GUARD_BYTES; ++index) {
        if (prefix[index] != CAMPP_TENSOR_ARENA_GUARD_PATTERN ||
            suffix[index] != CAMPP_TENSOR_ARENA_GUARD_PATTERN) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
    }
    return CAMPP_STATUS_OK;
}
