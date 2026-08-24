#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "campp_runtime/tensor_descriptor.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"
#include "memory_management/tensor_arena.h"
#include "model_loading/compiled_model_validator.h"

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
            fprintf(stderr,                                                     \
                    "STATUS failed at line %d: got %s, expected %s\n",       \
                    __LINE__, campp_status_name(actual_status),                 \
                    campp_status_name(expected));                               \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static void init_arena_tensor(
    CamppTensorDescriptor *descriptor, uint32_t tensor_id,
    uint8_t storage_type, uint64_t offset,
    uint32_t first_use, uint32_t last_use)
{
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->tensor_id = tensor_id;
    descriptor->dtype = CAMPP_DTYPE_FLOAT32;
    descriptor->rank = 1u;
    descriptor->storage_type = storage_type;
    descriptor->flags =
        CAMPP_TENSOR_FLAG_CONTIGUOUS | CAMPP_TENSOR_FLAG_DENSE_SLAB;
    descriptor->dimensions[0] = 16u;
    descriptor->dimensions[1] = 1u;
    descriptor->dimensions[2] = 1u;
    descriptor->dimensions[3] = 1u;
    descriptor->byte_strides[0] = 4u;
    descriptor->data_offset = offset;
    descriptor->logical_byte_size = 64u;
    descriptor->storage_span_bytes = 64u;
    descriptor->alias_of_tensor_id = CAMPP_INVALID_TENSOR_ID;
    descriptor->quantization_index = CAMPP_INVALID_QUANTIZATION_INDEX;
    descriptor->first_use = first_use;
    descriptor->last_use = last_use;
}

static int test_synthetic_arena(void)
{
    CamppTensorDescriptor descriptors[3];
    CamppRuntimeModel model;
    CamppActivationStorage storage;
    bool uses_arena = false;
    size_t arena_size = 0u;
    void *first;
    void *second;
    void *reused;
    size_t buffer_size;

    init_arena_tensor(
        &descriptors[0], 0u, CAMPP_TENSOR_STORAGE_ACTIVATION,
        0u, 0u, 0u);
    init_arena_tensor(
        &descriptors[1], 1u, CAMPP_TENSOR_STORAGE_ACTIVATION,
        64u, 0u, 1u);
    init_arena_tensor(
        &descriptors[2], 2u, CAMPP_TENSOR_STORAGE_OUTPUT,
        0u, 1u, 2u);

    memset(&model, 0, sizeof(model));
    model.tensors = descriptors;
    model.tensor_count = 3u;
    model.operator_count = 3u;

    CHECK_STATUS(
        campp_tensor_arena_model_uses_arena(&model, &uses_arena),
        CAMPP_STATUS_OK);
    CHECK_TRUE(uses_arena);
    CHECK_STATUS(
        campp_tensor_arena_validate_layout(&model, &arena_size),
        CAMPP_STATUS_OK);
    CHECK_TRUE(arena_size == 128u);

    memset(&storage, 0, sizeof(storage));
    CHECK_STATUS(
        campp_tensor_arena_create(&model, &storage), CAMPP_STATUS_OK);
    CHECK_TRUE(storage.mode == CAMPP_ACTIVATION_STORAGE_ARENA);
    CHECK_TRUE(storage.arena_size == 128u);
    CHECK_TRUE(storage.total_bytes == 128u);
    CHECK_TRUE(((uintptr_t)storage.arena_base & 63u) == 0u);
    CHECK_STATUS(
        campp_tensor_arena_buffer(
            &storage, 0u, &first, &buffer_size),
        CAMPP_STATUS_OK);
    CHECK_TRUE(buffer_size == 64u);
    CHECK_STATUS(
        campp_tensor_arena_buffer(
            &storage, 1u, &second, &buffer_size),
        CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_tensor_arena_buffer(
            &storage, 2u, &reused, &buffer_size),
        CAMPP_STATUS_OK);
    CHECK_TRUE(first == reused);
    CHECK_TRUE((uint8_t *)second == (uint8_t *)first + 64u);
    CHECK_STATUS(
        campp_tensor_arena_check_guards(&storage), CAMPP_STATUS_OK);

    storage.arena_base[storage.arena_size] = 0u;
    CHECK_STATUS(
        campp_tensor_arena_check_guards(&storage),
        CAMPP_STATUS_BUFFER_OVERFLOW);
    storage.arena_base[storage.arena_size] =
        CAMPP_TENSOR_ARENA_GUARD_PATTERN;
    campp_tensor_arena_release(&storage);
    CHECK_TRUE(storage.arena_base == NULL);
    CHECK_TRUE(storage.buffers == NULL);

    /* offset을 공유하는 Tensor의 수명이 겹치면 손상된 plan이다. */
    descriptors[2].first_use = 0u;
    CHECK_STATUS(
        campp_tensor_arena_validate_layout(&model, &arena_size),
        CAMPP_STATUS_CORRUPT_PLAN);
    descriptors[2].first_use = 1u;

    /* Arena와 Reference descriptor를 한 plan에서 섞지 않는다. */
    descriptors[2].flags = CAMPP_TENSOR_FLAG_CONTIGUOUS;
    descriptors[2].data_offset = CAMPP_INVALID_DATA_OFFSET;
    CHECK_STATUS(
        campp_tensor_arena_model_uses_arena(&model, &uses_arena),
        CAMPP_STATUS_CORRUPT_PLAN);
    return 0;
}

static int test_compiled_arena_plan(
    const char *plan_path, const char *weights_path)
{
    CamppRuntimeModel model;
    CamppActivationStorage storage;
    size_t arena_size;
    uint32_t tensor_id;
    uint32_t reused_pairs = 0u;

    memset(&model, 0, sizeof(model));
    CHECK_STATUS(
        campp_runtime_model_load(plan_path, weights_path, &model),
        CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_tensor_arena_validate_layout(&model, &arena_size),
        CAMPP_STATUS_OK);
    memset(&storage, 0, sizeof(storage));
    CHECK_STATUS(
        campp_tensor_arena_create(&model, &storage), CAMPP_STATUS_OK);
    CHECK_TRUE(storage.arena_size == arena_size);

    for (tensor_id = 0u; tensor_id < model.tensor_count; ++tensor_id) {
        const CamppTensorDescriptor *descriptor = &model.tensors[tensor_id];
        if (descriptor->storage_type != CAMPP_TENSOR_STORAGE_ACTIVATION &&
            descriptor->storage_type != CAMPP_TENSOR_STORAGE_OUTPUT) {
            continue;
        }
        CHECK_TRUE(
            storage.buffers[tensor_id] ==
            storage.arena_base + (size_t)descriptor->data_offset);
        if (tensor_id != 0u) {
            uint32_t previous;
            for (previous = 0u; previous < tensor_id; ++previous) {
                if (storage.buffers[previous] == storage.buffers[tensor_id]) {
                    reused_pairs += 1u;
                    break;
                }
            }
        }
    }
    CHECK_TRUE(reused_pairs != 0u);
    printf(
        "arena plan: bucket=%" PRIu32 " tensors=%" PRIu32
        " arena=%zu bytes reused_offsets=%" PRIu32 "\n",
        model.bucket_frames, model.tensor_count, arena_size, reused_pairs);

    campp_tensor_arena_release(&storage);
    campp_runtime_model_release(&model);
    return 0;
}

int main(int argc, char **argv)
{
    if (test_synthetic_arena() != 0) {
        return 1;
    }
    if (argc == 3 && test_compiled_arena_plan(argv[1], argv[2]) != 0) {
        return 1;
    }
    if (argc != 1 && argc != 3) {
        fprintf(stderr, "usage: %s [arena_plan.bin weights.bin]\n", argv[0]);
        return 2;
    }
    puts("tensor arena tests: PASS");
    return 0;
}
