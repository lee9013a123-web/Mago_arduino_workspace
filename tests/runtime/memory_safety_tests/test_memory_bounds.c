#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "campp_runtime/operator_descriptor.h"
#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"
#include "memory_management/memory_bounds_checker.h"
#include "memory_management/reference_tensor_storage.h"

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

static CamppStatus test_kernel_run(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    (void)model;
    (void)op;
    (void)inputs;
    (void)input_count;
    (void)outputs;
    (void)output_count;
    (void)scratch;
    (void)scratch_size;
    return CAMPP_STATUS_OK;
}

static void test_descriptor_init(
    CamppTensorDescriptor *descriptor, uint32_t tensor_id, uint8_t dtype,
    uint8_t storage_type, uint8_t rank, uint32_t dim0, uint32_t dim1,
    uint64_t byte_size)
{
    uint32_t dtype_size;

    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->tensor_id = tensor_id;
    descriptor->dtype = dtype;
    descriptor->rank = rank;
    descriptor->storage_type = storage_type;
    descriptor->flags = CAMPP_TENSOR_FLAG_CONTIGUOUS;
    if (storage_type == CAMPP_TENSOR_STORAGE_CONSTANT) {
        descriptor->flags = (uint8_t)(descriptor->flags |
            CAMPP_TENSOR_FLAG_READ_ONLY | CAMPP_TENSOR_FLAG_EXTERNAL);
    }
    descriptor->dimensions[0] = dim0;
    descriptor->dimensions[1] = rank > 1u ? dim1 : 1u;
    descriptor->dimensions[2] = 1u;
    descriptor->dimensions[3] = 1u;
    dtype_size = campp_dtype_byte_size(dtype);
    if (rank == 2u) {
        descriptor->byte_strides[0] = dim1 * dtype_size;
        descriptor->byte_strides[1] = dtype_size;
    } else {
        descriptor->byte_strides[0] = dtype_size;
    }
    descriptor->data_offset = CAMPP_INVALID_DATA_OFFSET;
    descriptor->logical_byte_size = byte_size;
    descriptor->storage_span_bytes = byte_size;
    descriptor->alias_of_tensor_id = CAMPP_INVALID_TENSOR_ID;
    descriptor->quantization_index = CAMPP_INVALID_QUANTIZATION_INDEX;
    descriptor->first_use = 0u;
    descriptor->last_use = 0u;
}

static int test_synthetic_context(void)
{
    uint8_t weights[8] = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
    CamppTensorDescriptor descriptors[4];
    CamppOperatorDescriptor op;
    CamppRuntimeModel model;
    CamppKernelEntry entry;
    CamppKernelRegistry registry;
    CamppRuntimeContext context;
    float input[2] = {1.0f, -2.0f};
    uint32_t input_dimensions[CAMPP_TENSOR_MAX_RANK] = {1u, 2u, 1u, 1u};
    uint32_t wrong_bucket[CAMPP_TENSOR_MAX_RANK] = {1u, 3u, 1u, 1u};
    uint32_t wrong_shape[CAMPP_TENSOR_MAX_RANK] = {2u, 2u, 1u, 1u};
    const CamppTensorView *output;
    CamppTensorView *view;
    void *stable_input_pointer;
    uint8_t *output_bytes;
    uint64_t saved_span;

    test_descriptor_init(
        &descriptors[0], 0u, CAMPP_DTYPE_FLOAT32,
        CAMPP_TENSOR_STORAGE_INPUT, 2u, 1u, 2u, 8u);
    test_descriptor_init(
        &descriptors[1], 1u, CAMPP_DTYPE_INT8,
        CAMPP_TENSOR_STORAGE_CONSTANT, 1u, 4u, 1u, 4u);
    descriptors[1].data_offset = 2u;
    test_descriptor_init(
        &descriptors[2], 2u, CAMPP_DTYPE_FLOAT32,
        CAMPP_TENSOR_STORAGE_ACTIVATION, 1u, 2u, 1u, 8u);
    test_descriptor_init(
        &descriptors[3], 3u, CAMPP_DTYPE_FLOAT32,
        CAMPP_TENSOR_STORAGE_OUTPUT, 1u, 2u, 1u, 8u);

    memset(&op, 0, sizeof(op));
    op.operator_id = 0u;
    op.opcode = CAMPP_OP_RELU;
    op.input_count = 1u;
    op.output_count = 1u;
    op.input_tensor_ids[0] = 0u;
    op.output_tensor_ids[0] = 3u;
    op.backend_id = CAMPP_BACKEND_AUTO;
    op.kernel_id = CAMPP_DEFAULT_KERNEL_ID;

    memset(&model, 0, sizeof(model));
    model.weights = weights;
    model.weights_size = sizeof(weights);
    model.bucket_frames = 2u;
    model.tensors = descriptors;
    model.tensor_count = 4u;
    model.operators = &op;
    model.operator_count = 1u;

    memset(&entry, 0, sizeof(entry));
    entry.opcode = CAMPP_OP_RELU;
    entry.kernel_id = CAMPP_DEFAULT_KERNEL_ID;
    entry.run = test_kernel_run;
    entry.name = "test_relu";
    memset(&registry, 0, sizeof(registry));
    registry.backend_id = CAMPP_BACKEND_CPU_REFERENCE;
    registry.name = "test";
    registry.entries = &entry;
    registry.entry_count = 1u;

    memset(&context, 0, sizeof(context));
    CHECK_STATUS(
        campp_runtime_context_create(&model, &registry, &context),
        CAMPP_STATUS_OK);
    CHECK_TRUE(context.tensor_count == 4u);
    CHECK_TRUE(context.activations.buffer_count == 4u);
    CHECK_TRUE(context.activations.total_bytes == 16u);
    CHECK_TRUE(context.activations.buffers[0] == NULL);
    CHECK_TRUE(context.activations.buffers[1] == NULL);
    CHECK_TRUE(context.activations.buffers[2] != NULL);
    CHECK_TRUE(context.activations.buffers[3] != NULL);
    CHECK_TRUE(context.tensors[0].data == NULL);
    CHECK_TRUE(context.tensors[1].data == weights + 2u);
    CHECK_TRUE(context.tensors[2].data == context.activations.buffers[2]);
    CHECK_TRUE(context.tensors[3].data == context.activations.buffers[3]);
    CHECK_STATUS(
        campp_memory_bounds_check_all(&context),
        CAMPP_STATUS_TENSOR_NOT_BOUND);

    CHECK_STATUS(
        campp_runtime_context_bind_input(
            &context, 0u, input, sizeof(input), CAMPP_DTYPE_FLOAT32, 2u,
            input_dimensions),
        CAMPP_STATUS_OK);
    stable_input_pointer = context.tensors[0].data;
    CHECK_STATUS(campp_memory_bounds_check_all(&context), CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_memory_bounds_check_operator(&context, &op), CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_runtime_context_output(&context, 3u, &output), CAMPP_STATUS_OK);
    CHECK_TRUE(output->data == context.activations.buffers[3]);

    CHECK_STATUS(
        campp_runtime_context_bind_input(
            &context, 0u, input, sizeof(input), CAMPP_DTYPE_FLOAT32, 2u,
            wrong_bucket),
        CAMPP_STATUS_BUCKET_MISMATCH);
    CHECK_STATUS(
        campp_runtime_context_bind_input(
            &context, 0u, input, sizeof(input), CAMPP_DTYPE_FLOAT32, 2u,
            wrong_shape),
        CAMPP_STATUS_SHAPE_MISMATCH);
    CHECK_STATUS(
        campp_runtime_context_bind_input(
            &context, 0u, input, 4u, CAMPP_DTYPE_FLOAT32, 2u,
            input_dimensions),
        CAMPP_STATUS_BUFFER_OVERFLOW);
    CHECK_STATUS(
        campp_runtime_context_bind_input(
            &context, 0u, input, sizeof(input), CAMPP_DTYPE_INT8, 2u,
            input_dimensions),
        CAMPP_STATUS_UNSUPPORTED_DTYPE);
    CHECK_TRUE(context.tensors[0].data == stable_input_pointer);

    CHECK_STATUS(
        campp_runtime_context_tensor(&context, 4u, &view),
        CAMPP_STATUS_INVALID_ARGUMENT);
    CHECK_STATUS(campp_runtime_context_reset(&context), CAMPP_STATUS_OK);
    CHECK_TRUE(context.tensors[0].data == stable_input_pointer);

    saved_span = context.tensors[3].storage_span_bytes;
    context.tensors[3].storage_span_bytes = 7u;
    CHECK_STATUS(
        campp_memory_bounds_check_tensor(&context, 3u),
        CAMPP_STATUS_BUFFER_OVERFLOW);
    context.tensors[3].storage_span_bytes = saved_span;

    output_bytes = (uint8_t *)context.activations.buffers[3];
    output_bytes[context.activations.buffer_sizes[3]] = 0u;
    CHECK_STATUS(
        campp_memory_bounds_check_guards(&context),
        CAMPP_STATUS_BUFFER_OVERFLOW);
    output_bytes[context.activations.buffer_sizes[3]] =
        CAMPP_REFERENCE_GUARD_PATTERN;
    CHECK_STATUS(campp_memory_bounds_check_guards(&context), CAMPP_STATUS_OK);

    campp_runtime_context_release(&context);
    CHECK_TRUE(context.model == NULL);
    CHECK_TRUE(context.tensors == NULL);
    CHECK_TRUE(context.activations.buffers == NULL);
    return 0;
}

static int test_compiled_plan(const char *plan_path, const char *weights_path)
{
    CamppRuntimeModel model;
    CamppRuntimeContext context;
    CamppKernelEntry entries[CAMPP_KERNEL_OPCODE_COUNT];
    CamppKernelRegistry registry;
    const CamppTensorDescriptor *input_descriptor;
    uint32_t input_tensor_id;
    void *input_data;
    uint32_t tensor_id;
    uint32_t operator_id;
    uint32_t owned_buffers = 0u;
    CamppStatus status;

    memset(&model, 0, sizeof(model));
    CHECK_STATUS(
        campp_runtime_model_load(plan_path, weights_path, &model),
        CAMPP_STATUS_OK);

    memset(entries, 0, sizeof(entries));
    for (tensor_id = 0u; tensor_id < CAMPP_KERNEL_OPCODE_COUNT; ++tensor_id) {
        entries[tensor_id].opcode = (uint16_t)(tensor_id + 1u);
        entries[tensor_id].kernel_id = CAMPP_DEFAULT_KERNEL_ID;
        entries[tensor_id].run = test_kernel_run;
        entries[tensor_id].name = "bounds_test_stub";
    }
    memset(&registry, 0, sizeof(registry));
    registry.backend_id = CAMPP_BACKEND_CPU_REFERENCE;
    registry.name = "bounds_test";
    registry.entries = entries;
    registry.entry_count = CAMPP_KERNEL_OPCODE_COUNT;

    memset(&context, 0, sizeof(context));
    CHECK_STATUS(
        campp_runtime_context_create(&model, &registry, &context),
        CAMPP_STATUS_OK);
    CHECK_TRUE(context.tensor_count == model.tensor_count);
    CHECK_TRUE(context.resolved_kernel_count == model.operator_count);

    CHECK_TRUE(model.input_count == 1u);
    input_tensor_id = model.input_tensor_ids[0];
    input_descriptor = &model.tensors[input_tensor_id];
    CHECK_TRUE(input_descriptor->storage_span_bytes <= (uint64_t)SIZE_MAX);
    input_data = malloc((size_t)input_descriptor->storage_span_bytes);
    CHECK_TRUE(input_data != NULL);
    memset(input_data, 0, (size_t)input_descriptor->storage_span_bytes);
    CHECK_STATUS(
        campp_runtime_context_bind_input(
            &context, input_tensor_id, input_data,
            (size_t)input_descriptor->storage_span_bytes,
            input_descriptor->dtype, input_descriptor->rank,
            input_descriptor->dimensions),
        CAMPP_STATUS_OK);

    CHECK_STATUS(campp_memory_bounds_check_all(&context), CAMPP_STATUS_OK);
    for (operator_id = 0u; operator_id < model.operator_count; ++operator_id) {
        status = campp_memory_bounds_check_operator(
            &context, &model.operators[operator_id]);
        if (status != CAMPP_STATUS_OK) {
            fprintf(stderr, "operator %" PRIu32 " bounds failed: %s\n",
                    operator_id, campp_status_name(status));
            free(input_data);
            campp_runtime_context_release(&context);
            campp_runtime_model_release(&model);
            return 1;
        }
    }
    for (tensor_id = 0u; tensor_id < context.tensor_count; ++tensor_id) {
        CHECK_TRUE(context.tensors[tensor_id].data != NULL);
        if (context.activations.buffers[tensor_id] != NULL) {
            owned_buffers += 1u;
        }
    }

    printf(
        "compiled plan storage: tensors=%" PRIu32
        " buffers=%" PRIu32 " payload=%zu bytes\n",
        model.tensor_count, owned_buffers, context.activations.total_bytes);

    free(input_data);
    campp_runtime_context_release(&context);
    campp_runtime_model_release(&model);
    return 0;
}

int main(int argc, char **argv)
{
    if (test_synthetic_context() != 0) {
        return 1;
    }
    if (argc == 3 && test_compiled_plan(argv[1], argv[2]) != 0) {
        return 1;
    }
    if (argc != 1 && argc != 3) {
        fprintf(stderr, "usage: %s [plan.bin weights.bin]\n", argv[0]);
        return 2;
    }
    puts("reference tensor storage tests: PASS");
    return 0;
}
