#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "campp_runtime/operator_descriptor.h"
#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "backends/cpu_reference/reference_kernel_utils.h"
#include "execution/graph_executor.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"

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

static void init_tensor_descriptor(
    CamppTensorDescriptor *descriptor, uint32_t tensor_id,
    uint8_t storage_type, uint32_t dim0, uint32_t dim1)
{
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->tensor_id = tensor_id;
    descriptor->dtype = CAMPP_DTYPE_FLOAT32;
    descriptor->rank = 2u;
    descriptor->storage_type = storage_type;
    descriptor->dimensions[0] = dim0;
    descriptor->dimensions[1] = dim1;
    descriptor->dimensions[2] = 1u;
    descriptor->dimensions[3] = 1u;
    descriptor->byte_strides[0] = dim1 * 4u;
    descriptor->byte_strides[1] = 4u;
    descriptor->logical_byte_size = (uint64_t)dim0 * dim1 * 4u;
    descriptor->storage_span_bytes = descriptor->logical_byte_size;
    descriptor->data_offset = CAMPP_INVALID_DATA_OFFSET;
    descriptor->alias_of_tensor_id = CAMPP_INVALID_TENSOR_ID;
    descriptor->quantization_index = CAMPP_INVALID_QUANTIZATION_INDEX;
    descriptor->first_use = 0u;
    descriptor->last_use = 0u;
}

typedef struct TestTensorReadyState {
    uint32_t calls;
    uint32_t operator_id;
    uint32_t tensor_id;
    float values[2];
} TestTensorReadyState;

static CamppStatus test_tensor_ready(
    void *user_data, uint32_t operator_id, uint32_t tensor_id,
    const CamppTensorView *view)
{
    TestTensorReadyState *state = (TestTensorReadyState *)user_data;
    if (state == NULL || view == NULL || view->data == NULL ||
        view->dtype != CAMPP_DTYPE_FLOAT32 ||
        view->logical_byte_size != sizeof(state->values)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    state->calls += 1u;
    state->operator_id = operator_id;
    state->tensor_id = tensor_id;
    memcpy(state->values, view->data, sizeof(state->values));
    return CAMPP_STATUS_OK;
}

static int test_synthetic_execution(void)
{
    uint8_t unused_weights = 0u;
    float input[2] = {-1.0f, 2.0f};
    uint32_t dimensions[CAMPP_TENSOR_MAX_RANK] = {1u, 2u, 1u, 1u};
    CamppTensorDescriptor tensors[2];
    CamppOperatorDescriptor operator_descriptor;
    CamppRuntimeModel model;
    CamppRuntimeContext context;
    TestTensorReadyState ready_state;
    const CamppKernelRegistry *registry = campp_cpu_reference_registry();

    init_tensor_descriptor(
        &tensors[0], 0u, CAMPP_TENSOR_STORAGE_INPUT, 1u, 2u);
    init_tensor_descriptor(
        &tensors[1], 1u, CAMPP_TENSOR_STORAGE_OUTPUT, 1u, 2u);

    memset(&operator_descriptor, 0, sizeof(operator_descriptor));
    operator_descriptor.operator_id = 0u;
    operator_descriptor.opcode = CAMPP_OP_RELU;
    operator_descriptor.input_count = 1u;
    operator_descriptor.output_count = 1u;
    operator_descriptor.input_tensor_ids[0] = 0u;
    operator_descriptor.output_tensor_ids[0] = 1u;
    operator_descriptor.backend_id = CAMPP_BACKEND_AUTO;
    operator_descriptor.kernel_id = CAMPP_DEFAULT_KERNEL_ID;

    memset(&model, 0, sizeof(model));
    model.weights = &unused_weights;
    model.weights_size = 1u;
    model.bucket_frames = 2u;
    model.tensors = tensors;
    model.tensor_count = 2u;
    model.operators = &operator_descriptor;
    model.operator_count = 1u;

    CHECK_TRUE(registry != NULL);
    CHECK_TRUE(registry->entry_count == CAMPP_KERNEL_OPCODE_COUNT);
    memset(&context, 0, sizeof(context));
    CHECK_STATUS(
        campp_runtime_context_create(&model, registry, &context),
        CAMPP_STATUS_OK);
    CHECK_STATUS(
        campp_runtime_context_bind_input(
            &context, 0u, input, sizeof(input), CAMPP_DTYPE_FLOAT32, 2u,
            dimensions),
        CAMPP_STATUS_OK);
    CHECK_TRUE(context.resolved_kernel_count == 1u);
    CHECK_TRUE(context.resolved_kernels[0] != NULL);
    CHECK_TRUE(context.resolved_kernels[0]->opcode == CAMPP_OP_RELU);
    CHECK_TRUE(
        strcmp(context.resolved_kernels[0]->name, "relu_reference") == 0);

    memset(&ready_state, 0, sizeof(ready_state));
    context.diagnostics.tensor_ready = test_tensor_ready;
    context.diagnostics.tensor_ready_user_data = &ready_state;

    CHECK_STATUS(campp_graph_execute(&context), CAMPP_STATUS_OK);
    CHECK_TRUE(context.diagnostics.current_operator_id == 0u);
    CHECK_TRUE(context.diagnostics.executed_operator_count == 1u);
    CHECK_TRUE(context.last_status == CAMPP_STATUS_OK);
    CHECK_TRUE(((const float *)context.tensors[1].data)[0] == 0.0f);
    CHECK_TRUE(((const float *)context.tensors[1].data)[1] == 2.0f);
    CHECK_TRUE(ready_state.calls == 1u);
    CHECK_TRUE(ready_state.operator_id == 0u);
    CHECK_TRUE(ready_state.tensor_id == 1u);
    CHECK_TRUE(ready_state.values[0] == 0.0f);
    CHECK_TRUE(ready_state.values[1] == 2.0f);

    campp_runtime_context_release(&context);
    return 0;
}

static int test_compiled_plan_execution(
    const char *plan_path, const char *weights_path)
{
    CamppRuntimeModel model;
    CamppRuntimeContext context;
    const CamppKernelRegistry *registry = campp_cpu_reference_registry();
    const CamppTensorDescriptor *input_descriptor;
    uint32_t input_tensor_id;
    void *input_data;
    const CamppTensorView *output_view;
    uint64_t output_index;
    CamppStatus status;

    memset(&model, 0, sizeof(model));
    CHECK_STATUS(
        campp_runtime_model_load(plan_path, weights_path, &model),
        CAMPP_STATUS_OK);
    CHECK_TRUE(model.operator_count == 1438u);
    CHECK_TRUE(model.operators[0].operator_id == 0u);
    CHECK_TRUE(model.operators[0].opcode == CAMPP_OP_TRANSPOSE);
    CHECK_TRUE(model.operators[0].backend_id == CAMPP_BACKEND_AUTO);

    memset(&context, 0, sizeof(context));
    status = campp_runtime_context_create(&model, registry, &context);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "context create failed: %s\n", campp_status_name(status));
        campp_runtime_model_release(&model);
        return 1;
    }
    CHECK_TRUE(context.resolved_kernel_count == model.operator_count);
    CHECK_TRUE(context.resolved_kernels[0] != NULL);
    CHECK_TRUE(context.resolved_kernels[0]->opcode == CAMPP_OP_TRANSPOSE);
    CHECK_TRUE(
        strcmp(context.resolved_kernels[0]->name, "transpose_reference") == 0);

    CHECK_TRUE(model.input_count == 1u);
    input_tensor_id = model.input_tensor_ids[0];
    input_descriptor = &model.tensors[input_tensor_id];
    CHECK_TRUE(input_descriptor->storage_span_bytes <= (uint64_t)SIZE_MAX);
    input_data = malloc((size_t)input_descriptor->storage_span_bytes);
    CHECK_TRUE(input_data != NULL);
    memset(input_data, 0, (size_t)input_descriptor->storage_span_bytes);
    status = campp_runtime_context_bind_input(
        &context, input_tensor_id, input_data,
        (size_t)input_descriptor->storage_span_bytes, input_descriptor->dtype,
        input_descriptor->rank, input_descriptor->dimensions);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "input bind failed: %s\n", campp_status_name(status));
        free(input_data);
        campp_runtime_context_release(&context);
        campp_runtime_model_release(&model);
        return 1;
    }

    CHECK_STATUS(campp_graph_execute(&context), CAMPP_STATUS_OK);
    CHECK_TRUE(
        context.diagnostics.current_operator_id == model.operator_count - 1u);
    CHECK_TRUE(
        context.diagnostics.executed_operator_count == model.operator_count);
    CHECK_TRUE(context.last_status == CAMPP_STATUS_OK);
    CHECK_TRUE(model.output_count == 1u);
    CHECK_STATUS(
        campp_runtime_context_output(
            &context, model.output_tensor_ids[0], &output_view),
        CAMPP_STATUS_OK);
    CHECK_TRUE(output_view->dtype == CAMPP_DTYPE_FLOAT32);
    CHECK_TRUE(campp_tensor_view_element_count(output_view) == 192u);
    for (output_index = 0u; output_index < 192u; ++output_index) {
        float value;
        const uint64_t offset =
            campp_reference_offset_for_linear(output_view, output_index);
        memcpy(
            &value, (const uint8_t *)output_view->data + offset,
            sizeof(value));
        CHECK_TRUE(isfinite(value));
    }
    printf(
        "plan dispatch: bucket=%" PRIu32 " operators=%" PRIu32
        " executed=%" PRIu32 " embedding=192\n",
        model.bucket_frames, model.operator_count,
        context.diagnostics.executed_operator_count);

    free(input_data);
    campp_runtime_context_release(&context);
    campp_runtime_model_release(&model);
    return 0;
}

int main(int argc, char **argv)
{
    if (test_synthetic_execution() != 0) {
        return 1;
    }
    if (argc == 3 &&
        test_compiled_plan_execution(argv[1], argv[2]) != 0) {
        return 1;
    }
    if (argc != 1 && argc != 3) {
        fprintf(stderr, "usage: %s [plan.bin weights.bin]\n", argv[0]);
        return 2;
    }
    puts("graph executor tests: PASS");
    return 0;
}
