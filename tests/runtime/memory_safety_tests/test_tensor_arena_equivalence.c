#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backends/cpu_reference/reference_kernel_utils.h"
#include "execution/graph_executor.h"
#include "execution/tensor_registry.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"

static int tensor_bytes_equal(
    const CamppTensorView *reference, const CamppTensorView *arena)
{
    uint64_t element_count;
    uint64_t index;
    uint32_t element_size;

    if (reference == NULL || arena == NULL || reference->data == NULL ||
        arena->data == NULL || reference->dtype != arena->dtype ||
        reference->rank != arena->rank ||
        reference->logical_byte_size != arena->logical_byte_size) {
        return 0;
    }
    if (memcmp(
            reference->dimensions, arena->dimensions,
            sizeof(reference->dimensions)) != 0) {
        return 0;
    }
    if (campp_tensor_view_is_contiguous(reference) &&
        campp_tensor_view_is_contiguous(arena)) {
        return memcmp(
                   reference->data, arena->data,
                   (size_t)reference->logical_byte_size) == 0;
    }

    element_size = campp_dtype_byte_size(reference->dtype);
    element_count = campp_tensor_view_element_count(reference);
    if (element_size == 0u ||
        element_count != campp_tensor_view_element_count(arena)) {
        return 0;
    }
    for (index = 0u; index < element_count; ++index) {
        const uint64_t reference_offset =
            campp_reference_offset_for_linear(reference, index);
        const uint64_t arena_offset =
            campp_reference_offset_for_linear(arena, index);
        if (memcmp(
                (const uint8_t *)reference->data + reference_offset,
                (const uint8_t *)arena->data + arena_offset,
                element_size) != 0) {
            return 0;
        }
    }
    return 1;
}

static int fail_status(
    const char *step, CamppStatus status, uint32_t operator_id)
{
    fprintf(
        stderr, "%s failed at operator %" PRIu32 ": %s\n",
        step, operator_id, campp_status_name(status));
    return 1;
}

int main(int argc, char **argv)
{
    CamppRuntimeModel reference_model;
    CamppRuntimeModel arena_model;
    CamppRuntimeContext reference_context;
    CamppRuntimeContext arena_context;
    const CamppKernelRegistry *registry = campp_cpu_reference_registry();
    const CamppTensorDescriptor *input_descriptor;
    uint32_t input_id;
    void *input = NULL;
    uint32_t operator_id;
    uint32_t compared = 0u;
    CamppStatus status;
    int result = 1;

    if (argc != 5) {
        fprintf(
            stderr,
            "usage: %s <reference_plan> <arena_plan> "
            "<reference_weights> <arena_weights>\n",
            argv[0]);
        return 2;
    }
    memset(&reference_model, 0, sizeof(reference_model));
    memset(&arena_model, 0, sizeof(arena_model));
    memset(&reference_context, 0, sizeof(reference_context));
    memset(&arena_context, 0, sizeof(arena_context));

    status = campp_runtime_model_load(argv[1], argv[3], &reference_model);
    if (status != CAMPP_STATUS_OK) {
        return fail_status("reference model load", status, 0u);
    }
    status = campp_runtime_model_load(argv[2], argv[4], &arena_model);
    if (status != CAMPP_STATUS_OK) {
        fail_status("arena model load", status, 0u);
        goto cleanup;
    }
    if (reference_model.bucket_frames != arena_model.bucket_frames ||
        reference_model.tensor_count != arena_model.tensor_count ||
        reference_model.operator_count != arena_model.operator_count ||
        reference_model.input_count != 1u || arena_model.input_count != 1u) {
        fprintf(stderr, "reference and arena graph metadata differ\n");
        goto cleanup;
    }

    status = campp_runtime_context_create(
        &reference_model, registry, &reference_context);
    if (status != CAMPP_STATUS_OK) {
        fail_status("reference context create", status, 0u);
        goto cleanup;
    }
    status = campp_runtime_context_create(&arena_model, registry, &arena_context);
    if (status != CAMPP_STATUS_OK) {
        fail_status("arena context create", status, 0u);
        goto cleanup;
    }
    if (reference_context.activations.mode !=
            CAMPP_ACTIVATION_STORAGE_REFERENCE ||
        arena_context.activations.mode != CAMPP_ACTIVATION_STORAGE_ARENA) {
        fprintf(stderr, "unexpected activation storage mode\n");
        goto cleanup;
    }

    input_id = reference_model.input_tensor_ids[0];
    if (input_id != arena_model.input_tensor_ids[0]) {
        fprintf(stderr, "reference and arena input IDs differ\n");
        goto cleanup;
    }
    input_descriptor = &reference_model.tensors[input_id];
    if (input_descriptor->storage_span_bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "input is too large for this host\n");
        goto cleanup;
    }
    input = calloc(1u, (size_t)input_descriptor->storage_span_bytes);
    if (input == NULL) {
        fprintf(stderr, "input allocation failed\n");
        goto cleanup;
    }
    status = campp_runtime_context_bind_input(
        &reference_context, input_id, input,
        (size_t)input_descriptor->storage_span_bytes,
        input_descriptor->dtype, input_descriptor->rank,
        input_descriptor->dimensions);
    if (status != CAMPP_STATUS_OK) {
        fail_status("reference input bind", status, 0u);
        goto cleanup;
    }
    status = campp_runtime_context_bind_input(
        &arena_context, input_id, input,
        (size_t)input_descriptor->storage_span_bytes,
        input_descriptor->dtype, input_descriptor->rank,
        input_descriptor->dimensions);
    if (status != CAMPP_STATUS_OK) {
        fail_status("arena input bind", status, 0u);
        goto cleanup;
    }

    for (operator_id = 0u; operator_id < reference_model.operator_count;
         ++operator_id) {
        const CamppOperatorDescriptor *reference_operator =
            &reference_model.operators[operator_id];
        const CamppOperatorDescriptor *arena_operator =
            &arena_model.operators[operator_id];
        uint8_t slot;

        status = campp_graph_execute_operator(&reference_context, operator_id);
        if (status != CAMPP_STATUS_OK) {
            fail_status("reference execute", status, operator_id);
            goto cleanup;
        }
        status = campp_graph_execute_operator(&arena_context, operator_id);
        if (status != CAMPP_STATUS_OK) {
            fail_status("arena execute", status, operator_id);
            goto cleanup;
        }
        if (reference_operator->opcode != arena_operator->opcode ||
            reference_operator->output_count != arena_operator->output_count) {
            fprintf(stderr, "operator metadata differs at %" PRIu32 "\n",
                    operator_id);
            goto cleanup;
        }
        for (slot = 0u; slot < reference_operator->output_count; ++slot) {
            const uint32_t reference_id =
                reference_operator->output_tensor_ids[slot];
            const uint32_t arena_id = arena_operator->output_tensor_ids[slot];
            if (reference_id != arena_id ||
                !tensor_bytes_equal(
                    &reference_context.tensors[reference_id],
                    &arena_context.tensors[arena_id])) {
                fprintf(
                    stderr,
                    "first Tensor mismatch: operator=%" PRIu32
                    " tensor=%" PRIu32 "\n",
                    operator_id, reference_id);
                goto cleanup;
            }
            compared += 1u;
        }
    }

    printf(
        "arena equivalence: bucket=%" PRIu32 " operators=%" PRIu32
        " tensors=%" PRIu32 " reference=%zu arena=%zu bit-exact=YES\n",
        reference_model.bucket_frames, reference_model.operator_count,
        compared, reference_context.activations.total_bytes,
        arena_context.activations.total_bytes);
    result = 0;

cleanup:
    free(input);
    campp_runtime_context_release(&arena_context);
    campp_runtime_context_release(&reference_context);
    campp_runtime_model_release(&arena_model);
    campp_runtime_model_release(&reference_model);
    return result;
}
