/* Kernel에 pointer를 넘기기 전에 Tensor의 논리 shape와 실제 byte 범위를 검사한다. */

#include "memory_bounds_checker.h"

#include <stddef.h>
#include <stdint.h>

#include "memory_management/reference_tensor_storage.h"
#include "memory_management/tensor_arena.h"

static int campp_u64_multiply_overflows(uint64_t left, uint64_t right)
{
    return right != 0u && left > UINT64_MAX / right;
}

static int campp_u64_add_overflows(uint64_t left, uint64_t right)
{
    return left > UINT64_MAX - right;
}

static CamppStatus campp_memory_required_span(
    const CamppTensorDescriptor *descriptor, uint64_t *out_logical_bytes,
    uint64_t *out_required_span)
{
    uint32_t dtype_bytes;
    uint64_t element_count = 1u;
    uint64_t maximum_offset = 0u;
    uint8_t axis;

    if (descriptor == NULL || out_logical_bytes == NULL ||
        out_required_span == NULL || descriptor->rank > CAMPP_TENSOR_MAX_RANK) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    dtype_bytes = campp_dtype_byte_size(descriptor->dtype);
    if (dtype_bytes == 0u) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }

    for (axis = 0u; axis < descriptor->rank; ++axis) {
        uint64_t extent;
        uint64_t stride_term;

        if (descriptor->dimensions[axis] == 0u) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        if (campp_u64_multiply_overflows(
                element_count, descriptor->dimensions[axis])) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
        element_count *= descriptor->dimensions[axis];

        extent = (uint64_t)descriptor->dimensions[axis] - 1u;
        if (campp_u64_multiply_overflows(
                extent, descriptor->byte_strides[axis])) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
        stride_term = extent * descriptor->byte_strides[axis];
        if (campp_u64_add_overflows(maximum_offset, stride_term)) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
        maximum_offset += stride_term;
    }
    for (axis = descriptor->rank; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        if (descriptor->dimensions[axis] != 1u ||
            descriptor->byte_strides[axis] != 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }

    if (campp_u64_multiply_overflows(element_count, dtype_bytes)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    *out_logical_bytes = element_count * dtype_bytes;
    if (campp_u64_add_overflows(maximum_offset, dtype_bytes)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    *out_required_span = maximum_offset + dtype_bytes;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_memory_bounds_check_view(
    const CamppTensorDescriptor *descriptor, const CamppTensorView *view)
{
    CamppStatus status;
    uint64_t expected_logical_bytes;
    uint64_t required_span;
    uint8_t axis;

    if (descriptor == NULL || view == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (view->data == NULL) {
        return CAMPP_STATUS_TENSOR_NOT_BOUND;
    }
    if (view->dtype != descriptor->dtype || view->rank != descriptor->rank) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    if (view->reserved != 0u) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        if (view->dimensions[axis] != descriptor->dimensions[axis] ||
            view->byte_strides[axis] != descriptor->byte_strides[axis]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }

    status = campp_memory_required_span(
        descriptor, &expected_logical_bytes, &required_span);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (descriptor->logical_byte_size != expected_logical_bytes ||
        view->logical_byte_size != descriptor->logical_byte_size) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    if (descriptor->storage_span_bytes < required_span ||
        view->storage_span_bytes < descriptor->storage_span_bytes ||
        view->storage_span_bytes < required_span) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_memory_bounds_check_constant(
    const CamppRuntimeModel *model,
    const CamppTensorDescriptor *descriptor)
{
    uint64_t end;

    if (model == NULL || descriptor == NULL ||
        descriptor->storage_type != CAMPP_TENSOR_STORAGE_CONSTANT ||
        model->weights == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor->data_offset == CAMPP_INVALID_DATA_OFFSET ||
        campp_u64_add_overflows(
            descriptor->data_offset, descriptor->storage_span_bytes)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    end = descriptor->data_offset + descriptor->storage_span_bytes;
    if (end > (uint64_t)model->weights_size) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_memory_bounds_check_tensor(
    const CamppRuntimeContext *context, uint32_t tensor_id)
{
    const CamppTensorDescriptor *descriptor;
    const CamppTensorView *view;
    CamppStatus status;

    if (context == NULL || context->model == NULL || context->tensors == NULL ||
        tensor_id >= context->tensor_count ||
        tensor_id >= context->model->tensor_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    descriptor = &context->model->tensors[tensor_id];
    view = &context->tensors[tensor_id];
    if (descriptor->tensor_id != tensor_id) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    status = campp_memory_bounds_check_view(descriptor, view);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    switch (descriptor->storage_type) {
    case CAMPP_TENSOR_STORAGE_INPUT:
        if ((view->flags & CAMPP_TENSOR_FLAG_EXTERNAL) == 0u) {
            return CAMPP_STATUS_TENSOR_NOT_BOUND;
        }
        return CAMPP_STATUS_OK;

    case CAMPP_TENSOR_STORAGE_CONSTANT: {
        const void *expected;

        status = campp_memory_bounds_check_constant(context->model, descriptor);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        expected = context->model->weights + descriptor->data_offset;
        if ((const void *)view->data != expected ||
            (view->flags & CAMPP_TENSOR_FLAG_READ_ONLY) == 0u) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
        return CAMPP_STATUS_OK;
    }

    case CAMPP_TENSOR_STORAGE_ACTIVATION:
    case CAMPP_TENSOR_STORAGE_OUTPUT: {
        if (context->activations.buffers == NULL ||
            context->activations.buffer_sizes == NULL ||
            tensor_id >= context->activations.buffer_count ||
            context->activations.buffers[tensor_id] != view->data ||
            (uint64_t)context->activations.buffer_sizes[tensor_id] <
                descriptor->storage_span_bytes) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
        if (context->activations.mode == CAMPP_ACTIVATION_STORAGE_ARENA) {
            const uint8_t *expected;
            uint64_t end;

            if ((descriptor->flags & CAMPP_TENSOR_FLAG_DENSE_SLAB) == 0u ||
                descriptor->data_offset == CAMPP_INVALID_DATA_OFFSET ||
                campp_u64_add_overflows(
                    descriptor->data_offset,
                    descriptor->storage_span_bytes)) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            end = descriptor->data_offset + descriptor->storage_span_bytes;
            if (context->activations.arena_base == NULL ||
                end > (uint64_t)context->activations.arena_size) {
                return CAMPP_STATUS_BUFFER_OVERFLOW;
            }
            expected = context->activations.arena_base +
                       (size_t)descriptor->data_offset;
            if ((const void *)view->data != (const void *)expected) {
                return CAMPP_STATUS_BUFFER_OVERFLOW;
            }
            return campp_tensor_arena_check_guards(&context->activations);
        }
        if ((descriptor->flags & CAMPP_TENSOR_FLAG_DENSE_SLAB) != 0u ||
            descriptor->data_offset != CAMPP_INVALID_DATA_OFFSET) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        return campp_reference_tensor_storage_check_guard(
            &context->activations, tensor_id);
    }

    case CAMPP_TENSOR_STORAGE_VIEW: {
        const CamppTensorDescriptor *base_descriptor;
        const CamppTensorView *base_view;
        const uint8_t *expected;
        uint64_t remaining;

        if (descriptor->alias_of_tensor_id >= context->tensor_count ||
            descriptor->alias_of_tensor_id == tensor_id ||
            descriptor->data_offset == CAMPP_INVALID_DATA_OFFSET) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        base_descriptor =
            &context->model->tensors[descriptor->alias_of_tensor_id];
        base_view = &context->tensors[descriptor->alias_of_tensor_id];
        if (base_descriptor->storage_type == CAMPP_TENSOR_STORAGE_VIEW ||
            base_view->data == NULL ||
            descriptor->data_offset > base_view->storage_span_bytes ||
            descriptor->data_offset > (uint64_t)SIZE_MAX) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        remaining = base_view->storage_span_bytes - descriptor->data_offset;
        if (descriptor->storage_span_bytes > remaining) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
        expected = (const uint8_t *)base_view->data +
                   (size_t)descriptor->data_offset;
        if ((const void *)view->data != (const void *)expected ||
            view->storage_span_bytes != descriptor->storage_span_bytes ||
            (view->flags & CAMPP_TENSOR_FLAG_ALIASED) == 0u) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
        if (context->activations.mode == CAMPP_ACTIVATION_STORAGE_ARENA) {
            return campp_tensor_arena_check_guards(&context->activations);
        }
        return campp_reference_tensor_storage_check_guard(
            &context->activations, descriptor->alias_of_tensor_id);
    }

    default:
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
}

CamppStatus campp_memory_bounds_check_all(
    const CamppRuntimeContext *context)
{
    uint32_t tensor_id;
    CamppStatus status;

    if (context == NULL || context->model == NULL ||
        context->tensor_count != context->model->tensor_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    for (tensor_id = 0u; tensor_id < context->tensor_count; ++tensor_id) {
        status = campp_memory_bounds_check_tensor(context, tensor_id);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_memory_bounds_check_operator(
    const CamppRuntimeContext *context,
    const CamppOperatorDescriptor *operator_descriptor)
{
    uint8_t slot;
    CamppStatus status;

    if (context == NULL || operator_descriptor == NULL ||
        operator_descriptor->input_count > CAMPP_OPERATOR_INPUT_CAPACITY ||
        operator_descriptor->output_count > CAMPP_OPERATOR_OUTPUT_CAPACITY) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    for (slot = 0u; slot < operator_descriptor->input_count; ++slot) {
        status = campp_memory_bounds_check_tensor(
            context, operator_descriptor->input_tensor_ids[slot]);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    for (slot = 0u; slot < operator_descriptor->output_count; ++slot) {
        uint32_t tensor_id = operator_descriptor->output_tensor_ids[slot];
        const CamppTensorDescriptor *descriptor;

        status = campp_memory_bounds_check_tensor(context, tensor_id);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        descriptor = &context->model->tensors[tensor_id];
        if (descriptor->storage_type == CAMPP_TENSOR_STORAGE_INPUT ||
            descriptor->storage_type == CAMPP_TENSOR_STORAGE_CONSTANT ||
            (context->tensors[tensor_id].flags &
             CAMPP_TENSOR_FLAG_READ_ONLY) != 0u) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_memory_bounds_check_guards(
    const CamppRuntimeContext *context)
{
    if (context == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (context->activations.mode == CAMPP_ACTIVATION_STORAGE_ARENA) {
        return campp_tensor_arena_check_guards(&context->activations);
    }
    return campp_reference_tensor_storage_check_all_guards(
        &context->activations);
}
