/* Tensor ID를 inference에서 사용할 실제 pointer가 든 TensorView로 연결한다. */

#include "tensor_registry.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "memory_management/memory_bounds_checker.h"
#include "memory_management/reference_tensor_storage.h"

static void campp_tensor_view_copy_metadata(
    const CamppTensorDescriptor *descriptor, CamppTensorView *view)
{
    uint8_t axis;

    memset(view, 0, sizeof(*view));
    view->dtype = descriptor->dtype;
    view->rank = descriptor->rank;
    /* 연속성은 아래에서 stride로 다시 계산한다. */
    view->flags = (uint8_t)(
        descriptor->flags & (uint8_t)~CAMPP_TENSOR_FLAG_CONTIGUOUS);
    view->reserved = 0u;
    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        view->dimensions[axis] = descriptor->dimensions[axis];
        view->byte_strides[axis] = descriptor->byte_strides[axis];
    }
    view->logical_byte_size = descriptor->logical_byte_size;
    view->storage_span_bytes = descriptor->storage_span_bytes;
}

CamppStatus campp_tensor_registry_create(
    const CamppRuntimeModel *model,
    const CamppActivationStorage *activation_storage,
    CamppTensorView **out_tensors, uint32_t *out_tensor_count)
{
    CamppTensorView *views;
    uint32_t tensor_id;
    CamppStatus status = CAMPP_STATUS_OK;

    if (model == NULL || activation_storage == NULL || out_tensors == NULL ||
        out_tensor_count == NULL || model->tensors == NULL ||
        model->tensor_count == 0u || activation_storage->buffers == NULL ||
        activation_storage->buffer_sizes == NULL ||
        activation_storage->buffer_count != model->tensor_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if ((size_t)model->tensor_count > SIZE_MAX / sizeof(*views)) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }

    views = (CamppTensorView *)calloc(
        (size_t)model->tensor_count, sizeof(*views));
    if (views == NULL) {
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }

    for (tensor_id = 0u; tensor_id < model->tensor_count; ++tensor_id) {
        const CamppTensorDescriptor *descriptor = &model->tensors[tensor_id];
        CamppTensorView *view = &views[tensor_id];
        if (descriptor->tensor_id != tensor_id) {
            status = CAMPP_STATUS_CORRUPT_PLAN;
            goto failed;
        }
        campp_tensor_view_copy_metadata(descriptor, view);

        switch (descriptor->storage_type) {
        case CAMPP_TENSOR_STORAGE_INPUT:
            /* bind_input 전까지 의도적으로 NULL이다. */
            view->data = NULL;
            break;

        case CAMPP_TENSOR_STORAGE_CONSTANT: {
            const void *constant_data = NULL;

            status = campp_memory_bounds_check_constant(model, descriptor);
            if (status != CAMPP_STATUS_OK) {
                goto failed;
            }
            status = campp_runtime_model_constant_data(
                model, descriptor, &constant_data);
            if (status != CAMPP_STATUS_OK) {
                goto failed;
            }
            /* TensorView는 공통 표현이라 void*지만 READ_ONLY flag를 유지한다. */
            memcpy(&view->data, &constant_data, sizeof(view->data));
            break;
        }

        case CAMPP_TENSOR_STORAGE_ACTIVATION:
        case CAMPP_TENSOR_STORAGE_OUTPUT: {
            void *buffer = NULL;
            size_t buffer_size = 0u;

            status = campp_reference_tensor_storage_buffer(
                activation_storage, tensor_id, &buffer, &buffer_size);
            if (status != CAMPP_STATUS_OK) {
                goto failed;
            }
            if ((uint64_t)buffer_size < descriptor->storage_span_bytes) {
                status = CAMPP_STATUS_BUFFER_OVERFLOW;
                goto failed;
            }
            view->data = buffer;
            view->storage_span_bytes = (uint64_t)buffer_size;
            break;
        }

        case CAMPP_TENSOR_STORAGE_VIEW:
            status = CAMPP_STATUS_NOT_IMPLEMENTED;
            goto failed;

        default:
            status = CAMPP_STATUS_CORRUPT_PLAN;
            goto failed;
        }

        if (campp_tensor_view_is_contiguous(view)) {
            view->flags =
                (uint8_t)(view->flags | CAMPP_TENSOR_FLAG_CONTIGUOUS);
        }
        if (descriptor->storage_type != CAMPP_TENSOR_STORAGE_INPUT) {
            status = campp_memory_bounds_check_view(descriptor, view);
            if (status != CAMPP_STATUS_OK) {
                goto failed;
            }
        }
    }

    *out_tensors = views;
    *out_tensor_count = model->tensor_count;
    return CAMPP_STATUS_OK;

failed:
    free(views);
    return status;
}

void campp_tensor_registry_release(
    CamppTensorView **tensors, uint32_t *tensor_count)
{
    if (tensors != NULL) {
        free(*tensors);
        *tensors = NULL;
    }
    if (tensor_count != NULL) {
        *tensor_count = 0u;
    }
}

CamppStatus campp_runtime_context_bind_input(
    CamppRuntimeContext *context, uint32_t tensor_id, void *data,
    size_t byte_size, uint8_t dtype, uint8_t rank,
    const uint32_t dimensions[CAMPP_TENSOR_MAX_RANK])
{
    const CamppTensorDescriptor *descriptor;
    CamppTensorView candidate;
    uint8_t axis;
    uint8_t mismatch_count = 0u;
    uint8_t mismatched_axis = 0u;

    if (context == NULL || context->model == NULL || context->tensors == NULL ||
        dimensions == NULL || data == NULL ||
        tensor_id >= context->tensor_count ||
        tensor_id >= context->model->tensor_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    descriptor = &context->model->tensors[tensor_id];
    if (descriptor->storage_type != CAMPP_TENSOR_STORAGE_INPUT) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (dtype != descriptor->dtype) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (rank != descriptor->rank) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        if (dimensions[axis] != descriptor->dimensions[axis]) {
            mismatch_count += 1u;
            mismatched_axis = axis;
        }
    }
    if (mismatch_count != 0u) {
        if (mismatch_count == 1u &&
            descriptor->dimensions[mismatched_axis] ==
                context->model->bucket_frames) {
            return CAMPP_STATUS_BUCKET_MISMATCH;
        }
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    if ((uint64_t)byte_size < descriptor->storage_span_bytes) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }

    /* 검증이 모두 끝난 뒤 한 번에 반영하므로 실패해도 이전 binding은 유지된다. */
    candidate = context->tensors[tensor_id];
    candidate.data = data;
    candidate.storage_span_bytes = (uint64_t)byte_size;
    candidate.flags = (uint8_t)(candidate.flags | CAMPP_TENSOR_FLAG_EXTERNAL);
    candidate.reserved = 0u;
    if (campp_tensor_view_is_contiguous(&candidate)) {
        candidate.flags =
            (uint8_t)(candidate.flags | CAMPP_TENSOR_FLAG_CONTIGUOUS);
    } else {
        candidate.flags =
            (uint8_t)(candidate.flags &
                      (uint8_t)~CAMPP_TENSOR_FLAG_CONTIGUOUS);
    }
    context->tensors[tensor_id] = candidate;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_runtime_context_output(
    const CamppRuntimeContext *context, uint32_t tensor_id,
    const CamppTensorView **out_view)
{
    const CamppTensorDescriptor *descriptor;

    if (context == NULL || context->model == NULL || out_view == NULL ||
        context->tensors == NULL || tensor_id >= context->tensor_count ||
        tensor_id >= context->model->tensor_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    descriptor = &context->model->tensors[tensor_id];
    if (descriptor->storage_type != CAMPP_TENSOR_STORAGE_OUTPUT) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (context->tensors[tensor_id].data == NULL) {
        return CAMPP_STATUS_TENSOR_NOT_BOUND;
    }
    *out_view = &context->tensors[tensor_id];
    return CAMPP_STATUS_OK;
}

CamppStatus campp_runtime_context_tensor(
    const CamppRuntimeContext *context, uint32_t tensor_id,
    CamppTensorView **out_view)
{
    if (context == NULL || out_view == NULL || context->tensors == NULL ||
        tensor_id >= context->tensor_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    *out_view = &context->tensors[tensor_id];
    return CAMPP_STATUS_OK;
}
