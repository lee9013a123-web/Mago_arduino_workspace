#include "packed_iteration.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static uint64_t campp_packed_max_offset(
    const CamppPackedIteration *plan)
{
    return (uint64_t)(plan->batches - 1u) * plan->strides[0] +
        (uint64_t)(plan->channels - 1u) * plan->strides[1] +
        (uint64_t)(plan->height - 1u) * plan->strides[2] +
        (uint64_t)(plan->width - 1u) * plan->strides[3];
}

bool campp_packed_iteration_create(
    const CamppTensorView *view, CamppPackedIteration *out_plan)
{
    uint8_t axis;

    if (view == NULL || out_plan == NULL || view->data == NULL ||
        view->rank < 2u || view->rank > 4u) {
        return false;
    }
    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->view = view;
    out_plan->element_size = campp_dtype_byte_size(view->dtype);
    if (out_plan->element_size == 0u) return false;
    out_plan->batches = view->dimensions[0];
    out_plan->channels = view->dimensions[1];
    out_plan->height = view->rank == 4u ? view->dimensions[2] : 1u;
    out_plan->width = view->rank >= 3u
        ? view->dimensions[view->rank - 1u] : 1u;
    if (out_plan->batches == 0u || out_plan->channels == 0u ||
        out_plan->height == 0u || out_plan->width == 0u) {
        return false;
    }
    out_plan->strides[0] = view->byte_strides[0];
    out_plan->strides[1] = view->byte_strides[1];
    if (view->rank == 3u) {
        out_plan->strides[3] = view->byte_strides[2];
    } else if (view->rank == 4u) {
        out_plan->strides[2] = view->byte_strides[2];
        out_plan->strides[3] = view->byte_strides[3];
    }
    if (out_plan->strides[1] != out_plan->element_size ||
        campp_packed_max_offset(out_plan) + out_plan->element_size >
            view->storage_span_bytes) {
        return false;
    }
    if (out_plan->element_size > 1u) {
        if ((uintptr_t)view->data % out_plan->element_size != 0u) return false;
        for (axis = 0u; axis < view->rank; ++axis) {
            if (view->byte_strides[axis] % out_plan->element_size != 0u) {
                return false;
            }
        }
    }
    return true;
}

bool campp_packed_iteration_same_shape(
    const CamppPackedIteration *left, const CamppPackedIteration *right)
{
    return left != NULL && right != NULL &&
        left->batches == right->batches &&
        left->channels == right->channels &&
        left->height == right->height && left->width == right->width;
}

uint64_t campp_packed_iteration_offset(
    const CamppPackedIteration *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width)
{
    return (uint64_t)batch * plan->strides[0] +
        (uint64_t)channel * plan->strides[1] +
        (uint64_t)height * plan->strides[2] +
        (uint64_t)width * plan->strides[3];
}

const uint8_t *campp_packed_iteration_const_pointer(
    const CamppPackedIteration *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width)
{
    return (const uint8_t *)plan->view->data +
        campp_packed_iteration_offset(plan, batch, channel, height, width);
}

uint8_t *campp_packed_iteration_pointer(
    const CamppPackedIteration *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width)
{
    return (uint8_t *)plan->view->data +
        campp_packed_iteration_offset(plan, batch, channel, height, width);
}
