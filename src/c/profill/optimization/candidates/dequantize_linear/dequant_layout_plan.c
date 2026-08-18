#include "dequant_layout_plan.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static uint64_t campp_dequant_max_offset(
    const uint32_t dimensions[4], const uint32_t strides[4])
{
    uint64_t offset = 0u;
    uint8_t axis;
    for (axis = 0u; axis < 4u; ++axis) {
        offset += (uint64_t)(dimensions[axis] - 1u) * strides[axis];
    }
    return offset;
}

bool campp_dequant_layout_plan_create(
    const CamppTensorView *input, const CamppTensorView *output,
    CamppDequantLayoutPlan *out_plan)
{
    uint32_t dimensions[4];
    uint8_t axis;

    if (input == NULL || output == NULL || out_plan == NULL ||
        input->data == NULL || output->data == NULL ||
        (input->dtype != CAMPP_DTYPE_UINT8 &&
         input->dtype != CAMPP_DTYPE_INT8) ||
        output->dtype != CAMPP_DTYPE_FLOAT32 ||
        input->rank < 2u || input->rank > 4u ||
        input->rank != output->rank) {
        return false;
    }
    for (axis = 0u; axis < input->rank; ++axis) {
        if (input->dimensions[axis] != output->dimensions[axis]) return false;
    }

    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->batches = input->dimensions[0];
    out_plan->channels = input->dimensions[1];
    out_plan->height = input->rank == 4u ? input->dimensions[2] : 1u;
    out_plan->width = input->rank == 2u
        ? 1u : input->dimensions[input->rank - 1u];
    if (out_plan->batches == 0u || out_plan->channels == 0u ||
        out_plan->height == 0u || out_plan->width == 0u) {
        return false;
    }

    out_plan->input_strides[0] = input->byte_strides[0];
    out_plan->input_strides[1] = input->byte_strides[1];
    out_plan->output_strides[0] = output->byte_strides[0];
    out_plan->output_strides[1] = output->byte_strides[1];
    if (input->rank == 3u) {
        out_plan->input_strides[3] = input->byte_strides[2];
        out_plan->output_strides[3] = output->byte_strides[2];
    } else if (input->rank == 4u) {
        out_plan->input_strides[2] = input->byte_strides[2];
        out_plan->input_strides[3] = input->byte_strides[3];
        out_plan->output_strides[2] = output->byte_strides[2];
        out_plan->output_strides[3] = output->byte_strides[3];
    }

    dimensions[0] = out_plan->batches;
    dimensions[1] = out_plan->channels;
    dimensions[2] = out_plan->height;
    dimensions[3] = out_plan->width;
    if (campp_dequant_max_offset(dimensions, out_plan->input_strides) + 1u >
            input->storage_span_bytes ||
        campp_dequant_max_offset(dimensions, out_plan->output_strides) +
            sizeof(float) > output->storage_span_bytes) {
        return false;
    }
    for (axis = 0u; axis < output->rank; ++axis) {
        if (output->byte_strides[axis] % _Alignof(float) != 0u) {
            return false;
        }
    }
    out_plan->channel_contiguous =
        out_plan->input_strides[1] == 1u &&
        out_plan->output_strides[1] == sizeof(float) &&
        ((uintptr_t)output->data % _Alignof(float)) == 0u;
    return out_plan->channel_contiguous;
}

uint64_t campp_dequant_input_offset(
    const CamppDequantLayoutPlan *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width)
{
    return (uint64_t)batch * plan->input_strides[0] +
        (uint64_t)channel * plan->input_strides[1] +
        (uint64_t)height * plan->input_strides[2] +
        (uint64_t)width * plan->input_strides[3];
}

uint64_t campp_dequant_output_offset(
    const CamppDequantLayoutPlan *plan, uint32_t batch, uint32_t channel,
    uint32_t height, uint32_t width)
{
    return (uint64_t)batch * plan->output_strides[0] +
        (uint64_t)channel * plan->output_strides[1] +
        (uint64_t)height * plan->output_strides[2] +
        (uint64_t)width * plan->output_strides[3];
}
