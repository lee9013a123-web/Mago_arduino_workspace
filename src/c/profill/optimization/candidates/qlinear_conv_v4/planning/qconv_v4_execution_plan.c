#include "qconv_v4_execution_plan.h"

#include <limits.h>
#include <math.h>
#include <string.h>

#include "backends/cpu_reference/reference_kernel_utils.h"
#include "internal/runtime_model.h"
#include "qconv_mac_neon.h"

static CamppStatus campp_qconv_v4_attribute(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, int64_t *values, uint8_t capacity, uint8_t expected)
{
    uint8_t count = 0u;
    CamppStatus status = campp_runtime_model_attribute_ints(
        model, op, key, values, capacity, &count);
    if (status != CAMPP_STATUS_OK) return status;
    return count == expected ? CAMPP_STATUS_OK : CAMPP_STATUS_CORRUPT_PLAN;
}

CamppStatus campp_qconv_v4_execution_plan_create(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppQconvV4ExecutionPlan *out_plan)
{
    int64_t group_value[1] = {1};
    CamppStatus status;
    uint8_t axis;

    if (out_plan == NULL) return CAMPP_STATUS_INVALID_ARGUMENT;
    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->kernel_shape[0] = 1;
    out_plan->kernel_shape[1] = 1;
    out_plan->strides[0] = 1;
    out_plan->strides[1] = 1;
    out_plan->dilations[0] = 1;
    out_plan->dilations[1] = 1;
    out_plan->group = 1u;
    out_plan->kernel_elements = 1u;
    out_plan->output_spatial = 1u;

    status = campp_reference_validate_invocation(
        inputs, input_count, 8u, 9u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    out_plan->input = &inputs[0];
    out_plan->weight = &inputs[3];
    out_plan->output = &outputs[0];
    if ((out_plan->weight->flags & CAMPP_TENSOR_FLAG_PACKED_QCONV_O4I4) == 0u ||
        (out_plan->input->dtype != CAMPP_DTYPE_INT8 &&
         out_plan->input->dtype != CAMPP_DTYPE_UINT8) ||
        (out_plan->weight->dtype != CAMPP_DTYPE_INT8 &&
         out_plan->weight->dtype != CAMPP_DTYPE_UINT8) ||
        (out_plan->output->dtype != CAMPP_DTYPE_INT8 &&
         out_plan->output->dtype != CAMPP_DTYPE_UINT8) ||
        out_plan->input->rank < 3u || out_plan->input->rank > 4u ||
        out_plan->weight->rank != out_plan->input->rank ||
        out_plan->output->rank != out_plan->input->rank ||
        out_plan->output->dimensions[0] != out_plan->input->dimensions[0] ||
        out_plan->input->byte_strides[1] != 1u ||
        out_plan->output->byte_strides[1] != 1u) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }

    out_plan->spatial_rank = (uint8_t)(out_plan->input->rank - 2u);
    status = campp_qconv_v4_attribute(
        model, op, CAMPP_ATTR_KERNEL_SHAPE, out_plan->kernel_shape,
        2u, out_plan->spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_v4_attribute(
        model, op, CAMPP_ATTR_PADS, out_plan->pads, 4u,
        (uint8_t)(out_plan->spatial_rank * 2u));
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_v4_attribute(
        model, op, CAMPP_ATTR_STRIDES, out_plan->strides,
        2u, out_plan->spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_v4_attribute(
        model, op, CAMPP_ATTR_DILATIONS, out_plan->dilations,
        2u, out_plan->spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_v4_attribute(
        model, op, CAMPP_ATTR_GROUP, group_value, 1u, 1u);
    if (status != CAMPP_STATUS_OK || group_value[0] <= 0 ||
        group_value[0] > UINT32_MAX) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    out_plan->group = (uint32_t)group_value[0];
    out_plan->input_channels = out_plan->input->dimensions[1];
    out_plan->output_channels = out_plan->weight->dimensions[0];
    if (out_plan->input_channels % out_plan->group != 0u ||
        out_plan->output_channels % out_plan->group != 0u ||
        out_plan->output->dimensions[1] != out_plan->output_channels) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    out_plan->inputs_per_group = out_plan->input_channels / out_plan->group;
    out_plan->outputs_per_group = out_plan->output_channels / out_plan->group;
    if (out_plan->weight->dimensions[1] != out_plan->inputs_per_group ||
        (uint64_t)out_plan->inputs_per_group * UINT64_C(65025) > INT32_MAX) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    out_plan->input_blocks = (out_plan->inputs_per_group + 3u) / 4u;
    out_plan->output_blocks = (out_plan->outputs_per_group + 3u) / 4u;

    for (axis = 0u; axis < out_plan->spatial_rank; ++axis) {
        uint64_t effective;
        uint64_t padded;
        uint64_t expected;
        if (out_plan->kernel_shape[axis] <= 0 ||
            out_plan->strides[axis] <= 0 ||
            out_plan->dilations[axis] <= 0 || out_plan->pads[axis] < 0 ||
            out_plan->pads[axis + out_plan->spatial_rank] < 0 ||
            out_plan->kernel_shape[axis] !=
                out_plan->weight->dimensions[axis + 2u]) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        if ((uint64_t)out_plan->kernel_elements *
                (uint64_t)out_plan->kernel_shape[axis] > UINT32_MAX ||
            (uint64_t)out_plan->output_spatial *
                out_plan->output->dimensions[axis + 2u] > UINT32_MAX) {
            return CAMPP_STATUS_NOT_IMPLEMENTED;
        }
        out_plan->kernel_elements *= (uint32_t)out_plan->kernel_shape[axis];
        out_plan->output_spatial *= out_plan->output->dimensions[axis + 2u];
        effective = (uint64_t)out_plan->dilations[axis]
            * (uint64_t)(out_plan->kernel_shape[axis] - 1) + 1u;
        padded = (uint64_t)out_plan->input->dimensions[axis + 2u]
            + (uint64_t)out_plan->pads[axis]
            + (uint64_t)out_plan->pads[axis + out_plan->spatial_rank];
        if (padded < effective) return CAMPP_STATUS_SHAPE_MISMATCH;
        expected = (padded - effective) /
            (uint64_t)out_plan->strides[axis] + 1u;
        if (expected != out_plan->output->dimensions[axis + 2u]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    if (out_plan->kernel_elements > CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS ||
        (uint64_t)out_plan->kernel_elements * out_plan->inputs_per_group *
            UINT64_C(65025) > INT32_MAX) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    if (out_plan->weight->storage_span_bytes !=
        (uint64_t)out_plan->group * out_plan->output_blocks *
            out_plan->kernel_elements * out_plan->input_blocks * 16u) {
        return CAMPP_STATUS_CORRUPT_WEIGHTS;
    }
    {
        const uint64_t scale_count =
            campp_tensor_view_element_count(&inputs[4]);
        const uint64_t zero_count =
            campp_tensor_view_element_count(&inputs[5]);
        if ((scale_count != 1u && scale_count != out_plan->output_channels) ||
            (zero_count != 1u && zero_count != out_plan->output_channels) ||
            (input_count == 9u &&
             campp_tensor_view_element_count(&inputs[8]) !=
                out_plan->output_channels)) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        out_plan->scalar_weight_scale = scale_count == 1u;
        out_plan->scalar_weight_zero = zero_count == 1u;
        out_plan->has_bias = input_count == 9u;
    }

    status = campp_reference_read_f32(&inputs[1], 0u, &out_plan->input_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_quantized(
        &inputs[2], 0u, &out_plan->input_zero);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_f32(&inputs[6], 0u, &out_plan->output_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_quantized(
        &inputs[7], 0u, &out_plan->output_zero);
    if (status != CAMPP_STATUS_OK) return status;
    if (!(out_plan->input_scale > 0.0f) ||
        !(out_plan->output_scale > 0.0f) ||
        !isfinite(out_plan->input_scale) ||
        !isfinite(out_plan->output_scale)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    if (!campp_qconv_address_plan_create(
            out_plan->input, out_plan->output, out_plan->spatial_rank,
            out_plan->strides, out_plan->dilations, out_plan->pads,
            &out_plan->address)) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }

    out_plan->direct_channel_store = true;
    if (out_plan->spatial_rank == 1u && out_plan->kernel_shape[0] == 1 &&
        out_plan->dilations[0] == 1) {
        out_plan->preferred_path = CAMPP_QCONV_V4_PATH_1X1;
    } else if (out_plan->spatial_rank == 2u &&
               out_plan->kernel_shape[0] == 3 &&
               out_plan->kernel_shape[1] == 3 &&
               out_plan->dilations[0] == 1 &&
               out_plan->dilations[1] == 1) {
        out_plan->preferred_path = CAMPP_QCONV_V4_PATH_3X3_INTERIOR;
    } else {
        out_plan->preferred_path = CAMPP_QCONV_V4_PATH_GENERIC;
    }
    return CAMPP_STATUS_OK;
}
