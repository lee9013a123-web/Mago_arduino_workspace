#include "bn_v2_execution_plan.h"

#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "backends/cpu_reference/reference_kernel_utils.h"
#include "bn_iteration_fastpath.h"

static int campp_bn_v2_ranges_overlap(
    const CamppTensorView *left, const CamppTensorView *right)
{
    const uintptr_t left_begin = (uintptr_t)left->data;
    const uintptr_t right_begin = (uintptr_t)right->data;
    uintptr_t left_end;
    uintptr_t right_end;

    if (left->storage_span_bytes > UINTPTR_MAX - left_begin ||
        right->storage_span_bytes > UINTPTR_MAX - right_begin) {
        return 1;
    }
    left_end = left_begin + (uintptr_t)left->storage_span_bytes;
    right_end = right_begin + (uintptr_t)right->storage_span_bytes;
    return left_begin < right_end && right_begin < left_end;
}

static CamppStatus campp_bn_v2_read_zero(
    const CamppTensorView *view, uint8_t output_dtype, int32_t *out_value)
{
    if (view == NULL) {
        *out_value = 0;
        return CAMPP_STATUS_OK;
    }
    if (view->dtype != output_dtype ||
        campp_tensor_view_element_count(view) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_reference_read_quantized(view, 0u, out_value);
}

CamppStatus campp_bn_v2_execution_plan_create(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppBnV2ExecutionPlan *out_plan)
{
    const CamppTensorView *zero_point =
        input_count == 7u ? &inputs[6] : NULL;
    const CamppTensorView *parameters[4];
    CamppBnIterationPlan iteration;
    double epsilon;
    uint8_t parameter;
    CamppStatus status;

    if (out_plan == NULL) return CAMPP_STATUS_INVALID_ARGUMENT;
    memset(out_plan, 0, sizeof(*out_plan));
    status = campp_reference_validate_invocation(
        inputs, input_count, 6u, 7u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    if (!campp_bn_iteration_plan_create(&inputs[0], &outputs[0], &iteration) ||
        !iteration.channel_contiguous ||
        campp_bn_v2_ranges_overlap(&inputs[0], &outputs[0])) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    parameters[0] = &inputs[1];
    parameters[1] = &inputs[2];
    parameters[2] = &inputs[3];
    parameters[3] = &inputs[4];
    for (parameter = 0u; parameter < 4u; ++parameter) {
        if (parameters[parameter]->dtype != CAMPP_DTYPE_FLOAT32 ||
            parameters[parameter]->rank != 1u ||
            parameters[parameter]->dimensions[0] != iteration.channels) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    if (inputs[5].dtype != CAMPP_DTYPE_FLOAT32 ||
        campp_tensor_view_element_count(&inputs[5]) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    status = campp_runtime_model_attribute_double(
        model, op, CAMPP_ATTR_EPSILON, &epsilon);
    if (status != CAMPP_STATUS_OK) return status;
    if (epsilon < 0.0) return CAMPP_STATUS_CORRUPT_PLAN;
    status = campp_reference_read_f32(
        &inputs[5], 0u, &out_plan->quant_scale);
    if (status != CAMPP_STATUS_OK || !(out_plan->quant_scale > 0.0f) ||
        !isfinite(out_plan->quant_scale)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    status = campp_bn_v2_read_zero(
        zero_point, outputs[0].dtype, &out_plan->quant_zero);
    if (status != CAMPP_STATUS_OK) return status;
    if (fegetround() != FE_TONEAREST) return CAMPP_STATUS_NOT_IMPLEMENTED;

    out_plan->input = &inputs[0];
    out_plan->output = &outputs[0];
    out_plan->scale = &inputs[1];
    out_plan->bias = &inputs[2];
    out_plan->mean = &inputs[3];
    out_plan->variance = &inputs[4];
    out_plan->batches = iteration.batches;
    out_plan->channels = iteration.channels;
    out_plan->height = iteration.height;
    out_plan->width = iteration.width;
    memcpy(
        out_plan->input_strides, iteration.input_strides,
        sizeof(out_plan->input_strides));
    memcpy(
        out_plan->output_strides, iteration.output_strides,
        sizeof(out_plan->output_strides));
    out_plan->epsilon = (float)epsilon;
    out_plan->output_dtype = outputs[0].dtype;
    return CAMPP_STATUS_OK;
}

const float *campp_bn_v2_input_pointer(
    const CamppBnV2ExecutionPlan *plan, uint32_t batch,
    uint32_t height, uint32_t width)
{
    const uint64_t offset =
        (uint64_t)batch * plan->input_strides[0] +
        (uint64_t)height * plan->input_strides[2] +
        (uint64_t)width * plan->input_strides[3];
    return (const float *)((const uint8_t *)plan->input->data + offset);
}

uint8_t *campp_bn_v2_output_pointer(
    const CamppBnV2ExecutionPlan *plan, uint32_t batch,
    uint32_t height, uint32_t width)
{
    const uint64_t offset =
        (uint64_t)batch * plan->output_strides[0] +
        (uint64_t)height * plan->output_strides[2] +
        (uint64_t)width * plan->output_strides[3];
    return (uint8_t *)plan->output->data + offset;
}
