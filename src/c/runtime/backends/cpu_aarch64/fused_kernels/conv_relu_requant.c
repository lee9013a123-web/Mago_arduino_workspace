/* Quantized CAM++ epilogues and QuantizeLinear -> packed QLinearConv. */

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_aarch64/fused_kernels/fused_quant_qconv_internal.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "backends/cpu_reference/reference_kernel_utils.h"
#include "internal/runtime_model.h"

#if defined(CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS)
#include "campp_profill/optimization/stage_probe.h"
#else
#define CAMPP_OPTIMIZATION_STAGE_BEGIN(variable) ((void)0)
#define CAMPP_OPTIMIZATION_STAGE_END(stage, variable) ((void)0)
#endif

static CamppStatus campp_fused_read_scalar_f32(
    const CamppTensorView *view, float *out_value)
{
    if (view->dtype != CAMPP_DTYPE_FLOAT32 ||
        campp_tensor_view_element_count(view) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_reference_read_f32(view, 0u, out_value);
}

static CamppStatus campp_fused_read_scalar_quant(
    const CamppTensorView *view, int32_t *out_value)
{
    if (campp_tensor_view_element_count(view) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_reference_read_quantized(view, 0u, out_value);
}

static CamppStatus campp_fused_quantized_value(
    CamppTensorView *output, uint64_t index, float value,
    float scale, int32_t zero_point)
{
    float scaled;
    int64_t rounded;

    if (!(scale > 0.0f) || !isfinite(scale) || !isfinite(value)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    scaled = value / scale;
    if (scaled <= -2147483648.0f) {
        rounded = INT32_MIN;
    } else if (scaled >= 2147483520.0f) {
        rounded = INT32_MAX;
    } else {
        rounded = (int64_t)nearbyintf(scaled);
    }
    return campp_reference_write_quantized(
        output, index, rounded + zero_point);
}

static CamppStatus campp_fused_dequantized_value(
    const CamppTensorView *input, uint64_t index,
    float scale, int32_t zero_point, float *out_value)
{
    int32_t value;
    CamppStatus status = campp_reference_read_quantized(input, index, &value);
    if (status != CAMPP_STATUS_OK) return status;
    *out_value = (float)(value - zero_point) * scale;
    return CAMPP_STATUS_OK;
}

static uint64_t campp_fused_broadcast_offset(
    const CamppTensorView *input, const CamppTensorView *output,
    const uint32_t output_coordinates[CAMPP_TENSOR_MAX_RANK])
{
    uint32_t input_coordinates[CAMPP_TENSOR_MAX_RANK] = {0u, 0u, 0u, 0u};
    const uint8_t rank_delta = (uint8_t)(output->rank - input->rank);
    uint8_t axis;
    for (axis = 0u; axis < input->rank; ++axis) {
        input_coordinates[axis] = input->dimensions[axis] == 1u
            ? 0u : output_coordinates[axis + rank_delta];
    }
    return campp_tensor_view_byte_offset(input, input_coordinates);
}

static CamppStatus campp_fused_validate_broadcast(
    const CamppTensorView *input, const CamppTensorView *output)
{
    uint8_t axis;
    uint8_t delta;
    if (input->rank > output->rank) return CAMPP_STATUS_SHAPE_MISMATCH;
    delta = (uint8_t)(output->rank - input->rank);
    for (axis = 0u; axis < input->rank; ++axis) {
        const uint32_t dimension = input->dimensions[axis];
        const uint32_t target = output->dimensions[axis + delta];
        if (dimension != 1u && dimension != target) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_fused_quant_qlinear_conv_scratch_bytes(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    size_t *out_bytes)
{
    const CamppTensorDescriptor *input;
    CamppStatus status;
    if (model == NULL || op == NULL || out_bytes == NULL || op->input_count < 1u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_runtime_model_tensor(model, op->input_tensor_ids[0], &input);
    if (status != CAMPP_STATUS_OK) return status;
    if (input->dtype != CAMPP_DTYPE_FLOAT32 ||
        input->storage_span_bytes % sizeof(float) != 0u ||
        input->storage_span_bytes / sizeof(float) > SIZE_MAX) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    *out_bytes = (size_t)(input->storage_span_bytes / sizeof(float));
    return CAMPP_STATUS_OK;
}

CamppStatus campp_fused_quantize_input_scalar(
    const CamppTensorView *input, CamppTensorView *output,
    float scale, int32_t zero_point)
{
    uint64_t index;
    CamppStatus status;

    if (input == NULL || output == NULL ||
        input->dtype != CAMPP_DTYPE_FLOAT32 ||
        (output->dtype != CAMPP_DTYPE_UINT8 &&
         output->dtype != CAMPP_DTYPE_INT8) ||
        campp_tensor_view_element_count(input) !=
            campp_tensor_view_element_count(output)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    for (index = 0u; index < campp_tensor_view_element_count(input); ++index) {
        float value;
        status = campp_reference_read_f32(input, index, &value);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_fused_quantized_value(
            output, index, value, scale, zero_point);
        if (status != CAMPP_STATUS_OK) return status;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_fused_quant_qlinear_conv_with_components(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size, CamppFusedInputQuantizeRun quantize_run,
    CamppKernelRun qconv_run)
{
    CamppTensorView quantized;
    CamppTensorView qconv_inputs[CAMPP_OPERATOR_INPUT_CAPACITY];
    size_t required;
    float scale;
    int32_t zero;
    uint8_t axis;
    CamppStatus status;

    if (quantize_run == NULL || qconv_run == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_reference_validate_invocation(
        inputs, input_count, 8u, 9u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_quant_qlinear_conv_scratch_bytes(
        model, op, &required);
    if (status != CAMPP_STATUS_OK) return status;
    if (scratch == NULL || scratch_size < required ||
        inputs[0].dtype != CAMPP_DTYPE_FLOAT32) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_fused_read_scalar_f32(&inputs[1], &scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_quant(&inputs[2], &zero);
    if (status != CAMPP_STATUS_OK) return status;

    quantized = inputs[0];
    quantized.data = scratch;
    quantized.dtype = inputs[2].dtype;
    quantized.logical_byte_size /= sizeof(float);
    quantized.storage_span_bytes /= sizeof(float);
    for (axis = 0u; axis < quantized.rank; ++axis) {
        if (quantized.byte_strides[axis] % sizeof(float) != 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        quantized.byte_strides[axis] /= sizeof(float);
    }
    CAMPP_OPTIMIZATION_STAGE_BEGIN(quantize_started_ns);
    status = quantize_run(&inputs[0], &quantized, scale, zero);
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_FUSED_INPUT_QUANTIZE, quantize_started_ns);
    if (status != CAMPP_STATUS_OK) return status;
    qconv_inputs[0] = quantized;
    for (axis = 1u; axis < input_count; ++axis) qconv_inputs[axis] = inputs[axis];
    CAMPP_OPTIMIZATION_STAGE_BEGIN(qconv_started_ns);
    status = qconv_run(
        model, op, qconv_inputs, input_count, outputs, output_count,
        NULL, 0u);
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_FUSED_QCONV, qconv_started_ns);
    return status;
}

CamppStatus campp_fused_quant_qlinear_conv_with_runner(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size, CamppKernelRun qconv_run)
{
    return campp_fused_quant_qlinear_conv_with_components(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size, campp_fused_quantize_input_scalar,
        qconv_run);
}

CamppStatus campp_fused_quant_qlinear_conv_o4i4(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    return campp_fused_quant_qlinear_conv_with_runner(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size, campp_aarch64_qlinear_conv_o4i4);
}

CamppStatus campp_fused_dequant_relu_quant(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    float old_scale;
    float new_scale;
    int32_t old_zero;
    int32_t new_zero;
    uint64_t index;
    CamppStatus status;
    (void)model;
    (void)op;
    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 5u, 5u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_f32(&inputs[1], &old_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_quant(&inputs[2], &old_zero);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_f32(&inputs[3], &new_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_quant(&inputs[4], &new_zero);
    if (status != CAMPP_STATUS_OK) return status;
    if (!campp_reference_shapes_equal(&inputs[0], &outputs[0])) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    for (index = 0u; index < campp_tensor_view_element_count(&outputs[0]); ++index) {
        float value;
        status = campp_fused_dequantized_value(
            &inputs[0], index, old_scale, old_zero, &value);
        if (status != CAMPP_STATUS_OK) return status;
        if (value < 0.0f) value = 0.0f;
        status = campp_fused_quantized_value(
            &outputs[0], index, value, new_scale, new_zero);
        if (status != CAMPP_STATUS_OK) return status;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_fused_dequant_sigmoid_mul(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    float scale;
    int32_t zero;
    uint64_t index;
    CamppStatus status;
    (void)model;
    (void)op;
    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 4u, 4u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    if (inputs[3].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        !campp_reference_shapes_equal(&inputs[0], &outputs[0])) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    status = campp_fused_validate_broadcast(&inputs[3], &outputs[0]);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_f32(&inputs[1], &scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_quant(&inputs[2], &zero);
    if (status != CAMPP_STATUS_OK) return status;
    for (index = 0u; index < campp_tensor_view_element_count(&outputs[0]); ++index) {
        uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
        uint64_t other_offset;
        uint64_t output_offset;
        float value;
        float sigmoid;
        float other;
        float result;
        status = campp_fused_dequantized_value(
            &inputs[0], index, scale, zero, &value);
        if (status != CAMPP_STATUS_OK) return status;
        if (value >= 0.0f) {
            sigmoid = 1.0f / (1.0f + expf(-value));
        } else {
            const float exponential = expf(value);
            sigmoid = exponential / (1.0f + exponential);
        }
        campp_reference_unravel_index(
            index, outputs[0].rank, outputs[0].dimensions, coordinates);
        other_offset = campp_fused_broadcast_offset(
            &inputs[3], &outputs[0], coordinates);
        output_offset = campp_reference_offset_for_linear(&outputs[0], index);
        memcpy(&other, (const uint8_t *)inputs[3].data + other_offset, sizeof(other));
        result = other * sigmoid;
        memcpy((uint8_t *)outputs[0].data + output_offset, &result, sizeof(result));
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_fused_qdq_elementwise_quant(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    float old_scale;
    float new_scale;
    int32_t old_zero;
    int32_t new_zero;
    uint64_t index;
    CamppStatus status;
    (void)model;
    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 6u, 6u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    if (inputs[3].dtype != CAMPP_DTYPE_FLOAT32 ||
        (op->opcode != CAMPP_OP_ADD && op->opcode != CAMPP_OP_MUL)) {
        return CAMPP_STATUS_UNSUPPORTED_OPCODE;
    }
    status = campp_fused_validate_broadcast(&inputs[3], &outputs[0]);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_f32(&inputs[1], &old_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_quant(&inputs[2], &old_zero);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_f32(&inputs[4], &new_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_read_scalar_quant(&inputs[5], &new_zero);
    if (status != CAMPP_STATUS_OK) return status;
    for (index = 0u; index < campp_tensor_view_element_count(&outputs[0]); ++index) {
        uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
        uint64_t other_offset;
        float value;
        float other;
        status = campp_fused_dequantized_value(
            &inputs[0], index, old_scale, old_zero, &value);
        if (status != CAMPP_STATUS_OK) return status;
        campp_reference_unravel_index(
            index, outputs[0].rank, outputs[0].dimensions, coordinates);
        other_offset = campp_fused_broadcast_offset(
            &inputs[3], &outputs[0], coordinates);
        memcpy(&other, (const uint8_t *)inputs[3].data + other_offset, sizeof(other));
        value = op->opcode == CAMPP_OP_ADD ? value + other : value * other;
        status = campp_fused_quantized_value(
            &outputs[0], index, value, new_scale, new_zero);
        if (status != CAMPP_STATUS_OK) return status;
    }
    return CAMPP_STATUS_OK;
}
