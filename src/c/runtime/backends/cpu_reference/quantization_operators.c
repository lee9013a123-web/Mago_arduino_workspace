#include "reference_kernels.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "internal/runtime_model.h"
#include "reference_kernel_utils.h"

#if defined(CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS)
#include "campp_profill/optimization/stage_probe.h"
#else
#define CAMPP_OPTIMIZATION_STAGE_BEGIN(variable) ((void)0)
#define CAMPP_OPTIMIZATION_STAGE_END(stage, variable) ((void)0)
#endif

static CamppStatus campp_quantization_axis(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint8_t tensor_rank, uint8_t *out_axis)
{
    int64_t axis_value[1];
    uint8_t count;
    CamppStatus status = campp_runtime_model_attribute_ints(
        model, op, CAMPP_ATTR_AXIS, axis_value, 1u, &count);

    if (status == CAMPP_STATUS_MISSING_ATTRIBUTE) {
        axis_value[0] = 1;
        count = 1u;
        status = CAMPP_STATUS_OK;
    }
    if (status != CAMPP_STATUS_OK) return status;
    if (count != 1u) return CAMPP_STATUS_CORRUPT_PLAN;
    return campp_reference_normalize_axis(axis_value[0], tensor_rank, out_axis);
}

static CamppStatus campp_validate_quantization_parameters(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *data, const CamppTensorView *scale,
    const CamppTensorView *zero_point, uint8_t zero_dtype,
    uint8_t *out_axis, bool *out_per_axis)
{
    const uint64_t scale_count = campp_tensor_view_element_count(scale);
    uint64_t zero_count = 1u;
    CamppStatus status;

    if (scale->dtype != CAMPP_DTYPE_FLOAT32 || scale->rank > 1u ||
        (scale_count != 1u && scale->rank != 1u)) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    if (zero_point != NULL) {
        zero_count = campp_tensor_view_element_count(zero_point);
        if (zero_point->dtype != zero_dtype || zero_point->rank > 1u ||
            zero_count != scale_count ||
            (zero_count != 1u && zero_point->rank != 1u)) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    if (scale_count == 1u) {
        *out_axis = 0u;
        *out_per_axis = false;
        return CAMPP_STATUS_OK;
    }
    status = campp_quantization_axis(model, op, data->rank, out_axis);
    if (status != CAMPP_STATUS_OK) return status;
    if (scale_count != data->dimensions[*out_axis]) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    *out_per_axis = true;
    return CAMPP_STATUS_OK;
}

static uint64_t campp_quantization_parameter_index(
    uint64_t data_index, const CamppTensorView *data, uint8_t axis,
    bool per_axis)
{
    uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
    if (!per_axis) return 0u;
    campp_reference_unravel_index(
        data_index, data->rank, data->dimensions, coordinates);
    return coordinates[axis];
}

static CamppStatus campp_read_zero_point(
    const CamppTensorView *zero_point, uint64_t index, int32_t *out_value)
{
    if (zero_point == NULL) {
        *out_value = 0;
        return CAMPP_STATUS_OK;
    }
    return campp_reference_read_quantized(zero_point, index, out_value);
}

CamppStatus campp_reference_quantize_linear(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    const CamppTensorView *zero_point = input_count == 3u ? &inputs[2] : NULL;
    uint8_t axis;
    bool per_axis;
    uint64_t count;
    uint64_t index;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 2u, 3u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    if (inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        (outputs[0].dtype != CAMPP_DTYPE_UINT8 &&
         outputs[0].dtype != CAMPP_DTYPE_INT8)) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (!campp_reference_shapes_equal(&inputs[0], &outputs[0])) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    status = campp_validate_quantization_parameters(
        model, op, &inputs[0], &inputs[1], zero_point, outputs[0].dtype,
        &axis, &per_axis);
    if (status != CAMPP_STATUS_OK) return status;

    count = campp_tensor_view_element_count(&outputs[0]);
    for (index = 0u; index < count; ++index) {
        const uint64_t parameter_index = campp_quantization_parameter_index(
            index, &inputs[0], axis, per_axis);
        float value;
        float scale;
        float scaled;
        int32_t zero;
        int64_t rounded;

        status = campp_reference_read_f32(&inputs[0], index, &value);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_read_f32(
            &inputs[1], parameter_index, &scale);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_read_zero_point(zero_point, parameter_index, &zero);
        if (status != CAMPP_STATUS_OK) return status;
        if (!(scale > 0.0f) || !isfinite(scale) || !isfinite(value)) {
            return CAMPP_STATUS_KERNEL_FAILED;
        }
        /* ONNX QuantizeLinear은 x/scale을 nearest-even으로 반올림한 뒤
         * 정수 zero point를 더한다. zero point를 반올림 전에 더하면 .5 tie에서
         * zero point의 홀짝에 따라 결과가 달라진다. */
        scaled = value / scale;
        if (scaled <= -2147483648.0f) {
            rounded = INT32_MIN;
        } else if (scaled >= 2147483520.0f) {
            rounded = INT32_MAX;
        } else {
            rounded = (int64_t)nearbyintf(scaled);
        }
        rounded += zero;
        status = campp_reference_write_quantized(&outputs[0], index, rounded);
        if (status != CAMPP_STATUS_OK) return status;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_dequantize_linear(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    const CamppTensorView *zero_point = input_count == 3u ? &inputs[2] : NULL;
    uint8_t axis;
    bool per_axis;
    uint64_t count;
    uint64_t index;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    CAMPP_OPTIMIZATION_STAGE_BEGIN(setup_started_ns);
    status = campp_reference_validate_invocation(
        inputs, input_count, 2u, 3u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    if ((inputs[0].dtype != CAMPP_DTYPE_UINT8 &&
         inputs[0].dtype != CAMPP_DTYPE_INT8) ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (!campp_reference_shapes_equal(&inputs[0], &outputs[0])) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    status = campp_validate_quantization_parameters(
        model, op, &inputs[0], &inputs[1], zero_point, inputs[0].dtype,
        &axis, &per_axis);
    if (status != CAMPP_STATUS_OK) return status;
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_DEQUANT_SETUP, setup_started_ns);

    count = campp_tensor_view_element_count(&outputs[0]);
    CAMPP_OPTIMIZATION_STAGE_BEGIN(elementwise_started_ns);
    for (index = 0u; index < count; ++index) {
        const uint64_t parameter_index = campp_quantization_parameter_index(
            index, &inputs[0], axis, per_axis);
        int32_t value;
        int32_t zero;
        float scale;
        float result;
        const uint64_t output_offset =
            campp_reference_offset_for_linear(&outputs[0], index);

        status = campp_reference_read_quantized(&inputs[0], index, &value);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_read_f32(
            &inputs[1], parameter_index, &scale);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_read_zero_point(zero_point, parameter_index, &zero);
        if (status != CAMPP_STATUS_OK) return status;
        if (!(scale > 0.0f) || !isfinite(scale)) {
            return CAMPP_STATUS_KERNEL_FAILED;
        }
        result = (float)(value - zero) * scale;
        memcpy(
            (uint8_t *)outputs[0].data + output_offset, &result,
            sizeof(result));
    }
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_DEQUANT_ELEMENTWISE, elementwise_started_ns);
    return CAMPP_STATUS_OK;
}
