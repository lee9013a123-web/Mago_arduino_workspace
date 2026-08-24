#include "reference_kernels.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "internal/runtime_model.h"
#include "reference_kernel_utils.h"

static CamppStatus campp_qconv_attribute(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, int64_t *values, uint8_t capacity, uint8_t expected_count)
{
    uint8_t count;
    CamppStatus status = campp_runtime_model_attribute_ints(
        model, op, key, values, capacity, &count);
    if (status != CAMPP_STATUS_OK) return status;
    return count == expected_count ? CAMPP_STATUS_OK
                                   : CAMPP_STATUS_CORRUPT_PLAN;
}

static CamppStatus campp_qconv_read_at(
    const CamppTensorView *view,
    const uint32_t coordinates[CAMPP_TENSOR_MAX_RANK], int32_t *out_value)
{
    const uint64_t offset = campp_tensor_view_byte_offset(view, coordinates);
    if (view->dtype == CAMPP_DTYPE_UINT8) {
        *out_value = *((const uint8_t *)view->data + offset);
        return CAMPP_STATUS_OK;
    }
    if (view->dtype == CAMPP_DTYPE_INT8) {
        int8_t value;
        memcpy(&value, (const uint8_t *)view->data + offset, sizeof(value));
        *out_value = value;
        return CAMPP_STATUS_OK;
    }
    return CAMPP_STATUS_UNSUPPORTED_DTYPE;
}

static CamppStatus campp_qconv_validate_parameters(
    const CamppTensorView *inputs, uint8_t input_count,
    const CamppTensorView *output, uint32_t output_channels)
{
    const uint64_t weight_scale_count =
        campp_tensor_view_element_count(&inputs[4]);
    const uint64_t weight_zero_count =
        campp_tensor_view_element_count(&inputs[5]);

    if (campp_tensor_view_element_count(&inputs[1]) != 1u ||
        inputs[1].dtype != CAMPP_DTYPE_FLOAT32 ||
        campp_tensor_view_element_count(&inputs[2]) != 1u ||
        inputs[2].dtype != inputs[0].dtype ||
        inputs[4].dtype != CAMPP_DTYPE_FLOAT32 ||
        (weight_scale_count != 1u && weight_scale_count != output_channels) ||
        (weight_scale_count != 1u && inputs[4].rank != 1u) ||
        inputs[5].dtype != inputs[3].dtype ||
        (weight_zero_count != 1u && weight_zero_count != output_channels) ||
        (weight_zero_count != 1u && inputs[5].rank != 1u) ||
        campp_tensor_view_element_count(&inputs[6]) != 1u ||
        inputs[6].dtype != CAMPP_DTYPE_FLOAT32 ||
        campp_tensor_view_element_count(&inputs[7]) != 1u ||
        inputs[7].dtype != output->dtype) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    if (input_count == 9u &&
        (inputs[8].dtype != CAMPP_DTYPE_INT32 || inputs[8].rank != 1u ||
         inputs[8].dimensions[0] != output_channels)) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_qlinear_conv(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[2];
    int64_t dilations[2];
    int64_t group_value[1];
    const CamppTensorView *x;
    const CamppTensorView *weight;
    CamppTensorView *y;
    uint8_t spatial_rank;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t channels_per_group;
    uint32_t outputs_per_group;
    uint32_t group;
    float x_scale;
    float y_scale;
    int32_t x_zero;
    int32_t y_zero;
    uint64_t output_index;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 8u, 9u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    x = &inputs[0];
    weight = &inputs[3];
    y = &outputs[0];
    if ((x->dtype != CAMPP_DTYPE_UINT8 && x->dtype != CAMPP_DTYPE_INT8) ||
        (weight->dtype != CAMPP_DTYPE_UINT8 &&
         weight->dtype != CAMPP_DTYPE_INT8) ||
        (y->dtype != CAMPP_DTYPE_UINT8 && y->dtype != CAMPP_DTYPE_INT8)) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (x->rank < 3u || x->rank > 4u || weight->rank != x->rank ||
        y->rank != x->rank || y->dimensions[0] != x->dimensions[0]) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    spatial_rank = (uint8_t)(x->rank - 2u);
    input_channels = x->dimensions[1];
    output_channels = weight->dimensions[0];
    if (y->dimensions[1] != output_channels) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_KERNEL_SHAPE, kernel_shape, 2u, spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_PADS, pads, 4u,
        (uint8_t)(spatial_rank * 2u));
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_STRIDES, strides, 2u, spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_DILATIONS, dilations, 2u, spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_GROUP, group_value, 1u, 1u);
    if (status != CAMPP_STATUS_OK) return status;
    if (group_value[0] <= 0 || group_value[0] > UINT32_MAX) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    group = (uint32_t)group_value[0];
    if (input_channels % group != 0u || output_channels % group != 0u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    channels_per_group = input_channels / group;
    outputs_per_group = output_channels / group;
    if (weight->dimensions[1] != channels_per_group) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    {
        uint8_t spatial_axis;
        for (spatial_axis = 0u; spatial_axis < spatial_rank; ++spatial_axis) {
            uint64_t effective_kernel;
            uint64_t padded_input;
            uint64_t numerator;
            uint64_t expected_output;
            if (kernel_shape[spatial_axis] <= 0 ||
                kernel_shape[spatial_axis] !=
                    weight->dimensions[spatial_axis + 2u] ||
                pads[spatial_axis] < 0 ||
                pads[spatial_axis + spatial_rank] < 0 ||
                strides[spatial_axis] <= 0 ||
                dilations[spatial_axis] <= 0) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            if (kernel_shape[spatial_axis] > 1 &&
                (uint64_t)dilations[spatial_axis] >
                    (UINT64_MAX - 1u) /
                        (uint64_t)(kernel_shape[spatial_axis] - 1)) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            effective_kernel = (uint64_t)dilations[spatial_axis] *
                (uint64_t)(kernel_shape[spatial_axis] - 1) + 1u;
            padded_input = x->dimensions[spatial_axis + 2u];
            if ((uint64_t)pads[spatial_axis] >
                    UINT64_MAX - padded_input) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            padded_input += (uint64_t)pads[spatial_axis];
            if ((uint64_t)pads[spatial_axis + spatial_rank] >
                    UINT64_MAX - padded_input) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            padded_input +=
                (uint64_t)pads[spatial_axis + spatial_rank];
            if (padded_input < effective_kernel) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
            numerator = padded_input - effective_kernel;
            expected_output =
                numerator / (uint64_t)strides[spatial_axis] + 1u;
            if (expected_output != y->dimensions[spatial_axis + 2u]) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
        }
    }
    status = campp_qconv_validate_parameters(
        inputs, input_count, y, output_channels);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_f32(&inputs[1], 0u, &x_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_quantized(&inputs[2], 0u, &x_zero);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_f32(&inputs[6], 0u, &y_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_quantized(&inputs[7], 0u, &y_zero);
    if (status != CAMPP_STATUS_OK) return status;
    if (!(x_scale > 0.0f) || !(y_scale > 0.0f) ||
        !isfinite(x_scale) || !isfinite(y_scale)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }

    for (output_index = 0u;
         output_index < campp_tensor_view_element_count(y); ++output_index) {
        uint32_t output_coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t input_coordinates[CAMPP_TENSOR_MAX_RANK] = {0u, 0u, 0u, 0u};
        uint32_t weight_coordinates[CAMPP_TENSOR_MAX_RANK] = {0u, 0u, 0u, 0u};
        uint32_t output_channel;
        uint32_t group_index;
        uint64_t kernel_elements = 1u;
        uint64_t kernel_index;
        uint32_t input_channel;
        int64_t accumulator = 0;
        uint64_t parameter_index;
        float weight_scale;
        float multiplier;
        int32_t weight_zero;
        int64_t rounded;

        campp_reference_unravel_index(
            output_index, y->rank, y->dimensions, output_coordinates);
        output_channel = output_coordinates[1];
        group_index = output_channel / outputs_per_group;
        input_coordinates[0] = output_coordinates[0];
        weight_coordinates[0] = output_channel;
        {
            uint8_t spatial_axis;
            for (spatial_axis = 0u; spatial_axis < spatial_rank;
                 ++spatial_axis) {
                kernel_elements *= (uint64_t)kernel_shape[spatial_axis];
            }
        }
        parameter_index =
            campp_tensor_view_element_count(&inputs[4]) == 1u
                ? 0u : output_channel;
        status = campp_reference_read_f32(
            &inputs[4], parameter_index, &weight_scale);
        if (status != CAMPP_STATUS_OK) return status;
        parameter_index =
            campp_tensor_view_element_count(&inputs[5]) == 1u
                ? 0u : output_channel;
        status = campp_reference_read_quantized(
            &inputs[5], parameter_index, &weight_zero);
        if (status != CAMPP_STATUS_OK) return status;
        if (!(weight_scale > 0.0f) || !isfinite(weight_scale)) {
            return CAMPP_STATUS_KERNEL_FAILED;
        }

        for (input_channel = 0u; input_channel < channels_per_group;
             ++input_channel) {
            input_coordinates[1] =
                group_index * channels_per_group + input_channel;
            weight_coordinates[1] = input_channel;
            for (kernel_index = 0u; kernel_index < kernel_elements;
                 ++kernel_index) {
                uint64_t remaining = kernel_index;
                bool valid = true;
                uint8_t spatial_axis;

                for (spatial_axis = spatial_rank; spatial_axis > 0u;
                     --spatial_axis) {
                    const uint8_t index_axis = (uint8_t)(spatial_axis - 1u);
                    const uint32_t kernel_coordinate =
                        (uint32_t)(remaining %
                                  (uint64_t)kernel_shape[index_axis]);
                    const int64_t input_coordinate =
                        (int64_t)output_coordinates[index_axis + 2u] *
                            strides[index_axis] +
                        (int64_t)kernel_coordinate * dilations[index_axis] -
                        pads[index_axis];
                    remaining /= (uint64_t)kernel_shape[index_axis];
                    weight_coordinates[index_axis + 2u] = kernel_coordinate;
                    if (input_coordinate < 0 ||
                        input_coordinate >=
                            (int64_t)x->dimensions[index_axis + 2u]) {
                        valid = false;
                    } else {
                        input_coordinates[index_axis + 2u] =
                            (uint32_t)input_coordinate;
                    }
                }
                if (valid) {
                    int32_t x_value;
                    int32_t weight_value;
                    status = campp_qconv_read_at(
                        x, input_coordinates, &x_value);
                    if (status != CAMPP_STATUS_OK) return status;
                    status = campp_qconv_read_at(
                        weight, weight_coordinates, &weight_value);
                    if (status != CAMPP_STATUS_OK) return status;
                    accumulator +=
                        (int64_t)(x_value - x_zero) *
                        (int64_t)(weight_value - weight_zero);
                }
            }
        }
        if (input_count == 9u) {
            int32_t bias;
            status = campp_reference_read_quantized(
                &inputs[8], output_channel, &bias);
            if (status != CAMPP_STATUS_OK) return status;
            accumulator += bias;
        }
        if (accumulator < INT32_MIN || accumulator > INT32_MAX) {
            return CAMPP_STATUS_KERNEL_FAILED;
        }
        multiplier = x_scale * weight_scale / y_scale;
        if (!isfinite(multiplier)) return CAMPP_STATUS_KERNEL_FAILED;
        /* ONNX QLinearConv은 accumulator에 multiplier를 곱한 값을 nearest-even으로
         * 반올림한 뒤 정수 zero point를 더한다. zero point를 반올림 전에 더하면
         * float32가 소수부를 잃어 없던 .5 tie가 생기고 결과가 1만큼 어긋난다. */
        {
            const float scaled = (float)(int32_t)accumulator * multiplier;
            if (scaled <= -2147483648.0f) {
                rounded = INT32_MIN;
            } else if (scaled >= 2147483520.0f) {
                rounded = INT32_MAX;
            } else {
                rounded = (int64_t)nearbyintf(scaled);
            }
            rounded += y_zero;
        }
        status = campp_reference_write_quantized(y, output_index, rounded);
        if (status != CAMPP_STATUS_OK) return status;
    }
    return CAMPP_STATUS_OK;
}
