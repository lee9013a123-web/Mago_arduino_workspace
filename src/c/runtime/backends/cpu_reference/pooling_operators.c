#include "reference_kernels.h"

#include <stdint.h>
#include <string.h>

#include "internal/runtime_model.h"
#include "reference_kernel_utils.h"

static CamppStatus campp_pool_attribute(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, int64_t *values, uint8_t capacity, uint8_t expected_count)
{
    uint8_t count;
    CamppStatus status = campp_runtime_model_attribute_ints(
        model, op, key, values, capacity, &count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return count == expected_count ? CAMPP_STATUS_OK
                                   : CAMPP_STATUS_CORRUPT_PLAN;
}

CamppStatus campp_reference_average_pool(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[2];
    int64_t ceil_mode[1];
    int64_t count_include_pad[1];
    uint8_t spatial_rank;
    uint64_t output_elements;
    uint64_t output_index;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 1u, 1u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (inputs[0].rank < 3u || inputs[0].rank > 4u ||
        outputs[0].rank != inputs[0].rank ||
        outputs[0].dimensions[0] != inputs[0].dimensions[0] ||
        outputs[0].dimensions[1] != inputs[0].dimensions[1]) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    spatial_rank = (uint8_t)(inputs[0].rank - 2u);
    status = campp_pool_attribute(
        model, op, CAMPP_ATTR_KERNEL_SHAPE, kernel_shape, 2u, spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_pool_attribute(
        model, op, CAMPP_ATTR_PADS, pads, 4u,
        (uint8_t)(spatial_rank * 2u));
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_pool_attribute(
        model, op, CAMPP_ATTR_STRIDES, strides, 2u, spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_pool_attribute(
        model, op, CAMPP_ATTR_CEIL_MODE, ceil_mode, 1u, 1u);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_pool_attribute(
        model, op, CAMPP_ATTR_COUNT_INCLUDE_PAD, count_include_pad, 1u, 1u);
    if (status != CAMPP_STATUS_OK) return status;
    if ((ceil_mode[0] != 0 && ceil_mode[0] != 1) ||
        (count_include_pad[0] != 0 && count_include_pad[0] != 1)) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    {
        uint8_t spatial_axis;
        for (spatial_axis = 0u; spatial_axis < spatial_rank; ++spatial_axis) {
            uint64_t padded_input;
            uint64_t stride;
            uint64_t expected_output;
            if (kernel_shape[spatial_axis] <= 0 ||
                strides[spatial_axis] <= 0 || pads[spatial_axis] < 0 ||
                pads[spatial_axis + spatial_rank] < 0) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            padded_input = inputs[0].dimensions[spatial_axis + 2u];
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
            stride = (uint64_t)strides[spatial_axis];
            if (padded_input >= (uint64_t)kernel_shape[spatial_axis]) {
                const uint64_t numerator =
                    padded_input - (uint64_t)kernel_shape[spatial_axis];
                if (ceil_mode[0] != 0) {
                    expected_output = numerator / stride +
                        (numerator % stride == 0u ? 0u : 1u) + 1u;
                } else {
                    expected_output = numerator / stride + 1u;
                }
            } else {
                const uint64_t difference =
                    (uint64_t)kernel_shape[spatial_axis] - padded_input;
                uint64_t removed_windows;
                if (ceil_mode[0] != 0) {
                    removed_windows = difference / stride;
                } else {
                    removed_windows = difference / stride +
                        (difference % stride == 0u ? 0u : 1u);
                }
                if (removed_windows >= 1u) {
                    return CAMPP_STATUS_SHAPE_MISMATCH;
                }
                expected_output = 1u;
            }
            if (ceil_mode[0] != 0) {
                if (expected_output > 0u &&
                    (expected_output - 1u > UINT64_MAX / stride ||
                     (expected_output - 1u) * stride >=
                        (uint64_t)inputs[0].dimensions[spatial_axis + 2u] +
                            (uint64_t)pads[spatial_axis])) {
                    --expected_output;
                }
            }
            if (expected_output !=
                outputs[0].dimensions[spatial_axis + 2u]) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
        }
    }

    output_elements = campp_tensor_view_element_count(&outputs[0]);
    for (output_index = 0u; output_index < output_elements; ++output_index) {
        uint32_t output_coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t input_coordinates[CAMPP_TENSOR_MAX_RANK] = {0u, 0u, 0u, 0u};
        uint64_t kernel_elements = 1u;
        uint64_t kernel_index;
        uint64_t valid_count = 0u;
        float sum = 0.0f;
        float result;
        uint64_t output_offset;
        uint8_t spatial_axis;

        campp_reference_unravel_index(
            output_index, outputs[0].rank, outputs[0].dimensions,
            output_coordinates);
        input_coordinates[0] = output_coordinates[0];
        input_coordinates[1] = output_coordinates[1];
        for (spatial_axis = 0u; spatial_axis < spatial_rank; ++spatial_axis) {
            kernel_elements *= (uint64_t)kernel_shape[spatial_axis];
        }
        for (kernel_index = 0u; kernel_index < kernel_elements; ++kernel_index) {
            uint64_t remaining = kernel_index;
            bool valid = true;

            for (spatial_axis = spatial_rank; spatial_axis > 0u;
                 --spatial_axis) {
                const uint8_t index_axis = (uint8_t)(spatial_axis - 1u);
                const int64_t kernel_coordinate =
                    (int64_t)(remaining %
                              (uint64_t)kernel_shape[index_axis]);
                const int64_t input_coordinate =
                    (int64_t)output_coordinates[index_axis + 2u] *
                        strides[index_axis] +
                    kernel_coordinate - pads[index_axis];
                remaining /= (uint64_t)kernel_shape[index_axis];
                if (input_coordinate < 0 ||
                    input_coordinate >=
                        (int64_t)inputs[0].dimensions[index_axis + 2u]) {
                    valid = false;
                } else {
                    input_coordinates[index_axis + 2u] =
                        (uint32_t)input_coordinate;
                }
            }
            if (valid) {
                const uint64_t input_offset = campp_tensor_view_byte_offset(
                    &inputs[0], input_coordinates);
                float value;
                memcpy(
                    &value, (const uint8_t *)inputs[0].data + input_offset,
                    sizeof(value));
                sum += value;
                ++valid_count;
            }
        }
        if (valid_count == 0u) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        result = sum /
            (float)(count_include_pad[0] != 0 ? kernel_elements : valid_count);
        output_offset = campp_tensor_view_byte_offset(
            &outputs[0], output_coordinates);
        memcpy(
            (uint8_t *)outputs[0].data + output_offset, &result,
            sizeof(result));
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_reduce_mean(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t axes_values[CAMPP_TENSOR_MAX_RANK];
    int64_t keepdims_value[1];
    bool reduced[CAMPP_TENSOR_MAX_RANK] = {false, false, false, false};
    uint8_t axes_count;
    uint8_t keepdims_count;
    uint8_t axis;
    uint8_t output_axis;
    uint64_t reduction_count = 1u;
    uint64_t index;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 1u, 1u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    if (inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    status = campp_runtime_model_attribute_ints(
        model, op, CAMPP_ATTR_AXES, axes_values, CAMPP_TENSOR_MAX_RANK,
        &axes_count);
    if (status == CAMPP_STATUS_MISSING_ATTRIBUTE) {
        axes_count = inputs[0].rank;
        for (axis = 0u; axis < axes_count; ++axis) axes_values[axis] = axis;
    } else if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_runtime_model_attribute_ints(
        model, op, CAMPP_ATTR_KEEPDIMS, keepdims_value, 1u,
        &keepdims_count);
    if (status != CAMPP_STATUS_OK) return status;
    if (keepdims_count != 1u ||
        (keepdims_value[0] != 0 && keepdims_value[0] != 1)) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    for (axis = 0u; axis < axes_count; ++axis) {
        uint8_t normalized;
        status = campp_reference_normalize_axis(
            axes_values[axis], inputs[0].rank, &normalized);
        if (status != CAMPP_STATUS_OK || reduced[normalized]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        reduced[normalized] = true;
        reduction_count *= inputs[0].dimensions[normalized];
    }
    output_axis = 0u;
    for (axis = 0u; axis < inputs[0].rank; ++axis) {
        if (keepdims_value[0] != 0) {
            const uint32_t expected = reduced[axis]
                ? 1u : inputs[0].dimensions[axis];
            if (outputs[0].rank != inputs[0].rank ||
                outputs[0].dimensions[axis] != expected) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
        } else if (!reduced[axis]) {
            if (output_axis >= outputs[0].rank ||
                outputs[0].dimensions[output_axis] !=
                    inputs[0].dimensions[axis]) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
            ++output_axis;
        }
    }
    if (keepdims_value[0] == 0 && output_axis != outputs[0].rank) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    for (index = 0u; index < campp_tensor_view_element_count(&outputs[0]);
         ++index) {
        const uint64_t offset =
            campp_reference_offset_for_linear(&outputs[0], index);
        const float zero = 0.0f;
        memcpy((uint8_t *)outputs[0].data + offset, &zero, sizeof(zero));
    }
    for (index = 0u; index < campp_tensor_view_element_count(&inputs[0]);
         ++index) {
        uint32_t input_coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t output_coordinates[CAMPP_TENSOR_MAX_RANK] = {0u, 0u, 0u, 0u};
        uint64_t input_offset;
        uint64_t output_offset;
        float input_value;
        float output_value;

        campp_reference_unravel_index(
            index, inputs[0].rank, inputs[0].dimensions, input_coordinates);
        output_axis = 0u;
        for (axis = 0u; axis < inputs[0].rank; ++axis) {
            if (keepdims_value[0] != 0) {
                output_coordinates[axis] = reduced[axis]
                    ? 0u : input_coordinates[axis];
            } else if (!reduced[axis]) {
                output_coordinates[output_axis++] = input_coordinates[axis];
            }
        }
        input_offset = campp_tensor_view_byte_offset(
            &inputs[0], input_coordinates);
        output_offset = campp_tensor_view_byte_offset(
            &outputs[0], output_coordinates);
        memcpy(
            &input_value, (const uint8_t *)inputs[0].data + input_offset,
            sizeof(input_value));
        memcpy(
            &output_value, (const uint8_t *)outputs[0].data + output_offset,
            sizeof(output_value));
        output_value += input_value;
        memcpy(
            (uint8_t *)outputs[0].data + output_offset, &output_value,
            sizeof(output_value));
    }
    for (index = 0u; index < campp_tensor_view_element_count(&outputs[0]);
         ++index) {
        const uint64_t offset =
            campp_reference_offset_for_linear(&outputs[0], index);
        float value;
        memcpy(
            &value, (const uint8_t *)outputs[0].data + offset, sizeof(value));
        value /= (float)reduction_count;
        memcpy(
            (uint8_t *)outputs[0].data + offset, &value, sizeof(value));
    }
    return CAMPP_STATUS_OK;
}
