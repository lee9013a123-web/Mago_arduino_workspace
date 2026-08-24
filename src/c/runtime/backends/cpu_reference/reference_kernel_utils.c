#include "reference_kernel_utils.h"

#include <limits.h>
#include <string.h>

CamppStatus campp_reference_validate_invocation(
    const CamppTensorView *inputs, uint8_t input_count,
    uint8_t minimum_inputs, uint8_t maximum_inputs,
    CamppTensorView *outputs, uint8_t output_count)
{
    uint8_t index;

    if (inputs == NULL || outputs == NULL || input_count < minimum_inputs ||
        input_count > maximum_inputs || output_count != 1u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    for (index = 0u; index < input_count; ++index) {
        if (inputs[index].data == NULL) {
            return CAMPP_STATUS_TENSOR_NOT_BOUND;
        }
        if (inputs[index].rank > CAMPP_TENSOR_MAX_RANK) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    if (outputs[0].data == NULL) {
        return CAMPP_STATUS_TENSOR_NOT_BOUND;
    }
    if (outputs[0].rank > CAMPP_TENSOR_MAX_RANK) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_normalize_axis(
    int64_t axis, uint8_t rank, uint8_t *out_axis)
{
    int64_t normalized = axis;

    if (out_axis == NULL || rank == 0u || rank > CAMPP_TENSOR_MAX_RANK) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (normalized < 0) {
        normalized += (int64_t)rank;
    }
    if (normalized < 0 || normalized >= (int64_t)rank) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    *out_axis = (uint8_t)normalized;
    return CAMPP_STATUS_OK;
}

void campp_reference_unravel_index(
    uint64_t linear_index, uint8_t rank,
    const uint32_t dimensions[CAMPP_TENSOR_MAX_RANK],
    uint32_t coordinates[CAMPP_TENSOR_MAX_RANK])
{
    uint8_t axis;

    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        coordinates[axis] = 0u;
    }
    for (axis = rank; axis > 0u; --axis) {
        const uint32_t dimension = dimensions[axis - 1u];
        coordinates[axis - 1u] = (uint32_t)(linear_index % dimension);
        linear_index /= dimension;
    }
}

uint64_t campp_reference_offset_for_linear(
    const CamppTensorView *view, uint64_t linear_index)
{
    uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];

    campp_reference_unravel_index(
        linear_index, view->rank, view->dimensions, coordinates);
    return campp_tensor_view_byte_offset(view, coordinates);
}

bool campp_reference_shapes_equal(
    const CamppTensorView *first, const CamppTensorView *second)
{
    uint8_t axis;

    if (first == NULL || second == NULL || first->rank != second->rank) {
        return false;
    }
    for (axis = 0u; axis < first->rank; ++axis) {
        if (first->dimensions[axis] != second->dimensions[axis]) {
            return false;
        }
    }
    return true;
}

CamppStatus campp_reference_copy_tensor(
    const CamppTensorView *input, CamppTensorView *output)
{
    uint64_t input_count;
    uint64_t output_count;
    uint32_t element_size;
    uint64_t index;

    if (input == NULL || output == NULL || input->data == NULL ||
        output->data == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    input_count = campp_tensor_view_element_count(input);
    output_count = campp_tensor_view_element_count(output);
    element_size = campp_dtype_byte_size(input->dtype);
    if (input->dtype != output->dtype) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (input_count != output_count || element_size == 0u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    if (input_count > (uint64_t)SIZE_MAX / element_size) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    if (campp_tensor_view_is_contiguous(input) &&
        campp_tensor_view_is_contiguous(output)) {
        memcpy(output->data, input->data, (size_t)(input_count * element_size));
        return CAMPP_STATUS_OK;
    }
    for (index = 0u; index < input_count; ++index) {
        const uint64_t input_offset =
            campp_reference_offset_for_linear(input, index);
        const uint64_t output_offset =
            campp_reference_offset_for_linear(output, index);
        memcpy(
            (uint8_t *)output->data + output_offset,
            (const uint8_t *)input->data + input_offset, element_size);
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_read_f32(
    const CamppTensorView *view, uint64_t linear_index, float *out_value)
{
    uint64_t offset;

    if (view == NULL || out_value == NULL || view->data == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (view->dtype != CAMPP_DTYPE_FLOAT32) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (linear_index >= campp_tensor_view_element_count(view)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    offset = campp_reference_offset_for_linear(view, linear_index);
    memcpy(out_value, (const uint8_t *)view->data + offset, sizeof(*out_value));
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_read_i64(
    const CamppTensorView *view, uint64_t linear_index, int64_t *out_value)
{
    uint64_t offset;

    if (view == NULL || out_value == NULL || view->data == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (view->dtype != CAMPP_DTYPE_INT64) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (linear_index >= campp_tensor_view_element_count(view)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    offset = campp_reference_offset_for_linear(view, linear_index);
    memcpy(out_value, (const uint8_t *)view->data + offset, sizeof(*out_value));
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_read_quantized(
    const CamppTensorView *view, uint64_t linear_index, int32_t *out_value)
{
    uint64_t offset;

    if (view == NULL || out_value == NULL || view->data == NULL ||
        linear_index >= campp_tensor_view_element_count(view)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    offset = campp_reference_offset_for_linear(view, linear_index);
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
    if (view->dtype == CAMPP_DTYPE_INT32) {
        int32_t value;
        memcpy(&value, (const uint8_t *)view->data + offset, sizeof(value));
        *out_value = value;
        return CAMPP_STATUS_OK;
    }
    return CAMPP_STATUS_UNSUPPORTED_DTYPE;
}

CamppStatus campp_reference_write_quantized(
    CamppTensorView *view, uint64_t linear_index, int64_t value)
{
    uint64_t offset;

    if (view == NULL || view->data == NULL ||
        linear_index >= campp_tensor_view_element_count(view)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    offset = campp_reference_offset_for_linear(view, linear_index);
    if (view->dtype == CAMPP_DTYPE_UINT8) {
        const uint8_t stored = (uint8_t)(value < 0 ? 0 : value > 255 ? 255 : value);
        memcpy((uint8_t *)view->data + offset, &stored, sizeof(stored));
        return CAMPP_STATUS_OK;
    }
    if (view->dtype == CAMPP_DTYPE_INT8) {
        const int8_t stored =
            (int8_t)(value < -128 ? -128 : value > 127 ? 127 : value);
        memcpy((uint8_t *)view->data + offset, &stored, sizeof(stored));
        return CAMPP_STATUS_OK;
    }
    if (view->dtype == CAMPP_DTYPE_INT32) {
        const int32_t stored =
            (int32_t)(value < INT32_MIN ? INT32_MIN :
                      value > INT32_MAX ? INT32_MAX : value);
        memcpy((uint8_t *)view->data + offset, &stored, sizeof(stored));
        return CAMPP_STATUS_OK;
    }
    return CAMPP_STATUS_UNSUPPORTED_DTYPE;
}
