#include "reference_kernels.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "reference_kernel_utils.h"

typedef float (*CamppReferenceBinaryOperation)(float left, float right);

static float campp_add_value(float left, float right) { return left + right; }
static float campp_mul_value(float left, float right) { return left * right; }
static float campp_sub_value(float left, float right) { return left - right; }
static float campp_div_value(float left, float right) { return left / right; }

static CamppStatus campp_validate_broadcast_input(
    const CamppTensorView *input, const CamppTensorView *output)
{
    uint8_t input_axis;
    uint8_t rank_delta;

    if (input->rank > output->rank) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    rank_delta = (uint8_t)(output->rank - input->rank);
    for (input_axis = 0u; input_axis < input->rank; ++input_axis) {
        const uint32_t input_dimension = input->dimensions[input_axis];
        const uint32_t output_dimension =
            output->dimensions[input_axis + rank_delta];
        if (input_dimension != 1u && input_dimension != output_dimension) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_validate_binary_output_shape(
    const CamppTensorView *left, const CamppTensorView *right,
    const CamppTensorView *output)
{
    uint8_t output_axis;
    const uint8_t left_delta = (uint8_t)(output->rank - left->rank);
    const uint8_t right_delta = (uint8_t)(output->rank - right->rank);

    for (output_axis = 0u; output_axis < output->rank; ++output_axis) {
        const uint32_t left_dimension = output_axis < left_delta
            ? 1u : left->dimensions[output_axis - left_delta];
        const uint32_t right_dimension = output_axis < right_delta
            ? 1u : right->dimensions[output_axis - right_delta];
        const uint32_t expected =
            left_dimension > right_dimension ? left_dimension : right_dimension;
        if ((left_dimension != right_dimension && left_dimension != 1u &&
             right_dimension != 1u) || output->dimensions[output_axis] != expected) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    return CAMPP_STATUS_OK;
}

static uint64_t campp_broadcast_offset(
    const CamppTensorView *input, const CamppTensorView *output,
    const uint32_t output_coordinates[CAMPP_TENSOR_MAX_RANK])
{
    uint32_t input_coordinates[CAMPP_TENSOR_MAX_RANK] = {0u, 0u, 0u, 0u};
    const uint8_t rank_delta = (uint8_t)(output->rank - input->rank);
    uint8_t input_axis;

    for (input_axis = 0u; input_axis < input->rank; ++input_axis) {
        input_coordinates[input_axis] =
            input->dimensions[input_axis] == 1u
                ? 0u
                : output_coordinates[input_axis + rank_delta];
    }
    return campp_tensor_view_byte_offset(input, input_coordinates);
}

static CamppStatus campp_reference_binary(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppReferenceBinaryOperation operation)
{
    CamppTensorView *output;
    uint64_t count;
    uint64_t index;
    CamppStatus status;

    status = campp_reference_validate_invocation(
        inputs, input_count, 2u, 2u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    output = &outputs[0];
    if (inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        inputs[1].dtype != CAMPP_DTYPE_FLOAT32 ||
        output->dtype != CAMPP_DTYPE_FLOAT32) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    status = campp_validate_broadcast_input(&inputs[0], output);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_validate_broadcast_input(&inputs[1], output);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_validate_binary_output_shape(
        &inputs[0], &inputs[1], output);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    count = campp_tensor_view_element_count(output);
    for (index = 0u; index < count; ++index) {
        uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
        const uint64_t output_offset =
            campp_reference_offset_for_linear(output, index);
        uint64_t left_offset;
        uint64_t right_offset;
        float left;
        float right;
        float result;

        campp_reference_unravel_index(
            index, output->rank, output->dimensions, coordinates);
        left_offset = campp_broadcast_offset(&inputs[0], output, coordinates);
        right_offset = campp_broadcast_offset(&inputs[1], output, coordinates);
        memcpy(
            &left, (const uint8_t *)inputs[0].data + left_offset,
            sizeof(left));
        memcpy(
            &right, (const uint8_t *)inputs[1].data + right_offset,
            sizeof(right));
        result = operation(left, right);
        memcpy((uint8_t *)output->data + output_offset, &result, sizeof(result));
    }
    return CAMPP_STATUS_OK;
}

#define CAMPP_DEFINE_BINARY_KERNEL(function_name, operation_function)          \
    CamppStatus function_name(                                                 \
        const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,     \
        const CamppTensorView *inputs, uint8_t input_count,                    \
        CamppTensorView *outputs, uint8_t output_count, void *scratch,         \
        size_t scratch_size)                                                   \
    {                                                                          \
        (void)model;                                                           \
        (void)op;                                                              \
        (void)scratch;                                                         \
        (void)scratch_size;                                                    \
        return campp_reference_binary(                                         \
            inputs, input_count, outputs, output_count, operation_function);   \
    }

CAMPP_DEFINE_BINARY_KERNEL(campp_reference_add, campp_add_value)
CAMPP_DEFINE_BINARY_KERNEL(campp_reference_mul, campp_mul_value)
CAMPP_DEFINE_BINARY_KERNEL(campp_reference_sub, campp_sub_value)
CAMPP_DEFINE_BINARY_KERNEL(campp_reference_div, campp_div_value)

#undef CAMPP_DEFINE_BINARY_KERNEL

CamppStatus campp_reference_sqrt(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    CamppStatus status;
    uint64_t count;
    uint64_t index;

    (void)model;
    (void)op;
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
    if (!campp_reference_shapes_equal(&inputs[0], &outputs[0])) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    count = campp_tensor_view_element_count(&outputs[0]);
    for (index = 0u; index < count; ++index) {
        const uint64_t input_offset =
            campp_reference_offset_for_linear(&inputs[0], index);
        const uint64_t output_offset =
            campp_reference_offset_for_linear(&outputs[0], index);
        float value;
        float result;

        memcpy(
            &value, (const uint8_t *)inputs[0].data + input_offset,
            sizeof(value));
        result = sqrtf(value);
        memcpy(
            (uint8_t *)outputs[0].data + output_offset, &result,
            sizeof(result));
    }
    return CAMPP_STATUS_OK;
}
