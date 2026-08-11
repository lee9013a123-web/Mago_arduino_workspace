#include "reference_kernels.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "reference_kernel_utils.h"

typedef float (*CamppReferenceActivation)(float value);

static float campp_relu_value(float value)
{
    /* x < 0일 때만 0으로 바꾸면 NaN과 양수, -0은 입력 bit를 보존한다. */
    return value < 0.0f ? 0.0f : value;
}

static float campp_sigmoid_value(float value)
{
    if (value >= 0.0f) {
        return 1.0f / (1.0f + expf(-value));
    }
    {
        const float exponential = expf(value);
        return exponential / (1.0f + exponential);
    }
}

static CamppStatus campp_reference_activation(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppReferenceActivation activation)
{
    const CamppTensorView *input;
    CamppTensorView *output;
    uint64_t count;
    uint64_t index;
    CamppStatus status;

    status = campp_reference_validate_invocation(
        inputs, input_count, 1u, 1u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    input = &inputs[0];
    output = &outputs[0];
    if (input->dtype != CAMPP_DTYPE_FLOAT32 ||
        output->dtype != CAMPP_DTYPE_FLOAT32) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (!campp_reference_shapes_equal(input, output)) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }

    count = campp_tensor_view_element_count(output);
    for (index = 0u; index < count; ++index) {
        const uint64_t input_offset =
            campp_reference_offset_for_linear(input, index);
        const uint64_t output_offset =
            campp_reference_offset_for_linear(output, index);
        float value;
        float result;

        memcpy(
            &value, (const uint8_t *)input->data + input_offset,
            sizeof(value));
        result = activation(value);
        memcpy((uint8_t *)output->data + output_offset, &result, sizeof(result));
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_reference_relu(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    (void)model;
    (void)op;
    (void)scratch;
    (void)scratch_size;
    return campp_reference_activation(
        inputs, input_count, outputs, output_count, campp_relu_value);
}

CamppStatus campp_reference_sigmoid(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    (void)model;
    (void)op;
    (void)scratch;
    (void)scratch_size;
    return campp_reference_activation(
        inputs, input_count, outputs, output_count, campp_sigmoid_value);
}
