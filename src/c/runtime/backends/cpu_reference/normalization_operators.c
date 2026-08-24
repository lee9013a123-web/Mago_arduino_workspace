#include "reference_kernels.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "internal/runtime_model.h"
#include "reference_kernel_utils.h"

CamppStatus campp_reference_batch_normalization(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    double epsilon_value;
    uint32_t channels;
    uint8_t parameter_index;
    uint64_t index;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 5u, 5u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    if (inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (inputs[0].rank < 2u ||
        !campp_reference_shapes_equal(&inputs[0], &outputs[0])) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    channels = inputs[0].dimensions[1];
    for (parameter_index = 1u; parameter_index < 5u; ++parameter_index) {
        if (inputs[parameter_index].dtype != CAMPP_DTYPE_FLOAT32 ||
            inputs[parameter_index].rank != 1u ||
            inputs[parameter_index].dimensions[0] != channels) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    status = campp_runtime_model_attribute_double(
        model, op, CAMPP_ATTR_EPSILON, &epsilon_value);
    if (status != CAMPP_STATUS_OK) return status;
    if (epsilon_value < 0.0) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    for (index = 0u; index < campp_tensor_view_element_count(&outputs[0]);
         ++index) {
        uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t channel_coordinates[CAMPP_TENSOR_MAX_RANK] = {0u, 0u, 0u, 0u};
        uint64_t input_offset;
        uint64_t output_offset;
        uint64_t parameter_offset[4];
        float x;
        float scale;
        float bias;
        float mean;
        float variance;
        float result;

        campp_reference_unravel_index(
            index, outputs[0].rank, outputs[0].dimensions, coordinates);
        channel_coordinates[0] = coordinates[1];
        input_offset = campp_tensor_view_byte_offset(&inputs[0], coordinates);
        output_offset = campp_tensor_view_byte_offset(&outputs[0], coordinates);
        for (parameter_index = 0u; parameter_index < 4u; ++parameter_index) {
            parameter_offset[parameter_index] = campp_tensor_view_byte_offset(
                &inputs[parameter_index + 1u], channel_coordinates);
        }
        memcpy(&x, (const uint8_t *)inputs[0].data + input_offset, sizeof(x));
        memcpy(&scale, (const uint8_t *)inputs[1].data + parameter_offset[0], sizeof(scale));
        memcpy(&bias, (const uint8_t *)inputs[2].data + parameter_offset[1], sizeof(bias));
        memcpy(&mean, (const uint8_t *)inputs[3].data + parameter_offset[2], sizeof(mean));
        memcpy(&variance, (const uint8_t *)inputs[4].data + parameter_offset[3], sizeof(variance));
        /*
         * Legacy direct form (algebraically equivalent, but not necessarily
         * float32 rounding-equivalent to ONNX Runtime):
         *
         * result = scale * (x - mean) /
         *              sqrtf(variance + (float)epsilon_value) +
         *          bias;
         *
         * ONNX Runtime's CPU BatchNormalization kernel first turns the
         * channel parameters into an affine transform, then evaluates
         * x * new_scale + new_bias.  Keep every intermediate as float so a
         * compiler cannot silently promote part of the calculation to
         * double precision.  This ordering is important near a following
         * QuantizeLinear half-way rounding boundary.
         */
        {
            const float epsilon = (float)epsilon_value;
            const float inv_std = 1.0f / sqrtf(variance + epsilon);
            const float new_scale = inv_std * scale;
            const float new_bias = bias - mean * new_scale;

            result = x * new_scale + new_bias;
        }
        memcpy(
            (uint8_t *)outputs[0].data + output_offset, &result,
            sizeof(result));
    }
    return CAMPP_STATUS_OK;
}
