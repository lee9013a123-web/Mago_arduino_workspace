/* BatchNorm affine + ReLU + QuantizeLinear without FP32 intermediates. */

#include "backends/cpu_aarch64/aarch64_kernels.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "backends/cpu_reference/reference_kernel_utils.h"
#include "internal/runtime_model.h"

static CamppStatus campp_fused_scalar_zero(
    const CamppTensorView *view, int32_t *out_value)
{
    if (view == NULL) {
        *out_value = 0;
        return CAMPP_STATUS_OK;
    }
    if (campp_tensor_view_element_count(view) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_reference_read_quantized(view, 0u, out_value);
}

CamppStatus campp_fused_bn_relu_quant(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    const CamppTensorView *zero_point = input_count == 7u ? &inputs[6] : NULL;
    double epsilon_value;
    float quant_scale;
    int32_t quant_zero;
    uint32_t channels;
    uint64_t count;
    uint64_t index;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 6u, 7u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    if (inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        (outputs[0].dtype != CAMPP_DTYPE_UINT8 &&
         outputs[0].dtype != CAMPP_DTYPE_INT8) ||
        inputs[0].rank < 2u ||
        !campp_reference_shapes_equal(&inputs[0], &outputs[0])) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    channels = inputs[0].dimensions[1];
    for (index = 1u; index < 5u; ++index) {
        if (inputs[index].dtype != CAMPP_DTYPE_FLOAT32 ||
            inputs[index].rank != 1u ||
            inputs[index].dimensions[0] != channels) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    if (inputs[5].dtype != CAMPP_DTYPE_FLOAT32 ||
        campp_tensor_view_element_count(&inputs[5]) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    status = campp_runtime_model_attribute_double(
        model, op, CAMPP_ATTR_EPSILON, &epsilon_value);
    if (status != CAMPP_STATUS_OK) return status;
    if (epsilon_value < 0.0) return CAMPP_STATUS_CORRUPT_PLAN;
    status = campp_reference_read_f32(&inputs[5], 0u, &quant_scale);
    if (status != CAMPP_STATUS_OK || !(quant_scale > 0.0f) ||
        !isfinite(quant_scale)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    status = campp_fused_scalar_zero(zero_point, &quant_zero);
    if (status != CAMPP_STATUS_OK) return status;

    count = campp_tensor_view_element_count(&outputs[0]);
    for (index = 0u; index < count; ++index) {
        uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t channel;
        float x;
        float scale;
        float bias;
        float mean;
        float variance;
        float inv_std;
        float new_scale;
        float new_bias;
        float value;
        float scaled;
        int64_t rounded;

        campp_reference_unravel_index(
            index, outputs[0].rank, outputs[0].dimensions, coordinates);
        channel = coordinates[1];
        status = campp_reference_read_f32(&inputs[0], index, &x);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_read_f32(&inputs[1], channel, &scale);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_read_f32(&inputs[2], channel, &bias);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_read_f32(&inputs[3], channel, &mean);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_read_f32(&inputs[4], channel, &variance);
        if (status != CAMPP_STATUS_OK) return status;

        inv_std = 1.0f / sqrtf(variance + (float)epsilon_value);
        new_scale = inv_std * scale;
        new_bias = bias - mean * new_scale;
        value = x * new_scale + new_bias;
        if (value < 0.0f) value = 0.0f;
        if (!isfinite(value)) return CAMPP_STATUS_KERNEL_FAILED;
        scaled = value / quant_scale;
        if (scaled <= -2147483648.0f) {
            rounded = INT32_MIN;
        } else if (scaled >= 2147483520.0f) {
            rounded = INT32_MAX;
        } else {
            rounded = (int64_t)nearbyintf(scaled);
        }
        status = campp_reference_write_quantized(
            &outputs[0], index, rounded + quant_zero);
        if (status != CAMPP_STATUS_OK) return status;
    }
    return CAMPP_STATUS_OK;
}
