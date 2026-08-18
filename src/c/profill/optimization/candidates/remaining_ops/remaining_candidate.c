#include "remaining_candidate.h"

#include <fenv.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_reference/reference_kernel_utils.h"
#include "backends/cpu_reference/reference_kernels.h"
#include "block_copy_fastpath.h"
#include "elementwise_neon.h"
#include "internal/runtime_model.h"
#include "packed_iteration.h"
#include "reduction_neon.h"
#include "sigmoid_lut.h"

static CamppStatus campp_remaining_scalar_f32(
    const CamppTensorView *view, float *out_value)
{
    if (view == NULL || out_value == NULL || view->dtype != CAMPP_DTYPE_FLOAT32 ||
        campp_tensor_view_element_count(view) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_reference_read_f32(view, 0u, out_value);
}

static CamppStatus campp_remaining_scalar_quantized(
    const CamppTensorView *view, uint8_t dtype, int32_t *out_value)
{
    if (view == NULL) {
        *out_value = 0;
        return CAMPP_STATUS_OK;
    }
    if (view->dtype != dtype || campp_tensor_view_element_count(view) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_reference_read_quantized(view, 0u, out_value);
}

static CamppStatus campp_remaining_attribute_ints(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, int64_t *values, uint8_t capacity, uint8_t expected_count)
{
    uint8_t count = 0u;
    CamppStatus status = campp_runtime_model_attribute_ints(
        model, op, key, values, capacity, &count);
    if (status != CAMPP_STATUS_OK) return status;
    return count == expected_count ? CAMPP_STATUS_OK
                                   : CAMPP_STATUS_CORRUPT_PLAN;
}

static bool campp_remaining_same_logical_shape(
    const CamppTensorView *left, const CamppTensorView *right)
{
    return left->rank == right->rank &&
        campp_reference_shapes_equal(left, right);
}

static CamppStatus campp_remaining_add_optimized(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    CamppPackedIteration left;
    CamppPackedIteration right;
    CamppPackedIteration output;
    uint32_t batch;
    if (campp_reference_validate_invocation(
            inputs, input_count, 2u, 2u, outputs, output_count) !=
            CAMPP_STATUS_OK ||
        inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        inputs[1].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        inputs[0].rank != outputs[0].rank ||
        inputs[1].rank != outputs[0].rank ||
        !campp_packed_iteration_create(&inputs[0], &left) ||
        !campp_packed_iteration_create(&inputs[1], &right) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        left.channels != output.channels || right.channels != output.channels ||
        left.batches != output.batches || right.batches != output.batches ||
        (left.height != 1u && left.height != output.height) ||
        (right.height != 1u && right.height != output.height) ||
        (left.width != 1u && left.width != output.width) ||
        (right.width != 1u && right.width != output.width)) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (batch = 0u; batch < output.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < output.height; ++height) {
            uint32_t width;
            for (width = 0u; width < output.width; ++width) {
                const float *left_values = (const float *)
                    campp_packed_iteration_const_pointer(
                        &left, batch, 0u,
                        left.height == 1u ? 0u : height,
                        left.width == 1u ? 0u : width);
                const float *right_values = (const float *)
                    campp_packed_iteration_const_pointer(
                        &right, batch, 0u,
                        right.height == 1u ? 0u : height,
                        right.width == 1u ? 0u : width);
                float *output_values = (float *)campp_packed_iteration_pointer(
                    &output, batch, 0u, height, width);
                campp_elementwise_add_f32(
                    left_values, right_values, output_values, output.channels);
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_remaining_relu_optimized(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    CamppPackedIteration input;
    CamppPackedIteration output;
    uint32_t batch;
    if (campp_reference_validate_invocation(
            inputs, input_count, 1u, 1u, outputs, output_count) !=
            CAMPP_STATUS_OK ||
        inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        !campp_remaining_same_logical_shape(&inputs[0], &outputs[0]) ||
        !campp_packed_iteration_create(&inputs[0], &input) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        !campp_packed_iteration_same_shape(&input, &output)) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (batch = 0u; batch < output.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < output.height; ++height) {
            uint32_t width;
            for (width = 0u; width < output.width; ++width) {
                campp_elementwise_relu_f32(
                    (const float *)campp_packed_iteration_const_pointer(
                        &input, batch, 0u, height, width),
                    (float *)campp_packed_iteration_pointer(
                        &output, batch, 0u, height, width),
                    output.channels);
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_remaining_quantize_optimized(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    const CamppTensorView *zero_view = input_count == 3u ? &inputs[2] : NULL;
    CamppPackedIteration input;
    CamppPackedIteration output;
    float scale;
    int32_t zero;
    uint32_t batch;
    if (campp_reference_validate_invocation(
            inputs, input_count, 2u, 3u, outputs, output_count) !=
            CAMPP_STATUS_OK ||
        inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        (outputs[0].dtype != CAMPP_DTYPE_UINT8 &&
         outputs[0].dtype != CAMPP_DTYPE_INT8) ||
        !campp_remaining_same_logical_shape(&inputs[0], &outputs[0]) ||
        campp_remaining_scalar_f32(&inputs[1], &scale) != CAMPP_STATUS_OK ||
        campp_remaining_scalar_quantized(
            zero_view, outputs[0].dtype, &zero) != CAMPP_STATUS_OK ||
        !(scale > 0.0f) || !isfinite(scale) || fegetround() != FE_TONEAREST ||
        !campp_packed_iteration_create(&inputs[0], &input) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        !campp_packed_iteration_same_shape(&input, &output)) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (batch = 0u; batch < input.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < input.height; ++height) {
            uint32_t width;
            for (width = 0u; width < input.width; ++width) {
                const float *values = (const float *)
                    campp_packed_iteration_const_pointer(
                        &input, batch, 0u, height, width);
                uint32_t channel;
                for (channel = 0u; channel < input.channels; ++channel) {
                    if (!isfinite(values[channel])) {
                        return CAMPP_STATUS_KERNEL_FAILED;
                    }
                }
            }
        }
    }
    for (batch = 0u; batch < input.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < input.height; ++height) {
            uint32_t width;
            for (width = 0u; width < input.width; ++width) {
                campp_elementwise_quantize_f32(
                    (const float *)campp_packed_iteration_const_pointer(
                        &input, batch, 0u, height, width),
                    campp_packed_iteration_pointer(
                        &output, batch, 0u, height, width),
                    input.channels, scale, zero, outputs[0].dtype);
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_remaining_expand_optimized(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    CamppPackedIteration input;
    CamppPackedIteration output;
    uint64_t shape_count;
    uint8_t axis;
    uint32_t batch;
    if (campp_reference_validate_invocation(
            inputs, input_count, 2u, 2u, outputs, output_count) !=
            CAMPP_STATUS_OK ||
        inputs[0].dtype != outputs[0].dtype || inputs[0].rank != 4u ||
        outputs[0].rank != 4u || inputs[1].dtype != CAMPP_DTYPE_INT64 ||
        !campp_packed_iteration_create(&inputs[0], &input) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        input.batches != output.batches || input.channels != output.channels ||
        input.height != 1u || input.width != 1u || output.height != 1u) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    shape_count = campp_tensor_view_element_count(&inputs[1]);
    if (shape_count != outputs[0].rank) return CAMPP_STATUS_NOT_IMPLEMENTED;
    for (axis = 0u; axis < outputs[0].rank; ++axis) {
        int64_t requested;
        if (campp_reference_read_i64(&inputs[1], axis, &requested) !=
                CAMPP_STATUS_OK ||
            requested != (int64_t)outputs[0].dimensions[axis]) {
            return CAMPP_STATUS_NOT_IMPLEMENTED;
        }
    }
    for (batch = 0u; batch < output.batches; ++batch) {
        campp_copy_repeated_block(
            campp_packed_iteration_const_pointer(&input, batch, 0u, 0u, 0u),
            (size_t)input.channels * input.element_size,
            campp_packed_iteration_pointer(&output, batch, 0u, 0u, 0u),
            output.strides[3], output.width);
    }
    return CAMPP_STATUS_OK;
}

static int64_t campp_remaining_slice_index(int64_t value, int64_t dimension)
{
    if (value == INT64_MIN) return 0;
    if (value == INT64_MAX) return dimension;
    if (value < 0) value += dimension;
    if (value < 0) return 0;
    return value > dimension ? dimension : value;
}

static CamppStatus campp_remaining_slice_optimized(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    CamppPackedIteration input;
    CamppPackedIteration output;
    int64_t start;
    int64_t end;
    int64_t axis;
    int64_t step;
    int64_t normalized_start;
    int64_t normalized_end;
    uint32_t batch;
    if (campp_reference_validate_invocation(
            inputs, input_count, 3u, 5u, outputs, output_count) !=
            CAMPP_STATUS_OK || input_count != 5u ||
        inputs[0].dtype != outputs[0].dtype || inputs[0].rank != 3u ||
        outputs[0].rank != 3u ||
        campp_tensor_view_element_count(&inputs[1]) != 1u ||
        campp_tensor_view_element_count(&inputs[2]) != 1u ||
        campp_tensor_view_element_count(&inputs[3]) != 1u ||
        campp_tensor_view_element_count(&inputs[4]) != 1u ||
        campp_reference_read_i64(&inputs[1], 0u, &start) != CAMPP_STATUS_OK ||
        campp_reference_read_i64(&inputs[2], 0u, &end) != CAMPP_STATUS_OK ||
        campp_reference_read_i64(&inputs[3], 0u, &axis) != CAMPP_STATUS_OK ||
        campp_reference_read_i64(&inputs[4], 0u, &step) != CAMPP_STATUS_OK ||
        (axis != 2 && axis != -1) || step != 1 ||
        !campp_packed_iteration_create(&inputs[0], &input) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        input.batches != output.batches || input.channels != output.channels ||
        input.height != output.height) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    normalized_start = campp_remaining_slice_index(start, input.width);
    normalized_end = campp_remaining_slice_index(end, input.width);
    if (normalized_end < normalized_start ||
        (uint64_t)(normalized_end - normalized_start) != output.width) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (batch = 0u; batch < output.batches; ++batch) {
        const size_t block_size =
            (size_t)output.channels * output.element_size;
        if (input.strides[3] == block_size &&
            output.strides[3] == block_size) {
            memcpy(
                campp_packed_iteration_pointer(
                    &output, batch, 0u, 0u, 0u),
                campp_packed_iteration_const_pointer(
                    &input, batch, 0u, 0u, (uint32_t)normalized_start),
                block_size * output.width);
            continue;
        }
        uint32_t width;
        for (width = 0u; width < output.width; ++width) {
            memcpy(
                campp_packed_iteration_pointer(
                    &output, batch, 0u, 0u, width),
                campp_packed_iteration_const_pointer(
                    &input, batch, 0u, 0u,
                    (uint32_t)normalized_start + width),
                (size_t)output.channels * output.element_size);
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_remaining_reduce_mean_optimized(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    CamppPackedIteration input;
    CamppPackedIteration output;
    int64_t axes[4];
    int64_t keepdims[1];
    uint8_t axes_count = 0u;
    uint8_t keepdims_count = 0u;
    uint8_t normalized;
    uint32_t batch;
    CamppStatus status;
    if (campp_reference_validate_invocation(
            inputs, input_count, 1u, 1u, outputs, output_count) !=
            CAMPP_STATUS_OK || inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32 || inputs[0].rank != 3u ||
        outputs[0].rank != 3u) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    status = campp_runtime_model_attribute_ints(
        model, op, CAMPP_ATTR_AXES, axes, 4u, &axes_count);
    if (status != CAMPP_STATUS_OK || axes_count != 1u ||
        campp_reference_normalize_axis(axes[0], 3u, &normalized) !=
            CAMPP_STATUS_OK || normalized != 2u ||
        campp_runtime_model_attribute_ints(
            model, op, CAMPP_ATTR_KEEPDIMS, keepdims, 1u,
            &keepdims_count) != CAMPP_STATUS_OK ||
        keepdims_count != 1u || keepdims[0] != 1 ||
        !campp_packed_iteration_create(&inputs[0], &input) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        input.batches != output.batches || input.channels != output.channels ||
        output.height != 1u || output.width != 1u) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (batch = 0u; batch < input.batches; ++batch) {
        campp_reduction_mean_channels_f32(
            campp_packed_iteration_const_pointer(&input, batch, 0u, 0u, 0u),
            input.strides[3], input.channels, input.width,
            campp_packed_iteration_pointer(&output, batch, 0u, 0u, 0u));
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_remaining_average_pool_optimized(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    CamppPackedIteration input;
    CamppPackedIteration output;
    int64_t kernel[1];
    int64_t pads[2];
    int64_t strides[1];
    int64_t ceil_mode[1];
    int64_t include_pad[1];
    uint32_t batch;
    if (campp_reference_validate_invocation(
            inputs, input_count, 1u, 1u, outputs, output_count) !=
            CAMPP_STATUS_OK || inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32 || inputs[0].rank != 3u ||
        outputs[0].rank != 3u ||
        campp_remaining_attribute_ints(
            model, op, CAMPP_ATTR_KERNEL_SHAPE, kernel, 1u, 1u) !=
            CAMPP_STATUS_OK ||
        campp_remaining_attribute_ints(
            model, op, CAMPP_ATTR_PADS, pads, 2u, 2u) != CAMPP_STATUS_OK ||
        campp_remaining_attribute_ints(
            model, op, CAMPP_ATTR_STRIDES, strides, 1u, 1u) !=
            CAMPP_STATUS_OK ||
        campp_remaining_attribute_ints(
            model, op, CAMPP_ATTR_CEIL_MODE, ceil_mode, 1u, 1u) !=
            CAMPP_STATUS_OK ||
        campp_remaining_attribute_ints(
            model, op, CAMPP_ATTR_COUNT_INCLUDE_PAD, include_pad, 1u, 1u) !=
            CAMPP_STATUS_OK || pads[0] != 0 || pads[1] != 0 ||
        strides[0] != 1 || (ceil_mode[0] != 0 && ceil_mode[0] != 1) ||
        (include_pad[0] != 0 && include_pad[0] != 1) ||
        !campp_packed_iteration_create(&inputs[0], &input) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        kernel[0] != input.width || input.batches != output.batches ||
        input.channels != output.channels || output.height != 1u ||
        output.width != 1u) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (batch = 0u; batch < input.batches; ++batch) {
        campp_reduction_mean_channels_f32(
            campp_packed_iteration_const_pointer(&input, batch, 0u, 0u, 0u),
            input.strides[3], input.channels, input.width,
            campp_packed_iteration_pointer(&output, batch, 0u, 0u, 0u));
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_remaining_sigmoid_mul_optimized(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    CamppPackedIteration quantized;
    CamppPackedIteration other;
    CamppPackedIteration output;
    float table[256];
    float scale;
    int32_t zero;
    uint32_t batch;
    if (campp_reference_validate_invocation(
            inputs, input_count, 4u, 4u, outputs, output_count) !=
            CAMPP_STATUS_OK ||
        (inputs[0].dtype != CAMPP_DTYPE_UINT8 &&
         inputs[0].dtype != CAMPP_DTYPE_INT8) ||
        inputs[3].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        campp_remaining_scalar_f32(&inputs[1], &scale) != CAMPP_STATUS_OK ||
        campp_remaining_scalar_quantized(
            &inputs[2], inputs[0].dtype, &zero) != CAMPP_STATUS_OK ||
        !(scale > 0.0f) || !isfinite(scale) ||
        !campp_packed_iteration_create(&inputs[0], &quantized) ||
        !campp_packed_iteration_create(&inputs[3], &other) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        !campp_packed_iteration_same_shape(&quantized, &other) ||
        !campp_packed_iteration_same_shape(&quantized, &output)) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    campp_sigmoid_lut_build(inputs[0].dtype, scale, zero, table);
    for (batch = 0u; batch < output.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < output.height; ++height) {
            uint32_t width;
            for (width = 0u; width < output.width; ++width) {
                campp_elementwise_sigmoid_lut_mul_f32(
                    campp_packed_iteration_const_pointer(
                        &quantized, batch, 0u, height, width),
                    (const float *)campp_packed_iteration_const_pointer(
                        &other, batch, 0u, height, width),
                    (float *)campp_packed_iteration_pointer(
                        &output, batch, 0u, height, width),
                    output.channels, table);
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static bool campp_remaining_reshape_request_matches(
    const CamppTensorView *input, const CamppTensorView *shape,
    const CamppTensorView *output)
{
    uint64_t known_product = 1u;
    uint64_t input_count = campp_tensor_view_element_count(input);
    int8_t inferred_axis = -1;
    uint8_t axis;
    if (shape->dtype != CAMPP_DTYPE_INT64 ||
        campp_tensor_view_element_count(shape) != output->rank) {
        return false;
    }
    for (axis = 0u; axis < output->rank; ++axis) {
        int64_t requested;
        uint64_t dimension;
        if (campp_reference_read_i64(shape, axis, &requested) !=
            CAMPP_STATUS_OK) {
            return false;
        }
        if (requested == -1) {
            if (inferred_axis >= 0) return false;
            inferred_axis = (int8_t)axis;
            continue;
        }
        if (requested == 0) {
            if (axis >= input->rank) return false;
            dimension = input->dimensions[axis];
        } else if (requested > 0) {
            dimension = (uint64_t)requested;
        } else {
            return false;
        }
        if (dimension != output->dimensions[axis] ||
            known_product > UINT64_MAX / dimension) {
            return false;
        }
        known_product *= dimension;
    }
    if (inferred_axis >= 0) {
        return known_product != 0u && input_count % known_product == 0u &&
            input_count / known_product ==
                output->dimensions[(uint8_t)inferred_axis];
    }
    return known_product == input_count;
}

static CamppStatus campp_remaining_reshape_optimized(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    CamppPackedIteration input;
    CamppPackedIteration output;
    uint32_t batch;
    if (campp_reference_validate_invocation(
            inputs, input_count, 2u, 2u, outputs, output_count) !=
            CAMPP_STATUS_OK || inputs[0].dtype != outputs[0].dtype ||
        inputs[0].rank != 4u || outputs[0].rank != 3u ||
        !campp_remaining_reshape_request_matches(
            &inputs[0], &inputs[1], &outputs[0]) ||
        !campp_packed_iteration_create(&inputs[0], &input) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        input.batches != output.batches || input.width != output.width ||
        input.element_size != sizeof(uint32_t) ||
        input.channels > UINT32_MAX / input.height ||
        output.channels != input.channels * input.height) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (batch = 0u; batch < input.batches; ++batch) {
        uint32_t channel;
        for (channel = 0u; channel < input.channels; ++channel) {
            uint32_t height;
            for (height = 0u; height < input.height; ++height) {
                const uint32_t output_channel =
                    channel * input.height + height;
                uint32_t width;
                for (width = 0u; width < input.width; ++width) {
                    uint32_t bits;
                    memcpy(
                        &bits,
                        campp_packed_iteration_const_pointer(
                            &input, batch, channel, height, width),
                        sizeof(bits));
                    memcpy(
                        campp_packed_iteration_pointer(
                            &output, batch, output_channel, 0u, width),
                        &bits, sizeof(bits));
                }
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_remaining_statistics_optimized(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count)
{
    CamppPackedIteration input;
    CamppPackedIteration output;
    float multiplier;
    float divisor;
    uint32_t batch;
    if (campp_reference_validate_invocation(
            inputs, input_count, 3u, 3u, outputs, output_count) !=
            CAMPP_STATUS_OK || inputs[0].dtype != CAMPP_DTYPE_FLOAT32 ||
        outputs[0].dtype != CAMPP_DTYPE_FLOAT32 || inputs[0].rank != 3u ||
        outputs[0].rank != 2u ||
        campp_remaining_scalar_f32(&inputs[1], &multiplier) !=
            CAMPP_STATUS_OK ||
        campp_remaining_scalar_f32(&inputs[2], &divisor) != CAMPP_STATUS_OK ||
        divisor == 0.0f ||
        !campp_packed_iteration_create(&inputs[0], &input) ||
        !campp_packed_iteration_create(&outputs[0], &output) ||
        input.batches != output.batches ||
        input.channels > UINT32_MAX / 2u ||
        output.channels != input.channels * 2u) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (batch = 0u; batch < input.batches; ++batch) {
        campp_reduction_statistics_channels_f32(
            campp_packed_iteration_const_pointer(&input, batch, 0u, 0u, 0u),
            input.strides[3], input.channels, input.width,
            multiplier, divisor,
            campp_packed_iteration_pointer(&output, batch, 0u, 0u, 0u),
            campp_packed_iteration_pointer(
                &output, batch, input.channels, 0u, 0u));
    }
    return CAMPP_STATUS_OK;
}

#define CAMPP_REMAINING_WRAPPER(function_name, optimized_call, fallback_call) \
    static CamppStatus function_name(                                         \
        const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,    \
        const CamppTensorView *inputs, uint8_t input_count,                   \
        CamppTensorView *outputs, uint8_t output_count,                       \
        void *scratch, size_t scratch_size)                                   \
    {                                                                         \
        CamppStatus status = (optimized_call);                                \
        if (status != CAMPP_STATUS_NOT_IMPLEMENTED) return status;            \
        return (fallback_call)(                                               \
            model, op, inputs, input_count, outputs, output_count,            \
            scratch, scratch_size);                                           \
    }

CAMPP_REMAINING_WRAPPER(
    campp_remaining_add,
    campp_remaining_add_optimized(inputs, input_count, outputs, output_count),
    campp_reference_add)
CAMPP_REMAINING_WRAPPER(
    campp_remaining_relu,
    campp_remaining_relu_optimized(inputs, input_count, outputs, output_count),
    campp_reference_relu)
CAMPP_REMAINING_WRAPPER(
    campp_remaining_quantize,
    campp_remaining_quantize_optimized(
        inputs, input_count, outputs, output_count),
    campp_reference_quantize_linear)
CAMPP_REMAINING_WRAPPER(
    campp_remaining_expand,
    campp_remaining_expand_optimized(
        inputs, input_count, outputs, output_count),
    campp_reference_expand)
CAMPP_REMAINING_WRAPPER(
    campp_remaining_slice,
    campp_remaining_slice_optimized(inputs, input_count, outputs, output_count),
    campp_reference_slice)
CAMPP_REMAINING_WRAPPER(
    campp_remaining_reduce_mean,
    campp_remaining_reduce_mean_optimized(
        model, op, inputs, input_count, outputs, output_count),
    campp_reference_reduce_mean)
CAMPP_REMAINING_WRAPPER(
    campp_remaining_average_pool,
    campp_remaining_average_pool_optimized(
        model, op, inputs, input_count, outputs, output_count),
    campp_reference_average_pool)
CAMPP_REMAINING_WRAPPER(
    campp_remaining_sigmoid_mul,
    campp_remaining_sigmoid_mul_optimized(
        inputs, input_count, outputs, output_count),
    campp_fused_dequant_sigmoid_mul)
CAMPP_REMAINING_WRAPPER(
    campp_remaining_reshape,
    campp_remaining_reshape_optimized(
        inputs, input_count, outputs, output_count),
    campp_reference_reshape)
CAMPP_REMAINING_WRAPPER(
    campp_remaining_statistics,
    campp_remaining_statistics_optimized(
        inputs, input_count, outputs, output_count),
    campp_fused_statistics_pooling)

#undef CAMPP_REMAINING_WRAPPER

static const CamppKernelEntry CAMPP_REMAINING_ENTRIES[] = {
    {CAMPP_OP_ADD, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_remaining_add, NULL, "add_stride"},
    {CAMPP_OP_RELU, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_remaining_relu, NULL, "relu_stride"},
    {CAMPP_OP_EXPAND, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_remaining_expand, NULL, "expand_stride"},
    {CAMPP_OP_SLICE, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_remaining_slice, NULL, "slice_stride"},
    {CAMPP_OP_QUANTIZE_LINEAR, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_remaining_quantize, NULL, "quantize_linear_stride"},
    {CAMPP_OP_REDUCE_MEAN, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_remaining_reduce_mean, NULL, "reduce_mean_stride"},
    {CAMPP_OP_MUL, CAMPP_FUSION_EPILOGUE_KERNEL_ID,
     campp_remaining_sigmoid_mul, NULL, "fused_dequant_sigmoid_mul"},
    {CAMPP_OP_AVERAGE_POOL, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_remaining_average_pool, NULL, "average_pool_stride"},
    {CAMPP_OP_RESHAPE, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_remaining_reshape, NULL, "reshape_stride"},
    {CAMPP_OP_CONCAT, CAMPP_FUSION_STATS_POOLING_KERNEL_ID,
     campp_remaining_statistics, NULL, "fused_statistics_pooling"}
};

const char *campp_remaining_candidate_mode_name(
    CamppRemainingCandidateMode mode)
{
    if (mode == CAMPP_REMAINING_CANDIDATE_BASELINE) return "baseline";
    if (mode == CAMPP_REMAINING_CANDIDATE_OPTIMIZED) return "optimized";
    return "invalid";
}

int campp_remaining_candidate_mode_parse(
    const char *text, CamppRemainingCandidateMode *out_mode)
{
    if (text == NULL || out_mode == NULL) return 1;
    if (strcmp(text, "baseline") == 0) {
        *out_mode = CAMPP_REMAINING_CANDIDATE_BASELINE;
        return 0;
    }
    if (strcmp(text, "optimized") == 0) {
        *out_mode = CAMPP_REMAINING_CANDIDATE_OPTIMIZED;
        return 0;
    }
    return 1;
}

const CamppKernelEntry *campp_remaining_candidate_entry(
    CamppRemainingCandidateMode mode, uint16_t opcode, uint16_t kernel_id)
{
    size_t index;
    if (mode != CAMPP_REMAINING_CANDIDATE_OPTIMIZED) return NULL;
    for (index = 0u;
         index < sizeof(CAMPP_REMAINING_ENTRIES) /
             sizeof(CAMPP_REMAINING_ENTRIES[0]);
         ++index) {
        if (CAMPP_REMAINING_ENTRIES[index].opcode == opcode &&
            CAMPP_REMAINING_ENTRIES[index].kernel_id == kernel_id) {
            return &CAMPP_REMAINING_ENTRIES[index];
        }
    }
    return NULL;
}
