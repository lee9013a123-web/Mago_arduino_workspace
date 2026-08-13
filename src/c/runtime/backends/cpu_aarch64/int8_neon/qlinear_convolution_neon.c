/* O4I4 packed INT8 QLinearConv. SDOT을 가정하지 않는 AArch64 NEON 경로다. */

#include "backends/cpu_aarch64/aarch64_kernels.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "backends/cpu_reference/reference_kernel_utils.h"
#include "internal/runtime_model.h"

#define CAMPP_QCONV_OUTPUT_BLOCK 4u
#define CAMPP_QCONV_INPUT_BLOCK 4u
#define CAMPP_QCONV_SPATIAL_TILE 8u

static CamppStatus campp_qconv_attribute(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, int64_t *values, uint8_t capacity, uint8_t expected)
{
    uint8_t count = 0u;
    CamppStatus status = campp_runtime_model_attribute_ints(
        model, op, key, values, capacity, &count);
    if (status != CAMPP_STATUS_OK) return status;
    return count == expected ? CAMPP_STATUS_OK : CAMPP_STATUS_CORRUPT_PLAN;
}

static int32_t campp_read_qbyte(const CamppTensorView *view, uint64_t offset)
{
    if (view->dtype == CAMPP_DTYPE_UINT8) {
        return *((const uint8_t *)view->data + offset);
    }
    {
        int8_t value;
        memcpy(&value, (const uint8_t *)view->data + offset, sizeof(value));
        return value;
    }
}

static int32_t campp_dot4_i16(
    const int16_t input[CAMPP_QCONV_INPUT_BLOCK],
    const int16_t weight[CAMPP_QCONV_INPUT_BLOCK])
{
#if defined(__aarch64__) && defined(__ARM_NEON)
    const int16x4_t x = vld1_s16(input);
    const int16x4_t w = vld1_s16(weight);
    return vaddvq_s32(vmull_s16(x, w));
#else
    uint32_t lane;
    int32_t result = 0;
    for (lane = 0u; lane < CAMPP_QCONV_INPUT_BLOCK; ++lane) {
        result += (int32_t)input[lane] * (int32_t)weight[lane];
    }
    return result;
#endif
}

static void campp_spatial_coordinates(
    uint8_t spatial_rank, uint32_t linear, uint32_t width,
    uint32_t coordinates[2])
{
    if (spatial_rank == 1u) {
        coordinates[0] = linear;
        coordinates[1] = 0u;
    } else {
        coordinates[0] = linear / width;
        coordinates[1] = linear % width;
    }
}

static CamppStatus campp_qconv_write(
    CamppTensorView *output, uint64_t logical_index, int64_t accumulator,
    float multiplier, int32_t output_zero)
{
    int64_t rounded;
    float scaled;
    if (accumulator < INT32_MIN || accumulator > INT32_MAX ||
        !isfinite(multiplier)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    scaled = (float)(int32_t)accumulator * multiplier;
    if (scaled <= -2147483648.0f) {
        rounded = INT32_MIN;
    } else if (scaled >= 2147483520.0f) {
        rounded = INT32_MAX;
    } else {
        rounded = (int64_t)nearbyintf(scaled);
    }
    return campp_reference_write_quantized(
        output, logical_index, rounded + output_zero);
}

CamppStatus campp_aarch64_qlinear_conv_o4i4(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    int64_t kernel_shape[2] = {1, 1};
    int64_t pads[4] = {0, 0, 0, 0};
    int64_t strides[2] = {1, 1};
    int64_t dilations[2] = {1, 1};
    int64_t group_value[1] = {1};
    const CamppTensorView *x;
    const CamppTensorView *weight;
    CamppTensorView *y;
    uint8_t spatial_rank;
    uint32_t group;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t inputs_per_group;
    uint32_t outputs_per_group;
    uint32_t input_blocks;
    uint32_t output_blocks;
    uint32_t kernel_elements = 1u;
    uint32_t output_spatial = 1u;
    uint32_t batch;
    float input_scale;
    float output_scale;
    int32_t input_zero;
    int32_t output_zero;
    CamppStatus status;

    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 8u, 9u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    x = &inputs[0];
    weight = &inputs[3];
    y = &outputs[0];
    if ((weight->flags & CAMPP_TENSOR_FLAG_PACKED_QCONV_O4I4) == 0u ||
        (x->dtype != CAMPP_DTYPE_INT8 && x->dtype != CAMPP_DTYPE_UINT8) ||
        (weight->dtype != CAMPP_DTYPE_INT8 &&
         weight->dtype != CAMPP_DTYPE_UINT8) ||
        (y->dtype != CAMPP_DTYPE_INT8 && y->dtype != CAMPP_DTYPE_UINT8) ||
        x->rank < 3u || x->rank > 4u || weight->rank != x->rank ||
        y->rank != x->rank || y->dimensions[0] != x->dimensions[0]) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    spatial_rank = (uint8_t)(x->rank - 2u);
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_KERNEL_SHAPE, kernel_shape, 2u, spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_PADS, pads, 4u, (uint8_t)(spatial_rank * 2u));
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_STRIDES, strides, 2u, spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_DILATIONS, dilations, 2u, spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_attribute(
        model, op, CAMPP_ATTR_GROUP, group_value, 1u, 1u);
    if (status != CAMPP_STATUS_OK || group_value[0] <= 0 ||
        group_value[0] > UINT32_MAX) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    group = (uint32_t)group_value[0];
    input_channels = x->dimensions[1];
    output_channels = weight->dimensions[0];
    if (input_channels % group != 0u || output_channels % group != 0u ||
        y->dimensions[1] != output_channels) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    inputs_per_group = input_channels / group;
    outputs_per_group = output_channels / group;
    if (weight->dimensions[1] != inputs_per_group) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    input_blocks = (inputs_per_group + 3u) / 4u;
    output_blocks = (outputs_per_group + 3u) / 4u;
    {
        uint8_t axis;
        for (axis = 0u; axis < spatial_rank; ++axis) {
            uint64_t effective;
            uint64_t padded;
            uint64_t expected;
            if (kernel_shape[axis] <= 0 || strides[axis] <= 0 ||
                dilations[axis] <= 0 || pads[axis] < 0 ||
                pads[axis + spatial_rank] < 0 ||
                kernel_shape[axis] != weight->dimensions[axis + 2u]) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            kernel_elements *= (uint32_t)kernel_shape[axis];
            output_spatial *= y->dimensions[axis + 2u];
            effective = (uint64_t)dilations[axis] *
                (uint64_t)(kernel_shape[axis] - 1) + 1u;
            padded = (uint64_t)x->dimensions[axis + 2u] +
                (uint64_t)pads[axis] + (uint64_t)pads[axis + spatial_rank];
            if (padded < effective) return CAMPP_STATUS_SHAPE_MISMATCH;
            expected = (padded - effective) / (uint64_t)strides[axis] + 1u;
            if (expected != y->dimensions[axis + 2u]) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
        }
    }
    if (weight->storage_span_bytes !=
        (uint64_t)group * output_blocks * kernel_elements * input_blocks * 16u) {
        return CAMPP_STATUS_CORRUPT_WEIGHTS;
    }
    status = campp_reference_read_f32(&inputs[1], 0u, &input_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_quantized(&inputs[2], 0u, &input_zero);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_f32(&inputs[6], 0u, &output_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_quantized(&inputs[7], 0u, &output_zero);
    if (status != CAMPP_STATUS_OK) return status;
    if (!(input_scale > 0.0f) || !(output_scale > 0.0f) ||
        !isfinite(input_scale) || !isfinite(output_scale)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }

    for (batch = 0u; batch < x->dimensions[0]; ++batch) {
        uint32_t group_index;
        for (group_index = 0u; group_index < group; ++group_index) {
            uint32_t output_block;
            for (output_block = 0u; output_block < output_blocks; ++output_block) {
                int32_t weight_zero[4] = {0, 0, 0, 0};
                float multiplier[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                int32_t bias[4] = {0, 0, 0, 0};
                bool valid_output[4] = {false, false, false, false};
                uint32_t output_lane;
                uint32_t tile_start;
                for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                    const uint32_t within = output_block * 4u + output_lane;
                    const uint32_t channel =
                        group_index * outputs_per_group + within;
                    uint64_t parameter_index;
                    float weight_scale;
                    if (within >= outputs_per_group) continue;
                    valid_output[output_lane] = true;
                    parameter_index =
                        campp_tensor_view_element_count(&inputs[4]) == 1u
                            ? 0u : channel;
                    status = campp_reference_read_f32(
                        &inputs[4], parameter_index, &weight_scale);
                    if (status != CAMPP_STATUS_OK) return status;
                    parameter_index =
                        campp_tensor_view_element_count(&inputs[5]) == 1u
                            ? 0u : channel;
                    status = campp_reference_read_quantized(
                        &inputs[5], parameter_index, &weight_zero[output_lane]);
                    if (status != CAMPP_STATUS_OK) return status;
                    if (!(weight_scale > 0.0f) || !isfinite(weight_scale)) {
                        return CAMPP_STATUS_KERNEL_FAILED;
                    }
                    multiplier[output_lane] =
                        input_scale * weight_scale / output_scale;
                    if (input_count == 9u) {
                        status = campp_reference_read_quantized(
                            &inputs[8], channel, &bias[output_lane]);
                        if (status != CAMPP_STATUS_OK) return status;
                    }
                }

                for (tile_start = 0u; tile_start < output_spatial;
                     tile_start += CAMPP_QCONV_SPATIAL_TILE) {
                    const uint32_t tile_count =
                        output_spatial - tile_start < CAMPP_QCONV_SPATIAL_TILE
                            ? output_spatial - tile_start
                            : CAMPP_QCONV_SPATIAL_TILE;
                    int64_t accum[CAMPP_QCONV_SPATIAL_TILE][4];
                    uint32_t tile;
                    uint32_t kernel_index;
                    for (tile = 0u; tile < tile_count; ++tile) {
                        for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                            accum[tile][output_lane] = bias[output_lane];
                        }
                    }
                    for (kernel_index = 0u; kernel_index < kernel_elements;
                         ++kernel_index) {
                        uint32_t kernel_coordinates[2];
                        uint32_t input_block;
                        campp_spatial_coordinates(
                            spatial_rank, kernel_index,
                            spatial_rank == 1u ? 1u : (uint32_t)kernel_shape[1],
                            kernel_coordinates);
                        for (input_block = 0u; input_block < input_blocks;
                             ++input_block) {
                            const uint64_t packed_offset =
                                (((uint64_t)group_index * output_blocks + output_block)
                                  * kernel_elements + kernel_index)
                                    * input_blocks * 16u
                                + (uint64_t)input_block * 16u;
                            int16_t centered_weight[4][4];
                            for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                                uint32_t input_lane;
                                for (input_lane = 0u; input_lane < 4u; ++input_lane) {
                                    centered_weight[output_lane][input_lane] =
                                        (int16_t)(campp_read_qbyte(
                                            weight, packed_offset +
                                                output_lane * 4u + input_lane)
                                            - weight_zero[output_lane]);
                                }
                            }
                            for (tile = 0u; tile < tile_count; ++tile) {
                                const uint32_t output_linear = tile_start + tile;
                                uint32_t output_coordinates[2];
                                int64_t input_coordinates[2];
                                bool spatial_valid = true;
                                int16_t centered_input[4] = {0, 0, 0, 0};
                                uint32_t input_lane;
                                uint8_t axis;
                                campp_spatial_coordinates(
                                    spatial_rank, output_linear,
                                    spatial_rank == 1u ? 1u : y->dimensions[3],
                                    output_coordinates);
                                for (axis = 0u; axis < spatial_rank; ++axis) {
                                    input_coordinates[axis] =
                                        (int64_t)output_coordinates[axis] * strides[axis]
                                        + (int64_t)kernel_coordinates[axis] *
                                            dilations[axis]
                                        - pads[axis];
                                    if (input_coordinates[axis] < 0 ||
                                        input_coordinates[axis] >=
                                            (int64_t)x->dimensions[axis + 2u]) {
                                        spatial_valid = false;
                                    }
                                }
                                if (spatial_valid) {
                                    for (input_lane = 0u; input_lane < 4u; ++input_lane) {
                                        const uint32_t within =
                                            input_block * 4u + input_lane;
                                        uint32_t coordinates[CAMPP_TENSOR_MAX_RANK] =
                                            {batch, 0u, 0u, 0u};
                                        uint64_t offset;
                                        if (within >= inputs_per_group) continue;
                                        coordinates[1] =
                                            group_index * inputs_per_group + within;
                                        for (axis = 0u; axis < spatial_rank; ++axis) {
                                            coordinates[axis + 2u] =
                                                (uint32_t)input_coordinates[axis];
                                        }
                                        offset = campp_tensor_view_byte_offset(
                                            x, coordinates);
                                        centered_input[input_lane] =
                                            (int16_t)(campp_read_qbyte(x, offset) -
                                                      input_zero);
                                    }
                                }
                                for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                                    if (valid_output[output_lane]) {
                                        accum[tile][output_lane] += campp_dot4_i16(
                                            centered_input,
                                            centered_weight[output_lane]);
                                    }
                                }
                            }
                        }
                    }
                    for (tile = 0u; tile < tile_count; ++tile) {
                        for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                            const uint32_t within = output_block * 4u + output_lane;
                            const uint32_t channel =
                                group_index * outputs_per_group + within;
                            const uint64_t logical_index =
                                ((uint64_t)batch * output_channels + channel) *
                                    output_spatial + tile_start + tile;
                            if (!valid_output[output_lane]) continue;
                            status = campp_qconv_write(
                                y, logical_index, accum[tile][output_lane],
                                multiplier[output_lane], output_zero);
                            if (status != CAMPP_STATUS_OK) return status;
                        }
                    }
                }
            }
        }
    }
    return CAMPP_STATUS_OK;
}
