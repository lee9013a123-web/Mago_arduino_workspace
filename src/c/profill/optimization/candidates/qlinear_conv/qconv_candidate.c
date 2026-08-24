#include "qconv_candidate.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_reference/reference_kernel_utils.h"
#include "conv_layer_hybrid_plan.h"
#include "internal/runtime_model.h"
#include "microkernels/qconv_mac_4x8.h"
#include "qconv_address_fastpath.h"
#include "qconv_hybrid_dispatch.h"
#include "qconv_mac_neon.h"
#include "qconv_v4_dispatch.h"
#include "qconv_v5_dispatch.h"

#if defined(CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS)
#include "campp_profill/optimization/stage_probe.h"
#else
#define CAMPP_OPTIMIZATION_STAGE_BEGIN(variable) ((void)0)
#define CAMPP_OPTIMIZATION_STAGE_END(stage, variable) ((void)0)
#endif

typedef struct CamppQconvCandidateContext {
    const CamppTensorView *input;
    const CamppTensorView *weight;
    CamppTensorView *output;
    CamppQconvAddressPlan address;
    int64_t kernel_shape[2];
    int64_t pads[4];
    int64_t strides[2];
    int64_t dilations[2];
    uint8_t spatial_rank;
    uint32_t group;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t inputs_per_group;
    uint32_t outputs_per_group;
    uint32_t input_blocks;
    uint32_t output_blocks;
    uint32_t kernel_elements;
    uint32_t output_spatial;
    float input_scale;
    float output_scale;
    int32_t input_zero;
    int32_t output_zero;
} CamppQconvCandidateContext;

static CamppStatus campp_qconv_candidate_attribute(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, int64_t *values, uint8_t capacity, uint8_t expected)
{
    uint8_t count = 0u;
    CamppStatus status = campp_runtime_model_attribute_ints(
        model, op, key, values, capacity, &count);
    if (status != CAMPP_STATUS_OK) return status;
    return count == expected ? CAMPP_STATUS_OK : CAMPP_STATUS_CORRUPT_PLAN;
}

static void campp_qconv_candidate_spatial_coordinates(
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

static CamppStatus campp_qconv_candidate_write(
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

static CamppStatus campp_qconv_candidate_prepare(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppQconvCandidateContext *context)
{
    int64_t group_value[1] = {1};
    CamppStatus status;
    uint8_t axis;

    memset(context, 0, sizeof(*context));
    context->kernel_shape[0] = 1;
    context->kernel_shape[1] = 1;
    context->strides[0] = 1;
    context->strides[1] = 1;
    context->dilations[0] = 1;
    context->dilations[1] = 1;
    context->group = 1u;
    context->kernel_elements = 1u;
    context->output_spatial = 1u;

    status = campp_reference_validate_invocation(
        inputs, input_count, 8u, 9u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    context->input = &inputs[0];
    context->weight = &inputs[3];
    context->output = &outputs[0];
    if ((context->weight->flags &
         CAMPP_TENSOR_FLAG_PACKED_QCONV_O4I4) == 0u ||
        (context->input->dtype != CAMPP_DTYPE_INT8 &&
         context->input->dtype != CAMPP_DTYPE_UINT8) ||
        (context->weight->dtype != CAMPP_DTYPE_INT8 &&
         context->weight->dtype != CAMPP_DTYPE_UINT8) ||
        (context->output->dtype != CAMPP_DTYPE_INT8 &&
         context->output->dtype != CAMPP_DTYPE_UINT8) ||
        context->input->rank < 3u || context->input->rank > 4u ||
        context->weight->rank != context->input->rank ||
        context->output->rank != context->input->rank ||
        context->output->dimensions[0] != context->input->dimensions[0]) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    context->spatial_rank = (uint8_t)(context->input->rank - 2u);
    status = campp_qconv_candidate_attribute(
        model, op, CAMPP_ATTR_KERNEL_SHAPE, context->kernel_shape,
        2u, context->spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_candidate_attribute(
        model, op, CAMPP_ATTR_PADS, context->pads, 4u,
        (uint8_t)(context->spatial_rank * 2u));
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_candidate_attribute(
        model, op, CAMPP_ATTR_STRIDES, context->strides,
        2u, context->spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_candidate_attribute(
        model, op, CAMPP_ATTR_DILATIONS, context->dilations,
        2u, context->spatial_rank);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_candidate_attribute(
        model, op, CAMPP_ATTR_GROUP, group_value, 1u, 1u);
    if (status != CAMPP_STATUS_OK || group_value[0] <= 0 ||
        group_value[0] > UINT32_MAX) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    context->group = (uint32_t)group_value[0];
    context->input_channels = context->input->dimensions[1];
    context->output_channels = context->weight->dimensions[0];
    if (context->input_channels % context->group != 0u ||
        context->output_channels % context->group != 0u ||
        context->output->dimensions[1] != context->output_channels) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    context->inputs_per_group =
        context->input_channels / context->group;
    context->outputs_per_group =
        context->output_channels / context->group;
    if (context->weight->dimensions[1] != context->inputs_per_group ||
        (uint64_t)context->inputs_per_group * UINT64_C(65025) > INT32_MAX) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    context->input_blocks =
        (context->inputs_per_group + 3u) / 4u;
    context->output_blocks =
        (context->outputs_per_group + 3u) / 4u;
    for (axis = 0u; axis < context->spatial_rank; ++axis) {
        uint64_t effective;
        uint64_t padded;
        uint64_t expected;
        if (context->kernel_shape[axis] <= 0 ||
            context->strides[axis] <= 0 ||
            context->dilations[axis] <= 0 || context->pads[axis] < 0 ||
            context->pads[axis + context->spatial_rank] < 0 ||
            context->kernel_shape[axis] !=
                context->weight->dimensions[axis + 2u]) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        context->kernel_elements *=
            (uint32_t)context->kernel_shape[axis];
        context->output_spatial *=
            context->output->dimensions[axis + 2u];
        effective = (uint64_t)context->dilations[axis]
            * (uint64_t)(context->kernel_shape[axis] - 1) + 1u;
        padded = (uint64_t)context->input->dimensions[axis + 2u]
            + (uint64_t)context->pads[axis]
            + (uint64_t)context->pads[axis + context->spatial_rank];
        if (padded < effective) return CAMPP_STATUS_SHAPE_MISMATCH;
        expected = (padded - effective)
            / (uint64_t)context->strides[axis] + 1u;
        if (expected != context->output->dimensions[axis + 2u]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    if (context->weight->storage_span_bytes !=
        (uint64_t)context->group * context->output_blocks
            * context->kernel_elements * context->input_blocks * 16u) {
        return CAMPP_STATUS_CORRUPT_WEIGHTS;
    }
    status = campp_reference_read_f32(
        &inputs[1], 0u, &context->input_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_quantized(
        &inputs[2], 0u, &context->input_zero);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_f32(
        &inputs[6], 0u, &context->output_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_quantized(
        &inputs[7], 0u, &context->output_zero);
    if (status != CAMPP_STATUS_OK) return status;
    if (!(context->input_scale > 0.0f) ||
        !(context->output_scale > 0.0f) ||
        !isfinite(context->input_scale) ||
        !isfinite(context->output_scale)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    if (!campp_qconv_address_plan_create(
            context->input, context->output, context->spatial_rank,
            context->strides, context->dilations, context->pads,
            &context->address)) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_qconv_candidate_run_v1(
    CamppQconvCandidateMode mode, const CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op, const CamppTensorView *inputs,
    uint8_t input_count, CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    CamppQconvCandidateContext context;
    CamppStatus status;
    uint32_t batch;

    CAMPP_OPTIMIZATION_STAGE_BEGIN(setup_started_ns);
    status = campp_qconv_candidate_prepare(
        model, op, inputs, input_count, outputs, output_count, &context);
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_QCONV_SETUP, setup_started_ns);
    if (status == CAMPP_STATUS_NOT_IMPLEMENTED) {
        return campp_aarch64_qlinear_conv_o4i4(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    if (status != CAMPP_STATUS_OK) return status;

    for (batch = 0u; batch < context.input->dimensions[0]; ++batch) {
        uint32_t group_index;
        for (group_index = 0u;
             group_index < context.group; ++group_index) {
            uint32_t output_block;
            for (output_block = 0u;
                 output_block < context.output_blocks; ++output_block) {
                int32_t weight_zero[4] = {0, 0, 0, 0};
                float multiplier[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                int32_t bias[4] = {0, 0, 0, 0};
                bool valid_output[4] = {false, false, false, false};
                uint32_t output_lane;
                uint32_t tile_start;

                for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                    const uint32_t within =
                        output_block * 4u + output_lane;
                    const uint32_t channel =
                        group_index * context.outputs_per_group + within;
                    uint64_t parameter_index;
                    float weight_scale;
                    if (within >= context.outputs_per_group) continue;
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
                        &inputs[5], parameter_index,
                        &weight_zero[output_lane]);
                    if (status != CAMPP_STATUS_OK) return status;
                    if (!(weight_scale > 0.0f) || !isfinite(weight_scale)) {
                        return CAMPP_STATUS_KERNEL_FAILED;
                    }
                    multiplier[output_lane] =
                        context.input_scale * weight_scale
                        / context.output_scale;
                    if (input_count == 9u) {
                        status = campp_reference_read_quantized(
                            &inputs[8], channel, &bias[output_lane]);
                        if (status != CAMPP_STATUS_OK) return status;
                    }
                }

                for (tile_start = 0u;
                     tile_start < context.output_spatial;
                     tile_start += CAMPP_QCONV_CANDIDATE_TILE) {
                    const uint32_t tile_count =
                        context.output_spatial - tile_start
                            < CAMPP_QCONV_CANDIDATE_TILE
                        ? context.output_spatial - tile_start
                        : CAMPP_QCONV_CANDIDATE_TILE;
                    uint32_t output_coordinates
                        [CAMPP_QCONV_CANDIDATE_TILE][2];
                    int64_t accum[CAMPP_QCONV_CANDIDATE_TILE][4];
                    uint32_t kernel_index;
                    uint32_t tile;

                    campp_qconv_address_output_tile(
                        &context.address, tile_start, tile_count,
                        output_coordinates);
                    for (tile = 0u; tile < tile_count; ++tile) {
                        for (output_lane = 0u;
                             output_lane < 4u; ++output_lane) {
                            accum[tile][output_lane] = bias[output_lane];
                        }
                    }
                    CAMPP_OPTIMIZATION_STAGE_BEGIN(mac_started_ns);
                    for (kernel_index = 0u;
                         kernel_index < context.kernel_elements;
                         ++kernel_index) {
                        const uint64_t packed_offset =
                            (((uint64_t)group_index
                              * context.output_blocks + output_block)
                             * context.kernel_elements + kernel_index)
                            * context.input_blocks * 16u;
                        const uint8_t *input_points
                            [CAMPP_QCONV_CANDIDATE_TILE] = {NULL};
                        int32_t contribution
                            [CAMPP_QCONV_CANDIDATE_TILE][4];
                        uint32_t kernel_coordinates[2];

                        campp_qconv_candidate_spatial_coordinates(
                            context.spatial_rank, kernel_index,
                            context.spatial_rank == 1u
                                ? 1u
                                : (uint32_t)context.kernel_shape[1],
                            kernel_coordinates);
                        for (tile = 0u; tile < tile_count; ++tile) {
                            input_points[tile] =
                                campp_qconv_address_input_base(
                                    &context.address, batch,
                                    group_index
                                        * context.inputs_per_group,
                                    output_coordinates[tile],
                                    kernel_coordinates,
                                    mode != CAMPP_QCONV_CANDIDATE_MAC);
                        }
                        if (mode == CAMPP_QCONV_CANDIDATE_ADDRESS) {
                            campp_qconv_mac_scalar_tile(
                                input_points, tile_count,
                                (const uint8_t *)context.weight->data
                                    + packed_offset,
                                context.inputs_per_group,
                                context.input->dtype,
                                context.weight->dtype,
                                context.input_zero, weight_zero,
                                contribution);
                        } else {
                            campp_qconv_mac_neon_tile(
                                input_points, tile_count,
                                (const uint8_t *)context.weight->data
                                    + packed_offset,
                                context.inputs_per_group,
                                context.input->dtype,
                                context.weight->dtype,
                                context.input_zero, weight_zero,
                                contribution);
                        }
                        for (tile = 0u; tile < tile_count; ++tile) {
                            for (output_lane = 0u;
                                 output_lane < 4u; ++output_lane) {
                                accum[tile][output_lane] +=
                                    contribution[tile][output_lane];
                            }
                        }
                    }
                    CAMPP_OPTIMIZATION_STAGE_END(
                        CAMPP_OPT_STAGE_QCONV_MAC_ADDRESS,
                        mac_started_ns);
                    CAMPP_OPTIMIZATION_STAGE_BEGIN(requant_started_ns);
                    for (tile = 0u; tile < tile_count; ++tile) {
                        for (output_lane = 0u;
                             output_lane < 4u; ++output_lane) {
                            const uint32_t within =
                                output_block * 4u + output_lane;
                            const uint32_t channel =
                                group_index * context.outputs_per_group
                                + within;
                            const uint64_t logical_index =
                                ((uint64_t)batch
                                 * context.output_channels + channel)
                                * context.output_spatial
                                + tile_start + tile;
                            if (!valid_output[output_lane]) continue;
                            status = campp_qconv_candidate_write(
                                context.output, logical_index,
                                accum[tile][output_lane],
                                multiplier[output_lane],
                                context.output_zero);
                            if (status != CAMPP_STATUS_OK) return status;
                        }
                    }
                    CAMPP_OPTIMIZATION_STAGE_END(
                        CAMPP_OPT_STAGE_QCONV_REQUANT_WRITE,
                        requant_started_ns);
                }
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_qconv_candidate_load_parameters(
    const CamppQconvCandidateContext *context,
    const CamppTensorView *inputs, uint8_t input_count,
    uint32_t group_index, uint32_t first_within, uint32_t capacity,
    int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    float multiplier[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t *out_valid_outputs)
{
    const uint32_t remaining =
        context->outputs_per_group - first_within;
    const uint32_t valid_outputs = remaining < capacity
        ? remaining : capacity;
    uint32_t output_lane;

    memset(
        weight_zero, 0,
        sizeof(*weight_zero) * CAMPP_QCONV_CANDIDATE_OUTPUT_TILE);
    memset(
        multiplier, 0,
        sizeof(*multiplier) * CAMPP_QCONV_CANDIDATE_OUTPUT_TILE);
    memset(
        bias, 0,
        sizeof(*bias) * CAMPP_QCONV_CANDIDATE_OUTPUT_TILE);
    for (output_lane = 0u;
         output_lane < valid_outputs; ++output_lane) {
        const uint32_t channel =
            group_index * context->outputs_per_group
            + first_within + output_lane;
        uint64_t parameter_index;
        float weight_scale;
        CamppStatus status;

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
            context->input_scale * weight_scale / context->output_scale;
        if (input_count == 9u) {
            status = campp_reference_read_quantized(
                &inputs[8], channel, &bias[output_lane]);
            if (status != CAMPP_STATUS_OK) return status;
        }
    }
    *out_valid_outputs = valid_outputs;
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_qconv_candidate_run_v2(
    CamppQconvCandidateMode mode, const CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op, const CamppTensorView *inputs,
    uint8_t input_count, CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    CamppQconvCandidateContext context;
    CamppStatus status;
    uint32_t batch;

    CAMPP_OPTIMIZATION_STAGE_BEGIN(setup_started_ns);
    status = campp_qconv_candidate_prepare(
        model, op, inputs, input_count, outputs, output_count, &context);
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_QCONV_SETUP, setup_started_ns);
    if (status == CAMPP_STATUS_NOT_IMPLEMENTED) {
        return campp_aarch64_qlinear_conv_o4i4(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    if (status != CAMPP_STATUS_OK) return status;
    if (context.kernel_elements >
            CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS ||
        (uint64_t)context.kernel_elements * context.inputs_per_group
            * UINT64_C(65025) > INT32_MAX ||
        context.output->byte_strides[1] != 1u) {
        return campp_aarch64_qlinear_conv_o4i4(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }

    for (batch = 0u; batch < context.input->dimensions[0]; ++batch) {
        uint32_t group_index;
        for (group_index = 0u;
             group_index < context.group; ++group_index) {
            uint32_t output_block;
            for (output_block = 0u;
                 output_block < context.output_blocks;
                 output_block += 2u) {
                const uint32_t first_within = output_block
                    * CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK;
                int32_t weight_zero
                    [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
                float multiplier[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
                int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
                const uint8_t *packed_weights[2] = {NULL, NULL};
                uint32_t valid_outputs;
                uint32_t tile_start;

                status = campp_qconv_candidate_load_parameters(
                    &context, inputs, input_count, group_index,
                    first_within, CAMPP_QCONV_CANDIDATE_OUTPUT_TILE,
                    weight_zero, multiplier, bias, &valid_outputs);
                if (status != CAMPP_STATUS_OK) return status;
                packed_weights[0] =
                    (const uint8_t *)context.weight->data
                    + (((uint64_t)group_index * context.output_blocks
                        + output_block)
                       * context.kernel_elements * context.input_blocks
                       * CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK
                       * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK);
                if (output_block + 1u < context.output_blocks) {
                    packed_weights[1] =
                        (const uint8_t *)context.weight->data
                        + (((uint64_t)group_index * context.output_blocks
                            + output_block + 1u)
                           * context.kernel_elements * context.input_blocks
                           * CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK
                           * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK);
                }

                for (tile_start = 0u;
                     tile_start < context.output_spatial;
                     tile_start += CAMPP_QCONV_CANDIDATE_TILE) {
                    const uint32_t tile_count =
                        context.output_spatial - tile_start
                            < CAMPP_QCONV_CANDIDATE_TILE
                        ? context.output_spatial - tile_start
                        : CAMPP_QCONV_CANDIDATE_TILE;
                    uint32_t output_coordinates
                        [CAMPP_QCONV_CANDIDATE_TILE][2];
                    const uint8_t *input_points
                        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
                        [CAMPP_QCONV_CANDIDATE_TILE] = {{NULL}};
                    int32_t accumulators
                        [CAMPP_QCONV_CANDIDATE_TILE]
                        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
                    uint32_t kernel_index;

                    campp_qconv_address_output_tile(
                        &context.address, tile_start, tile_count,
                        output_coordinates);
                    CAMPP_OPTIMIZATION_STAGE_BEGIN(mac_started_ns);
                    for (kernel_index = 0u;
                         kernel_index < context.kernel_elements;
                         ++kernel_index) {
                        uint32_t kernel_coordinates[2];
                        uint32_t tile;
                        campp_qconv_candidate_spatial_coordinates(
                            context.spatial_rank, kernel_index,
                            context.spatial_rank == 1u
                                ? 1u
                                : (uint32_t)context.kernel_shape[1],
                            kernel_coordinates);
                        for (tile = 0u; tile < tile_count; ++tile) {
                            input_points[kernel_index][tile] =
                                campp_qconv_address_input_base(
                                    &context.address, batch,
                                    group_index
                                        * context.inputs_per_group,
                                    output_coordinates[tile],
                                    kernel_coordinates,
                                    mode == CAMPP_QCONV_CANDIDATE_COMBINED);
                        }
                    }
                    if (mode == CAMPP_QCONV_CANDIDATE_MAC_FIXED ||
                        mode == CAMPP_QCONV_CANDIDATE_MAC_ASM) {
                        const CamppQconvMac4x8Implementation implementation =
                            mode == CAMPP_QCONV_CANDIDATE_MAC_ASM
                            ? CAMPP_QCONV_MAC_4X8_ASSEMBLY
                            : CAMPP_QCONV_MAC_4X8_INTRINSICS;
                        const CamppQconvMac4x8Result fixed_result =
                            campp_qconv_mac_4x8_try_tile(
                                input_points, context.kernel_elements,
                                tile_count, packed_weights,
                                context.inputs_per_group,
                                context.input->dtype,
                                context.weight->dtype,
                                context.input_zero, weight_zero, bias,
                                valid_outputs, implementation,
                                accumulators);
                        if (fixed_result == CAMPP_QCONV_MAC_4X8_FAILED) {
                            return CAMPP_STATUS_KERNEL_FAILED;
                        }
                        if (fixed_result == CAMPP_QCONV_MAC_4X8_UNSUPPORTED &&
                            campp_qconv_mac_neon_tile_v2(
                                input_points, context.kernel_elements,
                                tile_count, packed_weights,
                                context.inputs_per_group,
                                context.input->dtype,
                                context.weight->dtype,
                                context.input_zero, weight_zero, bias,
                                valid_outputs, accumulators) != 0) {
                            return CAMPP_STATUS_KERNEL_FAILED;
                        }
                    } else if (campp_qconv_mac_neon_tile_v2(
                                   input_points, context.kernel_elements,
                                   tile_count, packed_weights,
                                   context.inputs_per_group,
                                   context.input->dtype,
                                   context.weight->dtype,
                                   context.input_zero, weight_zero, bias,
                                   valid_outputs, accumulators) != 0) {
                        return CAMPP_STATUS_KERNEL_FAILED;
                    }
                    CAMPP_OPTIMIZATION_STAGE_END(
                        CAMPP_OPT_STAGE_QCONV_MAC_ADDRESS,
                        mac_started_ns);

                    CAMPP_OPTIMIZATION_STAGE_BEGIN(requant_started_ns);
                    if (campp_qconv_requantize_store_neon_tile(
                            context.output, batch,
                            group_index * context.outputs_per_group
                                + first_within,
                            context.spatial_rank, output_coordinates,
                            tile_count, valid_outputs, accumulators,
                            multiplier, context.output_zero) != 0) {
                        return CAMPP_STATUS_KERNEL_FAILED;
                    }
                    CAMPP_OPTIMIZATION_STAGE_END(
                        CAMPP_OPT_STAGE_QCONV_REQUANT_WRITE,
                        requant_started_ns);
                }
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_qconv_candidate_run(
    CamppQconvCandidateMode mode, const CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op, const CamppTensorView *inputs,
    uint8_t input_count, CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    if (mode == CAMPP_QCONV_CANDIDATE_ADDRESS) {
        return campp_qconv_candidate_run_v1(
            mode, model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    return campp_qconv_candidate_run_v2(
        mode, model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size);
}

#define CAMPP_DEFINE_QCONV_CANDIDATE(function_name, candidate_mode) \
    static CamppStatus function_name(                                  \
        const CamppRuntimeModel *model,                                \
        const CamppOperatorDescriptor *op,                             \
        const CamppTensorView *inputs, uint8_t input_count,            \
        CamppTensorView *outputs, uint8_t output_count,                 \
        void *scratch, size_t scratch_size)                            \
    {                                                                  \
        return campp_qconv_candidate_run(                              \
            (candidate_mode), model, op, inputs, input_count,          \
            outputs, output_count, scratch, scratch_size);             \
    }

CAMPP_DEFINE_QCONV_CANDIDATE(
    campp_qconv_candidate_address, CAMPP_QCONV_CANDIDATE_ADDRESS)
CAMPP_DEFINE_QCONV_CANDIDATE(
    campp_qconv_candidate_mac, CAMPP_QCONV_CANDIDATE_MAC)
CAMPP_DEFINE_QCONV_CANDIDATE(
    campp_qconv_candidate_combined, CAMPP_QCONV_CANDIDATE_COMBINED)
CAMPP_DEFINE_QCONV_CANDIDATE(
    campp_qconv_candidate_mac_fixed, CAMPP_QCONV_CANDIDATE_MAC_FIXED)
CAMPP_DEFINE_QCONV_CANDIDATE(
    campp_qconv_candidate_mac_asm, CAMPP_QCONV_CANDIDATE_MAC_ASM)

static CamppStatus campp_qconv_candidate_v4(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    return campp_qconv_v4_run(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size);
}

static CamppStatus campp_qconv_candidate_hybrid(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    return campp_qconv_hybrid_run(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size);
}

static CamppStatus campp_qconv_candidate_v5(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    return campp_qconv_v5_run(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size);
}

static CamppStatus campp_qconv_candidate_layer_hybrid_v3(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    const CamppQconvCandidateMode selected =
        campp_conv_layer_hybrid_select_qconv(model, op);
    const CamppKernelEntry *entry = campp_qconv_candidate_entry(selected);

    if (entry == NULL || entry->run == NULL ||
        selected == CAMPP_QCONV_CANDIDATE_LAYER_HYBRID_V3) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    return entry->run(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size);
}

static const CamppKernelEntry CAMPP_QCONV_CANDIDATE_ENTRIES[] = {
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_address,
        NULL,
        "qlinear_conv_o4i4_neon"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_mac,
        NULL,
        "qlinear_conv_o4i4_neon"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_combined,
        NULL,
        "qlinear_conv_o4i4_neon"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_mac_fixed,
        NULL,
        "qlinear_conv_o4i4_neon"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_mac_asm,
        NULL,
        "qlinear_conv_o4i4_neon"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_v4,
        NULL,
        "qlinear_conv_o4i4_v4"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_hybrid,
        NULL,
        "qlinear_conv_o4i4_hybrid"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_v5,
        NULL,
        "qlinear_conv_o4i4_v5"
    },
    {
        CAMPP_OP_QLINEAR_CONV,
        CAMPP_AARCH64_PACKED_KERNEL_ID,
        campp_qconv_candidate_layer_hybrid_v3,
        NULL,
        "qlinear_conv_o4i4_layer_hybrid_v3"
    }
};

const char *campp_qconv_candidate_mode_name(CamppQconvCandidateMode mode)
{
    static const char *const names[] = {
        "baseline", "address", "mac", "combined", "mac_fixed", "mac_asm",
        "v4", "hybrid", "v5", "layer_hybrid_v3"
    };
    return mode >= CAMPP_QCONV_CANDIDATE_BASELINE &&
        mode <= CAMPP_QCONV_CANDIDATE_LAYER_HYBRID_V3
        ? names[mode] : "invalid";
}

int campp_qconv_candidate_mode_parse(
    const char *text, CamppQconvCandidateMode *out_mode)
{
    CamppQconvCandidateMode mode;
    if (text == NULL || out_mode == NULL) return 1;
    for (mode = CAMPP_QCONV_CANDIDATE_BASELINE;
         mode <= CAMPP_QCONV_CANDIDATE_LAYER_HYBRID_V3;
         mode = (CamppQconvCandidateMode)(mode + 1)) {
        if (strcmp(text, campp_qconv_candidate_mode_name(mode)) == 0) {
            *out_mode = mode;
            return 0;
        }
    }
    return 1;
}

const CamppKernelEntry *campp_qconv_candidate_entry(
    CamppQconvCandidateMode mode)
{
    if (mode == CAMPP_QCONV_CANDIDATE_BASELINE) return NULL;
    if (mode < CAMPP_QCONV_CANDIDATE_ADDRESS ||
        mode > CAMPP_QCONV_CANDIDATE_LAYER_HYBRID_V3) {
        return NULL;
    }
    return &CAMPP_QCONV_CANDIDATE_ENTRIES[mode - 1];
}

#undef CAMPP_DEFINE_QCONV_CANDIDATE
