#include "qconv_v5_dispatch.h"

#include <stdbool.h>
#include <stdint.h>

#include "microkernels/qconv_mac_1x1_8x8_real.h"
#include "microkernels/qconv_mac_3x3_cin32_unroll.h"
#include "microkernels/qconv_mac_3x3_sliding_8x8.h"
#include "microkernels/qconv_mac_tail_v5.h"
#include "parameters/qconv_v4_parameters.h"
#include "parameters/qconv_v5_zero_point_fastpath.h"
#include "planning/qconv_v4_execution_plan.h"
#include "planning/qconv_v4_tile_plan.h"
#include "planning/qconv_v5_validated_raw_plan.h"
#include "qconv_candidate.h"
#include "requant/qconv_requant_neon8.h"
#include "store/qconv_store_channel_packed.h"

#if defined(CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS)
#include "campp_profill/optimization/stage_probe.h"
#else
#define CAMPP_OPTIMIZATION_STAGE_BEGIN(variable) ((void)0)
#define CAMPP_OPTIMIZATION_STAGE_END(stage, variable) ((void)0)
#endif

static CamppStatus campp_qconv_v5_mac_fixed_fallback(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    const CamppKernelEntry *entry = campp_qconv_candidate_entry(
        CAMPP_QCONV_CANDIDATE_MAC_FIXED);
    if (entry == NULL || entry->run == NULL) return CAMPP_STATUS_MISSING_KERNEL;
    return entry->run(
        model, op, inputs, input_count, outputs, output_count,
        scratch, scratch_size);
}

static uint32_t campp_qconv_v5_spatial_index(
    const CamppQconvV4ExecutionPlan *plan,
    const uint32_t coordinates[2])
{
    return plan->spatial_rank == 1u
        ? coordinates[0]
        : coordinates[0] * plan->output->dimensions[3] + coordinates[1];
}

static bool campp_qconv_v5_prepare_bias(
    const CamppQconvV4ExecutionPlan *plan,
    const CamppQconvV4ParameterBlock *parameters,
    const uint8_t *const packed_weights[2],
    const CamppQconvV5ValidatedRawPlan *raw_plan,
    int32_t out_bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    bool *out_zero_point_fastpath)
{
    uint32_t lane;

    if (out_zero_point_fastpath == NULL || out_bias == NULL ||
        plan == NULL || parameters == NULL || raw_plan == NULL) {
        return false;
    }
    for (lane = 0u; lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE; ++lane) {
        out_bias[lane] = parameters->bias[lane];
    }
    *out_zero_point_fastpath = false;
    if (raw_plan->weight_zero_is_all_zero) {
        int32_t weight_sums[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
        if (campp_qconv_v5_weight_sums_o4i4(
                packed_weights, plan->kernel_elements,
                plan->inputs_per_group, weight_sums) &&
            campp_qconv_v5_correct_bias_block(
                parameters->bias, plan->input_zero,
                weight_sums, out_bias)) {
            *out_zero_point_fastpath = true;
        }
    }
    return true;
}

static CamppQconvMac4x8Result campp_qconv_v5_try_mac(
    const CamppQconvV4ExecutionPlan *plan,
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2],
    const CamppQconvV4ParameterBlock *parameters,
    const CamppQconvV5ValidatedRawPlan *raw_plan,
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    bool zero_point_fastpath,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    if (plan == NULL || tile_plan == NULL || packed_weights == NULL ||
        parameters == NULL || raw_plan == NULL || bias == NULL ||
        accumulators == NULL ||
        tile_plan->tile_count != CAMPP_QCONV_CANDIDATE_TILE ||
        packed_weights[0] == NULL || packed_weights[1] == NULL) {
        return CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
    }
    if (tile_plan->path == CAMPP_QCONV_V4_PATH_1X1) {
        return campp_qconv_v5_mac_1x1_8x8_real(
            tile_plan, packed_weights, plan->inputs_per_group,
            plan->input_zero, zero_point_fastpath,
            parameters->weight_zero, bias, accumulators);
    }
    if (tile_plan->path == CAMPP_QCONV_V4_PATH_3X3_INTERIOR) {
        CamppQconvMac4x8Result result =
            campp_qconv_v5_mac_3x3_sliding_8x8(
                tile_plan, packed_weights, plan->inputs_per_group,
                raw_plan->sliding_3x3_eligible,
                plan->input_zero, zero_point_fastpath,
                parameters->weight_zero, bias, accumulators);
        if (result != CAMPP_QCONV_MAC_4X8_UNSUPPORTED ||
            plan->inputs_per_group != 32u) {
            return result;
        }
        return campp_qconv_v5_mac_3x3_cin32_unroll(
            tile_plan, packed_weights, plan->input_zero,
            zero_point_fastpath, parameters->weight_zero,
            bias, accumulators);
    }
    return CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
}

CamppStatus campp_qconv_v5_run(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    CamppQconvV4ExecutionPlan plan;
    CamppQconvV4StorePlan store_plan;
    CamppStatus status;
    uint32_t batch;

    CAMPP_OPTIMIZATION_STAGE_BEGIN(setup_started_ns);
    status = campp_qconv_v4_execution_plan_create(
        model, op, inputs, input_count, outputs, output_count, &plan);
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_QCONV_SETUP, setup_started_ns);
    if (status == CAMPP_STATUS_NOT_IMPLEMENTED) {
        return campp_qconv_v5_mac_fixed_fallback(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_qconv_v4_store_plan_create(plan.output, &store_plan);
    if (status == CAMPP_STATUS_NOT_IMPLEMENTED) {
        return campp_qconv_v5_mac_fixed_fallback(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    if (status != CAMPP_STATUS_OK) return status;

    for (batch = 0u; batch < plan.input->dimensions[0]; ++batch) {
        uint32_t group_index;
        for (group_index = 0u; group_index < plan.group; ++group_index) {
            uint32_t output_block;
            for (output_block = 0u; output_block < plan.output_blocks;
                 output_block += 2u) {
                const uint32_t first_within = output_block * 4u;
                const uint32_t output_channel =
                    group_index * plan.outputs_per_group + first_within;
                CamppQconvV4ParameterBlock parameters;
                CamppQconvV5ValidatedRawPlan raw_plan;
                const uint8_t *packed_weights[2] = {NULL, NULL};
                int32_t mac_bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
                bool zero_point_fastpath = false;
                bool fixed_mac_block_eligible;
                uint32_t tile_start;

                status = campp_qconv_v4_parameters_load(
                    &plan, inputs, input_count, group_index,
                    first_within, &parameters);
                if (status != CAMPP_STATUS_OK) return status;
                packed_weights[0] = (const uint8_t *)plan.weight->data
                    + (((uint64_t)group_index * plan.output_blocks
                        + output_block) * plan.kernel_elements
                       * plan.input_blocks * 16u);
                if (output_block + 1u < plan.output_blocks) {
                    packed_weights[1] = (const uint8_t *)plan.weight->data
                        + (((uint64_t)group_index * plan.output_blocks
                            + output_block + 1u) * plan.kernel_elements
                           * plan.input_blocks * 16u);
                }
                raw_plan = campp_qconv_v5_validated_raw_plan_create(
                    &plan, &parameters);
                fixed_mac_block_eligible =
                    raw_plan.enabled &&
                    packed_weights[0] != NULL && packed_weights[1] != NULL;
                if (!campp_qconv_v5_prepare_bias(
                        &plan, &parameters, packed_weights, &raw_plan,
                        mac_bias, &zero_point_fastpath)) {
                    return CAMPP_STATUS_KERNEL_FAILED;
                }

                for (tile_start = 0u; tile_start < plan.output_spatial;
                     tile_start += CAMPP_QCONV_CANDIDATE_TILE) {
                    const uint32_t remaining =
                        plan.output_spatial - tile_start;
                    const uint32_t tile_count =
                        remaining < CAMPP_QCONV_CANDIDATE_TILE
                        ? remaining : CAMPP_QCONV_CANDIDATE_TILE;
                    CamppQconvV4TilePlan tile_plan;
                    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
                        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
                    CamppQconvMac4x8Result mac_result =
                        CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
                    uint32_t tile;

                    CAMPP_OPTIMIZATION_STAGE_BEGIN(mac_started_ns);
                    status = campp_qconv_v4_tile_plan_create(
                        &plan, batch,
                        group_index * plan.inputs_per_group,
                        tile_start, tile_count, &tile_plan);
                    if (status != CAMPP_STATUS_OK) return status;
                    if (fixed_mac_block_eligible) {
                        mac_result = campp_qconv_v5_try_mac(
                            &plan, &tile_plan, packed_weights, &parameters,
                            &raw_plan, mac_bias, zero_point_fastpath,
                            accumulators);
                    }
                    if (mac_result == CAMPP_QCONV_MAC_4X8_FAILED) {
                        return CAMPP_STATUS_KERNEL_FAILED;
                    }
                    if (mac_result != CAMPP_QCONV_MAC_4X8_OK &&
                        campp_qconv_v5_mac_tail(
                            &plan, &tile_plan, packed_weights,
                            parameters.weight_zero, parameters.bias,
                            parameters.valid_outputs, accumulators) != 0) {
                        return CAMPP_STATUS_KERNEL_FAILED;
                    }
                    CAMPP_OPTIMIZATION_STAGE_END(
                        CAMPP_OPT_STAGE_QCONV_MAC_ADDRESS, mac_started_ns);

                    CAMPP_OPTIMIZATION_STAGE_BEGIN(requant_started_ns);
                    for (tile = 0u; tile < tile_count; ++tile) {
                        uint8_t bytes[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE];
                        campp_qconv_v4_requantize8_validated(
                            accumulators[tile], parameters.multiplier,
                            plan.output->dtype, plan.output_zero, bytes);
                        status = campp_qconv_v4_store_channel_packed(
                            &store_plan, batch, output_channel,
                            campp_qconv_v5_spatial_index(
                                &plan, tile_plan.output_coordinates[tile]),
                            bytes, parameters.valid_outputs);
                        if (status != CAMPP_STATUS_OK) return status;
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
