#include "bn_candidate.h"

#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_reference/reference_kernel_utils.h"
#include "bn_affine_fastpath.h"
#include "bn_iteration_fastpath.h"
#include "bn_quant_neon.h"
#include "internal/runtime_model.h"

#if defined(CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS)
#include "campp_profill/optimization/stage_probe.h"
#else
#define CAMPP_OPTIMIZATION_STAGE_BEGIN(variable) ((void)0)
#define CAMPP_OPTIMIZATION_STAGE_END(stage, variable) ((void)0)
#endif

#define CAMPP_BN_CANDIDATE_MAX_CHANNELS 4096u

typedef struct CamppBnCandidateContext {
    const CamppTensorView *input;
    CamppTensorView *output;
    const CamppTensorView *scale;
    const CamppTensorView *bias;
    const CamppTensorView *mean;
    const CamppTensorView *variance;
    CamppBnIterationPlan iteration;
    float multiplier[CAMPP_BN_CANDIDATE_MAX_CHANNELS];
    float additive[CAMPP_BN_CANDIDATE_MAX_CHANNELS];
    float epsilon;
    float quant_scale;
    int32_t quant_zero;
} CamppBnCandidateContext;

static CamppStatus campp_bn_scalar_zero(
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

static CamppStatus campp_bn_candidate_prepare(
    CamppBnCandidateMode mode, const CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op, const CamppTensorView *inputs,
    uint8_t input_count, CamppTensorView *outputs, uint8_t output_count,
    CamppBnCandidateContext *context)
{
    const CamppTensorView *zero_point = input_count == 7u ? &inputs[6] : NULL;
    double epsilon_value;
    uint32_t channel;
    CamppStatus status;

    memset(context, 0, sizeof(*context));
    status = campp_reference_validate_invocation(
        inputs, input_count, 6u, 7u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    context->input = &inputs[0];
    context->output = &outputs[0];
    context->scale = &inputs[1];
    context->bias = &inputs[2];
    context->mean = &inputs[3];
    context->variance = &inputs[4];
    if (!campp_bn_iteration_plan_create(
            context->input, context->output, &context->iteration) ||
        context->iteration.channels > CAMPP_BN_CANDIDATE_MAX_CHANNELS) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    {
        const CamppTensorView *parameters[4] = {
            context->scale, context->bias, context->mean, context->variance
        };
        uint8_t parameter;
        for (parameter = 0u; parameter < 4u; ++parameter) {
            if (parameters[parameter]->dtype != CAMPP_DTYPE_FLOAT32 ||
                parameters[parameter]->rank != 1u ||
                parameters[parameter]->dimensions[0] !=
                    context->iteration.channels) {
                return CAMPP_STATUS_SHAPE_MISMATCH;
            }
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
    context->epsilon = (float)epsilon_value;
    status = campp_reference_read_f32(
        &inputs[5], 0u, &context->quant_scale);
    if (status != CAMPP_STATUS_OK || !(context->quant_scale > 0.0f) ||
        !isfinite(context->quant_scale)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    status = campp_bn_scalar_zero(zero_point, &context->quant_zero);
    if (status != CAMPP_STATUS_OK) return status;
    if ((mode == CAMPP_BN_CANDIDATE_QUANT ||
         mode == CAMPP_BN_CANDIDATE_COMBINED) &&
        fegetround() != FE_TONEAREST) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    if (mode == CAMPP_BN_CANDIDATE_AFFINE ||
        mode == CAMPP_BN_CANDIDATE_COMBINED) {
        for (channel = 0u; channel < context->iteration.channels; ++channel) {
            CamppBnAffine affine;
            status = campp_bn_affine_channel(
                context->scale, context->bias, context->mean,
                context->variance, channel, context->epsilon, &affine);
            if (status != CAMPP_STATUS_OK) return status;
            context->multiplier[channel] = affine.multiplier;
            context->additive[channel] = affine.additive;
        }
    }
    return CAMPP_STATUS_OK;
}

static void campp_bn_store_direct(
    CamppTensorView *output, uint64_t offset, int32_t value)
{
    if (output->dtype == CAMPP_DTYPE_UINT8) {
        *((uint8_t *)output->data + offset) = (uint8_t)value;
    } else {
        const int8_t stored = (int8_t)value;
        memcpy((uint8_t *)output->data + offset, &stored, sizeof(stored));
    }
}

static CamppStatus campp_bn_run_address(CamppBnCandidateContext *context)
{
    uint32_t batch;
    for (batch = 0u; batch < context->iteration.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < context->iteration.height; ++height) {
            uint32_t width;
            for (width = 0u; width < context->iteration.width; ++width) {
                uint32_t channel;
                for (channel = 0u;
                     channel < context->iteration.channels; ++channel) {
                    const uint64_t input_offset = campp_bn_input_offset(
                        &context->iteration, batch, channel, height, width);
                    const uint64_t output_offset = campp_bn_output_offset(
                        &context->iteration, batch, channel, height, width);
                    CamppBnAffine affine;
                    float input_value;
                    int32_t output_value;
                    CamppStatus status;
                    memcpy(
                        &input_value,
                        (const uint8_t *)context->input->data + input_offset,
                        sizeof(input_value));
                    status = campp_bn_affine_channel(
                        context->scale, context->bias, context->mean,
                        context->variance, channel, context->epsilon, &affine);
                    if (status != CAMPP_STATUS_OK) return status;
                    status = campp_bn_quantize_scalar(
                        input_value, affine.multiplier, affine.additive,
                        context->quant_scale, context->quant_zero,
                        context->output->dtype, &output_value);
                    if (status != CAMPP_STATUS_OK) return status;
                    campp_bn_store_direct(
                        context->output, output_offset, output_value);
                }
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_bn_run_affine(CamppBnCandidateContext *context)
{
    const uint64_t count = campp_tensor_view_element_count(context->output);
    uint64_t index;
    for (index = 0u; index < count; ++index) {
        uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
        uint32_t channel;
        float input_value;
        int32_t output_value;
        CamppStatus status;
        campp_reference_unravel_index(
            index, context->output->rank, context->output->dimensions,
            coordinates);
        channel = coordinates[1];
        status = campp_reference_read_f32(
            context->input, index, &input_value);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_bn_quantize_scalar(
            input_value, context->multiplier[channel],
            context->additive[channel], context->quant_scale,
            context->quant_zero, context->output->dtype, &output_value);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_write_quantized(
            context->output, index, output_value);
        if (status != CAMPP_STATUS_OK) return status;
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_bn_run_quant(CamppBnCandidateContext *context)
{
    const uint64_t count = campp_tensor_view_element_count(context->output);
    uint64_t index;
    for (index = 0u; index < count; index += 4u) {
        float input_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float multiplier[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float additive[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int32_t output_values[4];
        const uint32_t lanes = count - index < 4u
            ? (uint32_t)(count - index) : 4u;
        uint32_t lane;
        CamppStatus status;
        for (lane = 0u; lane < lanes; ++lane) {
            uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
            CamppBnAffine affine;
            campp_reference_unravel_index(
                index + lane, context->output->rank,
                context->output->dimensions, coordinates);
            status = campp_reference_read_f32(
                context->input, index + lane, &input_values[lane]);
            if (status != CAMPP_STATUS_OK) return status;
            status = campp_bn_affine_channel(
                context->scale, context->bias, context->mean,
                context->variance, coordinates[1], context->epsilon, &affine);
            if (status != CAMPP_STATUS_OK) return status;
            multiplier[lane] = affine.multiplier;
            additive[lane] = affine.additive;
        }
        if (lanes == 4u) {
            status = campp_bn_quantize_neon4(
                input_values, multiplier, additive, context->quant_scale,
                context->quant_zero, context->output->dtype, output_values);
            if (status != CAMPP_STATUS_OK) return status;
        } else {
            for (lane = 0u; lane < lanes; ++lane) {
                status = campp_bn_quantize_scalar(
                    input_values[lane], multiplier[lane], additive[lane],
                    context->quant_scale, context->quant_zero,
                    context->output->dtype, &output_values[lane]);
                if (status != CAMPP_STATUS_OK) return status;
            }
        }
        for (lane = 0u; lane < lanes; ++lane) {
            status = campp_reference_write_quantized(
                context->output, index + lane, output_values[lane]);
            if (status != CAMPP_STATUS_OK) return status;
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_bn_run_combined(CamppBnCandidateContext *context)
{
    uint32_t batch;
    for (batch = 0u; batch < context->iteration.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < context->iteration.height; ++height) {
            uint32_t width;
            for (width = 0u; width < context->iteration.width; ++width) {
                uint32_t channel = 0u;
                if (context->iteration.channel_contiguous) {
                    for (; channel + 4u <= context->iteration.channels;
                         channel += 4u) {
                        const uint64_t input_offset = campp_bn_input_offset(
                            &context->iteration, batch, channel, height, width);
                        const uint64_t output_offset = campp_bn_output_offset(
                            &context->iteration, batch, channel, height, width);
                        int32_t output_values[4];
                        uint32_t lane;
                        CamppStatus status = campp_bn_quantize_neon4(
                            (const float *)((const uint8_t *)
                                context->input->data + input_offset),
                            context->multiplier + channel,
                            context->additive + channel,
                            context->quant_scale, context->quant_zero,
                            context->output->dtype, output_values);
                        if (status != CAMPP_STATUS_OK) return status;
                        for (lane = 0u; lane < 4u; ++lane) {
                            campp_bn_store_direct(
                                context->output, output_offset + lane,
                                output_values[lane]);
                        }
                    }
                }
                for (; channel < context->iteration.channels; ++channel) {
                    const uint64_t input_offset = campp_bn_input_offset(
                        &context->iteration, batch, channel, height, width);
                    const uint64_t output_offset = campp_bn_output_offset(
                        &context->iteration, batch, channel, height, width);
                    float input_value;
                    int32_t output_value;
                    CamppStatus status;
                    memcpy(
                        &input_value,
                        (const uint8_t *)context->input->data + input_offset,
                        sizeof(input_value));
                    status = campp_bn_quantize_scalar(
                        input_value, context->multiplier[channel],
                        context->additive[channel], context->quant_scale,
                        context->quant_zero, context->output->dtype,
                        &output_value);
                    if (status != CAMPP_STATUS_OK) return status;
                    campp_bn_store_direct(
                        context->output, output_offset, output_value);
                }
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_bn_candidate_run(
    CamppBnCandidateMode mode, const CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op, const CamppTensorView *inputs,
    uint8_t input_count, CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    CamppBnCandidateContext context;
    CamppStatus status;
    (void)scratch;
    (void)scratch_size;
    CAMPP_OPTIMIZATION_STAGE_BEGIN(setup_started_ns);
    status = campp_bn_candidate_prepare(
        mode, model, op, inputs, input_count, outputs, output_count, &context);
    CAMPP_OPTIMIZATION_STAGE_END(CAMPP_OPT_STAGE_BN_SETUP, setup_started_ns);
    if (status == CAMPP_STATUS_NOT_IMPLEMENTED) {
        return campp_fused_bn_relu_quant(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    if (status != CAMPP_STATUS_OK) return status;
    CAMPP_OPTIMIZATION_STAGE_BEGIN(elementwise_started_ns);
    if (mode == CAMPP_BN_CANDIDATE_ADDRESS) {
        status = campp_bn_run_address(&context);
    } else if (mode == CAMPP_BN_CANDIDATE_AFFINE) {
        status = campp_bn_run_affine(&context);
    } else if (mode == CAMPP_BN_CANDIDATE_QUANT) {
        status = campp_bn_run_quant(&context);
    } else {
        status = campp_bn_run_combined(&context);
    }
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_BN_ELEMENTWISE, elementwise_started_ns);
    return status;
}

#define CAMPP_DEFINE_BN_CANDIDATE(function_name, candidate_mode)       \
    static CamppStatus function_name(                                  \
        const CamppRuntimeModel *model,                                \
        const CamppOperatorDescriptor *op,                             \
        const CamppTensorView *inputs, uint8_t input_count,            \
        CamppTensorView *outputs, uint8_t output_count,                 \
        void *scratch, size_t scratch_size)                             \
    {                                                                  \
        return campp_bn_candidate_run(                                 \
            (candidate_mode), model, op, inputs, input_count,          \
            outputs, output_count, scratch, scratch_size);             \
    }

CAMPP_DEFINE_BN_CANDIDATE(
    campp_bn_candidate_address, CAMPP_BN_CANDIDATE_ADDRESS)
CAMPP_DEFINE_BN_CANDIDATE(
    campp_bn_candidate_affine, CAMPP_BN_CANDIDATE_AFFINE)
CAMPP_DEFINE_BN_CANDIDATE(
    campp_bn_candidate_quant, CAMPP_BN_CANDIDATE_QUANT)
CAMPP_DEFINE_BN_CANDIDATE(
    campp_bn_candidate_combined, CAMPP_BN_CANDIDATE_COMBINED)

static const CamppKernelEntry CAMPP_BN_CANDIDATE_ENTRIES[] = {
    {CAMPP_OP_BATCH_NORMALIZATION, CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID,
     campp_bn_candidate_address, NULL, "fused_bn_relu_quant"},
    {CAMPP_OP_BATCH_NORMALIZATION, CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID,
     campp_bn_candidate_affine, NULL, "fused_bn_relu_quant"},
    {CAMPP_OP_BATCH_NORMALIZATION, CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID,
     campp_bn_candidate_quant, NULL, "fused_bn_relu_quant"},
    {CAMPP_OP_BATCH_NORMALIZATION, CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID,
     campp_bn_candidate_combined, NULL, "fused_bn_relu_quant"}
};

const char *campp_bn_candidate_mode_name(CamppBnCandidateMode mode)
{
    static const char *const names[] = {
        "baseline", "address", "affine", "quant", "combined"
    };
    return mode >= CAMPP_BN_CANDIDATE_BASELINE &&
        mode <= CAMPP_BN_CANDIDATE_COMBINED
        ? names[mode] : "invalid";
}

int campp_bn_candidate_mode_parse(
    const char *text, CamppBnCandidateMode *out_mode)
{
    CamppBnCandidateMode mode;
    if (text == NULL || out_mode == NULL) return 1;
    for (mode = CAMPP_BN_CANDIDATE_BASELINE;
         mode <= CAMPP_BN_CANDIDATE_COMBINED;
         mode = (CamppBnCandidateMode)(mode + 1)) {
        if (strcmp(text, campp_bn_candidate_mode_name(mode)) == 0) {
            *out_mode = mode;
            return 0;
        }
    }
    return 1;
}

const CamppKernelEntry *campp_bn_candidate_entry(CamppBnCandidateMode mode)
{
    if (mode == CAMPP_BN_CANDIDATE_BASELINE) return NULL;
    if (mode < CAMPP_BN_CANDIDATE_ADDRESS ||
        mode > CAMPP_BN_CANDIDATE_COMBINED) return NULL;
    return &CAMPP_BN_CANDIDATE_ENTRIES[mode - 1];
}

#undef CAMPP_DEFINE_BN_CANDIDATE
