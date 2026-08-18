#include "dequant_candidate.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_reference/reference_kernel_utils.h"
#include "backends/cpu_reference/reference_kernels.h"
#include "dequant_layout_plan.h"
#include "dequant_neon.h"
#include "dequant_scalar_fastpath.h"
#include "internal/runtime_model.h"

#if defined(CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS)
#include "campp_profill/optimization/stage_probe.h"
#else
#define CAMPP_OPTIMIZATION_STAGE_BEGIN(variable) ((void)0)
#define CAMPP_OPTIMIZATION_STAGE_END(stage, variable) ((void)0)
#endif

typedef struct CamppDequantCandidateContext {
    const CamppTensorView *input;
    const CamppTensorView *scale;
    const CamppTensorView *zero_point;
    CamppTensorView *output;
    CamppDequantLayoutPlan layout;
    uint8_t axis;
    bool per_axis;
    float scalar_scale;
    int32_t scalar_zero;
} CamppDequantCandidateContext;

static CamppStatus campp_dequant_axis(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint8_t rank, uint8_t *out_axis)
{
    int64_t axis_value[1];
    uint8_t count = 0u;
    CamppStatus status = campp_runtime_model_attribute_ints(
        model, op, CAMPP_ATTR_AXIS, axis_value, 1u, &count);
    if (status == CAMPP_STATUS_MISSING_ATTRIBUTE) {
        axis_value[0] = 1;
        count = 1u;
        status = CAMPP_STATUS_OK;
    }
    if (status != CAMPP_STATUS_OK) return status;
    if (count != 1u) return CAMPP_STATUS_CORRUPT_PLAN;
    return campp_reference_normalize_axis(axis_value[0], rank, out_axis);
}

static CamppStatus campp_dequant_read_zero(
    const CamppTensorView *view, uint64_t index, int32_t *out_value)
{
    if (view == NULL) {
        *out_value = 0;
        return CAMPP_STATUS_OK;
    }
    return campp_reference_read_quantized(view, index, out_value);
}

static CamppStatus campp_dequant_candidate_prepare(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppDequantCandidateContext *context)
{
    uint64_t scale_count;
    uint64_t zero_count = 1u;
    uint64_t parameter;
    CamppStatus status;

    memset(context, 0, sizeof(*context));
    status = campp_reference_validate_invocation(
        inputs, input_count, 2u, 3u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    context->input = &inputs[0];
    context->scale = &inputs[1];
    context->zero_point = input_count == 3u ? &inputs[2] : NULL;
    context->output = &outputs[0];
    if ((context->input->dtype != CAMPP_DTYPE_UINT8 &&
         context->input->dtype != CAMPP_DTYPE_INT8) ||
        context->output->dtype != CAMPP_DTYPE_FLOAT32) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (!campp_reference_shapes_equal(context->input, context->output)) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    scale_count = campp_tensor_view_element_count(context->scale);
    if (context->scale->dtype != CAMPP_DTYPE_FLOAT32 ||
        context->scale->rank > 1u ||
        (scale_count != 1u && context->scale->rank != 1u)) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    if (context->zero_point != NULL) {
        zero_count = campp_tensor_view_element_count(context->zero_point);
        if (context->zero_point->dtype != context->input->dtype ||
            context->zero_point->rank > 1u || zero_count != scale_count ||
            (zero_count != 1u && context->zero_point->rank != 1u)) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
    }
    context->per_axis = scale_count != 1u;
    if (context->per_axis) {
        status = campp_dequant_axis(
            model, op, context->input->rank, &context->axis);
        if (status != CAMPP_STATUS_OK) return status;
        if (scale_count != context->input->dimensions[context->axis]) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        if (context->axis != 1u) return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (parameter = 0u; parameter < scale_count; ++parameter) {
        float scale;
        int32_t zero;
        status = campp_reference_read_f32(
            context->scale, parameter, &scale);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_dequant_read_zero(
            context->zero_point, parameter, &zero);
        if (status != CAMPP_STATUS_OK) return status;
        if (!(scale > 0.0f) || !isfinite(scale)) {
            return CAMPP_STATUS_KERNEL_FAILED;
        }
        if (!context->per_axis) {
            context->scalar_scale = scale;
            context->scalar_zero = zero;
        }
    }
    if (!campp_dequant_layout_plan_create(
            context->input, context->output, &context->layout)) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    return CAMPP_STATUS_OK;
}

static void campp_dequant_parameter_direct(
    const CamppDequantCandidateContext *context, uint32_t channel,
    float *out_scale, int32_t *out_zero)
{
    if (!context->per_axis) {
        *out_scale = context->scalar_scale;
        *out_zero = context->scalar_zero;
        return;
    }
    memcpy(
        out_scale,
        (const uint8_t *)context->scale->data +
            (uint64_t)channel * context->scale->byte_strides[0],
        sizeof(*out_scale));
    if (context->zero_point == NULL) {
        *out_zero = 0;
    } else {
        *out_zero = campp_dequant_read_direct(
            (const uint8_t *)context->zero_point->data +
                (uint64_t)channel * context->zero_point->byte_strides[0],
            context->zero_point->dtype);
    }
}

static CamppStatus campp_dequant_run_address(
    CamppDequantCandidateContext *context)
{
    uint32_t batch;
    for (batch = 0u; batch < context->layout.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < context->layout.height; ++height) {
            uint32_t width;
            for (width = 0u; width < context->layout.width; ++width) {
                const uint64_t input_offset = campp_dequant_input_offset(
                    &context->layout, batch, 0u, height, width);
                const uint64_t output_offset = campp_dequant_output_offset(
                    &context->layout, batch, 0u, height, width);
                const uint8_t *input =
                    (const uint8_t *)context->input->data + input_offset;
                float *output = (float *)((uint8_t *)context->output->data +
                    output_offset);
                uint32_t channel;
                for (channel = 0u; channel < context->layout.channels;
                     ++channel) {
                    const uint64_t parameter =
                        context->per_axis ? channel : 0u;
                    float scale;
                    int32_t zero;
                    CamppStatus status = campp_reference_read_f32(
                        context->scale, parameter, &scale);
                    if (status != CAMPP_STATUS_OK) return status;
                    status = campp_dequant_read_zero(
                        context->zero_point, parameter, &zero);
                    if (status != CAMPP_STATUS_OK) return status;
                    if (!(scale > 0.0f) || !isfinite(scale)) {
                        return CAMPP_STATUS_KERNEL_FAILED;
                    }
                    campp_dequant_store_direct(
                        output + channel,
                        campp_dequant_scalar_value(
                            campp_dequant_read_direct(
                                input + channel, context->input->dtype),
                            scale, zero));
                }
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_dequant_run_parameter(
    CamppDequantCandidateContext *context)
{
    const uint64_t count = campp_tensor_view_element_count(context->input);
    uint64_t index;
    for (index = 0u; index < count; ++index) {
        uint32_t coordinates[CAMPP_TENSOR_MAX_RANK];
        float scale;
        float result;
        int32_t value;
        int32_t zero;
        uint64_t output_offset;
        CamppStatus status;
        campp_reference_unravel_index(
            index, context->input->rank, context->input->dimensions,
            coordinates);
        status = campp_reference_read_quantized(
            context->input, index, &value);
        if (status != CAMPP_STATUS_OK) return status;
        campp_dequant_parameter_direct(
            context, coordinates[1], &scale, &zero);
        result = campp_dequant_scalar_value(value, scale, zero);
        output_offset = campp_reference_offset_for_linear(
            context->output, index);
        memcpy(
            (uint8_t *)context->output->data + output_offset,
            &result, sizeof(result));
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_dequant_run_combined(
    CamppDequantCandidateContext *context, bool use_neon)
{
    uint32_t batch;
    for (batch = 0u; batch < context->layout.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < context->layout.height; ++height) {
            uint32_t width;
            for (width = 0u; width < context->layout.width; ++width) {
                const uint64_t input_offset = campp_dequant_input_offset(
                    &context->layout, batch, 0u, height, width);
                const uint64_t output_offset = campp_dequant_output_offset(
                    &context->layout, batch, 0u, height, width);
                const uint8_t *input =
                    (const uint8_t *)context->input->data + input_offset;
                float *output = (float *)((uint8_t *)context->output->data +
                    output_offset);
                uint32_t channel = 0u;
                if (use_neon) {
                    for (; channel + CAMPP_DEQUANT_NEON_LANES <=
                               context->layout.channels;
                         channel += CAMPP_DEQUANT_NEON_LANES) {
                        if (!context->per_axis) {
                            campp_dequant_neon_scalar16(
                                input + channel, context->input->dtype,
                                context->scalar_scale, context->scalar_zero,
                                output + channel);
                        } else {
                            float scales[CAMPP_DEQUANT_NEON_LANES];
                            int32_t zeros[CAMPP_DEQUANT_NEON_LANES];
                            uint32_t lane;
                            for (lane = 0u;
                                 lane < CAMPP_DEQUANT_NEON_LANES; ++lane) {
                                campp_dequant_parameter_direct(
                                    context, channel + lane,
                                    &scales[lane], &zeros[lane]);
                            }
                            campp_dequant_neon_per_axis16(
                                input + channel, context->input->dtype,
                                scales, zeros, output + channel);
                        }
                    }
                }
                for (; channel < context->layout.channels; ++channel) {
                    float scale;
                    int32_t zero;
                    campp_dequant_parameter_direct(
                        context, channel, &scale, &zero);
                    campp_dequant_store_direct(
                        output + channel,
                        campp_dequant_scalar_value(
                            campp_dequant_read_direct(
                                input + channel, context->input->dtype),
                            scale, zero));
                }
            }
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_dequant_candidate_run(
    CamppDequantCandidateMode mode, const CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op, const CamppTensorView *inputs,
    uint8_t input_count, CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    CamppDequantCandidateContext context;
    CamppStatus status;
    (void)scratch;
    (void)scratch_size;
    CAMPP_OPTIMIZATION_STAGE_BEGIN(setup_started_ns);
    status = campp_dequant_candidate_prepare(
        model, op, inputs, input_count, outputs, output_count, &context);
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_DEQUANT_SETUP, setup_started_ns);
    if (status == CAMPP_STATUS_NOT_IMPLEMENTED) {
        return campp_reference_dequantize_linear(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    if (status != CAMPP_STATUS_OK) return status;
    CAMPP_OPTIMIZATION_STAGE_BEGIN(elementwise_started_ns);
    if (mode == CAMPP_DEQUANT_CANDIDATE_ADDRESS) {
        status = campp_dequant_run_address(&context);
    } else if (mode == CAMPP_DEQUANT_CANDIDATE_PARAMETER) {
        status = campp_dequant_run_parameter(&context);
    } else if (mode == CAMPP_DEQUANT_CANDIDATE_SCALAR_COMBINED) {
        status = campp_dequant_run_combined(&context, false);
    } else {
        status = campp_dequant_run_combined(&context, true);
    }
    CAMPP_OPTIMIZATION_STAGE_END(
        CAMPP_OPT_STAGE_DEQUANT_ELEMENTWISE, elementwise_started_ns);
    return status;
}

#define CAMPP_DEFINE_DEQUANT_CANDIDATE(function_name, candidate_mode)  \
    static CamppStatus function_name(                                  \
        const CamppRuntimeModel *model,                                \
        const CamppOperatorDescriptor *op,                             \
        const CamppTensorView *inputs, uint8_t input_count,            \
        CamppTensorView *outputs, uint8_t output_count,                 \
        void *scratch, size_t scratch_size)                            \
    {                                                                  \
        return campp_dequant_candidate_run(                            \
            (candidate_mode), model, op, inputs, input_count,          \
            outputs, output_count, scratch, scratch_size);             \
    }

CAMPP_DEFINE_DEQUANT_CANDIDATE(
    campp_dequant_candidate_address, CAMPP_DEQUANT_CANDIDATE_ADDRESS)
CAMPP_DEFINE_DEQUANT_CANDIDATE(
    campp_dequant_candidate_parameter, CAMPP_DEQUANT_CANDIDATE_PARAMETER)
CAMPP_DEFINE_DEQUANT_CANDIDATE(
    campp_dequant_candidate_scalar_combined,
    CAMPP_DEQUANT_CANDIDATE_SCALAR_COMBINED)
CAMPP_DEFINE_DEQUANT_CANDIDATE(
    campp_dequant_candidate_neon_combined,
    CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED)

static const CamppKernelEntry CAMPP_DEQUANT_CANDIDATE_ENTRIES[] = {
    {CAMPP_OP_DEQUANTIZE_LINEAR, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_dequant_candidate_address, NULL, "dequantize_linear_stride"},
    {CAMPP_OP_DEQUANTIZE_LINEAR, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_dequant_candidate_parameter, NULL, "dequantize_linear_stride"},
    {CAMPP_OP_DEQUANTIZE_LINEAR, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_dequant_candidate_scalar_combined, NULL,
     "dequantize_linear_stride"},
    {CAMPP_OP_DEQUANTIZE_LINEAR, CAMPP_AARCH64_PACKED_KERNEL_ID,
     campp_dequant_candidate_neon_combined, NULL,
     "dequantize_linear_stride"}
};

const char *campp_dequant_candidate_mode_name(
    CamppDequantCandidateMode mode)
{
    static const char *const names[] = {
        "baseline", "address", "parameter", "scalar_combined",
        "neon_combined"
    };
    return mode >= CAMPP_DEQUANT_CANDIDATE_BASELINE &&
        mode <= CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED
        ? names[mode] : "invalid";
}

int campp_dequant_candidate_mode_parse(
    const char *text, CamppDequantCandidateMode *out_mode)
{
    CamppDequantCandidateMode mode;
    if (text == NULL || out_mode == NULL) return 1;
    for (mode = CAMPP_DEQUANT_CANDIDATE_BASELINE;
         mode <= CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED;
         mode = (CamppDequantCandidateMode)(mode + 1)) {
        if (strcmp(text, campp_dequant_candidate_mode_name(mode)) == 0) {
            *out_mode = mode;
            return 0;
        }
    }
    return 1;
}

const CamppKernelEntry *campp_dequant_candidate_entry(
    CamppDequantCandidateMode mode)
{
    if (mode == CAMPP_DEQUANT_CANDIDATE_BASELINE) return NULL;
    if (mode < CAMPP_DEQUANT_CANDIDATE_ADDRESS ||
        mode > CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED) {
        return NULL;
    }
    return &CAMPP_DEQUANT_CANDIDATE_ENTRIES[mode - 1];
}

#undef CAMPP_DEFINE_DEQUANT_CANDIDATE
