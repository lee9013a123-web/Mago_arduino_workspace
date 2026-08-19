#include "fused_dequant_relu_quant_candidate.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_reference/reference_kernel_utils.h"
#include "dequant_scalar_fastpath.h"
#include "internal/runtime_model.h"
#include "packed_iteration.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define CAMPP_FUSED_DQRQ_LANES 16u

typedef struct CamppFusedDqRqContext {
    CamppPackedIteration input;
    CamppPackedIteration output;
    float old_scale;
    float new_scale;
    int32_t old_zero;
    int32_t new_zero;
} CamppFusedDqRqContext;

static CamppStatus campp_fused_dqrq_read_f32(
    const CamppTensorView *view, float *out_value)
{
    if (view->dtype != CAMPP_DTYPE_FLOAT32 ||
        campp_tensor_view_element_count(view) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_reference_read_f32(view, 0u, out_value);
}

static CamppStatus campp_fused_dqrq_read_quant(
    const CamppTensorView *view, int32_t *out_value)
{
    if (campp_tensor_view_element_count(view) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    return campp_reference_read_quantized(view, 0u, out_value);
}

static CamppStatus campp_fused_dqrq_prepare(
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppFusedDqRqContext *context)
{
    CamppStatus status;
    memset(context, 0, sizeof(*context));
    status = campp_reference_validate_invocation(
        inputs, input_count, 5u, 5u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    if ((inputs[0].dtype != CAMPP_DTYPE_UINT8 &&
         inputs[0].dtype != CAMPP_DTYPE_INT8) ||
        (outputs[0].dtype != CAMPP_DTYPE_UINT8 &&
         outputs[0].dtype != CAMPP_DTYPE_INT8) ||
        inputs[2].dtype != inputs[0].dtype ||
        inputs[4].dtype != outputs[0].dtype) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    if (!campp_reference_shapes_equal(&inputs[0], &outputs[0])) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    status = campp_fused_dqrq_read_f32(&inputs[1], &context->old_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_dqrq_read_quant(&inputs[2], &context->old_zero);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_dqrq_read_f32(&inputs[3], &context->new_scale);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_fused_dqrq_read_quant(&inputs[4], &context->new_zero);
    if (status != CAMPP_STATUS_OK) return status;
    if (!isfinite(context->old_scale) ||
        !(context->new_scale > 0.0f) || !isfinite(context->new_scale)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    if (!campp_packed_iteration_create(&inputs[0], &context->input) ||
        !campp_packed_iteration_create(&outputs[0], &context->output) ||
        !campp_packed_iteration_same_shape(
            &context->input, &context->output)) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    return CAMPP_STATUS_OK;
}

static uint8_t campp_fused_dqrq_scalar_byte(
    uint8_t input, uint8_t input_dtype, uint8_t output_dtype,
    const CamppFusedDqRqContext *context)
{
    const int32_t minimum =
        output_dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
    const int32_t maximum =
        output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    float value = (float)(campp_dequant_read_direct(
        &input, input_dtype) - context->old_zero) * context->old_scale;
    float scaled;
    int64_t rounded;
    if (value < 0.0f) value = 0.0f;
    scaled = value / context->new_scale;
    if (scaled <= -2147483648.0f) {
        rounded = INT32_MIN;
    } else if (scaled >= 2147483520.0f) {
        rounded = INT32_MAX;
    } else {
        rounded = (int64_t)nearbyintf(scaled);
    }
    rounded += context->new_zero;
    if (rounded < minimum) rounded = minimum;
    if (rounded > maximum) rounded = maximum;
    return output_dtype == CAMPP_DTYPE_UINT8
        ? (uint8_t)rounded : (uint8_t)(int8_t)rounded;
}

#if defined(__aarch64__) && defined(__ARM_NEON)
static int32x4_t campp_fused_dqrq_convert4(
    int32x4_t input, int32x4_t old_zero, float32x4_t old_scale,
    float32x4_t new_scale, float32x4_t lower, float32x4_t upper,
    int32x4_t new_zero)
{
    float32x4_t value = vmulq_f32(
        vcvtq_f32_s32(vsubq_s32(input, old_zero)), old_scale);
    value = vmaxq_f32(value, vdupq_n_f32(0.0f));
    value = vdivq_f32(value, new_scale);
    value = vmaxq_f32(lower, vminq_f32(upper, value));
    return vaddq_s32(vcvtnq_s32_f32(value), new_zero);
}

static void campp_fused_dqrq_neon16(
    const uint8_t *input, uint8_t input_dtype, uint8_t *output,
    uint8_t output_dtype, const CamppFusedDqRqContext *context)
{
    int32x4_t values[4];
    const int32x4_t old_zero = vdupq_n_s32(context->old_zero);
    const int32x4_t new_zero = vdupq_n_s32(context->new_zero);
    const float32x4_t old_scale = vdupq_n_f32(context->old_scale);
    const float32x4_t new_scale = vdupq_n_f32(context->new_scale);
    const int32_t minimum =
        output_dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
    const int32_t maximum =
        output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    const float32x4_t lower =
        vdupq_n_f32((float)(minimum - context->new_zero));
    const float32x4_t upper =
        vdupq_n_f32((float)(maximum - context->new_zero));
    int32x4_t rounded[4];
    int16x8_t low;
    int16x8_t high;
    uint32_t quarter;

    if (input_dtype == CAMPP_DTYPE_UINT8) {
        const uint8x16_t bytes = vld1q_u8(input);
        const uint16x8_t low8 = vmovl_u8(vget_low_u8(bytes));
        const uint16x8_t high8 = vmovl_u8(vget_high_u8(bytes));
        values[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(low8)));
        values[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(low8)));
        values[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(high8)));
        values[3] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(high8)));
    } else {
        const int8x16_t bytes = vld1q_s8((const int8_t *)input);
        const int16x8_t low8 = vmovl_s8(vget_low_s8(bytes));
        const int16x8_t high8 = vmovl_s8(vget_high_s8(bytes));
        values[0] = vmovl_s16(vget_low_s16(low8));
        values[1] = vmovl_s16(vget_high_s16(low8));
        values[2] = vmovl_s16(vget_low_s16(high8));
        values[3] = vmovl_s16(vget_high_s16(high8));
    }
    for (quarter = 0u; quarter < 4u; ++quarter) {
        rounded[quarter] = campp_fused_dqrq_convert4(
            values[quarter], old_zero, old_scale, new_scale,
            lower, upper, new_zero);
    }
    low = vcombine_s16(vqmovn_s32(rounded[0]), vqmovn_s32(rounded[1]));
    high = vcombine_s16(vqmovn_s32(rounded[2]), vqmovn_s32(rounded[3]));
    if (output_dtype == CAMPP_DTYPE_UINT8) {
        vst1q_u8(output, vcombine_u8(vqmovun_s16(low), vqmovun_s16(high)));
    } else {
        vst1q_s8(
            (int8_t *)output,
            vcombine_s8(vqmovn_s16(low), vqmovn_s16(high)));
    }
}
#endif

static void campp_fused_dqrq_span(
    const uint8_t *input, uint8_t input_dtype, uint8_t *output,
    uint8_t output_dtype, uint32_t count, bool use_neon,
    const CamppFusedDqRqContext *context)
{
    uint32_t channel = 0u;
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (use_neon) {
        for (; channel + CAMPP_FUSED_DQRQ_LANES <= count;
             channel += CAMPP_FUSED_DQRQ_LANES) {
            campp_fused_dqrq_neon16(
                input + channel, input_dtype, output + channel,
                output_dtype, context);
        }
    }
#else
    (void)use_neon;
#endif
    for (; channel < count; ++channel) {
        output[channel] = campp_fused_dqrq_scalar_byte(
            input[channel], input_dtype, output_dtype, context);
    }
}

static CamppStatus campp_fused_dqrq_run(
    CamppFusedDqRqCandidateMode mode, const CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op, const CamppTensorView *inputs,
    uint8_t input_count, CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size)
{
    CamppFusedDqRqContext context;
    uint32_t batch;
    CamppStatus status = campp_fused_dqrq_prepare(
        inputs, input_count, outputs, output_count, &context);
    if (status == CAMPP_STATUS_NOT_IMPLEMENTED) {
        return campp_fused_dequant_relu_quant(
            model, op, inputs, input_count, outputs, output_count,
            scratch, scratch_size);
    }
    if (status != CAMPP_STATUS_OK) return status;
    for (batch = 0u; batch < context.input.batches; ++batch) {
        uint32_t height;
        for (height = 0u; height < context.input.height; ++height) {
            uint32_t width;
            for (width = 0u; width < context.input.width; ++width) {
                const uint8_t *source =
                    campp_packed_iteration_const_pointer(
                        &context.input, batch, 0u, height, width);
                uint8_t *target = campp_packed_iteration_pointer(
                    &context.output, batch, 0u, height, width);
                campp_fused_dqrq_span(
                    source, inputs[0].dtype, target, outputs[0].dtype,
                    context.input.channels,
                    mode == CAMPP_FUSED_DQRQ_CANDIDATE_NEON, &context);
            }
        }
    }
    return CAMPP_STATUS_OK;
}

#define CAMPP_DEFINE_FUSED_DQRQ_CANDIDATE(function_name, candidate_mode) \
    static CamppStatus function_name(                                  \
        const CamppRuntimeModel *model,                                \
        const CamppOperatorDescriptor *op,                             \
        const CamppTensorView *inputs, uint8_t input_count,            \
        CamppTensorView *outputs, uint8_t output_count,                 \
        void *scratch, size_t scratch_size)                            \
    {                                                                  \
        return campp_fused_dqrq_run(                                   \
            candidate_mode, model, op, inputs, input_count, outputs,  \
            output_count, scratch, scratch_size);                     \
    }

CAMPP_DEFINE_FUSED_DQRQ_CANDIDATE(
    campp_fused_dqrq_scalar, CAMPP_FUSED_DQRQ_CANDIDATE_SCALAR)
CAMPP_DEFINE_FUSED_DQRQ_CANDIDATE(
    campp_fused_dqrq_neon, CAMPP_FUSED_DQRQ_CANDIDATE_NEON)

static const CamppKernelEntry CAMPP_FUSED_DQRQ_ENTRIES[] = {
    {CAMPP_OP_DEQUANTIZE_LINEAR, CAMPP_FUSION_EPILOGUE_KERNEL_ID,
     campp_fused_dqrq_scalar, NULL, "fused_dequant_relu_quant"},
    {CAMPP_OP_DEQUANTIZE_LINEAR, CAMPP_FUSION_EPILOGUE_KERNEL_ID,
     campp_fused_dqrq_neon, NULL, "fused_dequant_relu_quant"}
};

const char *campp_fused_dqrq_candidate_mode_name(
    CamppFusedDqRqCandidateMode mode)
{
    static const char *const names[] = {"baseline", "scalar", "neon"};
    return mode >= CAMPP_FUSED_DQRQ_CANDIDATE_BASELINE &&
        mode <= CAMPP_FUSED_DQRQ_CANDIDATE_NEON
        ? names[mode] : "invalid";
}

int campp_fused_dqrq_candidate_mode_parse(
    const char *text, CamppFusedDqRqCandidateMode *out_mode)
{
    CamppFusedDqRqCandidateMode mode;
    if (text == NULL || out_mode == NULL) return 1;
    for (mode = CAMPP_FUSED_DQRQ_CANDIDATE_BASELINE;
         mode <= CAMPP_FUSED_DQRQ_CANDIDATE_NEON;
         mode = (CamppFusedDqRqCandidateMode)(mode + 1)) {
        if (strcmp(text, campp_fused_dqrq_candidate_mode_name(mode)) == 0) {
            *out_mode = mode;
            return 0;
        }
    }
    return 1;
}

const CamppKernelEntry *campp_fused_dqrq_candidate_entry(
    CamppFusedDqRqCandidateMode mode)
{
    if (mode == CAMPP_FUSED_DQRQ_CANDIDATE_BASELINE) return NULL;
    if (mode < CAMPP_FUSED_DQRQ_CANDIDATE_SCALAR ||
        mode > CAMPP_FUSED_DQRQ_CANDIDATE_NEON) {
        return NULL;
    }
    return &CAMPP_FUSED_DQRQ_ENTRIES[mode - 1];
}

#undef CAMPP_DEFINE_FUSED_DQRQ_CANDIDATE
