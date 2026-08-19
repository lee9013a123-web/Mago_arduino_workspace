#include "fused_input_quant_neon.h"

#include <math.h>
#include <stdint.h>

#include "backends/cpu_aarch64/fused_kernels/fused_quant_qconv_internal.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static int campp_fused_quant_dense_layout_matches(
    const CamppTensorView *input, const CamppTensorView *output)
{
    uint8_t axis;
    if (input == NULL || output == NULL || input->data == NULL ||
        output->data == NULL || input->dtype != CAMPP_DTYPE_FLOAT32 ||
        (output->dtype != CAMPP_DTYPE_UINT8 &&
         output->dtype != CAMPP_DTYPE_INT8) ||
        input->rank != output->rank ||
        input->logical_byte_size != input->storage_span_bytes ||
        output->logical_byte_size != output->storage_span_bytes ||
        input->logical_byte_size != output->logical_byte_size * sizeof(float)) {
        return 0;
    }
    for (axis = 0u; axis < input->rank; ++axis) {
        if (input->dimensions[axis] != output->dimensions[axis] ||
            input->byte_strides[axis] !=
                output->byte_strides[axis] * sizeof(float)) {
            return 0;
        }
    }
    return 1;
}

static uint8_t campp_fused_quantize_scalar_byte(
    float value, float scale, int32_t zero_point, uint8_t dtype)
{
    const int32_t minimum = dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
    const int32_t maximum = dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    float scaled = value / scale;
    int64_t rounded;
    const float lower = (float)(minimum - zero_point);
    const float upper = (float)(maximum - zero_point);
    if (scaled < lower) scaled = lower;
    if (scaled > upper) scaled = upper;
    rounded = (int64_t)nearbyintf(scaled) + zero_point;
    if (rounded < minimum) rounded = minimum;
    if (rounded > maximum) rounded = maximum;
    return dtype == CAMPP_DTYPE_UINT8
        ? (uint8_t)rounded : (uint8_t)(int8_t)rounded;
}

CamppStatus campp_fused_input_quantize_neon(
    const CamppTensorView *input, CamppTensorView *output,
    float scale, int32_t zero_point)
{
    const float *source;
    uint8_t *target;
    uint64_t count;
    uint64_t index = 0u;

    if (!(scale > 0.0f) || !isfinite(scale)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    if (!campp_fused_quant_dense_layout_matches(input, output)) {
        return campp_fused_quantize_input_scalar(
            input, output, scale, zero_point);
    }
    source = (const float *)input->data;
    target = (uint8_t *)output->data;
    count = campp_tensor_view_element_count(input);

#if defined(__aarch64__) && defined(__ARM_NEON)
    {
        const int32_t minimum =
            output->dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
        const int32_t maximum =
            output->dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
        const float32x4_t divisor = vdupq_n_f32(scale);
        const float32x4_t lower =
            vdupq_n_f32((float)(minimum - zero_point));
        const float32x4_t upper =
            vdupq_n_f32((float)(maximum - zero_point));
        const int32x4_t zero = vdupq_n_s32(zero_point);
        for (; index + 4u <= count; index += 4u) {
            const float32x4_t values = vld1q_f32(source + index);
            float lanes[4];
            int32_t rounded[4];
            uint32_t lane;
            vst1q_f32(lanes, values);
            for (lane = 0u; lane < 4u; ++lane) {
                if (!isfinite(lanes[lane])) {
                    return CAMPP_STATUS_KERNEL_FAILED;
                }
            }
            {
                float32x4_t scaled = vdivq_f32(values, divisor);
                scaled = vmaxq_f32(lower, vminq_f32(upper, scaled));
                vst1q_s32(
                    rounded,
                    vaddq_s32(vcvtnq_s32_f32(scaled), zero));
            }
            for (lane = 0u; lane < 4u; ++lane) {
                target[index + lane] = output->dtype == CAMPP_DTYPE_UINT8
                    ? (uint8_t)rounded[lane]
                    : (uint8_t)(int8_t)rounded[lane];
            }
        }
    }
#endif
    for (; index < count; ++index) {
        if (!isfinite(source[index])) return CAMPP_STATUS_KERNEL_FAILED;
        target[index] = campp_fused_quantize_scalar_byte(
            source[index], scale, zero_point, output->dtype);
    }
    return CAMPP_STATUS_OK;
}
