#include "bn_neon16_spatial2.h"

#include "bn_neon16_internal.h"

CamppStatus campp_bn_v2_exact16_spatial2(
    const float *input0, const float *input1,
    uint8_t *output0, uint8_t *output1, uint32_t channels,
    const float *multiplier, const float *additive, float quant_scale,
    int32_t quant_zero, uint8_t output_dtype)
{
    uint32_t channel = 0u;

    if (input0 == NULL || input1 == NULL || output0 == NULL ||
        output1 == NULL || multiplier == NULL || additive == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    {
        const int32_t maximum =
            output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
        const float32x4_t divisor = vdupq_n_f32(quant_scale);
        const float32x4_t upper =
            vdupq_n_f32((float)(maximum - quant_zero));
        const int32x4_t zero = vdupq_n_s32(quant_zero);
        for (; channel + 16u <= channels; channel += 16u) {
            int32x4_t quantized0[4];
            int32x4_t quantized1[4];
            uint32_t quarter;
            for (quarter = 0u; quarter < 4u; ++quarter) {
                const uint32_t offset = channel + quarter * 4u;
                const float32x4_t multiplier4 =
                    vld1q_f32(multiplier + offset);
                const float32x4_t additive4 =
                    vld1q_f32(additive + offset);
                CamppStatus status = campp_bn_v2_exact4_coefficients(
                    input0 + offset, multiplier4, additive4, divisor,
                    upper, zero, &quantized0[quarter]);
                if (status != CAMPP_STATUS_OK) return status;
                status = campp_bn_v2_exact4_coefficients(
                    input1 + offset, multiplier4, additive4, divisor,
                    upper, zero, &quantized1[quarter]);
                if (status != CAMPP_STATUS_OK) return status;
            }
            campp_bn_v2_pack16(
                quantized0, output_dtype, output0 + channel);
            campp_bn_v2_pack16(
                quantized1, output_dtype, output1 + channel);
        }
    }
#endif
    for (; channel < channels; ++channel) {
        int32_t value0;
        int32_t value1;
        CamppStatus status = campp_bn_quantize_scalar(
            input0[channel], multiplier[channel], additive[channel],
            quant_scale, quant_zero, output_dtype, &value0);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_bn_quantize_scalar(
            input1[channel], multiplier[channel], additive[channel],
            quant_scale, quant_zero, output_dtype, &value1);
        if (status != CAMPP_STATUS_OK) return status;
        output0[channel] = campp_bn_v2_store_byte(value0, output_dtype);
        output1[channel] = campp_bn_v2_store_byte(value1, output_dtype);
    }
    return CAMPP_STATUS_OK;
}
