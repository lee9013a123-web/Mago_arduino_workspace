#include "bn_neon16_prescaled.h"

#include "bn_neon16_internal.h"

static CamppStatus campp_bn_v2_prescaled16_impl(
    const float *input0, const float *input1,
    uint8_t *output0, uint8_t *output1, uint32_t channels,
    const float *multiplier, const float *additive,
    int32_t quant_zero, uint8_t output_dtype)
{
    uint32_t channel = 0u;
    const int paired = input1 != NULL;

    if (input0 == NULL || output0 == NULL || multiplier == NULL ||
        additive == NULL || (paired && output1 == NULL)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    {
        const int32_t maximum =
            output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
        const float32x4_t upper =
            vdupq_n_f32((float)(maximum - quant_zero));
        const int32x4_t zero = vdupq_n_s32(quant_zero);
        for (; channel + 16u <= channels; channel += 16u) {
            int32x4_t quantized0[4];
            int32x4_t quantized1[4];
            uint32_t quarter;
            for (quarter = 0u; quarter < 4u; ++quarter) {
                const uint32_t offset = channel + quarter * 4u;
                CamppStatus status = campp_bn_v2_prescaled4(
                    input0 + offset, multiplier + offset, additive + offset,
                    upper, zero, &quantized0[quarter]);
                if (status != CAMPP_STATUS_OK) return status;
                if (paired) {
                    status = campp_bn_v2_prescaled4(
                        input1 + offset, multiplier + offset,
                        additive + offset, upper, zero,
                        &quantized1[quarter]);
                    if (status != CAMPP_STATUS_OK) return status;
                }
            }
            campp_bn_v2_pack16(
                quantized0, output_dtype, output0 + channel);
            if (paired) {
                campp_bn_v2_pack16(
                    quantized1, output_dtype, output1 + channel);
            }
        }
    }
#endif
    for (; channel < channels; ++channel) {
        CamppStatus status = campp_bn_v2_prescaled_scalar(
            input0[channel], multiplier[channel], additive[channel],
            quant_zero, output_dtype, &output0[channel]);
        if (status != CAMPP_STATUS_OK) return status;
        if (paired) {
            status = campp_bn_v2_prescaled_scalar(
                input1[channel], multiplier[channel], additive[channel],
                quant_zero, output_dtype, &output1[channel]);
            if (status != CAMPP_STATUS_OK) return status;
        }
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_bn_v2_prescaled16_span(
    const float *input, uint8_t *output, uint32_t channels,
    const float *multiplier, const float *additive,
    int32_t quant_zero, uint8_t output_dtype)
{
    return campp_bn_v2_prescaled16_impl(
        input, NULL, output, NULL, channels, multiplier, additive,
        quant_zero, output_dtype);
}

CamppStatus campp_bn_v2_prescaled16_spatial2(
    const float *input0, const float *input1,
    uint8_t *output0, uint8_t *output1, uint32_t channels,
    const float *multiplier, const float *additive,
    int32_t quant_zero, uint8_t output_dtype)
{
    return campp_bn_v2_prescaled16_impl(
        input0, input1, output0, output1, channels, multiplier, additive,
        quant_zero, output_dtype);
}
