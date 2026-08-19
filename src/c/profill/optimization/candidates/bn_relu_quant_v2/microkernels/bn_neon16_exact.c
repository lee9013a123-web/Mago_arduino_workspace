#include "bn_neon16_exact.h"

#include "bn_neon16_internal.h"

CamppStatus campp_bn_v2_exact16_span(
    const float *input, uint8_t *output, uint32_t channels,
    const float *multiplier, const float *additive, float quant_scale,
    int32_t quant_zero, uint8_t output_dtype)
{
    uint32_t channel = 0u;

    if (input == NULL || output == NULL || multiplier == NULL ||
        additive == NULL) {
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
            int32x4_t quantized[4];
            uint32_t quarter;
            for (quarter = 0u; quarter < 4u; ++quarter) {
                const uint32_t offset = channel + quarter * 4u;
                CamppStatus status = campp_bn_v2_exact4(
                    input + offset, multiplier + offset, additive + offset,
                    divisor, upper, zero, &quantized[quarter]);
                if (status != CAMPP_STATUS_OK) return status;
            }
            campp_bn_v2_pack16(
                quantized, output_dtype, output + channel);
        }
    }
#endif
    for (; channel < channels; ++channel) {
        int32_t value;
        CamppStatus status = campp_bn_quantize_scalar(
            input[channel], multiplier[channel], additive[channel],
            quant_scale, quant_zero, output_dtype, &value);
        if (status != CAMPP_STATUS_OK) return status;
        output[channel] = campp_bn_v2_store_byte(value, output_dtype);
    }
    return CAMPP_STATUS_OK;
}
