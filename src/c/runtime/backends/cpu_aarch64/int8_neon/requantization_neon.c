/* QLinearConv INT32 accumulator requantization and channel-tail stores. */

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

#define CAMPP_QCONV_REQUANT_LANES 4u

static CamppStatus campp_qconv_requantize_scalar(
    int32_t accumulator, float multiplier, uint8_t output_dtype,
    int32_t output_zero, int32_t *out_value)
{
    const int64_t output_min =
        output_dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
    const int64_t output_max =
        output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    float scaled;
    int64_t rounded;

    if (out_value == NULL || !isfinite(multiplier)) {
        return CAMPP_STATUS_KERNEL_FAILED;
    }
    scaled = (float)accumulator * multiplier;
    if (scaled <= -2147483648.0f) {
        rounded = INT32_MIN;
    } else if (scaled >= 2147483520.0f) {
        rounded = INT32_MAX;
    } else {
        rounded = (int64_t)nearbyintf(scaled);
    }
    rounded += output_zero;
    if (rounded < output_min) rounded = output_min;
    if (rounded > output_max) rounded = output_max;
    *out_value = (int32_t)rounded;
    return CAMPP_STATUS_OK;
}

#if defined(__aarch64__) && defined(__ARM_NEON)
static uint32_t campp_qconv_requantize_neon4(
    const int32_t accumulators[CAMPP_QCONV_REQUANT_LANES],
    const float multipliers[CAMPP_QCONV_REQUANT_LANES],
    uint8_t output_dtype, int32_t output_zero)
{
    const int32_t output_min =
        output_dtype == CAMPP_DTYPE_UINT8 ? 0 : -128;
    const int32_t output_max =
        output_dtype == CAMPP_DTYPE_UINT8 ? 255 : 127;
    float32x4_t scaled = vmulq_f32(
        vcvtq_f32_s32(vld1q_s32(accumulators)),
        vld1q_f32(multipliers));
    int32x4_t quantized;

    /* Clamping before conversion also keeps FP values inside INT32 range. */
    scaled = vmaxq_f32(
        scaled, vdupq_n_f32((float)(output_min - output_zero)));
    scaled = vminq_f32(
        scaled, vdupq_n_f32((float)(output_max - output_zero)));
    quantized = vaddq_s32(
        vcvtnq_s32_f32(scaled), vdupq_n_s32(output_zero));
    if (output_dtype == CAMPP_DTYPE_UINT8) {
        const uint16x4_t narrowed16 = vqmovun_s32(quantized);
        const uint8x8_t narrowed8 = vqmovn_u16(
            vcombine_u16(narrowed16, vdup_n_u16(0u)));
        return vget_lane_u32(vreinterpret_u32_u8(narrowed8), 0);
    }
    {
        const int16x4_t narrowed16 = vqmovn_s32(quantized);
        const int8x8_t narrowed8 = vqmovn_s16(
            vcombine_s16(narrowed16, vdup_n_s16(0)));
        return vget_lane_u32(vreinterpret_u32_s8(narrowed8), 0);
    }
}
#endif

static bool campp_qconv_channel_store_offset(
    const CamppTensorView *output, uint32_t batch, uint32_t output_channel,
    uint32_t spatial_index, uint64_t *out_offset,
    uint32_t *out_channel_capacity)
{
    uint64_t offset;
    uint32_t channel_capacity;

    if (output == NULL || out_offset == NULL || out_channel_capacity == NULL ||
        output->rank < 3u || output->rank > 4u ||
        output->byte_strides[1] != 1u || batch >= output->dimensions[0] ||
        output_channel >= output->dimensions[1]) {
        return false;
    }
    offset = (uint64_t)batch * output->byte_strides[0] + output_channel;
    if (output->rank == 3u) {
        if (spatial_index >= output->dimensions[2]) return false;
        offset += (uint64_t)spatial_index * output->byte_strides[2];
        channel_capacity = output->byte_strides[2];
    } else {
        const uint32_t width = output->dimensions[3];
        uint32_t height_index;
        uint32_t width_index;
        if (width == 0u ||
            (uint64_t)spatial_index >=
                (uint64_t)output->dimensions[2] * width) {
            return false;
        }
        height_index = spatial_index / width;
        width_index = spatial_index % width;
        offset += (uint64_t)height_index * output->byte_strides[2] +
            (uint64_t)width_index * output->byte_strides[3];
        channel_capacity = output->byte_strides[3];
    }
    *out_offset = offset;
    *out_channel_capacity = channel_capacity;
    return true;
}

CamppStatus campp_aarch64_qconv_requantize_store4(
    CamppTensorView *output, uint32_t batch, uint32_t output_channel,
    uint32_t spatial_index,
    const int32_t accumulators[CAMPP_QCONV_REQUANT_LANES],
    const float multipliers[CAMPP_QCONV_REQUANT_LANES],
    uint32_t valid_outputs, int32_t output_zero)
{
    uint64_t spatial_count = 1u;
    uint64_t direct_offset;
    uint32_t channel_capacity;
    uint32_t lane;
    uint8_t axis;

    if (output == NULL || output->data == NULL || accumulators == NULL ||
        multipliers == NULL || valid_outputs == 0u ||
        valid_outputs > CAMPP_QCONV_REQUANT_LANES ||
        (output->dtype != CAMPP_DTYPE_UINT8 &&
         output->dtype != CAMPP_DTYPE_INT8) ||
        output->rank < 3u || output->rank > 4u ||
        batch >= output->dimensions[0] ||
        output_channel + valid_outputs > output->dimensions[1]) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    for (lane = 0u; lane < CAMPP_QCONV_REQUANT_LANES; ++lane) {
        if (!isfinite(multipliers[lane])) {
            return CAMPP_STATUS_KERNEL_FAILED;
        }
    }
    for (axis = 2u; axis < output->rank; ++axis) {
        spatial_count *= output->dimensions[axis];
    }
    if (spatial_index >= spatial_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    if (campp_qconv_channel_store_offset(
            output, batch, output_channel, spatial_index,
            &direct_offset, &channel_capacity)) {
        uint8_t bytes[CAMPP_QCONV_REQUANT_LANES];
        uint32_t store_count = valid_outputs;

#if defined(__aarch64__) && defined(__ARM_NEON)
        {
            const uint32_t packed = campp_qconv_requantize_neon4(
                accumulators, multipliers, output->dtype, output_zero);
            memcpy(bytes, &packed, sizeof(bytes));
        }
#else
        for (lane = 0u; lane < CAMPP_QCONV_REQUANT_LANES; ++lane) {
            int32_t value;
            CamppStatus status = campp_qconv_requantize_scalar(
                accumulators[lane], multipliers[lane], output->dtype,
                output_zero, &value);
            if (status != CAMPP_STATUS_OK) return status;
            bytes[lane] = output->dtype == CAMPP_DTYPE_UINT8
                ? (uint8_t)value : (uint8_t)(int8_t)value;
        }
#endif
        /*
         * A non-aliased channel-packed tensor owns its padded channel block,
         * so the tail can use the same four-byte store as a full block.
         */
        if (valid_outputs < CAMPP_QCONV_REQUANT_LANES &&
            output_channel + valid_outputs == output->dimensions[1] &&
            (output->flags & CAMPP_TENSOR_FLAG_ALIASED) == 0u &&
            output_channel + CAMPP_QCONV_REQUANT_LANES <= channel_capacity &&
            direct_offset + CAMPP_QCONV_REQUANT_LANES <=
                output->storage_span_bytes) {
            store_count = CAMPP_QCONV_REQUANT_LANES;
        }
        if (direct_offset + store_count > output->storage_span_bytes) {
            return CAMPP_STATUS_BUFFER_OVERFLOW;
        }
        memcpy((uint8_t *)output->data + direct_offset, bytes, store_count);
        return CAMPP_STATUS_OK;
    }

    for (lane = 0u; lane < valid_outputs; ++lane) {
        const uint64_t logical_index =
            ((uint64_t)batch * output->dimensions[1] +
             output_channel + lane) * spatial_count + spatial_index;
        int32_t value;
        CamppStatus status = campp_qconv_requantize_scalar(
            accumulators[lane], multipliers[lane], output->dtype,
            output_zero, &value);
        if (status != CAMPP_STATUS_OK) return status;
        status = campp_reference_write_quantized(
            output, logical_index, value);
        if (status != CAMPP_STATUS_OK) return status;
    }
    return CAMPP_STATUS_OK;
}
