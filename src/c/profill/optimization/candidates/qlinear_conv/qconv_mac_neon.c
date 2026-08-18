#include "qconv_mac_neon.h"

#include <string.h>

#include "campp_runtime/tensor_descriptor.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static int32_t campp_qconv_candidate_read(
    const uint8_t *data, uint8_t dtype)
{
    if (dtype == CAMPP_DTYPE_UINT8) return *data;
    {
        int8_t value;
        memcpy(&value, data, sizeof(value));
        return value;
    }
}

void campp_qconv_mac_scalar_tile(
    const uint8_t *const input_points[CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t tile_count, const uint8_t *packed_weight,
    uint32_t input_channels, uint8_t input_dtype, uint8_t weight_dtype,
    int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK],
    int32_t contribution[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK])
{
    const uint32_t input_blocks =
        (input_channels + CAMPP_QCONV_CANDIDATE_INPUT_BLOCK - 1u)
        / CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
    uint32_t tile;
    uint32_t input_block;

    memset(
        contribution, 0,
        sizeof(*contribution) * CAMPP_QCONV_CANDIDATE_TILE);
    for (input_block = 0u; input_block < input_blocks; ++input_block) {
        const uint8_t *weight_block = packed_weight
            + (size_t)input_block
                * CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK
                * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
        for (tile = 0u; tile < tile_count; ++tile) {
            uint32_t output_lane;
            if (input_points[tile] == NULL) continue;
            for (output_lane = 0u;
                 output_lane < CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK;
                 ++output_lane) {
                int64_t sum = contribution[tile][output_lane];
                uint32_t input_lane;
                for (input_lane = 0u;
                     input_lane < CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
                     ++input_lane) {
                    const uint32_t channel =
                        input_block * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK
                        + input_lane;
                    int32_t input_value;
                    int32_t weight_value;
                    if (channel >= input_channels) continue;
                    input_value = campp_qconv_candidate_read(
                        input_points[tile] + channel, input_dtype);
                    weight_value = campp_qconv_candidate_read(
                        weight_block
                            + output_lane
                                * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK
                            + input_lane,
                        weight_dtype);
                    sum += (int64_t)(input_value - input_zero)
                        * (weight_value - weight_zero[output_lane]);
                }
                contribution[tile][output_lane] = (int32_t)sum;
            }
        }
    }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
static int16x4_t campp_qconv_neon_input(
    const uint8_t *data, uint32_t valid_lanes,
    uint8_t dtype, int32_t zero_point)
{
    const uint8_t fill = (uint8_t)zero_point;
    uint32_t packed = (uint32_t)fill * UINT32_C(0x01010101);
    uint8x8_t bytes;
    int16x4_t values;

    if (valid_lanes == 4u) {
        memcpy(&packed, data, sizeof(packed));
    } else {
        uint8_t *destination = (uint8_t *)&packed;
        uint32_t lane;
        for (lane = 0u; lane < valid_lanes; ++lane) {
            destination[lane] = data[lane];
        }
    }
    bytes = vreinterpret_u8_u64(vcreate_u64(packed));
    if (dtype == CAMPP_DTYPE_UINT8) {
        values = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(bytes)));
    } else {
        values = vget_low_s16(vmovl_s8(vreinterpret_s8_u8(bytes)));
    }
    return vsub_s16(values, vdup_n_s16((int16_t)zero_point));
}

static void campp_qconv_neon_weight_columns(
    const uint8_t *data, uint8_t dtype,
    const int32_t zero_points[CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK],
    int16x4_t columns[CAMPP_QCONV_CANDIDATE_INPUT_BLOCK])
{
    int16x8_t low;
    int16x8_t high;
    int16x4_t rows[4];
    int16x4x2_t transpose01;
    int16x4x2_t transpose23;
    int32x2x2_t columns02;
    int32x2x2_t columns13;
    int16_t zero_values[4];
    int16x4_t zero;
    uint32_t lane;

    if (dtype == CAMPP_DTYPE_UINT8) {
        const uint8x16_t packed = vld1q_u8(data);
        low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(packed)));
        high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(packed)));
    } else {
        const int8x16_t packed = vld1q_s8((const int8_t *)data);
        low = vmovl_s8(vget_low_s8(packed));
        high = vmovl_s8(vget_high_s8(packed));
    }
    rows[0] = vget_low_s16(low);
    rows[1] = vget_high_s16(low);
    rows[2] = vget_low_s16(high);
    rows[3] = vget_high_s16(high);
    transpose01 = vtrn_s16(rows[0], rows[1]);
    transpose23 = vtrn_s16(rows[2], rows[3]);
    columns02 = vtrn_s32(
        vreinterpret_s32_s16(transpose01.val[0]),
        vreinterpret_s32_s16(transpose23.val[0]));
    columns13 = vtrn_s32(
        vreinterpret_s32_s16(transpose01.val[1]),
        vreinterpret_s32_s16(transpose23.val[1]));
    columns[0] = vreinterpret_s16_s32(columns02.val[0]);
    columns[1] = vreinterpret_s16_s32(columns13.val[0]);
    columns[2] = vreinterpret_s16_s32(columns02.val[1]);
    columns[3] = vreinterpret_s16_s32(columns13.val[1]);
    for (lane = 0u; lane < 4u; ++lane) {
        zero_values[lane] = (int16_t)zero_points[lane];
    }
    zero = vld1_s16(zero_values);
    for (lane = 0u; lane < 4u; ++lane) {
        columns[lane] = vsub_s16(columns[lane], zero);
    }
}
#endif

void campp_qconv_mac_neon_tile(
    const uint8_t *const input_points[CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t tile_count, const uint8_t *packed_weight,
    uint32_t input_channels, uint8_t input_dtype, uint8_t weight_dtype,
    int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK],
    int32_t contribution[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK])
{
#if defined(__aarch64__) && defined(__ARM_NEON)
    const uint32_t input_blocks =
        (input_channels + CAMPP_QCONV_CANDIDATE_INPUT_BLOCK - 1u)
        / CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
    int32x4_t accumulators[CAMPP_QCONV_CANDIDATE_TILE];
    uint32_t input_block;
    uint32_t tile;

    for (tile = 0u; tile < tile_count; ++tile) {
        accumulators[tile] = vdupq_n_s32(0);
    }
    for (input_block = 0u; input_block < input_blocks; ++input_block) {
        const uint32_t channel =
            input_block * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
        const uint32_t valid_lanes = input_channels - channel < 4u
            ? input_channels - channel : 4u;
        const uint8_t *weight_block = packed_weight
            + (size_t)input_block
                * CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK
                * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
        int16x4_t weight_columns[CAMPP_QCONV_CANDIDATE_INPUT_BLOCK];
        campp_qconv_neon_weight_columns(
            weight_block, weight_dtype, weight_zero, weight_columns);
        for (tile = 0u; tile < tile_count; ++tile) {
            int16x4_t input;
            if (input_points[tile] == NULL) continue;
            input = campp_qconv_neon_input(
                input_points[tile] + channel, valid_lanes,
                input_dtype, input_zero);
            accumulators[tile] = vmlal_lane_s16(
                accumulators[tile], weight_columns[0], input, 0);
            accumulators[tile] = vmlal_lane_s16(
                accumulators[tile], weight_columns[1], input, 1);
            accumulators[tile] = vmlal_lane_s16(
                accumulators[tile], weight_columns[2], input, 2);
            accumulators[tile] = vmlal_lane_s16(
                accumulators[tile], weight_columns[3], input, 3);
        }
    }
    memset(
        contribution, 0,
        sizeof(*contribution) * CAMPP_QCONV_CANDIDATE_TILE);
    for (tile = 0u; tile < tile_count; ++tile) {
        vst1q_s32(contribution[tile], accumulators[tile]);
    }
#else
    campp_qconv_mac_scalar_tile(
        input_points, tile_count, packed_weight, input_channels,
        input_dtype, weight_dtype, input_zero, weight_zero,
        contribution);
#endif
}
