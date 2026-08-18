#include "qconv_mac_neon.h"

#include <limits.h>
#include <math.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
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

#if defined(__aarch64__) && defined(__ARM_NEON) && \
    defined(__ARM_FEATURE_DOTPROD)
static int8x8_t campp_qconv_dot_input(
    const uint8_t *data, uint32_t valid_lanes,
    uint8_t dtype, int32_t zero_point)
{
    const uint8_t fill = (uint8_t)zero_point;
    uint32_t packed = (uint32_t)fill * UINT32_C(0x01010101);

    if (valid_lanes == 4u) {
        memcpy(&packed, data, sizeof(packed));
    } else {
        uint8_t *destination = (uint8_t *)&packed;
        uint32_t lane;
        for (lane = 0u; lane < valid_lanes; ++lane) {
            destination[lane] = data[lane];
        }
    }
    if (dtype == CAMPP_DTYPE_UINT8) {
        packed ^= UINT32_C(0x80808080);
    }
    return vreinterpret_s8_u64(vcreate_u64(packed));
}

static int8x16_t campp_qconv_dot_weight(
    const uint8_t *data, uint8_t dtype)
{
    uint8x16_t bytes = vld1q_u8(data);
    if (dtype == CAMPP_DTYPE_UINT8) {
        bytes = veorq_u8(bytes, vdupq_n_u8(UINT8_C(0x80)));
    }
    return vreinterpretq_s8_u8(bytes);
}

static int32x4_t campp_qconv_dot_weight_zero(
    const int32_t zero_points[CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK],
    uint8_t dtype)
{
    int32_t transformed[CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK];
    uint32_t lane;
    const int32_t offset = dtype == CAMPP_DTYPE_UINT8 ? 128 : 0;

    for (lane = 0u; lane < CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK; ++lane) {
        transformed[lane] = zero_points[lane] - offset;
    }
    return vld1q_s32(transformed);
}

static int32x4_t campp_qconv_dot_accumulate(
    int32x4_t accumulator, int8x16_t weight, int8x8_t input,
    int32x4_t weight_zero, int32_t input_zero)
{
    const int32_t packed_input =
        vget_lane_s32(vreinterpret_s32_s8(input), 0);
    const int8x16_t repeated_input = vreinterpretq_s8_s32(
        vdupq_n_s32(packed_input));
    const int32_t input_sum = vaddlv_s8(input);
    const int16x8_t weight_pairs = vpaddlq_s8(weight);
    const int32x4_t weight_sums = vpaddlq_s16(weight_pairs);

    accumulator = vdotq_s32(accumulator, weight, repeated_input);
    accumulator = vsubq_s32(
        accumulator, vmulq_n_s32(weight_zero, input_sum));
    accumulator = vsubq_s32(
        accumulator, vmulq_n_s32(weight_sums, input_zero));
    accumulator = vaddq_s32(
        accumulator, vmulq_n_s32(weight_zero, input_zero * 4));
    return accumulator;
}
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#if !defined(__ARM_FEATURE_DOTPROD)
static int32x4_t campp_qconv_smlal_accumulate(
    int32x4_t accumulator,
    const int16x4_t weight_columns[CAMPP_QCONV_CANDIDATE_INPUT_BLOCK],
    int16x4_t input)
{
    accumulator = vmlal_lane_s16(
        accumulator, weight_columns[0], input, 0);
    accumulator = vmlal_lane_s16(
        accumulator, weight_columns[1], input, 1);
    accumulator = vmlal_lane_s16(
        accumulator, weight_columns[2], input, 2);
    accumulator = vmlal_lane_s16(
        accumulator, weight_columns[3], input, 3);
    return accumulator;
}
#endif

static int campp_qconv_mac_neon_half_tile_v2(
    const uint8_t *const input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t kernel_elements, const uint8_t *const packed_weights[2],
    uint32_t input_channels, uint8_t input_dtype, uint8_t weight_dtype,
    int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs, uint32_t tile_offset,
    int32_t output[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    const uint32_t input_blocks =
        (input_channels + CAMPP_QCONV_CANDIDATE_INPUT_BLOCK - 1u)
        / CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
    int32x4_t accumulator00 = vdupq_n_s32(0);
    int32x4_t accumulator01 = vdupq_n_s32(0);
    int32x4_t accumulator02 = vdupq_n_s32(0);
    int32x4_t accumulator03 = vdupq_n_s32(0);
    int32x4_t accumulator10 = vdupq_n_s32(0);
    int32x4_t accumulator11 = vdupq_n_s32(0);
    int32x4_t accumulator12 = vdupq_n_s32(0);
    int32x4_t accumulator13 = vdupq_n_s32(0);
    uint32_t kernel;

    for (kernel = 0u; kernel < kernel_elements; ++kernel) {
        uint32_t input_block;
        for (input_block = 0u;
             input_block < input_blocks; ++input_block) {
            const uint32_t input_channel =
                input_block * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
            const uint32_t valid_lanes = input_channels - input_channel
                < CAMPP_QCONV_CANDIDATE_INPUT_BLOCK
                ? input_channels - input_channel
                : CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
            const size_t block_offset =
                ((size_t)kernel * input_blocks + input_block)
                * CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK
                * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
#if defined(__ARM_FEATURE_DOTPROD)
            const int8x16_t weight0 = campp_qconv_dot_weight(
                packed_weights[0] + block_offset, weight_dtype);
            const int32x4_t weight_zero0 =
                campp_qconv_dot_weight_zero(weight_zero, weight_dtype);
            int8x16_t weight1 = vdupq_n_s8(0);
            int32x4_t weight_zero1 = vdupq_n_s32(0);
            const int32_t transformed_input_zero = input_zero
                - (input_dtype == CAMPP_DTYPE_UINT8 ? 128 : 0);
#define CAMPP_QCONV_DOT_FULL_TILE(tile)                                \
            do {                                                       \
                const uint8_t *point =                               \
                    input_points[kernel][tile_offset + tile];          \
                if (point != NULL) {                                   \
                    const int8x8_t input = campp_qconv_dot_input(       \
                        point + input_channel, valid_lanes,             \
                        input_dtype, input_zero);                       \
                    accumulator0##tile = campp_qconv_dot_accumulate(   \
                        accumulator0##tile, weight0, input,             \
                        weight_zero0, transformed_input_zero);         \
                    if (valid_outputs >                                \
                        CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK) {          \
                        accumulator1##tile =                           \
                            campp_qconv_dot_accumulate(                \
                                accumulator1##tile, weight1, input,    \
                                weight_zero1,                          \
                                transformed_input_zero);              \
                    }                                                  \
                }                                                      \
            } while (0)
            if (valid_outputs > CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK) {
                weight1 = campp_qconv_dot_weight(
                    packed_weights[1] + block_offset, weight_dtype);
                weight_zero1 = campp_qconv_dot_weight_zero(
                    weight_zero + CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK,
                    weight_dtype);
            }
            CAMPP_QCONV_DOT_FULL_TILE(0);
            CAMPP_QCONV_DOT_FULL_TILE(1);
            CAMPP_QCONV_DOT_FULL_TILE(2);
            CAMPP_QCONV_DOT_FULL_TILE(3);
#undef CAMPP_QCONV_DOT_FULL_TILE
#else
            {
                int16x4_t weight_columns0
                    [CAMPP_QCONV_CANDIDATE_INPUT_BLOCK];
                int16x4_t weight_columns1
                    [CAMPP_QCONV_CANDIDATE_INPUT_BLOCK];
#define CAMPP_QCONV_SMLAL_FULL_TILE(tile)                              \
                do {                                                   \
                    const uint8_t *point =                            \
                        input_points[kernel][tile_offset + tile];      \
                    if (point != NULL) {                               \
                        const int16x4_t input =                         \
                            campp_qconv_neon_input(                    \
                                point + input_channel, valid_lanes,    \
                                input_dtype, input_zero);              \
                        accumulator0##tile =                           \
                            campp_qconv_smlal_accumulate(              \
                                accumulator0##tile,                   \
                                weight_columns0, input);              \
                        if (valid_outputs >                            \
                            CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK) {      \
                            accumulator1##tile =                       \
                                campp_qconv_smlal_accumulate(          \
                                    accumulator1##tile,               \
                                    weight_columns1, input);          \
                        }                                              \
                    }                                                  \
                } while (0)
                campp_qconv_neon_weight_columns(
                    packed_weights[0] + block_offset, weight_dtype,
                    weight_zero, weight_columns0);
                if (valid_outputs >
                    CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK) {
                    campp_qconv_neon_weight_columns(
                        packed_weights[1] + block_offset, weight_dtype,
                        weight_zero
                            + CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK,
                        weight_columns1);
                }
                CAMPP_QCONV_SMLAL_FULL_TILE(0);
                CAMPP_QCONV_SMLAL_FULL_TILE(1);
                CAMPP_QCONV_SMLAL_FULL_TILE(2);
                CAMPP_QCONV_SMLAL_FULL_TILE(3);
#undef CAMPP_QCONV_SMLAL_FULL_TILE
            }
#endif
        }
    }
    vst1q_s32(output[tile_offset], accumulator00);
    vst1q_s32(output[tile_offset + 1u], accumulator01);
    vst1q_s32(output[tile_offset + 2u], accumulator02);
    vst1q_s32(output[tile_offset + 3u], accumulator03);
    vst1q_s32(
        output[tile_offset] + CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK,
        accumulator10);
    vst1q_s32(
        output[tile_offset + 1u] + CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK,
        accumulator11);
    vst1q_s32(
        output[tile_offset + 2u] + CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK,
        accumulator12);
    vst1q_s32(
        output[tile_offset + 3u] + CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK,
        accumulator13);
    {
        uint32_t tile;
        for (tile = tile_offset; tile < tile_offset + 4u; ++tile) {
            uint32_t output_lane;
            for (output_lane = 0u;
                 output_lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE;
                 ++output_lane) {
                const int64_t value = (int64_t)output[tile][output_lane]
                    + bias[output_lane];
                if (output_lane < valid_outputs &&
                    (value < INT32_MIN || value > INT32_MAX)) {
                    return 1;
                }
                output[tile][output_lane] = (int32_t)value;
            }
        }
    }
    return 0;
}
#endif

#if !defined(__aarch64__) || !defined(__ARM_NEON)
static int campp_qconv_mac_scalar_tile_v2(
    const uint8_t *const input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t kernel_elements, uint32_t tile_count,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    uint8_t input_dtype, uint8_t weight_dtype, int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    const uint32_t input_blocks =
        (input_channels + CAMPP_QCONV_CANDIDATE_INPUT_BLOCK - 1u)
        / CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
    int64_t sums[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE] = {{0}};
    uint32_t kernel;
    uint32_t tile;

    for (kernel = 0u; kernel < kernel_elements; ++kernel) {
        uint32_t input_block;
        for (input_block = 0u; input_block < input_blocks; ++input_block) {
            const uint32_t input_channel =
                input_block * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
            uint32_t output_lane;
            for (tile = 0u; tile < tile_count; ++tile) {
                const uint8_t *input = input_points[kernel][tile];
                if (input == NULL) continue;
                for (output_lane = 0u;
                     output_lane < valid_outputs; ++output_lane) {
                    const uint32_t output_block =
                        output_lane / CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK;
                    const uint32_t block_lane =
                        output_lane % CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK;
                    const uint8_t *weight_block =
                        packed_weights[output_block]
                        + ((size_t)kernel * input_blocks + input_block)
                            * CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK
                            * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
                    uint32_t input_lane;
                    for (input_lane = 0u;
                         input_lane < CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
                         ++input_lane) {
                        const uint32_t channel = input_channel + input_lane;
                        if (channel >= input_channels) continue;
                        sums[tile][output_lane] +=
                            (int64_t)(campp_qconv_candidate_read(
                                input + channel, input_dtype) - input_zero)
                            * (campp_qconv_candidate_read(
                                weight_block
                                    + block_lane
                                        * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK
                                    + input_lane,
                                weight_dtype) - weight_zero[output_lane]);
                    }
                }
            }
        }
    }
    for (tile = 0u; tile < tile_count; ++tile) {
        uint32_t output_lane;
        for (output_lane = 0u;
             output_lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE;
             ++output_lane) {
            const int64_t value = sums[tile][output_lane]
                + bias[output_lane];
            if (output_lane < valid_outputs &&
                (value < INT32_MIN || value > INT32_MAX)) {
                return 1;
            }
            accumulators[tile][output_lane] = (int32_t)value;
        }
    }
    return 0;
}
#endif

int campp_qconv_mac_neon_tile_v2(
    const uint8_t *const input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t kernel_elements, uint32_t tile_count,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    uint8_t input_dtype, uint8_t weight_dtype, int32_t input_zero,
    const int32_t weight_zero[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs,
    int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    const uint64_t accumulation_bound =
        (uint64_t)kernel_elements * input_channels * UINT64_C(65025);

    if (input_points == NULL || packed_weights == NULL ||
        packed_weights[0] == NULL || weight_zero == NULL || bias == NULL ||
        accumulators == NULL || kernel_elements == 0u ||
        kernel_elements > CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS ||
        tile_count == 0u || tile_count > CAMPP_QCONV_CANDIDATE_TILE ||
        valid_outputs == 0u ||
        valid_outputs > CAMPP_QCONV_CANDIDATE_OUTPUT_TILE ||
        (valid_outputs > CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK &&
         packed_weights[1] == NULL) || accumulation_bound > INT32_MAX) {
        return 1;
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (tile_count == CAMPP_QCONV_CANDIDATE_TILE) {
        int result = campp_qconv_mac_neon_half_tile_v2(
            input_points, kernel_elements, packed_weights,
            input_channels, input_dtype, weight_dtype, input_zero,
            weight_zero, bias, valid_outputs, 0u, accumulators);
        if (result != 0) return result;
        return campp_qconv_mac_neon_half_tile_v2(
            input_points, kernel_elements, packed_weights,
            input_channels, input_dtype, weight_dtype, input_zero,
            weight_zero, bias, valid_outputs, 4u, accumulators);
    }
    {
        const uint32_t input_blocks =
            (input_channels + CAMPP_QCONV_CANDIDATE_INPUT_BLOCK - 1u)
            / CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
        int32x4_t accumulator0[CAMPP_QCONV_CANDIDATE_TILE];
        int32x4_t accumulator1[CAMPP_QCONV_CANDIDATE_TILE];
        uint32_t kernel;
        uint32_t tile;

        for (tile = 0u; tile < tile_count; ++tile) {
            accumulator0[tile] = vdupq_n_s32(0);
            accumulator1[tile] = vdupq_n_s32(0);
        }
        for (kernel = 0u; kernel < kernel_elements; ++kernel) {
            uint32_t input_block;
            for (input_block = 0u;
                 input_block < input_blocks; ++input_block) {
                const uint32_t input_channel =
                    input_block * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
                const uint32_t valid_lanes = input_channels - input_channel
                    < CAMPP_QCONV_CANDIDATE_INPUT_BLOCK
                    ? input_channels - input_channel
                    : CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
                const size_t block_offset =
                    ((size_t)kernel * input_blocks + input_block)
                    * CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK
                    * CAMPP_QCONV_CANDIDATE_INPUT_BLOCK;
#if defined(__ARM_FEATURE_DOTPROD)
                const int8x16_t weight0 = campp_qconv_dot_weight(
                    packed_weights[0] + block_offset, weight_dtype);
                const int32x4_t weight_zero0 =
                    campp_qconv_dot_weight_zero(weight_zero, weight_dtype);
                int8x16_t weight1 = vdupq_n_s8(0);
                int32x4_t weight_zero1 = vdupq_n_s32(0);
                const int32_t transformed_input_zero = input_zero
                    - (input_dtype == CAMPP_DTYPE_UINT8 ? 128 : 0);
                if (valid_outputs > CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK) {
                    weight1 = campp_qconv_dot_weight(
                        packed_weights[1] + block_offset, weight_dtype);
                    weight_zero1 = campp_qconv_dot_weight_zero(
                        weight_zero + CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK,
                        weight_dtype);
                }
                for (tile = 0u; tile < tile_count; ++tile) {
                    int8x8_t input;
                    if (input_points[kernel][tile] == NULL) continue;
                    input = campp_qconv_dot_input(
                        input_points[kernel][tile] + input_channel,
                        valid_lanes, input_dtype, input_zero);
                    accumulator0[tile] = campp_qconv_dot_accumulate(
                        accumulator0[tile], weight0, input,
                        weight_zero0, transformed_input_zero);
                    if (valid_outputs >
                        CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK) {
                        accumulator1[tile] = campp_qconv_dot_accumulate(
                            accumulator1[tile], weight1, input,
                            weight_zero1, transformed_input_zero);
                    }
                }
#else
                {
                    int16x4_t weight_columns0
                        [CAMPP_QCONV_CANDIDATE_INPUT_BLOCK];
                    int16x4_t weight_columns1
                        [CAMPP_QCONV_CANDIDATE_INPUT_BLOCK];
                    campp_qconv_neon_weight_columns(
                        packed_weights[0] + block_offset, weight_dtype,
                        weight_zero, weight_columns0);
                    if (valid_outputs >
                        CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK) {
                        campp_qconv_neon_weight_columns(
                            packed_weights[1] + block_offset, weight_dtype,
                            weight_zero
                                + CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK,
                            weight_columns1);
                    }
                    for (tile = 0u; tile < tile_count; ++tile) {
                        int16x4_t input;
                        if (input_points[kernel][tile] == NULL) continue;
                        input = campp_qconv_neon_input(
                            input_points[kernel][tile] + input_channel,
                            valid_lanes, input_dtype, input_zero);
                        accumulator0[tile] = vmlal_lane_s16(
                            accumulator0[tile], weight_columns0[0], input, 0);
                        accumulator0[tile] = vmlal_lane_s16(
                            accumulator0[tile], weight_columns0[1], input, 1);
                        accumulator0[tile] = vmlal_lane_s16(
                            accumulator0[tile], weight_columns0[2], input, 2);
                        accumulator0[tile] = vmlal_lane_s16(
                            accumulator0[tile], weight_columns0[3], input, 3);
                        if (valid_outputs >
                            CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK) {
                            accumulator1[tile] = vmlal_lane_s16(
                                accumulator1[tile], weight_columns1[0],
                                input, 0);
                            accumulator1[tile] = vmlal_lane_s16(
                                accumulator1[tile], weight_columns1[1],
                                input, 1);
                            accumulator1[tile] = vmlal_lane_s16(
                                accumulator1[tile], weight_columns1[2],
                                input, 2);
                            accumulator1[tile] = vmlal_lane_s16(
                                accumulator1[tile], weight_columns1[3],
                                input, 3);
                        }
                    }
                }
#endif
            }
        }
        for (tile = 0u; tile < tile_count; ++tile) {
            uint32_t output_lane;
            vst1q_s32(accumulators[tile], accumulator0[tile]);
            vst1q_s32(
                accumulators[tile]
                    + CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK,
                accumulator1[tile]);
            for (output_lane = 0u;
                 output_lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE;
                 ++output_lane) {
                const int64_t value =
                    (int64_t)accumulators[tile][output_lane]
                    + bias[output_lane];
                if (output_lane < valid_outputs &&
                    (value < INT32_MIN || value > INT32_MAX)) {
                    return 1;
                }
                accumulators[tile][output_lane] = (int32_t)value;
            }
        }
        return 0;
    }
#else
    return campp_qconv_mac_scalar_tile_v2(
        input_points, kernel_elements, tile_count, packed_weights,
        input_channels, input_dtype, weight_dtype, input_zero,
        weight_zero, bias, valid_outputs, accumulators);
#endif
}

int campp_qconv_requantize_store_neon_tile(
    CamppTensorView *output, uint32_t batch, uint32_t output_channel,
    uint8_t spatial_rank,
    const uint32_t output_coordinates[CAMPP_QCONV_CANDIDATE_TILE][2],
    uint32_t tile_count, uint32_t valid_outputs,
    const int32_t accumulators[CAMPP_QCONV_CANDIDATE_TILE]
        [CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    const float multipliers[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    int32_t output_zero)
{
    uint32_t tile;

    if (output == NULL || output->data == NULL ||
        output_coordinates == NULL || accumulators == NULL ||
        multipliers == NULL || spatial_rank < 1u || spatial_rank > 2u ||
        output->rank != spatial_rank + 2u ||
        (output->dtype != CAMPP_DTYPE_UINT8 &&
         output->dtype != CAMPP_DTYPE_INT8) ||
        tile_count == 0u || tile_count > CAMPP_QCONV_CANDIDATE_TILE ||
        valid_outputs == 0u ||
        valid_outputs > CAMPP_QCONV_CANDIDATE_OUTPUT_TILE) {
        return 1;
    }
    for (tile = 0u; tile < tile_count; ++tile) {
        uint32_t output_block;
        const uint32_t spatial_index = spatial_rank == 1u
            ? output_coordinates[tile][0]
            : output_coordinates[tile][0] * output->dimensions[3] +
                output_coordinates[tile][1];

        for (output_block = 0u;
             output_block < 2u; ++output_block) {
            const uint32_t lane_start =
                output_block * CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK;
            const uint32_t lanes = valid_outputs > lane_start
                ? (valid_outputs - lane_start <
                       CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK
                    ? valid_outputs - lane_start
                    : CAMPP_QCONV_CANDIDATE_OUTPUT_BLOCK)
                : 0u;
            if (lanes == 0u) continue;
            if (campp_aarch64_qconv_requantize_store4(
                    output, batch, output_channel + lane_start,
                    spatial_index, accumulators[tile] + lane_start,
                    multipliers + lane_start, lanes, output_zero) !=
                CAMPP_STATUS_OK) {
                return 1;
            }
        }
    }
    return 0;
}
