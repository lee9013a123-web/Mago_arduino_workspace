#include "qconv_mac_v5_common.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

static inline int16x4_t campp_qconv_v5_load_input(
    const uint8_t *data, int16x4_t input_zero, bool zero_point_fastpath)
{
    uint32_t packed;
    uint8x8_t bytes;
    memcpy(&packed, data, sizeof(packed));
    bytes = vreinterpret_u8_u64(vcreate_u64(packed));
    {
        const int16x4_t widened =
            vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(bytes)));
        return zero_point_fastpath ? widened : vsub_s16(widened, input_zero);
    }
}

#define CAMPP_QCONV_V5_LOAD_COLUMNS(prefix, address, zero_points, fastpath) \
    const int8x16_t prefix##_packed =                                      \
        vld1q_s8((const int8_t *)(address));                               \
    const int16x8_t prefix##_low =                                         \
        vmovl_s8(vget_low_s8(prefix##_packed));                            \
    const int16x8_t prefix##_high =                                        \
        vmovl_s8(vget_high_s8(prefix##_packed));                           \
    const int16x4_t prefix##_row0 = vget_low_s16(prefix##_low);            \
    const int16x4_t prefix##_row1 = vget_high_s16(prefix##_low);           \
    const int16x4_t prefix##_row2 = vget_low_s16(prefix##_high);           \
    const int16x4_t prefix##_row3 = vget_high_s16(prefix##_high);          \
    const int16x4x2_t prefix##_transpose01 =                               \
        vtrn_s16(prefix##_row0, prefix##_row1);                            \
    const int16x4x2_t prefix##_transpose23 =                               \
        vtrn_s16(prefix##_row2, prefix##_row3);                            \
    const int32x2x2_t prefix##_columns02 = vtrn_s32(                       \
        vreinterpret_s32_s16(prefix##_transpose01.val[0]),                 \
        vreinterpret_s32_s16(prefix##_transpose23.val[0]));                \
    const int32x2x2_t prefix##_columns13 = vtrn_s32(                       \
        vreinterpret_s32_s16(prefix##_transpose01.val[1]),                 \
        vreinterpret_s32_s16(prefix##_transpose23.val[1]));                \
    int16x4_t prefix##0 =                                                  \
        vreinterpret_s16_s32(prefix##_columns02.val[0]);                   \
    int16x4_t prefix##1 =                                                  \
        vreinterpret_s16_s32(prefix##_columns13.val[0]);                   \
    int16x4_t prefix##2 =                                                  \
        vreinterpret_s16_s32(prefix##_columns02.val[1]);                   \
    int16x4_t prefix##3 =                                                  \
        vreinterpret_s16_s32(prefix##_columns13.val[1]);                   \
    if (!(fastpath)) {                                                     \
        prefix##0 = vsub_s16(prefix##0, (zero_points));                   \
        prefix##1 = vsub_s16(prefix##1, (zero_points));                   \
        prefix##2 = vsub_s16(prefix##2, (zero_points));                   \
        prefix##3 = vsub_s16(prefix##3, (zero_points));                   \
    }

#define CAMPP_QCONV_V5_ACCUMULATE(accumulator, columns, input)            \
    do {                                                                  \
        (accumulator) = vmlal_lane_s16(                                   \
            (accumulator), columns##0, (input), 0);                       \
        (accumulator) = vmlal_lane_s16(                                   \
            (accumulator), columns##1, (input), 1);                       \
        (accumulator) = vmlal_lane_s16(                                   \
            (accumulator), columns##2, (input), 2);                       \
        (accumulator) = vmlal_lane_s16(                                   \
            (accumulator), columns##3, (input), 3);                       \
    } while (0)

static int campp_qconv_v5_add_bias(
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE])
{
    uint32_t tile;
    for (tile = 0u; tile < CAMPP_QCONV_V5_SPATIAL_TILE; ++tile) {
        uint32_t lane;
        for (lane = 0u; lane < CAMPP_QCONV_V5_OUTPUT_TILE; ++lane) {
            const int64_t value =
                (int64_t)accumulators[tile][lane] + bias[lane];
            if (value < INT32_MIN || value > INT32_MAX) return 1;
            accumulators[tile][lane] = (int32_t)value;
        }
    }
    return 0;
}

static void campp_qconv_v5_store_accumulators(
    int32x4_t low0, int32x4_t high0,
    int32x4_t low1, int32x4_t high1,
    int32x4_t low2, int32x4_t high2,
    int32x4_t low3, int32x4_t high3,
    int32x4_t low4, int32x4_t high4,
    int32x4_t low5, int32x4_t high5,
    int32x4_t low6, int32x4_t high6,
    int32x4_t low7, int32x4_t high7,
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE])
{
    vst1q_s32(accumulators[0], low0);
    vst1q_s32(accumulators[0] + 4u, high0);
    vst1q_s32(accumulators[1], low1);
    vst1q_s32(accumulators[1] + 4u, high1);
    vst1q_s32(accumulators[2], low2);
    vst1q_s32(accumulators[2] + 4u, high2);
    vst1q_s32(accumulators[3], low3);
    vst1q_s32(accumulators[3] + 4u, high3);
    vst1q_s32(accumulators[4], low4);
    vst1q_s32(accumulators[4] + 4u, high4);
    vst1q_s32(accumulators[5], low5);
    vst1q_s32(accumulators[5] + 4u, high5);
    vst1q_s32(accumulators[6], low6);
    vst1q_s32(accumulators[6] + 4u, high6);
    vst1q_s32(accumulators[7], low7);
    vst1q_s32(accumulators[7] + 4u, high7);
}

CamppQconvMac4x8Result campp_qconv_v5_mac_8x8_raw(
    const uint8_t *const input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t kernel_elements,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    int32_t input_zero, bool zero_point_fastpath,
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE])
{
    const uint32_t input_blocks = input_channels / 4u;
    const int16x4_t input_zero_v = vdup_n_s16((int16_t)input_zero);
    const int16x4_t weight_zero0 = vcreate_s16(
        (uint64_t)(uint16_t)weight_zero[0] |
        ((uint64_t)(uint16_t)weight_zero[1] << 16u) |
        ((uint64_t)(uint16_t)weight_zero[2] << 32u) |
        ((uint64_t)(uint16_t)weight_zero[3] << 48u));
    const int16x4_t weight_zero1 = vcreate_s16(
        (uint64_t)(uint16_t)weight_zero[4] |
        ((uint64_t)(uint16_t)weight_zero[5] << 16u) |
        ((uint64_t)(uint16_t)weight_zero[6] << 32u) |
        ((uint64_t)(uint16_t)weight_zero[7] << 48u));
    const uint8_t *weight0 = packed_weights[0];
    const uint8_t *weight1 = packed_weights[1];
    int32x4_t low0 = vdupq_n_s32(0);
    int32x4_t high0 = vdupq_n_s32(0);
    int32x4_t low1 = vdupq_n_s32(0);
    int32x4_t high1 = vdupq_n_s32(0);
    int32x4_t low2 = vdupq_n_s32(0);
    int32x4_t high2 = vdupq_n_s32(0);
    int32x4_t low3 = vdupq_n_s32(0);
    int32x4_t high3 = vdupq_n_s32(0);
    int32x4_t low4 = vdupq_n_s32(0);
    int32x4_t high4 = vdupq_n_s32(0);
    int32x4_t low5 = vdupq_n_s32(0);
    int32x4_t high5 = vdupq_n_s32(0);
    int32x4_t low6 = vdupq_n_s32(0);
    int32x4_t high6 = vdupq_n_s32(0);
    int32x4_t low7 = vdupq_n_s32(0);
    int32x4_t high7 = vdupq_n_s32(0);
    uint32_t kernel;

    for (kernel = 0u; kernel < kernel_elements; ++kernel) {
        const uint8_t *const *points =
            input_points[kernel];
        uint32_t input_block;
        for (input_block = 0u; input_block < input_blocks; ++input_block) {
            const uint32_t channel = input_block * 4u;
            const int16x4_t input0 = campp_qconv_v5_load_input(
                points[0] + channel, input_zero_v, zero_point_fastpath);
            const int16x4_t input1 = campp_qconv_v5_load_input(
                points[1] + channel, input_zero_v, zero_point_fastpath);
            const int16x4_t input2 = campp_qconv_v5_load_input(
                points[2] + channel, input_zero_v, zero_point_fastpath);
            const int16x4_t input3 = campp_qconv_v5_load_input(
                points[3] + channel, input_zero_v, zero_point_fastpath);
            const int16x4_t input4 = campp_qconv_v5_load_input(
                points[4] + channel, input_zero_v, zero_point_fastpath);
            const int16x4_t input5 = campp_qconv_v5_load_input(
                points[5] + channel, input_zero_v, zero_point_fastpath);
            const int16x4_t input6 = campp_qconv_v5_load_input(
                points[6] + channel, input_zero_v, zero_point_fastpath);
            const int16x4_t input7 = campp_qconv_v5_load_input(
                points[7] + channel, input_zero_v, zero_point_fastpath);
            {
                CAMPP_QCONV_V5_LOAD_COLUMNS(
                    w0, weight0, weight_zero0, zero_point_fastpath);
                CAMPP_QCONV_V5_ACCUMULATE(low0, w0, input0);
                CAMPP_QCONV_V5_ACCUMULATE(low1, w0, input1);
                CAMPP_QCONV_V5_ACCUMULATE(low2, w0, input2);
                CAMPP_QCONV_V5_ACCUMULATE(low3, w0, input3);
                CAMPP_QCONV_V5_ACCUMULATE(low4, w0, input4);
                CAMPP_QCONV_V5_ACCUMULATE(low5, w0, input5);
                CAMPP_QCONV_V5_ACCUMULATE(low6, w0, input6);
                CAMPP_QCONV_V5_ACCUMULATE(low7, w0, input7);
            }
            {
                CAMPP_QCONV_V5_LOAD_COLUMNS(
                    w1, weight1, weight_zero1, zero_point_fastpath);
                CAMPP_QCONV_V5_ACCUMULATE(high0, w1, input0);
                CAMPP_QCONV_V5_ACCUMULATE(high1, w1, input1);
                CAMPP_QCONV_V5_ACCUMULATE(high2, w1, input2);
                CAMPP_QCONV_V5_ACCUMULATE(high3, w1, input3);
                CAMPP_QCONV_V5_ACCUMULATE(high4, w1, input4);
                CAMPP_QCONV_V5_ACCUMULATE(high5, w1, input5);
                CAMPP_QCONV_V5_ACCUMULATE(high6, w1, input6);
                CAMPP_QCONV_V5_ACCUMULATE(high7, w1, input7);
            }
            weight0 += 16u;
            weight1 += 16u;
        }
    }
    campp_qconv_v5_store_accumulators(
        low0, high0, low1, high1, low2, high2, low3, high3,
        low4, high4, low5, high5, low6, high6, low7, high7,
        accumulators);
    return campp_qconv_v5_add_bias(bias, accumulators) == 0
        ? CAMPP_QCONV_MAC_4X8_OK : CAMPP_QCONV_MAC_4X8_FAILED;
}

CamppQconvMac4x8Result campp_qconv_v5_mac_3x3_sliding_raw(
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    int32_t input_zero, bool zero_point_fastpath,
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE])
{
    const uint32_t input_blocks = input_channels / 4u;
    const int16x4_t input_zero_v = vdup_n_s16((int16_t)input_zero);
    const int16x4_t weight_zero0 = vcreate_s16(
        (uint64_t)(uint16_t)weight_zero[0] |
        ((uint64_t)(uint16_t)weight_zero[1] << 16u) |
        ((uint64_t)(uint16_t)weight_zero[2] << 32u) |
        ((uint64_t)(uint16_t)weight_zero[3] << 48u));
    const int16x4_t weight_zero1 = vcreate_s16(
        (uint64_t)(uint16_t)weight_zero[4] |
        ((uint64_t)(uint16_t)weight_zero[5] << 16u) |
        ((uint64_t)(uint16_t)weight_zero[6] << 32u) |
        ((uint64_t)(uint16_t)weight_zero[7] << 48u));
    const ptrdiff_t column_stride =
        tile_plan->input_points[0][1] - tile_plan->input_points[0][0];
    int32x4_t low0 = vdupq_n_s32(0);
    int32x4_t high0 = vdupq_n_s32(0);
    int32x4_t low1 = vdupq_n_s32(0);
    int32x4_t high1 = vdupq_n_s32(0);
    int32x4_t low2 = vdupq_n_s32(0);
    int32x4_t high2 = vdupq_n_s32(0);
    int32x4_t low3 = vdupq_n_s32(0);
    int32x4_t high3 = vdupq_n_s32(0);
    int32x4_t low4 = vdupq_n_s32(0);
    int32x4_t high4 = vdupq_n_s32(0);
    int32x4_t low5 = vdupq_n_s32(0);
    int32x4_t high5 = vdupq_n_s32(0);
    int32x4_t low6 = vdupq_n_s32(0);
    int32x4_t high6 = vdupq_n_s32(0);
    int32x4_t low7 = vdupq_n_s32(0);
    int32x4_t high7 = vdupq_n_s32(0);
    uint32_t input_block;

    if (column_stride <= 0) return CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
    for (input_block = 0u; input_block < input_blocks; ++input_block) {
        const uint32_t channel = input_block * 4u;
        uint32_t kernel_row;
        for (kernel_row = 0u; kernel_row < 3u; ++kernel_row) {
            const uint8_t *const row_base =
                tile_plan->input_points[kernel_row * 3u][0] + channel;
            int16x4_t input[10];
            uint32_t column;
            for (column = 0u; column < 10u; ++column) {
                input[column] = campp_qconv_v5_load_input(
                    row_base + (ptrdiff_t)column * column_stride,
                    input_zero_v, zero_point_fastpath);
            }
            for (column = 0u; column < 3u; ++column) {
                const uint32_t kernel = kernel_row * 3u + column;
                const uint8_t *const weight0 = packed_weights[0] +
                    ((uint64_t)kernel * input_blocks + input_block) * 16u;
                const uint8_t *const weight1 = packed_weights[1] +
                    ((uint64_t)kernel * input_blocks + input_block) * 16u;
                {
                    CAMPP_QCONV_V5_LOAD_COLUMNS(
                        w0, weight0, weight_zero0, zero_point_fastpath);
                    CAMPP_QCONV_V5_ACCUMULATE(low0, w0, input[column + 0u]);
                    CAMPP_QCONV_V5_ACCUMULATE(low1, w0, input[column + 1u]);
                    CAMPP_QCONV_V5_ACCUMULATE(low2, w0, input[column + 2u]);
                    CAMPP_QCONV_V5_ACCUMULATE(low3, w0, input[column + 3u]);
                    CAMPP_QCONV_V5_ACCUMULATE(low4, w0, input[column + 4u]);
                    CAMPP_QCONV_V5_ACCUMULATE(low5, w0, input[column + 5u]);
                    CAMPP_QCONV_V5_ACCUMULATE(low6, w0, input[column + 6u]);
                    CAMPP_QCONV_V5_ACCUMULATE(low7, w0, input[column + 7u]);
                }
                {
                    CAMPP_QCONV_V5_LOAD_COLUMNS(
                        w1, weight1, weight_zero1, zero_point_fastpath);
                    CAMPP_QCONV_V5_ACCUMULATE(high0, w1, input[column + 0u]);
                    CAMPP_QCONV_V5_ACCUMULATE(high1, w1, input[column + 1u]);
                    CAMPP_QCONV_V5_ACCUMULATE(high2, w1, input[column + 2u]);
                    CAMPP_QCONV_V5_ACCUMULATE(high3, w1, input[column + 3u]);
                    CAMPP_QCONV_V5_ACCUMULATE(high4, w1, input[column + 4u]);
                    CAMPP_QCONV_V5_ACCUMULATE(high5, w1, input[column + 5u]);
                    CAMPP_QCONV_V5_ACCUMULATE(high6, w1, input[column + 6u]);
                    CAMPP_QCONV_V5_ACCUMULATE(high7, w1, input[column + 7u]);
                }
            }
        }
    }
    campp_qconv_v5_store_accumulators(
        low0, high0, low1, high1, low2, high2, low3, high3,
        low4, high4, low5, high5, low6, high6, low7, high7,
        accumulators);
    return campp_qconv_v5_add_bias(bias, accumulators) == 0
        ? CAMPP_QCONV_MAC_4X8_OK : CAMPP_QCONV_MAC_4X8_FAILED;
}

#undef CAMPP_QCONV_V5_ACCUMULATE
#undef CAMPP_QCONV_V5_LOAD_COLUMNS

#else

CamppQconvMac4x8Result campp_qconv_v5_mac_8x8_raw(
    const uint8_t *const input_points
        [CAMPP_QCONV_CANDIDATE_MAX_KERNEL_ELEMENTS]
        [CAMPP_QCONV_CANDIDATE_TILE],
    uint32_t kernel_elements,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    int32_t input_zero, bool zero_point_fastpath,
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE])
{
    (void)input_points;
    (void)kernel_elements;
    (void)packed_weights;
    (void)input_channels;
    (void)input_zero;
    (void)zero_point_fastpath;
    (void)weight_zero;
    (void)bias;
    (void)accumulators;
    return CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
}

CamppQconvMac4x8Result campp_qconv_v5_mac_3x3_sliding_raw(
    const CamppQconvV4TilePlan *tile_plan,
    const uint8_t *const packed_weights[2], uint32_t input_channels,
    int32_t input_zero, bool zero_point_fastpath,
    const int32_t weight_zero[CAMPP_QCONV_V5_OUTPUT_TILE],
    const int32_t bias[CAMPP_QCONV_V5_OUTPUT_TILE],
    int32_t accumulators[CAMPP_QCONV_V5_SPATIAL_TILE]
        [CAMPP_QCONV_V5_OUTPUT_TILE])
{
    (void)tile_plan;
    (void)packed_weights;
    (void)input_channels;
    (void)input_zero;
    (void)zero_point_fastpath;
    (void)weight_zero;
    (void)bias;
    (void)accumulators;
    return CAMPP_QCONV_MAC_4X8_UNSUPPORTED;
}

#endif
