#include "qconv_mac_4x8.h"

#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

static inline int16x4_t campp_qconv_mac_4x8_load_input(
    const uint8_t *data, int16x4_t input_zero)
{
    uint32_t packed;
    uint8x8_t bytes;
    memcpy(&packed, data, sizeof(packed));
    bytes = vreinterpret_u8_u64(vcreate_u64(packed));
    return vsub_s16(
        vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(bytes))),
        input_zero);
}

#define CAMPP_QCONV_LOAD_COLUMNS(prefix, address, zero_points)         \
    const int8x16_t prefix##_packed =                                  \
        vld1q_s8((const int8_t *)(address));                           \
    const int16x8_t prefix##_low =                                     \
        vmovl_s8(vget_low_s8(prefix##_packed));                        \
    const int16x8_t prefix##_high =                                    \
        vmovl_s8(vget_high_s8(prefix##_packed));                       \
    const int16x4_t prefix##_row0 = vget_low_s16(prefix##_low);        \
    const int16x4_t prefix##_row1 = vget_high_s16(prefix##_low);       \
    const int16x4_t prefix##_row2 = vget_low_s16(prefix##_high);       \
    const int16x4_t prefix##_row3 = vget_high_s16(prefix##_high);      \
    const int16x4x2_t prefix##_transpose01 =                           \
        vtrn_s16(prefix##_row0, prefix##_row1);                        \
    const int16x4x2_t prefix##_transpose23 =                           \
        vtrn_s16(prefix##_row2, prefix##_row3);                        \
    const int32x2x2_t prefix##_columns02 = vtrn_s32(                   \
        vreinterpret_s32_s16(prefix##_transpose01.val[0]),             \
        vreinterpret_s32_s16(prefix##_transpose23.val[0]));            \
    const int32x2x2_t prefix##_columns13 = vtrn_s32(                   \
        vreinterpret_s32_s16(prefix##_transpose01.val[1]),             \
        vreinterpret_s32_s16(prefix##_transpose23.val[1]));            \
    int16x4_t prefix##0 = vsub_s16(                                    \
        vreinterpret_s16_s32(prefix##_columns02.val[0]),               \
        (zero_points));                                                \
    int16x4_t prefix##1 = vsub_s16(                                    \
        vreinterpret_s16_s32(prefix##_columns13.val[0]),               \
        (zero_points));                                                \
    int16x4_t prefix##2 = vsub_s16(                                    \
        vreinterpret_s16_s32(prefix##_columns02.val[1]),               \
        (zero_points));                                                \
    int16x4_t prefix##3 = vsub_s16(                                    \
        vreinterpret_s16_s32(prefix##_columns13.val[1]),               \
        (zero_points))

#define CAMPP_QCONV_ACCUMULATE(accumulator, columns, input)            \
    do {                                                               \
        (accumulator) = vmlal_lane_s16(                                \
            (accumulator), columns##0, (input), 0);                    \
        (accumulator) = vmlal_lane_s16(                                \
            (accumulator), columns##1, (input), 1);                    \
        (accumulator) = vmlal_lane_s16(                                \
            (accumulator), columns##2, (input), 2);                    \
        (accumulator) = vmlal_lane_s16(                                \
            (accumulator), columns##3, (input), 3);                    \
    } while (0)

int campp_qconv_mac_4x8_intrinsics_raw(
    const CamppQconvMac4x8Params *params)
{
    const uint32_t input_blocks = params->input_channels / 4u;
    const int16x4_t input_zero =
        vdup_n_s16((int16_t)params->input_zero);
    const int16x4_t weight_zero0 = vcreate_s16(
        (uint64_t)(uint16_t)params->weight_zero[0] |
        ((uint64_t)(uint16_t)params->weight_zero[1] << 16u) |
        ((uint64_t)(uint16_t)params->weight_zero[2] << 32u) |
        ((uint64_t)(uint16_t)params->weight_zero[3] << 48u));
    const int16x4_t weight_zero1 = vcreate_s16(
        (uint64_t)(uint16_t)params->weight_zero[4] |
        ((uint64_t)(uint16_t)params->weight_zero[5] << 16u) |
        ((uint64_t)(uint16_t)params->weight_zero[6] << 32u) |
        ((uint64_t)(uint16_t)params->weight_zero[7] << 48u));
    const uint8_t *weight0 = params->packed_weight0;
    const uint8_t *weight1 = params->packed_weight1;
    int32x4_t accumulator00 = vdupq_n_s32(0);
    int32x4_t accumulator01 = vdupq_n_s32(0);
    int32x4_t accumulator02 = vdupq_n_s32(0);
    int32x4_t accumulator03 = vdupq_n_s32(0);
    int32x4_t accumulator10 = vdupq_n_s32(0);
    int32x4_t accumulator11 = vdupq_n_s32(0);
    int32x4_t accumulator12 = vdupq_n_s32(0);
    int32x4_t accumulator13 = vdupq_n_s32(0);
    uint32_t kernel;

    for (kernel = 0u; kernel < params->kernel_elements; ++kernel) {
        const uint8_t *const *points = params->input_points
            + (size_t)kernel * CAMPP_QCONV_CANDIDATE_TILE;
        uint32_t input_block;
        for (input_block = 0u; input_block < input_blocks; ++input_block) {
            const uint32_t channel = input_block * 4u;
            const int16x4_t input0 = campp_qconv_mac_4x8_load_input(
                points[0] + channel, input_zero);
            const int16x4_t input1 = campp_qconv_mac_4x8_load_input(
                points[1] + channel, input_zero);
            const int16x4_t input2 = campp_qconv_mac_4x8_load_input(
                points[2] + channel, input_zero);
            const int16x4_t input3 = campp_qconv_mac_4x8_load_input(
                points[3] + channel, input_zero);
            {
                CAMPP_QCONV_LOAD_COLUMNS(
                    weight_columns0, weight0, weight_zero0);
                CAMPP_QCONV_ACCUMULATE(
                    accumulator00, weight_columns0, input0);
                CAMPP_QCONV_ACCUMULATE(
                    accumulator01, weight_columns0, input1);
                CAMPP_QCONV_ACCUMULATE(
                    accumulator02, weight_columns0, input2);
                CAMPP_QCONV_ACCUMULATE(
                    accumulator03, weight_columns0, input3);
            }
            {
                CAMPP_QCONV_LOAD_COLUMNS(
                    weight_columns1, weight1, weight_zero1);
                CAMPP_QCONV_ACCUMULATE(
                    accumulator10, weight_columns1, input0);
                CAMPP_QCONV_ACCUMULATE(
                    accumulator11, weight_columns1, input1);
                CAMPP_QCONV_ACCUMULATE(
                    accumulator12, weight_columns1, input2);
                CAMPP_QCONV_ACCUMULATE(
                    accumulator13, weight_columns1, input3);
            }
            weight0 += 16u;
            weight1 += 16u;
        }
    }
    vst1q_s32(params->output, accumulator00);
    vst1q_s32(params->output + 4u, accumulator10);
    vst1q_s32(params->output + 8u, accumulator01);
    vst1q_s32(params->output + 12u, accumulator11);
    vst1q_s32(params->output + 16u, accumulator02);
    vst1q_s32(params->output + 20u, accumulator12);
    vst1q_s32(params->output + 24u, accumulator03);
    vst1q_s32(params->output + 28u, accumulator13);
    return 0;
}

#undef CAMPP_QCONV_ACCUMULATE
#undef CAMPP_QCONV_LOAD_COLUMNS

#else

int campp_qconv_mac_4x8_intrinsics_raw(
    const CamppQconvMac4x8Params *params)
{
    (void)params;
    return 1;
}

#endif
