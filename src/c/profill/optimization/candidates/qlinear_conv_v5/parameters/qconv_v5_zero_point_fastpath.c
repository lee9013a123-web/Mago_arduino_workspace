#include "qconv_v5_zero_point_fastpath.h"

#include <limits.h>
#include <stddef.h>

bool campp_qconv_v5_weight_zero_all_zero(
    const int32_t *weight_zero, uint32_t valid_outputs)
{
    uint32_t lane;
    if (weight_zero == NULL || valid_outputs > 8u) return false;
    for (lane = 0u; lane < valid_outputs; ++lane) {
        if (weight_zero[lane] != 0) return false;
    }
    return true;
}

int32_t campp_qconv_v5_correct_bias_for_input_zero(
    int32_t bias, int32_t input_zero, int32_t weight_sum)
{
    return bias - input_zero * weight_sum;
}

bool campp_qconv_v5_weight_sums_o4i4(
    const uint8_t *const packed_weights[2], uint32_t kernel_elements,
    uint32_t input_channels,
    int32_t weight_sums[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    const uint32_t input_blocks = input_channels / 4u;
    uint32_t lane;
    uint32_t kernel;

    if (packed_weights == NULL || packed_weights[0] == NULL ||
        packed_weights[1] == NULL || weight_sums == NULL ||
        kernel_elements == 0u || input_channels == 0u ||
        input_channels % 4u != 0u) {
        return false;
    }
    for (lane = 0u; lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE; ++lane) {
        weight_sums[lane] = 0;
    }
    for (kernel = 0u; kernel < kernel_elements; ++kernel) {
        uint32_t input_block;
        for (input_block = 0u; input_block < input_blocks; ++input_block) {
            const int8_t *const weight0 = (const int8_t *)packed_weights[0] +
                ((uint64_t)kernel * input_blocks + input_block) * 16u;
            const int8_t *const weight1 = (const int8_t *)packed_weights[1] +
                ((uint64_t)kernel * input_blocks + input_block) * 16u;
            uint32_t output_lane;
            for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                uint32_t input_lane;
                for (input_lane = 0u; input_lane < 4u; ++input_lane) {
                    weight_sums[output_lane] +=
                        weight0[output_lane * 4u + input_lane];
                    weight_sums[output_lane + 4u] +=
                        weight1[output_lane * 4u + input_lane];
                }
            }
        }
    }
    return true;
}

bool campp_qconv_v5_correct_bias_block(
    const int32_t bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    int32_t input_zero,
    const int32_t weight_sums[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    int32_t corrected_bias[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE])
{
    uint32_t lane;
    if (bias == NULL || weight_sums == NULL || corrected_bias == NULL) {
        return false;
    }
    for (lane = 0u; lane < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE; ++lane) {
        const int64_t value =
            (int64_t)bias[lane] - (int64_t)input_zero * weight_sums[lane];
        if (value < INT32_MIN || value > INT32_MAX) return false;
        corrected_bias[lane] = (int32_t)value;
    }
    return true;
}
