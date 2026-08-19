#include "qconv_v5_zero_point_fastpath.h"

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
