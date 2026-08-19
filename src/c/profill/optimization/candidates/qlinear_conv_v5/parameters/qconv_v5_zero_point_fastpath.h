#ifndef CAMPP_PROFILL_QCONV_V5_ZERO_POINT_FASTPATH_H
#define CAMPP_PROFILL_QCONV_V5_ZERO_POINT_FASTPATH_H

#include <stdbool.h>
#include <stdint.h>

bool campp_qconv_v5_weight_zero_all_zero(
    const int32_t *weight_zero, uint32_t valid_outputs);

int32_t campp_qconv_v5_correct_bias_for_input_zero(
    int32_t bias, int32_t input_zero, int32_t weight_sum);

#endif /* CAMPP_PROFILL_QCONV_V5_ZERO_POINT_FASTPATH_H */
