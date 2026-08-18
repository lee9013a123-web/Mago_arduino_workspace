#ifndef CAMPP_PROFILL_SIGMOID_LUT_H
#define CAMPP_PROFILL_SIGMOID_LUT_H

#include <stdint.h>

void campp_sigmoid_lut_build(
    uint8_t input_dtype, float scale, int32_t zero_point, float table[256]);

#endif /* CAMPP_PROFILL_SIGMOID_LUT_H */
