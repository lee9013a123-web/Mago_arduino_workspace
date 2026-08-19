#ifndef CAMPP_PROFILL_QCONV_V5_WEIGHT_PACK_H
#define CAMPP_PROFILL_QCONV_V5_WEIGHT_PACK_H

#include <stdbool.h>
#include <stdint.h>

typedef struct CamppQconvV5WeightPackView {
    const uint8_t *data;
    uint64_t byte_size;
    bool symmetric_zero_point;
} CamppQconvV5WeightPackView;

CamppQconvV5WeightPackView campp_qconv_v5_weight_pack_view(
    const void *data, uint64_t byte_size, bool symmetric_zero_point);

#endif /* CAMPP_PROFILL_QCONV_V5_WEIGHT_PACK_H */
