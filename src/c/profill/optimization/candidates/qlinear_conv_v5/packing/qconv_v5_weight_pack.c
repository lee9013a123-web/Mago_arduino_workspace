#include "qconv_v5_weight_pack.h"

CamppQconvV5WeightPackView campp_qconv_v5_weight_pack_view(
    const void *data, uint64_t byte_size, bool symmetric_zero_point)
{
    CamppQconvV5WeightPackView view;
    view.data = (const uint8_t *)data;
    view.byte_size = byte_size;
    view.symmetric_zero_point = symmetric_zero_point;
    return view;
}
