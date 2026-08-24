#include "qconv_v5_stage_tags.h"

const char *campp_qconv_v5_stage_tag_name(CamppQconvV5StageTag tag)
{
    static const char *const names[] = {
        "validated_raw",
        "zero_point_fastpath",
        "weight_pack",
        "real_8x8_mac",
        "3x3_unroll",
        "3x3_sliding",
        "address_plan",
        "address_1x1",
        "address_3x3",
        "address_generic"
    };
    return tag >= CAMPP_QCONV_V5_STAGE_VALIDATED_RAW &&
        tag <= CAMPP_QCONV_V5_STAGE_ADDRESS_GENERIC
        ? names[tag] : "invalid";
}
