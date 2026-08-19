#ifndef CAMPP_PROFILL_QCONV_ADDRESS_V2_PROVIDER_H
#define CAMPP_PROFILL_QCONV_ADDRESS_V2_PROVIDER_H

#include <stdint.h>

#include "../contracts/qconv_address_types.h"

typedef enum CamppQconvAddressStatus {
    CAMPP_QCONV_ADDRESS_OK = 0,
    CAMPP_QCONV_ADDRESS_INVALID_ARGUMENT = 1,
    CAMPP_QCONV_ADDRESS_UNSUPPORTED = 2,
    CAMPP_QCONV_ADDRESS_OUT_OF_STORAGE = 3
} CamppQconvAddressStatus;

CamppQconvAddressStatus campp_qconv_address_v2_plan_create(
    const CamppQconvAddressGeometry *geometry,
    CamppQconvAddressPlan *out_plan);

CamppQconvAddressStatus campp_qconv_address_v2_tile_one_by_one(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_start, uint32_t tile_count,
    CamppQconvAddressTile *out_tile);

CamppQconvAddressStatus campp_qconv_address_v2_tile_three_by_three(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_start, uint32_t tile_count,
    CamppQconvAddressTile *out_tile);

CamppQconvAddressStatus campp_qconv_address_v2_tile_generic(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_start, uint32_t tile_count,
    CamppQconvAddressTile *out_tile);

#endif /* CAMPP_PROFILL_QCONV_ADDRESS_V2_PROVIDER_H */
