#ifndef CAMPP_PROFILL_QCONV_ADDRESS_V2_PROVIDER_H
#define CAMPP_PROFILL_QCONV_ADDRESS_V2_PROVIDER_H

#include <stdint.h>

#include "../contracts/qconv_address_types.h"

typedef enum CamppQconvAddressStatus {
    CAMPP_QCONV_ADDRESS_OK = 0,
    CAMPP_QCONV_ADDRESS_INVALID_ARGUMENT = 1,
    CAMPP_QCONV_ADDRESS_UNSUPPORTED = 2,
    CAMPP_QCONV_ADDRESS_OUT_OF_STORAGE = 3,
    CAMPP_QCONV_ADDRESS_DONE = 4
} CamppQconvAddressStatus;

CamppQconvAddressStatus campp_qconv_address_v2_plan_create(
    const CamppQconvAddressGeometry *geometry,
    CamppQconvAddressPlan *out_plan);

void campp_qconv_address_v2_schedule_begin(
    CamppQconvAddressCursor *out_cursor);

CamppQconvAddressStatus campp_qconv_address_v2_schedule_next(
    const CamppQconvAddressPlan *plan, CamppQconvAddressCursor *cursor,
    CamppQconvAddressWorkItem *out_work);

CamppQconvAddressStatus campp_qconv_address_v2_tile_one_by_one(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_start, uint32_t tile_count,
    CamppQconvAddressTile *out_tile);

CamppQconvAddressStatus campp_qconv_address_v2_tile_three_by_three(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_row,
    uint32_t output_column, uint32_t tile_count,
    CamppQconvAddressTile *out_tile);

CamppQconvAddressStatus campp_qconv_address_v2_tile_generic_1d(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_position,
    uint32_t tile_count, CamppQconvAddressTile *out_tile);

CamppQconvAddressStatus campp_qconv_address_v2_tile_generic_2d(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_row,
    uint32_t output_column, uint32_t tile_count,
    CamppQconvAddressTile *out_tile);

#endif /* CAMPP_PROFILL_QCONV_ADDRESS_V2_PROVIDER_H */
