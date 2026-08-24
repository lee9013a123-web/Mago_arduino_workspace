#ifndef CAMPP_PROFILL_QCONV_V5_ADDRESS_H
#define CAMPP_PROFILL_QCONV_V5_ADDRESS_H

#include <stdint.h>

#include "qconv_v5_address_contract.h"

typedef enum CamppQconvV5AddressStatus {
    CAMPP_QCONV_V5_ADDRESS_OK = 0,
    CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT = 1,
    CAMPP_QCONV_V5_ADDRESS_UNSUPPORTED = 2,
    CAMPP_QCONV_V5_ADDRESS_OUT_OF_STORAGE = 3,
    CAMPP_QCONV_V5_ADDRESS_DONE = 4
} CamppQconvV5AddressStatus;

CamppQconvV5AddressStatus campp_qconv_v5_address_plan_create(
    const CamppQconvV5AddressGeometry *geometry,
    CamppQconvV5AddressPlan *out_plan);

void campp_qconv_v5_address_schedule_begin(
    CamppQconvV5AddressCursor *out_cursor);

CamppQconvV5AddressStatus campp_qconv_v5_address_schedule_next(
    const CamppQconvV5AddressPlan *plan,
    CamppQconvV5AddressCursor *cursor,
    CamppQconvV5AddressWorkItem *out_work);

CamppQconvV5AddressStatus campp_qconv_v5_address_tile_1x1(
    const CamppQconvV5AddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_position,
    uint32_t tile_count, CamppQconvV5AddressTile *out_tile);

CamppQconvV5AddressStatus campp_qconv_v5_address_tile_3x3(
    const CamppQconvV5AddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_row,
    uint32_t output_column, uint32_t tile_count,
    CamppQconvV5AddressTile *out_tile);

CamppQconvV5AddressStatus campp_qconv_v5_address_tile_generic_1d(
    const CamppQconvV5AddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_position,
    uint32_t tile_count, CamppQconvV5AddressTile *out_tile);

CamppQconvV5AddressStatus campp_qconv_v5_address_tile_generic_2d(
    const CamppQconvV5AddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_row,
    uint32_t output_column, uint32_t tile_count,
    CamppQconvV5AddressTile *out_tile);

#endif /* CAMPP_PROFILL_QCONV_V5_ADDRESS_H */
