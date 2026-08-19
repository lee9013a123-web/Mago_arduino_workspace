#include "../api/qconv_address_provider.h"

#include <stddef.h>
#include <stdint.h>

static uint32_t campp_qconv_address_v2_minimum(
    uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}

void campp_qconv_address_v2_schedule_begin(
    CamppQconvAddressCursor *out_cursor)
{
    if (out_cursor == NULL) return;
    out_cursor->output_row = 0u;
    out_cursor->output_column = 0u;
}

static void campp_qconv_address_v2_schedule_advance(
    const CamppQconvAddressPlan *plan, CamppQconvAddressCursor *cursor,
    uint32_t tile_count)
{
    cursor->output_column += tile_count;
    if (cursor->output_column ==
        plan->geometry.output_spatial[plan->geometry.spatial_rank - 1u]) {
        cursor->output_column = 0u;
        cursor->output_row += 1u;
    }
}

CamppQconvAddressStatus campp_qconv_address_v2_schedule_next(
    const CamppQconvAddressPlan *plan, CamppQconvAddressCursor *cursor,
    CamppQconvAddressWorkItem *out_work)
{
    uint32_t output_width;
    uint32_t remaining;
    uint32_t tile_count;
    CamppQconvAddressPath path = CAMPP_QCONV_ADDRESS_PATH_GENERIC;

    if (plan == NULL || cursor == NULL || out_work == NULL ||
        plan->geometry.spatial_rank < 1u ||
        plan->geometry.spatial_rank > 2u) {
        return CAMPP_QCONV_ADDRESS_INVALID_ARGUMENT;
    }
    output_width = plan->geometry.output_spatial[
        plan->geometry.spatial_rank - 1u];
    if (cursor->output_column >= output_width) {
        return CAMPP_QCONV_ADDRESS_INVALID_ARGUMENT;
    }
    if ((plan->geometry.spatial_rank == 1u &&
         cursor->output_row != 0u) ||
        (plan->geometry.spatial_rank == 2u &&
         cursor->output_row >= plan->geometry.output_spatial[0])) {
        return CAMPP_QCONV_ADDRESS_DONE;
    }

    remaining = output_width - cursor->output_column;
    tile_count = campp_qconv_address_v2_minimum(
        remaining, CAMPP_QCONV_ADDRESS_TILE_MAX);
    if (plan->preferred_path == CAMPP_QCONV_ADDRESS_PATH_ONE_BY_ONE) {
        if (cursor->output_column < plan->interior_begin[0]) {
            tile_count = campp_qconv_address_v2_minimum(
                tile_count,
                plan->interior_begin[0] - cursor->output_column);
        } else if (cursor->output_column < plan->interior_end[0]) {
            tile_count = campp_qconv_address_v2_minimum(
                tile_count, plan->interior_end[0] - cursor->output_column);
            path = CAMPP_QCONV_ADDRESS_PATH_ONE_BY_ONE;
        }
    } else if (plan->preferred_path ==
               CAMPP_QCONV_ADDRESS_PATH_THREE_BY_THREE_INTERIOR &&
               cursor->output_row >= plan->interior_begin[0] &&
               cursor->output_row < plan->interior_end[0]) {
        if (cursor->output_column < plan->interior_begin[1]) {
            tile_count = campp_qconv_address_v2_minimum(
                tile_count,
                plan->interior_begin[1] - cursor->output_column);
        } else if (cursor->output_column + CAMPP_QCONV_ADDRESS_TILE_MAX <=
                   plan->interior_end[1]) {
            tile_count = CAMPP_QCONV_ADDRESS_TILE_MAX;
            path = CAMPP_QCONV_ADDRESS_PATH_THREE_BY_THREE_INTERIOR;
        }
    }
    if (tile_count == 0u) return CAMPP_QCONV_ADDRESS_INVALID_ARGUMENT;

    out_work->output_row = cursor->output_row;
    out_work->output_column = cursor->output_column;
    out_work->tile_count = (uint8_t)tile_count;
    out_work->path = path;
    campp_qconv_address_v2_schedule_advance(plan, cursor, tile_count);
    return CAMPP_QCONV_ADDRESS_OK;
}
