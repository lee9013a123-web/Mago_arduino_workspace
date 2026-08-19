#include "../../api/qconv_address_provider.h"

#include <stdint.h>

#include "../qconv_address_internal.h"

CamppQconvAddressStatus campp_qconv_address_v2_tile_one_by_one(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_start, uint32_t tile_count,
    CamppQconvAddressTile *out_tile)
{
    int64_t first_input;
    int64_t last_input;
    uint64_t first_offset;
    uint64_t last_offset;
    uint64_t point_offset;
    uint32_t tile;

    if (plan == NULL || out_tile == NULL ||
        plan->preferred_path != CAMPP_QCONV_ADDRESS_PATH_ONE_BY_ONE ||
        plan->geometry.spatial_rank != 1u || tile_count == 0u ||
        tile_count > CAMPP_QCONV_ADDRESS_TILE_MAX ||
        tile_start >= plan->geometry.output_spatial[0] ||
        tile_count > plan->geometry.output_spatial[0] - tile_start) {
        return CAMPP_QCONV_ADDRESS_INVALID_ARGUMENT;
    }
    if (!campp_qconv_address_v2_coordinate(
            tile_start, plan->geometry.convolution_stride[0], 0u,
            plan->geometry.dilation[0], plan->geometry.pad_begin[0],
            &first_input) ||
        !campp_qconv_address_v2_coordinate(
            tile_start + tile_count - 1u,
            plan->geometry.convolution_stride[0], 0u,
            plan->geometry.dilation[0], plan->geometry.pad_begin[0],
            &last_input)) {
        return CAMPP_QCONV_ADDRESS_INVALID_ARGUMENT;
    }
    if (first_input < 0 || last_input < 0 ||
        last_input >= (int64_t)plan->geometry.input_spatial[0]) {
        return CAMPP_QCONV_ADDRESS_UNSUPPORTED;
    }
    if (!campp_qconv_address_v2_group_base(
            plan, batch, group_channel, &first_offset) ||
        !campp_qconv_address_v2_add_spatial_offset(
            plan, 0u, (uint32_t)first_input, &first_offset) ||
        !campp_qconv_address_v2_group_base(
            plan, batch, group_channel, &last_offset) ||
        !campp_qconv_address_v2_add_spatial_offset(
            plan, 0u, (uint32_t)last_input, &last_offset)) {
        return CAMPP_QCONV_ADDRESS_INVALID_ARGUMENT;
    }
    if (!campp_qconv_address_v2_point_in_storage(plan, first_offset) ||
        !campp_qconv_address_v2_point_in_storage(plan, last_offset)) {
        return CAMPP_QCONV_ADDRESS_OUT_OF_STORAGE;
    }

    out_tile->tile_count = (uint8_t)tile_count;
    out_tile->kernel_elements = 1u;
    out_tile->path = CAMPP_QCONV_ADDRESS_PATH_ONE_BY_ONE;
    point_offset = first_offset;
    for (tile = 0u; tile < tile_count; ++tile) {
        out_tile->output_linear[tile] = tile_start + tile;
        out_tile->input_points[0][tile] =
            campp_qconv_address_v2_pointer(plan, point_offset);
        if (tile + 1u < tile_count) {
            point_offset += plan->output_step_bytes[0];
        }
    }
    return CAMPP_QCONV_ADDRESS_OK;
}
