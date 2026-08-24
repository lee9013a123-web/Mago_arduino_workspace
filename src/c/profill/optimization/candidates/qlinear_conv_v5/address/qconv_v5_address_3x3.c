#include "qconv_v5_address.h"

#include <stdint.h>

#include "qconv_v5_address_internal.h"

CamppQconvV5AddressStatus campp_qconv_v5_address_tile_3x3(
    const CamppQconvV5AddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_row,
    uint32_t output_column, uint32_t tile_count,
    CamppQconvV5AddressTile *out_tile)
{
    int64_t first_input_row;
    int64_t first_input_column;
    int64_t last_input_row;
    int64_t last_input_column;
    uint64_t first_offset;
    uint64_t last_offset;
    uint64_t kernel_row_offset;
    uint32_t kernel_row;
    uint32_t tile;

    if (plan == NULL || out_tile == NULL ||
        plan->preferred_path !=
            CAMPP_QCONV_V5_ADDRESS_PATH_THREE_BY_THREE_INTERIOR ||
        plan->geometry.spatial_rank != 2u || tile_count == 0u ||
        tile_count > CAMPP_QCONV_V5_ADDRESS_TILE_MAX ||
        output_row >= plan->geometry.output_spatial[0] ||
        output_column >= plan->geometry.output_spatial[1] ||
        tile_count > plan->geometry.output_spatial[1] - output_column) {
        return CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT;
    }
    if (output_row < plan->interior_begin[0] ||
        output_row >= plan->interior_end[0] ||
        output_column < plan->interior_begin[1] ||
        tile_count > plan->interior_end[1] - output_column) {
        return CAMPP_QCONV_V5_ADDRESS_UNSUPPORTED;
    }
    if (!campp_qconv_v5_address_coordinate(
            output_row, plan->geometry.convolution_stride[0], 0u,
            plan->geometry.dilation[0], plan->geometry.pad_begin[0],
            &first_input_row) ||
        !campp_qconv_v5_address_coordinate(
            output_column, plan->geometry.convolution_stride[1], 0u,
            plan->geometry.dilation[1], plan->geometry.pad_begin[1],
            &first_input_column) ||
        !campp_qconv_v5_address_coordinate(
            output_row, plan->geometry.convolution_stride[0], 2u,
            plan->geometry.dilation[0], plan->geometry.pad_begin[0],
            &last_input_row) ||
        !campp_qconv_v5_address_coordinate(
            output_column + tile_count - 1u,
            plan->geometry.convolution_stride[1], 2u,
            plan->geometry.dilation[1], plan->geometry.pad_begin[1],
            &last_input_column) ||
        first_input_row < 0 || first_input_column < 0 ||
        last_input_row < 0 || last_input_column < 0 ||
        last_input_row >= (int64_t)plan->geometry.input_spatial[0] ||
        last_input_column >= (int64_t)plan->geometry.input_spatial[1]) {
        return CAMPP_QCONV_V5_ADDRESS_UNSUPPORTED;
    }
    if (!campp_qconv_v5_address_group_base(
            plan, batch, group_channel, &first_offset) ||
        !campp_qconv_v5_address_add_spatial_offset(
            plan, 0u, (uint32_t)first_input_row, &first_offset) ||
        !campp_qconv_v5_address_add_spatial_offset(
            plan, 1u, (uint32_t)first_input_column, &first_offset) ||
        !campp_qconv_v5_address_group_base(
            plan, batch, group_channel, &last_offset) ||
        !campp_qconv_v5_address_add_spatial_offset(
            plan, 0u, (uint32_t)last_input_row, &last_offset) ||
        !campp_qconv_v5_address_add_spatial_offset(
            plan, 1u, (uint32_t)last_input_column, &last_offset)) {
        return CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT;
    }
    if (!campp_qconv_v5_address_point_in_storage(plan, first_offset) ||
        !campp_qconv_v5_address_point_in_storage(plan, last_offset)) {
        return CAMPP_QCONV_V5_ADDRESS_OUT_OF_STORAGE;
    }

    out_tile->tile_count = (uint8_t)tile_count;
    out_tile->kernel_elements = 9u;
    out_tile->path =
        CAMPP_QCONV_V5_ADDRESS_PATH_THREE_BY_THREE_INTERIOR;
    for (tile = 0u; tile < tile_count; ++tile) {
        out_tile->output_linear[tile] =
            output_row * plan->geometry.output_spatial[1]
            + output_column + tile;
    }
    kernel_row_offset = first_offset;
    for (kernel_row = 0u; kernel_row < 3u; ++kernel_row) {
        uint64_t kernel_offset = kernel_row_offset;
        uint32_t kernel_column;
        for (kernel_column = 0u; kernel_column < 3u; ++kernel_column) {
            const uint32_t kernel = kernel_row * 3u + kernel_column;
            uint64_t point_offset = kernel_offset;
            for (tile = 0u; tile < tile_count; ++tile) {
                out_tile->input_points[kernel][tile] =
                    campp_qconv_v5_address_pointer(plan, point_offset);
                if (tile + 1u < tile_count) {
                    point_offset += plan->output_step_bytes[1];
                }
            }
            if (kernel_column + 1u < 3u) {
                kernel_offset += plan->kernel_step_bytes[1];
            }
        }
        if (kernel_row + 1u < 3u) {
            kernel_row_offset += plan->kernel_step_bytes[0];
        }
    }
    return CAMPP_QCONV_V5_ADDRESS_OK;
}
