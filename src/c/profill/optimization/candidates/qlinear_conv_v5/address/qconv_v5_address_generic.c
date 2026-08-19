#include "qconv_v5_address.h"

#include <stdint.h>

#include "qconv_v5_address_internal.h"

static CamppQconvV5AddressStatus campp_qconv_v5_address_generic_point(
    const CamppQconvV5AddressPlan *plan, uint32_t batch,
    uint32_t group_channel, const uint32_t output_coordinates[2],
    const uint32_t kernel_coordinates[2], const uint8_t **out_pointer)
{
    uint64_t offset;
    uint8_t axis;

    if (!campp_qconv_v5_address_group_base(
            plan, batch, group_channel, &offset)) {
        return CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT;
    }
    for (axis = 0u; axis < plan->geometry.spatial_rank; ++axis) {
        int64_t coordinate;
        if (!campp_qconv_v5_address_coordinate(
                output_coordinates[axis],
                plan->geometry.convolution_stride[axis],
                kernel_coordinates[axis], plan->geometry.dilation[axis],
                plan->geometry.pad_begin[axis], &coordinate)) {
            return CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT;
        }
        if (coordinate < 0 ||
            coordinate >= (int64_t)plan->geometry.input_spatial[axis]) {
            *out_pointer = NULL;
            return CAMPP_QCONV_V5_ADDRESS_OK;
        }
        if (!campp_qconv_v5_address_add_spatial_offset(
                plan, axis, (uint32_t)coordinate, &offset)) {
            return CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT;
        }
    }
    if (!campp_qconv_v5_address_point_in_storage(plan, offset)) {
        return CAMPP_QCONV_V5_ADDRESS_OUT_OF_STORAGE;
    }
    *out_pointer = campp_qconv_v5_address_pointer(plan, offset);
    return CAMPP_QCONV_V5_ADDRESS_OK;
}

CamppQconvV5AddressStatus campp_qconv_v5_address_tile_generic_1d(
    const CamppQconvV5AddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_position,
    uint32_t tile_count, CamppQconvV5AddressTile *out_tile)
{
    uint32_t kernel;

    if (plan == NULL || out_tile == NULL ||
        plan->geometry.spatial_rank != 1u || tile_count == 0u ||
        tile_count > CAMPP_QCONV_V5_ADDRESS_TILE_MAX ||
        output_position >= plan->geometry.output_spatial[0] ||
        tile_count > plan->geometry.output_spatial[0] - output_position) {
        return CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT;
    }
    out_tile->tile_count = (uint8_t)tile_count;
    out_tile->kernel_elements = plan->kernel_elements;
    out_tile->path = CAMPP_QCONV_V5_ADDRESS_PATH_GENERIC;
    for (kernel = 0u; kernel < plan->kernel_elements; ++kernel) {
        const uint32_t kernel_coordinates[2] = {kernel, 0u};
        uint32_t tile;
        for (tile = 0u; tile < tile_count; ++tile) {
            const uint32_t output_coordinates[2] = {
                output_position + tile, 0u
            };
            CamppQconvV5AddressStatus status;
            out_tile->output_linear[tile] = output_position + tile;
            status = campp_qconv_v5_address_generic_point(
                plan, batch, group_channel, output_coordinates,
                kernel_coordinates, &out_tile->input_points[kernel][tile]);
            if (status != CAMPP_QCONV_V5_ADDRESS_OK) return status;
        }
    }
    return CAMPP_QCONV_V5_ADDRESS_OK;
}

CamppQconvV5AddressStatus campp_qconv_v5_address_tile_generic_2d(
    const CamppQconvV5AddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t output_row,
    uint32_t output_column, uint32_t tile_count,
    CamppQconvV5AddressTile *out_tile)
{
    uint32_t kernel;

    if (plan == NULL || out_tile == NULL ||
        plan->geometry.spatial_rank != 2u || tile_count == 0u ||
        tile_count > CAMPP_QCONV_V5_ADDRESS_TILE_MAX ||
        output_row >= plan->geometry.output_spatial[0] ||
        output_column >= plan->geometry.output_spatial[1] ||
        tile_count > plan->geometry.output_spatial[1] - output_column) {
        return CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT;
    }
    out_tile->tile_count = (uint8_t)tile_count;
    out_tile->kernel_elements = plan->kernel_elements;
    out_tile->path = CAMPP_QCONV_V5_ADDRESS_PATH_GENERIC;
    for (kernel = 0u; kernel < plan->kernel_elements; ++kernel) {
        const uint32_t kernel_coordinates[2] = {
            kernel / plan->geometry.kernel_shape[1],
            kernel % plan->geometry.kernel_shape[1]
        };
        uint32_t tile;
        for (tile = 0u; tile < tile_count; ++tile) {
            const uint32_t output_coordinates[2] = {
                output_row, output_column + tile
            };
            CamppQconvV5AddressStatus status;
            out_tile->output_linear[tile] =
                output_row * plan->geometry.output_spatial[1]
                + output_column + tile;
            status = campp_qconv_v5_address_generic_point(
                plan, batch, group_channel, output_coordinates,
                kernel_coordinates, &out_tile->input_points[kernel][tile]);
            if (status != CAMPP_QCONV_V5_ADDRESS_OK) return status;
        }
    }
    return CAMPP_QCONV_V5_ADDRESS_OK;
}
