#include "qconv_v4_tile_plan.h"

#include <stddef.h>
#include <string.h>

static bool campp_qconv_v4_point_in_storage(
    const CamppQconvV4ExecutionPlan *plan, uint64_t offset)
{
    return offset <= plan->input->storage_span_bytes &&
        plan->inputs_per_group <= plan->input->storage_span_bytes - offset;
}

static bool campp_qconv_v4_plan_1x1(
    const CamppQconvV4ExecutionPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_start, uint32_t tile_count,
    CamppQconvV4TilePlan *tile_plan)
{
    uint32_t tile;

    if (plan->preferred_path != CAMPP_QCONV_V4_PATH_1X1 ||
        plan->spatial_rank != 1u || tile_count != CAMPP_QCONV_CANDIDATE_TILE) {
        return false;
    }
    for (tile = 0u; tile < tile_count; ++tile) {
        const int64_t input_position =
            (int64_t)(tile_start + tile) * plan->strides[0] - plan->pads[0];
        uint64_t offset;
        if (input_position < 0 ||
            input_position >= (int64_t)plan->input->dimensions[2]) {
            return false;
        }
        offset = (uint64_t)batch * plan->input->byte_strides[0]
            + group_channel
            + (uint64_t)input_position * plan->input->byte_strides[2];
        if (!campp_qconv_v4_point_in_storage(plan, offset)) return false;
        tile_plan->input_points[0][tile] =
            (const uint8_t *)plan->input->data + offset;
    }
    tile_plan->path = CAMPP_QCONV_V4_PATH_1X1;
    return true;
}

static bool campp_qconv_v4_plan_3x3_interior(
    const CamppQconvV4ExecutionPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_start, uint32_t tile_count,
    CamppQconvV4TilePlan *tile_plan)
{
    const uint32_t output_width = plan->output->dimensions[3];
    uint32_t output_row;
    uint32_t output_column;
    int64_t first_input_row;
    int64_t first_input_column;
    int64_t last_input_column;
    uint32_t kernel_row;

    if (plan->preferred_path != CAMPP_QCONV_V4_PATH_3X3_INTERIOR ||
        plan->spatial_rank != 2u || tile_count != CAMPP_QCONV_CANDIDATE_TILE ||
        output_width == 0u) {
        return false;
    }
    output_row = tile_start / output_width;
    output_column = tile_start % output_width;
    first_input_row =
        (int64_t)output_row * plan->strides[0] - plan->pads[0];
    first_input_column =
        (int64_t)output_column * plan->strides[1] - plan->pads[1];
    last_input_column =
        (int64_t)(output_column + tile_count - 1u) * plan->strides[1]
        - plan->pads[1] + 2;
    if (output_column + tile_count > output_width ||
        first_input_row < 0 || first_input_row + 2 >=
            (int64_t)plan->input->dimensions[2] ||
        first_input_column < 0 || last_input_column >=
            (int64_t)plan->input->dimensions[3]) {
        return false;
    }

    for (kernel_row = 0u; kernel_row < 3u; ++kernel_row) {
        uint32_t kernel_column;
        for (kernel_column = 0u; kernel_column < 3u; ++kernel_column) {
            const uint32_t kernel = kernel_row * 3u + kernel_column;
            uint32_t tile;
            for (tile = 0u; tile < tile_count; ++tile) {
                const uint64_t offset =
                    (uint64_t)batch * plan->input->byte_strides[0]
                    + group_channel
                    + (uint64_t)(first_input_row + kernel_row)
                        * plan->input->byte_strides[2]
                    + (uint64_t)(first_input_column +
                        (int64_t)tile * plan->strides[1] + kernel_column)
                        * plan->input->byte_strides[3];
                if (!campp_qconv_v4_point_in_storage(plan, offset)) {
                    return false;
                }
                tile_plan->input_points[kernel][tile] =
                    (const uint8_t *)plan->input->data + offset;
            }
        }
    }
    tile_plan->path = CAMPP_QCONV_V4_PATH_3X3_INTERIOR;
    return true;
}

static void campp_qconv_v4_plan_generic(
    const CamppQconvV4ExecutionPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_count,
    CamppQconvV4TilePlan *tile_plan)
{
    uint32_t kernel;
    for (kernel = 0u; kernel < plan->kernel_elements; ++kernel) {
        const uint32_t kernel_coordinates[2] = {
            plan->spatial_rank == 1u
                ? kernel
                : kernel / (uint32_t)plan->kernel_shape[1],
            plan->spatial_rank == 1u
                ? 0u
                : kernel % (uint32_t)plan->kernel_shape[1]
        };
        uint32_t tile;
        for (tile = 0u; tile < tile_count; ++tile) {
            tile_plan->input_points[kernel][tile] =
                campp_qconv_address_input_base(
                    &plan->address, batch, group_channel,
                    tile_plan->output_coordinates[tile],
                    kernel_coordinates, true);
        }
    }
    tile_plan->path = CAMPP_QCONV_V4_PATH_GENERIC;
}

CamppStatus campp_qconv_v4_tile_plan_create(
    const CamppQconvV4ExecutionPlan *plan, uint32_t batch,
    uint32_t group_channel, uint32_t tile_start, uint32_t tile_count,
    CamppQconvV4TilePlan *out_tile)
{
    if (plan == NULL || out_tile == NULL || batch >= plan->input->dimensions[0] ||
        group_channel >= plan->input_channels || tile_count == 0u ||
        tile_count > CAMPP_QCONV_CANDIDATE_TILE ||
        tile_start >= plan->output_spatial ||
        tile_count > plan->output_spatial - tile_start) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    memset(out_tile, 0, sizeof(*out_tile));
    out_tile->tile_count = tile_count;
    out_tile->full_spatial_tile =
        tile_count == CAMPP_QCONV_CANDIDATE_TILE;
    campp_qconv_address_output_tile(
        &plan->address, tile_start, tile_count,
        out_tile->output_coordinates);

    if (campp_qconv_v4_plan_1x1(
            plan, batch, group_channel, tile_start, tile_count, out_tile) ||
        campp_qconv_v4_plan_3x3_interior(
            plan, batch, group_channel, tile_start, tile_count, out_tile)) {
        return CAMPP_STATUS_OK;
    }
    campp_qconv_v4_plan_generic(
        plan, batch, group_channel, tile_count, out_tile);
    return CAMPP_STATUS_OK;
}
