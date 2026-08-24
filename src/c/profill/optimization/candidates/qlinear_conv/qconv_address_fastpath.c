#include "qconv_address_fastpath.h"

#include <stddef.h>
#include <string.h>

bool campp_qconv_address_plan_create(
    const CamppTensorView *input, const CamppTensorView *output,
    uint8_t spatial_rank, const int64_t strides[2],
    const int64_t dilations[2], const int64_t pads[4],
    CamppQconvAddressPlan *out_plan)
{
    uint8_t axis;

    if (input == NULL || output == NULL || strides == NULL ||
        dilations == NULL || pads == NULL || out_plan == NULL ||
        spatial_rank < 1u || spatial_rank > 2u ||
        input->rank != spatial_rank + 2u ||
        output->rank != spatial_rank + 2u ||
        input->data == NULL || output->data == NULL ||
        input->byte_strides[1] != 1u) {
        return false;
    }
    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->input = input;
    out_plan->output = output;
    out_plan->spatial_rank = spatial_rank;
    for (axis = 0u; axis < spatial_rank; ++axis) {
        if (strides[axis] <= 0 || dilations[axis] <= 0 || pads[axis] < 0) {
            return false;
        }
        out_plan->strides[axis] = strides[axis];
        out_plan->dilations[axis] = dilations[axis];
        out_plan->pads[axis] = pads[axis];
    }
    return true;
}

void campp_qconv_address_output_tile(
    const CamppQconvAddressPlan *plan, uint32_t tile_start,
    uint32_t tile_count,
    uint32_t coordinates[CAMPP_QCONV_CANDIDATE_TILE][2])
{
    uint32_t tile;

    if (plan->spatial_rank == 1u) {
        for (tile = 0u; tile < tile_count; ++tile) {
            coordinates[tile][0] = tile_start + tile;
            coordinates[tile][1] = 0u;
        }
        return;
    }
    {
        const uint32_t width = plan->output->dimensions[3];
        uint32_t row = tile_start / width;
        uint32_t column = tile_start % width;
        for (tile = 0u; tile < tile_count; ++tile) {
            coordinates[tile][0] = row;
            coordinates[tile][1] = column;
            column += 1u;
            if (column == width) {
                column = 0u;
                row += 1u;
            }
        }
    }
}

const uint8_t *campp_qconv_address_input_base(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, const uint32_t output_coordinates[2],
    const uint32_t kernel_coordinates[2], bool direct_offset)
{
    uint32_t coordinates[CAMPP_TENSOR_MAX_RANK] = {batch, group_channel, 0u, 0u};
    uint64_t offset;
    uint8_t axis;

    for (axis = 0u; axis < plan->spatial_rank; ++axis) {
        const int64_t coordinate =
            (int64_t)output_coordinates[axis] * plan->strides[axis]
            + (int64_t)kernel_coordinates[axis] * plan->dilations[axis]
            - plan->pads[axis];
        if (coordinate < 0 ||
            coordinate >= (int64_t)plan->input->dimensions[axis + 2u]) {
            return NULL;
        }
        coordinates[axis + 2u] = (uint32_t)coordinate;
    }
    if (direct_offset) {
        offset = (uint64_t)batch * plan->input->byte_strides[0]
            + (uint64_t)group_channel * plan->input->byte_strides[1];
        for (axis = 0u; axis < plan->spatial_rank; ++axis) {
            offset += (uint64_t)coordinates[axis + 2u]
                * plan->input->byte_strides[axis + 2u];
        }
    } else {
        offset = campp_tensor_view_byte_offset(plan->input, coordinates);
    }
    if (offset >= plan->input->storage_span_bytes) return NULL;
    return (const uint8_t *)plan->input->data + offset;
}
