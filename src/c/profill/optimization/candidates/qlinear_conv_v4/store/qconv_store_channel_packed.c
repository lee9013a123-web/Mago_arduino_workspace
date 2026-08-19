#include "qconv_store_channel_packed.h"

#include <limits.h>
#include <string.h>

CamppStatus campp_qconv_v4_store_plan_create(
    CamppTensorView *output, CamppQconvV4StorePlan *out_plan)
{
    uint64_t spatial_count = 1u;
    uint8_t axis;

    if (output == NULL || out_plan == NULL || output->data == NULL ||
        output->rank < 3u || output->rank > 4u ||
        output->byte_strides[1] != 1u ||
        (output->dtype != CAMPP_DTYPE_UINT8 &&
         output->dtype != CAMPP_DTYPE_INT8)) {
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    }
    for (axis = 2u; axis < output->rank; ++axis) {
        spatial_count *= output->dimensions[axis];
    }
    if (spatial_count == 0u || spatial_count > UINT32_MAX) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->data = (uint8_t *)output->data;
    out_plan->storage_span_bytes = output->storage_span_bytes;
    out_plan->batch_stride = output->byte_strides[0];
    out_plan->batches = output->dimensions[0];
    out_plan->output_channels = output->dimensions[1];
    out_plan->spatial_count = (uint32_t)spatial_count;
    out_plan->rank = output->rank;
    out_plan->padded_tail_store =
        (output->flags & CAMPP_TENSOR_FLAG_ALIASED) == 0u;
    if (output->rank == 3u) {
        out_plan->spatial_stride = output->byte_strides[2];
        out_plan->channel_capacity = output->byte_strides[2];
    } else {
        if (output->dimensions[3] == 0u) {
            return CAMPP_STATUS_SHAPE_MISMATCH;
        }
        out_plan->row_stride = output->byte_strides[2];
        out_plan->spatial_stride = output->byte_strides[3];
        out_plan->channel_capacity = output->byte_strides[3];
        out_plan->width = output->dimensions[3];
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_qconv_v4_store_channel_packed(
    const CamppQconvV4StorePlan *plan,
    uint32_t batch, uint32_t output_channel, uint32_t spatial_index,
    const uint8_t bytes[CAMPP_QCONV_CANDIDATE_OUTPUT_TILE],
    uint32_t valid_outputs)
{
    uint64_t offset;
    uint32_t store_count = valid_outputs;

    if (plan == NULL || plan->data == NULL || bytes == NULL ||
        batch >= plan->batches || spatial_index >= plan->spatial_count ||
        valid_outputs == 0u ||
        valid_outputs > CAMPP_QCONV_CANDIDATE_OUTPUT_TILE ||
        output_channel + valid_outputs > plan->output_channels) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    offset = (uint64_t)batch * plan->batch_stride + output_channel;
    if (plan->rank == 3u) {
        offset += (uint64_t)spatial_index * plan->spatial_stride;
    } else {
        const uint32_t row = spatial_index / plan->width;
        const uint32_t column = spatial_index % plan->width;
        offset += (uint64_t)row * plan->row_stride
            + (uint64_t)column * plan->spatial_stride;
    }
    if (valid_outputs < CAMPP_QCONV_CANDIDATE_OUTPUT_TILE &&
        output_channel + valid_outputs == plan->output_channels &&
        plan->padded_tail_store &&
        output_channel + CAMPP_QCONV_CANDIDATE_OUTPUT_TILE <=
            plan->channel_capacity &&
        offset + CAMPP_QCONV_CANDIDATE_OUTPUT_TILE <=
            plan->storage_span_bytes) {
        store_count = CAMPP_QCONV_CANDIDATE_OUTPUT_TILE;
    }
    if (offset + store_count > plan->storage_span_bytes) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }
    memcpy(plan->data + offset, bytes, store_count);
    return CAMPP_STATUS_OK;
}
