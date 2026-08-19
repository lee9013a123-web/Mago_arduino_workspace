#ifndef CAMPP_PROFILL_QCONV_ADDRESS_V2_INTERNAL_H
#define CAMPP_PROFILL_QCONV_ADDRESS_V2_INTERNAL_H

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../api/qconv_address_provider.h"

static inline bool campp_qconv_address_v2_add_u64(
    uint64_t left, uint64_t right, uint64_t *out_value)
{
    if (out_value == NULL || left > UINT64_MAX - right) return false;
    *out_value = left + right;
    return true;
}

static inline bool campp_qconv_address_v2_multiply_u64(
    uint64_t left, uint64_t right, uint64_t *out_value)
{
    if (out_value == NULL || (right != 0u && left > UINT64_MAX / right)) {
        return false;
    }
    *out_value = left * right;
    return true;
}

static inline bool campp_qconv_address_v2_coordinate(
    uint32_t output, int64_t stride, uint32_t kernel,
    int64_t dilation, int64_t pad, int64_t *out_coordinate)
{
    int64_t output_term;
    int64_t kernel_term;

    if (out_coordinate == NULL || stride <= 0 || dilation <= 0 || pad < 0 ||
        (uint64_t)output > (uint64_t)INT64_MAX / (uint64_t)stride ||
        (uint64_t)kernel > (uint64_t)INT64_MAX / (uint64_t)dilation) {
        return false;
    }
    output_term = (int64_t)output * stride;
    kernel_term = (int64_t)kernel * dilation;
    if (output_term > INT64_MAX - kernel_term) return false;
    *out_coordinate = output_term + kernel_term - pad;
    return true;
}

static inline bool campp_qconv_address_v2_group_base(
    const CamppQconvAddressPlan *plan, uint32_t batch,
    uint32_t group_channel, uint64_t *out_offset)
{
    uint64_t batch_offset;
    uint64_t channel_offset;

    if (plan == NULL || out_offset == NULL ||
        batch >= plan->geometry.batch_count ||
        group_channel > plan->geometry.input_channels ||
        plan->geometry.input_channels_per_group >
            plan->geometry.input_channels - group_channel ||
        !campp_qconv_address_v2_multiply_u64(
            batch, plan->geometry.batch_stride_bytes, &batch_offset) ||
        !campp_qconv_address_v2_multiply_u64(
            group_channel, plan->geometry.channel_stride_bytes,
            &channel_offset)) {
        return false;
    }
    return campp_qconv_address_v2_add_u64(
        batch_offset, channel_offset, out_offset);
}

static inline bool campp_qconv_address_v2_add_spatial_offset(
    const CamppQconvAddressPlan *plan, uint8_t axis,
    uint32_t coordinate, uint64_t *offset)
{
    uint64_t spatial_offset;

    if (plan == NULL || offset == NULL || axis >= plan->geometry.spatial_rank ||
        !campp_qconv_address_v2_multiply_u64(
            coordinate, plan->geometry.input_spatial_stride_bytes[axis],
            &spatial_offset)) {
        return false;
    }
    return campp_qconv_address_v2_add_u64(
        *offset, spatial_offset, offset);
}

static inline bool campp_qconv_address_v2_point_in_storage(
    const CamppQconvAddressPlan *plan, uint64_t offset)
{
    return plan != NULL &&
        offset <= plan->geometry.storage_span_bytes &&
        plan->geometry.input_channels_per_group <=
            plan->geometry.storage_span_bytes - offset;
}

static inline const uint8_t *campp_qconv_address_v2_pointer(
    const CamppQconvAddressPlan *plan, uint64_t offset)
{
    return plan->geometry.input_base + (size_t)offset;
}

#endif /* CAMPP_PROFILL_QCONV_ADDRESS_V2_INTERNAL_H */
