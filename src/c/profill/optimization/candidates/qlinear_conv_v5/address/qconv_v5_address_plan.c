#include "qconv_v5_address.h"

#include <stddef.h>
#include <stdint.h>

#include "qconv_v5_address_internal.h"

static bool campp_qconv_v5_address_interior_range(
    uint32_t input_size, uint32_t output_size, uint32_t kernel_size,
    int64_t stride, int64_t dilation, int64_t pad,
    uint32_t *out_begin, uint32_t *out_end)
{
    uint64_t effective_tail;
    uint64_t begin;
    uint64_t end;
    uint64_t last_numerator;

    if (out_begin == NULL || out_end == NULL || input_size == 0u ||
        output_size == 0u || kernel_size == 0u || stride <= 0 ||
        dilation <= 0 || pad < 0 ||
        !campp_qconv_v5_address_multiply_u64(
            kernel_size - 1u, (uint64_t)dilation, &effective_tail) ||
        (uint64_t)pad > UINT64_MAX - ((uint64_t)stride - 1u)) {
        return false;
    }
    begin = ((uint64_t)pad + (uint64_t)stride - 1u) / (uint64_t)stride;
    if ((uint64_t)input_size - 1u > UINT64_MAX - (uint64_t)pad) {
        return false;
    }
    last_numerator = (uint64_t)input_size - 1u + (uint64_t)pad;
    if (last_numerator < effective_tail) {
        end = 0u;
    } else {
        end = (last_numerator - effective_tail) / (uint64_t)stride + 1u;
    }
    if (begin > output_size) begin = output_size;
    if (end > output_size) end = output_size;
    if (end < begin) end = begin;
    *out_begin = (uint32_t)begin;
    *out_end = (uint32_t)end;
    return true;
}

CamppQconvV5AddressStatus campp_qconv_v5_address_plan_create(
    const CamppQconvV5AddressGeometry *geometry,
    CamppQconvV5AddressPlan *out_plan)
{
    uint64_t output_spatial_count = 1u;
    uint64_t kernel_elements = 1u;
    uint8_t axis;

    if (geometry == NULL || out_plan == NULL ||
        geometry->input_base == NULL || geometry->storage_span_bytes == 0u ||
        geometry->storage_span_bytes > (uint64_t)SIZE_MAX ||
        geometry->spatial_rank < 1u || geometry->spatial_rank > 2u ||
        geometry->batch_count == 0u || geometry->input_channels == 0u ||
        geometry->input_channels_per_group == 0u ||
        geometry->input_channels_per_group > geometry->input_channels) {
        return CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT;
    }
    if (geometry->channel_stride_bytes != 1u) {
        return CAMPP_QCONV_V5_ADDRESS_UNSUPPORTED;
    }

    out_plan->geometry = *geometry;
    out_plan->output_step_bytes[0] = 0u;
    out_plan->output_step_bytes[1] = 0u;
    out_plan->kernel_step_bytes[0] = 0u;
    out_plan->kernel_step_bytes[1] = 0u;
    out_plan->interior_begin[0] = 0u;
    out_plan->interior_begin[1] = 0u;
    out_plan->interior_end[0] = 1u;
    out_plan->interior_end[1] = 1u;

    for (axis = 0u; axis < geometry->spatial_rank; ++axis) {
        if (geometry->input_spatial[axis] == 0u ||
            geometry->output_spatial[axis] == 0u ||
            geometry->kernel_shape[axis] == 0u ||
            geometry->input_spatial_stride_bytes[axis] == 0u ||
            geometry->convolution_stride[axis] <= 0 ||
            geometry->dilation[axis] <= 0 || geometry->pad_begin[axis] < 0 ||
            !campp_qconv_v5_address_multiply_u64(
                (uint64_t)geometry->convolution_stride[axis],
                geometry->input_spatial_stride_bytes[axis],
                &out_plan->output_step_bytes[axis]) ||
            !campp_qconv_v5_address_multiply_u64(
                (uint64_t)geometry->dilation[axis],
                geometry->input_spatial_stride_bytes[axis],
                &out_plan->kernel_step_bytes[axis]) ||
            !campp_qconv_v5_address_multiply_u64(
                output_spatial_count, geometry->output_spatial[axis],
                &output_spatial_count) ||
            !campp_qconv_v5_address_multiply_u64(
                kernel_elements, geometry->kernel_shape[axis],
                &kernel_elements) ||
            !campp_qconv_v5_address_interior_range(
                geometry->input_spatial[axis],
                geometry->output_spatial[axis], geometry->kernel_shape[axis],
                geometry->convolution_stride[axis], geometry->dilation[axis],
                geometry->pad_begin[axis],
                &out_plan->interior_begin[axis],
                &out_plan->interior_end[axis])) {
            return CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT;
        }
    }
    if (output_spatial_count > UINT32_MAX ||
        kernel_elements > CAMPP_QCONV_V5_ADDRESS_KERNEL_MAX) {
        return CAMPP_QCONV_V5_ADDRESS_UNSUPPORTED;
    }
    out_plan->output_spatial_count = (uint32_t)output_spatial_count;
    out_plan->kernel_elements = (uint8_t)kernel_elements;
    if (geometry->spatial_rank == 1u && geometry->kernel_shape[0] == 1u &&
        geometry->dilation[0] == 1) {
        out_plan->preferred_path = CAMPP_QCONV_V5_ADDRESS_PATH_ONE_BY_ONE;
    } else if (geometry->spatial_rank == 2u &&
               geometry->kernel_shape[0] == 3u &&
               geometry->kernel_shape[1] == 3u &&
               geometry->dilation[0] == 1 &&
               geometry->dilation[1] == 1) {
        out_plan->preferred_path =
            CAMPP_QCONV_V5_ADDRESS_PATH_THREE_BY_THREE_INTERIOR;
    } else {
        out_plan->preferred_path = CAMPP_QCONV_V5_ADDRESS_PATH_GENERIC;
    }
    return CAMPP_QCONV_V5_ADDRESS_OK;
}
