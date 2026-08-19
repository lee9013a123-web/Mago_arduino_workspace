#include "qconv_v4_address_adapter.h"

#include <stddef.h>

CamppQconvAddressStatus campp_qconv_v4_address_v2_geometry(
    const CamppQconvV4ExecutionPlan *v4_plan,
    CamppQconvAddressGeometry *out_geometry)
{
    uint8_t axis;

    if (v4_plan == NULL || out_geometry == NULL || v4_plan->input == NULL ||
        v4_plan->output == NULL || v4_plan->input->data == NULL ||
        v4_plan->spatial_rank < 1u || v4_plan->spatial_rank > 2u) {
        return CAMPP_QCONV_ADDRESS_INVALID_ARGUMENT;
    }
    out_geometry->input_base = (const uint8_t *)v4_plan->input->data;
    out_geometry->storage_span_bytes =
        v4_plan->input->storage_span_bytes;
    out_geometry->batch_stride_bytes =
        v4_plan->input->byte_strides[0];
    out_geometry->channel_stride_bytes =
        v4_plan->input->byte_strides[1];
    out_geometry->batch_count = v4_plan->input->dimensions[0];
    out_geometry->input_channels = v4_plan->input_channels;
    out_geometry->input_channels_per_group = v4_plan->inputs_per_group;
    out_geometry->spatial_rank = v4_plan->spatial_rank;
    for (axis = 0u; axis < 2u; ++axis) {
        out_geometry->input_spatial_stride_bytes[axis] = 0u;
        out_geometry->input_spatial[axis] = 1u;
        out_geometry->output_spatial[axis] = 1u;
        out_geometry->kernel_shape[axis] = 1u;
        out_geometry->convolution_stride[axis] = 1;
        out_geometry->dilation[axis] = 1;
        out_geometry->pad_begin[axis] = 0;
    }
    for (axis = 0u; axis < v4_plan->spatial_rank; ++axis) {
        out_geometry->input_spatial_stride_bytes[axis] =
            v4_plan->input->byte_strides[axis + 2u];
        out_geometry->input_spatial[axis] =
            v4_plan->input->dimensions[axis + 2u];
        out_geometry->output_spatial[axis] =
            v4_plan->output->dimensions[axis + 2u];
        out_geometry->kernel_shape[axis] =
            (uint32_t)v4_plan->kernel_shape[axis];
        out_geometry->convolution_stride[axis] = v4_plan->strides[axis];
        out_geometry->dilation[axis] = v4_plan->dilations[axis];
        out_geometry->pad_begin[axis] = v4_plan->pads[axis];
    }
    return CAMPP_QCONV_ADDRESS_OK;
}
