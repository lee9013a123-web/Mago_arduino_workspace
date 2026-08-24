#include "qconv_v5_execution_plan.h"

#include <stddef.h>
#include <stdint.h>

static CamppStatus campp_qconv_v5_address_status(
    CamppQconvV5AddressStatus status)
{
    switch (status) {
    case CAMPP_QCONV_V5_ADDRESS_OK:
        return CAMPP_STATUS_OK;
    case CAMPP_QCONV_V5_ADDRESS_UNSUPPORTED:
        return CAMPP_STATUS_NOT_IMPLEMENTED;
    case CAMPP_QCONV_V5_ADDRESS_OUT_OF_STORAGE:
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    case CAMPP_QCONV_V5_ADDRESS_INVALID_ARGUMENT:
    case CAMPP_QCONV_V5_ADDRESS_DONE:
    default:
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
}

CamppStatus campp_qconv_v5_address_plan_from_v4(
    const CamppQconvV4ExecutionPlan *v4_plan,
    CamppQconvV5AddressPlan *out_address)
{
    CamppQconvV5AddressGeometry geometry;
    uint8_t axis;

    if (v4_plan == NULL || out_address == NULL || v4_plan->input == NULL ||
        v4_plan->output == NULL || v4_plan->input->data == NULL ||
        v4_plan->spatial_rank < 1u || v4_plan->spatial_rank > 2u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    geometry.input_base = (const uint8_t *)v4_plan->input->data;
    geometry.storage_span_bytes = v4_plan->input->storage_span_bytes;
    geometry.batch_stride_bytes = v4_plan->input->byte_strides[0];
    geometry.channel_stride_bytes = v4_plan->input->byte_strides[1];
    geometry.batch_count = v4_plan->input->dimensions[0];
    geometry.input_channels = v4_plan->input_channels;
    geometry.input_channels_per_group = v4_plan->inputs_per_group;
    geometry.spatial_rank = v4_plan->spatial_rank;
    for (axis = 0u; axis < 2u; ++axis) {
        geometry.input_spatial_stride_bytes[axis] = 0u;
        geometry.input_spatial[axis] = 1u;
        geometry.output_spatial[axis] = 1u;
        geometry.kernel_shape[axis] = 1u;
        geometry.convolution_stride[axis] = 1;
        geometry.dilation[axis] = 1;
        geometry.pad_begin[axis] = 0;
    }
    for (axis = 0u; axis < v4_plan->spatial_rank; ++axis) {
        if (v4_plan->kernel_shape[axis] <= 0 ||
            v4_plan->kernel_shape[axis] > UINT32_MAX) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        geometry.input_spatial_stride_bytes[axis] =
            v4_plan->input->byte_strides[axis + 2u];
        geometry.input_spatial[axis] =
            v4_plan->input->dimensions[axis + 2u];
        geometry.output_spatial[axis] =
            v4_plan->output->dimensions[axis + 2u];
        geometry.kernel_shape[axis] =
            (uint32_t)v4_plan->kernel_shape[axis];
        geometry.convolution_stride[axis] = v4_plan->strides[axis];
        geometry.dilation[axis] = v4_plan->dilations[axis];
        geometry.pad_begin[axis] = v4_plan->pads[axis];
    }
    return campp_qconv_v5_address_status(
        campp_qconv_v5_address_plan_create(&geometry, out_address));
}

CamppStatus campp_qconv_v5_execution_plan_create(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    CamppQconvV5ExecutionPlan *out_plan)
{
    CamppStatus status;

    if (out_plan == NULL) return CAMPP_STATUS_INVALID_ARGUMENT;
    status = campp_qconv_v4_execution_plan_create(
        model, op, inputs, input_count, outputs, output_count, &out_plan->v4);
    if (status != CAMPP_STATUS_OK) return status;
    return campp_qconv_v5_address_plan_from_v4(
        &out_plan->v4, &out_plan->address);
}
