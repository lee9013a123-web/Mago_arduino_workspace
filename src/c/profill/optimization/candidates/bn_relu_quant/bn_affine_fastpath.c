#include "bn_affine_fastpath.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static CamppStatus campp_bn_read_parameter(
    const CamppTensorView *view, uint32_t channel, float *out_value)
{
    uint64_t offset;
    if (view == NULL || out_value == NULL || view->data == NULL ||
        view->dtype != CAMPP_DTYPE_FLOAT32 || view->rank != 1u ||
        channel >= view->dimensions[0]) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    offset = (uint64_t)channel * view->byte_strides[0];
    if (offset + sizeof(*out_value) > view->storage_span_bytes) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    memcpy(out_value, (const uint8_t *)view->data + offset, sizeof(*out_value));
    return CAMPP_STATUS_OK;
}

CamppStatus campp_bn_affine_channel(
    const CamppTensorView *scale, const CamppTensorView *bias,
    const CamppTensorView *mean, const CamppTensorView *variance,
    uint32_t channel, float epsilon, CamppBnAffine *out_affine)
{
    float scale_value;
    float bias_value;
    float mean_value;
    float variance_value;
    float inv_std;
    CamppStatus status;

    if (out_affine == NULL) return CAMPP_STATUS_INVALID_ARGUMENT;
    status = campp_bn_read_parameter(scale, channel, &scale_value);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_bn_read_parameter(bias, channel, &bias_value);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_bn_read_parameter(mean, channel, &mean_value);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_bn_read_parameter(variance, channel, &variance_value);
    if (status != CAMPP_STATUS_OK) return status;
    inv_std = 1.0f / sqrtf(variance_value + epsilon);
    out_affine->multiplier = inv_std * scale_value;
    out_affine->additive =
        bias_value - mean_value * out_affine->multiplier;
    return isfinite(out_affine->multiplier) && isfinite(out_affine->additive)
        ? CAMPP_STATUS_OK : CAMPP_STATUS_KERNEL_FAILED;
}
