/* Fused CAM++ statistics pooling mean/std reduction. */

#include "backends/cpu_aarch64/aarch64_kernels.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "backends/cpu_reference/reference_kernel_utils.h"
#include "internal/runtime_model.h"

CamppStatus campp_fused_statistics_pooling(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count, void *scratch,
    size_t scratch_size)
{
    const CamppTensorView *source;
    CamppTensorView *output;
    uint32_t batch_count;
    uint32_t channels;
    uint32_t frames;
    float variance_multiplier;
    float variance_divisor;
    uint32_t batch;
    uint32_t channel;
    CamppStatus status;

    (void)model;
    (void)op;
    (void)scratch;
    (void)scratch_size;
    status = campp_reference_validate_invocation(
        inputs, input_count, 3u, 3u, outputs, output_count);
    if (status != CAMPP_STATUS_OK) return status;
    source = &inputs[0];
    output = &outputs[0];
    if (source->dtype != CAMPP_DTYPE_FLOAT32 ||
        inputs[1].dtype != CAMPP_DTYPE_FLOAT32 ||
        inputs[2].dtype != CAMPP_DTYPE_FLOAT32 ||
        output->dtype != CAMPP_DTYPE_FLOAT32 ||
        source->rank != 3u || output->rank != 2u ||
        output->dimensions[0] != source->dimensions[0] ||
        output->dimensions[1] != source->dimensions[1] * 2u ||
        campp_tensor_view_element_count(&inputs[1]) != 1u ||
        campp_tensor_view_element_count(&inputs[2]) != 1u) {
        return CAMPP_STATUS_SHAPE_MISMATCH;
    }
    status = campp_reference_read_f32(
        &inputs[1], 0u, &variance_multiplier);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_reference_read_f32(
        &inputs[2], 0u, &variance_divisor);
    if (status != CAMPP_STATUS_OK) return status;
    if (variance_divisor == 0.0f) return CAMPP_STATUS_KERNEL_FAILED;

    batch_count = source->dimensions[0];
    channels = source->dimensions[1];
    frames = source->dimensions[2];
    for (batch = 0u; batch < batch_count; ++batch) {
        for (channel = 0u; channel < channels; ++channel) {
            float sum = 0.0f;
            float variance_sum = 0.0f;
            float mean;
            float variance;
            float standard_deviation;
            uint32_t frame;
            uint32_t input_coordinates[CAMPP_TENSOR_MAX_RANK] = {
                batch, channel, 0u, 0u};
            uint32_t mean_coordinates[CAMPP_TENSOR_MAX_RANK] = {
                batch, channel, 0u, 0u};
            uint32_t std_coordinates[CAMPP_TENSOR_MAX_RANK] = {
                batch, channels + channel, 0u, 0u};

            for (frame = 0u; frame < frames; ++frame) {
                float value;
                uint64_t offset;
                input_coordinates[2] = frame;
                offset = campp_tensor_view_byte_offset(
                    source, input_coordinates);
                memcpy(
                    &value, (const uint8_t *)source->data + offset,
                    sizeof(value));
                sum += value;
            }
            mean = sum / (float)frames;
            for (frame = 0u; frame < frames; ++frame) {
                float value;
                float centered;
                uint64_t offset;
                input_coordinates[2] = frame;
                offset = campp_tensor_view_byte_offset(
                    source, input_coordinates);
                memcpy(
                    &value, (const uint8_t *)source->data + offset,
                    sizeof(value));
                centered = value - mean;
                variance_sum += centered * centered;
            }
            variance = variance_sum / (float)frames;
            variance = variance * variance_multiplier;
            variance = variance / variance_divisor;
            standard_deviation = sqrtf(variance);
            memcpy(
                (uint8_t *)output->data +
                    campp_tensor_view_byte_offset(output, mean_coordinates),
                &mean, sizeof(mean));
            memcpy(
                (uint8_t *)output->data +
                    campp_tensor_view_byte_offset(output, std_coordinates),
                &standard_deviation, sizeof(standard_deviation));
        }
    }
    return CAMPP_STATUS_OK;
}
