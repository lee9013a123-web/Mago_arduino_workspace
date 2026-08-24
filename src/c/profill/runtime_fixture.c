#include "campp_profill/runtime_fixture.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

int campp_profill_read_entire_file(
    const char *path, uint8_t **out_data, size_t *out_size)
{
    FILE *file;
    long length;
    uint8_t *buffer;

    if (path == NULL || out_data == NULL || out_size == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0) {
        fclose(file);
        return 1;
    }
    rewind(file);
    buffer = (uint8_t *)malloc((size_t)length == 0u ? 1u : (size_t)length);
    if (buffer == NULL) {
        fclose(file);
        return 1;
    }
    if (fread(buffer, 1u, (size_t)length, file) != (size_t)length) {
        free(buffer);
        fclose(file);
        return 1;
    }
    fclose(file);
    *out_data = buffer;
    *out_size = (size_t)length;
    return 0;
}

int campp_profill_infer_input_dimensions(
    const CamppRuntimeModel *model,
    const CamppTensorDescriptor *descriptor, size_t input_size,
    uint32_t dimensions[CAMPP_TENSOR_MAX_RANK])
{
    const uint32_t element_size = campp_dtype_byte_size(descriptor->dtype);
    uint64_t fixed_elements = 1u;
    uint64_t bytes_per_frame;
    uint64_t provided_frames;
    uint8_t time_axis = 0u;
    uint8_t time_axis_count = 0u;
    uint8_t axis;

    if (model == NULL || descriptor == NULL || dimensions == NULL ||
        element_size == 0u) {
        return 1;
    }
    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        dimensions[axis] = descriptor->dimensions[axis];
    }
    if ((uint64_t)input_size == descriptor->storage_span_bytes) {
        return 0;
    }
    for (axis = 0u; axis < descriptor->rank; ++axis) {
        const uint32_t dimension = descriptor->dimensions[axis];
        if (dimension == model->bucket_frames) {
            time_axis = axis;
            time_axis_count += 1u;
            continue;
        }
        if (dimension == 0u || fixed_elements > UINT64_MAX / dimension) {
            return 1;
        }
        fixed_elements *= dimension;
    }
    if (time_axis_count != 1u || fixed_elements > UINT64_MAX / element_size) {
        return 1;
    }
    bytes_per_frame = fixed_elements * element_size;
    if (bytes_per_frame == 0u || (uint64_t)input_size % bytes_per_frame != 0u) {
        return 1;
    }
    provided_frames = (uint64_t)input_size / bytes_per_frame;
    if (provided_frames == 0u || provided_frames > UINT32_MAX) {
        return 1;
    }
    dimensions[time_axis] = (uint32_t)provided_frames;
    return 0;
}

const CamppKernelRegistry *campp_profill_registry_for_model(
    const CamppRuntimeModel *model)
{
    uint32_t operator_id;

    if (model == NULL) {
        return NULL;
    }
    for (operator_id = 0u; operator_id < model->operator_count; ++operator_id) {
        if (model->operators[operator_id].kernel_id !=
            CAMPP_DEFAULT_KERNEL_ID) {
            return campp_cpu_aarch64_registry();
        }
    }
    return campp_cpu_reference_registry();
}
