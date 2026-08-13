#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "execution/tensor_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"
#include "memory_management/memory_bounds_checker.h"
#include "memory_management/reference_tensor_storage.h"

#define CHECK_TRUE(condition)                                                   \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__,       \
                    #condition);                                                \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define CHECK_STATUS(expression, expected)                                     \
    do {                                                                        \
        CamppStatus actual_status = (expression);                               \
        if (actual_status != (expected)) {                                      \
            fprintf(stderr, "STATUS failed at line %d: got %s, expected %s\n",\
                    __LINE__, campp_status_name(actual_status),                 \
                    campp_status_name(expected));                               \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static void init_tensor(
    CamppTensorDescriptor *descriptor, uint32_t tensor_id,
    uint8_t storage_type, uint32_t channels, uint64_t logical_bytes,
    uint64_t span_bytes)
{
    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->tensor_id = tensor_id;
    descriptor->dtype = CAMPP_DTYPE_FLOAT32;
    descriptor->rank = 3u;
    descriptor->storage_type = storage_type;
    descriptor->flags = CAMPP_TENSOR_FLAG_CONTIGUOUS;
    descriptor->dimensions[0] = 1u;
    descriptor->dimensions[1] = channels;
    descriptor->dimensions[2] = 3u;
    descriptor->dimensions[3] = 1u;
    descriptor->byte_strides[0] = channels * 3u * 4u;
    descriptor->byte_strides[1] = 3u * 4u;
    descriptor->byte_strides[2] = 4u;
    descriptor->byte_strides[3] = 0u;
    descriptor->data_offset = CAMPP_INVALID_DATA_OFFSET;
    descriptor->logical_byte_size = logical_bytes;
    descriptor->storage_span_bytes = span_bytes;
    descriptor->alias_of_tensor_id = CAMPP_INVALID_TENSOR_ID;
    descriptor->quantization_index = CAMPP_INVALID_QUANTIZATION_INDEX;
    descriptor->first_use = 0u;
    descriptor->last_use = 0u;
}

int main(void)
{
    CamppTensorDescriptor descriptors[3];
    CamppRuntimeModel model;
    CamppActivationStorage storage;
    CamppTensorView *views = NULL;
    uint32_t view_count = 0u;
    float *base;
    float *feature;

    init_tensor(
        &descriptors[0], 0u, CAMPP_TENSOR_STORAGE_ACTIVATION,
        2u, 24u, 48u);
    init_tensor(
        &descriptors[1], 1u, CAMPP_TENSOR_STORAGE_VIEW,
        3u, 36u, 36u);
    descriptors[1].flags = (uint8_t)(
        descriptors[1].flags | CAMPP_TENSOR_FLAG_ALIASED);
    descriptors[1].data_offset = 0u;
    descriptors[1].alias_of_tensor_id = 0u;
    init_tensor(
        &descriptors[2], 2u, CAMPP_TENSOR_STORAGE_VIEW,
        1u, 12u, 12u);
    descriptors[2].flags = (uint8_t)(
        descriptors[2].flags | CAMPP_TENSOR_FLAG_ALIASED);
    descriptors[2].data_offset = 24u;
    descriptors[2].alias_of_tensor_id = 0u;

    memset(&model, 0, sizeof(model));
    model.tensors = descriptors;
    model.tensor_count = 3u;
    model.operator_count = 1u;

    memset(&storage, 0, sizeof(storage));
    CHECK_STATUS(
        campp_reference_tensor_storage_create(&model, &storage),
        CAMPP_STATUS_OK);
    CHECK_TRUE(storage.total_bytes == 48u);
    CHECK_STATUS(
        campp_tensor_registry_create(
            &model, &storage, &views, &view_count),
        CAMPP_STATUS_OK);
    CHECK_TRUE(view_count == 3u);
    CHECK_TRUE(views[1].data == views[0].data);
    CHECK_TRUE(
        (uint8_t *)views[2].data == (uint8_t *)views[0].data + 24u);

    base = (float *)views[0].data;
    feature = (float *)views[2].data;
    feature[0] = 1.0f;
    feature[1] = 2.0f;
    feature[2] = 3.0f;
    CHECK_TRUE(base[6] == 1.0f && base[7] == 2.0f && base[8] == 3.0f);

    campp_tensor_registry_release(&views, &view_count);
    descriptors[2].data_offset = 40u;
    CHECK_STATUS(
        campp_tensor_registry_create(
            &model, &storage, &views, &view_count),
        CAMPP_STATUS_BUFFER_OVERFLOW);
    CHECK_TRUE(views == NULL);

    campp_reference_tensor_storage_release(&storage);

    /* INPUT-backed producerless VIEW is resolved when the caller binds input. */
    {
        CamppTensorDescriptor input_descriptors[2];
        CamppRuntimeModel input_model;
        CamppActivationStorage input_storage;
        CamppRuntimeContext input_context;
        void *buffers[2] = {NULL, NULL};
        size_t buffer_sizes[2] = {0u, 0u};
        float input_data[6] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        uint32_t dimensions[CAMPP_TENSOR_MAX_RANK] = {1u, 2u, 3u, 1u};

        init_tensor(
            &input_descriptors[0], 0u, CAMPP_TENSOR_STORAGE_INPUT,
            2u, 24u, 24u);
        init_tensor(
            &input_descriptors[1], 1u, CAMPP_TENSOR_STORAGE_VIEW,
            2u, 24u, 24u);
        input_descriptors[1].flags = (uint8_t)(
            input_descriptors[1].flags | CAMPP_TENSOR_FLAG_ALIASED);
        input_descriptors[1].data_offset = 0u;
        input_descriptors[1].alias_of_tensor_id = 0u;

        memset(&input_model, 0, sizeof(input_model));
        input_model.tensors = input_descriptors;
        input_model.tensor_count = 2u;
        input_model.operator_count = 1u;
        memset(&input_storage, 0, sizeof(input_storage));
        input_storage.mode = CAMPP_ACTIVATION_STORAGE_REFERENCE;
        input_storage.buffers = buffers;
        input_storage.buffer_sizes = buffer_sizes;
        input_storage.buffer_count = 2u;
        CHECK_STATUS(
            campp_tensor_registry_create(
                &input_model, &input_storage, &views, &view_count),
            CAMPP_STATUS_OK);
        CHECK_TRUE(views[0].data == NULL && views[1].data == NULL);
        memset(&input_context, 0, sizeof(input_context));
        input_context.model = &input_model;
        input_context.tensors = views;
        input_context.tensor_count = view_count;
        CHECK_STATUS(
            campp_runtime_context_bind_input(
                &input_context, 0u, input_data, sizeof(input_data),
                CAMPP_DTYPE_FLOAT32, 3u, dimensions),
            CAMPP_STATUS_OK);
        CHECK_TRUE(views[0].data == input_data && views[1].data == input_data);
        campp_tensor_registry_release(&views, &view_count);
    }
    puts("tensor VIEW alias tests: PASS");
    return 0;
}
