#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "backends/cpu_reference/reference_kernels.h"

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
        CamppStatus actual = (expression);                                      \
        if (actual != (expected)) {                                             \
            fprintf(stderr, "STATUS line %d: got %s expected %s\n",          \
                    __LINE__, campp_status_name(actual),                        \
                    campp_status_name(expected));                               \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define CHECK_CLOSE(actual, expected, tolerance)                               \
    CHECK_TRUE(fabsf((actual) - (expected)) <= (tolerance))

typedef struct TestAttribute {
    uint16_t key;
    uint8_t type;
    uint8_t count;
    int64_t ints[4];
    double floats[4];
} TestAttribute;

static void write_u16(uint8_t *target, uint16_t value)
{
    target[0] = (uint8_t)value;
    target[1] = (uint8_t)(value >> 8u);
}

static void write_u32(uint8_t *target, uint32_t value)
{
    uint8_t index;
    for (index = 0u; index < 4u; ++index) {
        target[index] = (uint8_t)(value >> (index * 8u));
    }
}

static void write_u64(uint8_t *target, uint64_t value)
{
    uint8_t index;
    for (index = 0u; index < 8u; ++index) {
        target[index] = (uint8_t)(value >> (index * 8u));
    }
}

static uint32_t encode_attributes(
    uint8_t *buffer, const TestAttribute *attributes, uint32_t attribute_count)
{
    uint32_t cursor = CAMPP_ATTRIBUTE_BLOCK_HEADER_SIZE;
    uint32_t attribute_index;

    for (attribute_index = 0u; attribute_index < attribute_count;
         ++attribute_index) {
        const TestAttribute *attribute = &attributes[attribute_index];
        uint8_t value_index;
        write_u16(buffer + cursor, attribute->key);
        buffer[cursor + 2u] = attribute->type;
        buffer[cursor + 3u] = attribute->count;
        write_u32(buffer + cursor + 4u, 0u);
        cursor += CAMPP_ATTRIBUTE_RECORD_HEADER_SIZE;
        for (value_index = 0u; value_index < attribute->count; ++value_index) {
            uint64_t bits;
            if (attribute->type == CAMPP_ATTR_VALUE_INT) {
                bits = (uint64_t)attribute->ints[value_index];
            } else {
                memcpy(&bits, &attribute->floats[value_index], sizeof(bits));
            }
            write_u64(buffer + cursor, bits);
            cursor += CAMPP_ATTRIBUTE_VALUE_SIZE;
        }
    }
    write_u32(buffer, attribute_count);
    write_u32(buffer + 4u, cursor);
    return cursor;
}

static void init_model_and_op(
    CamppRuntimeModel *model, CamppOperatorDescriptor *op, uint16_t opcode,
    uint8_t *attribute_buffer, const TestAttribute *attributes,
    uint32_t attribute_count)
{
    memset(model, 0, sizeof(*model));
    memset(op, 0, sizeof(*op));
    op->opcode = opcode;
    if (attribute_count != 0u) {
        op->attribute_size = encode_attributes(
            attribute_buffer, attributes, attribute_count);
        model->attribute_section = attribute_buffer;
        model->attribute_section_size = op->attribute_size;
    }
}

static void init_view(
    CamppTensorView *view, void *data, uint8_t dtype, uint8_t rank,
    const uint32_t *dimensions)
{
    uint32_t stride = campp_dtype_byte_size(dtype);
    uint8_t axis;

    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = dtype;
    view->rank = rank;
    view->flags = CAMPP_TENSOR_FLAG_CONTIGUOUS;
    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        view->dimensions[axis] = axis < rank ? dimensions[axis] : 1u;
    }
    for (axis = rank; axis > 0u; --axis) {
        view->byte_strides[axis - 1u] = stride;
        stride *= view->dimensions[axis - 1u];
    }
    view->logical_byte_size = rank == 0u
        ? campp_dtype_byte_size(dtype) : stride;
    view->storage_span_bytes = view->logical_byte_size;
}

static int test_relu_and_sigmoid(void)
{
    const uint32_t dims[1] = {5u};
    float input[5] = {-3.0f, -0.0f, 0.0f, 2.0f, NAN};
    float output[5] = {0};
    CamppTensorView inputs[1];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attributes[8];

    init_view(&inputs[0], input, CAMPP_DTYPE_FLOAT32, 1u, dims);
    init_view(&outputs[0], output, CAMPP_DTYPE_FLOAT32, 1u, dims);
    init_model_and_op(&model, &op, CAMPP_OP_RELU, attributes, NULL, 0u);
    CHECK_STATUS(
        campp_reference_relu(
            &model, &op, inputs, 1u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(output[0] == 0.0f && output[1] == 0.0f &&
               output[2] == 0.0f && output[3] == 2.0f &&
               isnan(output[4]));

    input[0] = -100.0f;
    input[1] = 0.0f;
    input[2] = 100.0f;
    init_model_and_op(&model, &op, CAMPP_OP_SIGMOID, attributes, NULL, 0u);
    CHECK_STATUS(
        campp_reference_sigmoid(
            &model, &op, inputs, 1u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(output[0] >= 0.0f && output[0] < 1e-20f);
    CHECK_CLOSE(output[1], 0.5f, 1e-7f);
    CHECK_CLOSE(output[2], 1.0f, 1e-7f);
    return 0;
}

static int test_elementwise_broadcasting(void)
{
    const uint32_t matrix_dims[2] = {2u, 3u};
    const uint32_t row_dims[1] = {3u};
    const uint32_t column_dims[2] = {2u, 1u};
    const uint32_t scalar_dims[1] = {1u};
    float matrix[6] = {1, 2, 3, 4, 5, 6};
    float row[3] = {10, 20, 30};
    float column[2] = {2, 3};
    float scalar = 2.0f;
    float output[6] = {0};
    CamppTensorView inputs[2];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attributes[8];

    init_view(&inputs[0], matrix, CAMPP_DTYPE_FLOAT32, 2u, matrix_dims);
    init_view(&inputs[1], row, CAMPP_DTYPE_FLOAT32, 1u, row_dims);
    init_view(&outputs[0], output, CAMPP_DTYPE_FLOAT32, 2u, matrix_dims);
    init_model_and_op(&model, &op, CAMPP_OP_ADD, attributes, NULL, 0u);
    CHECK_STATUS(
        campp_reference_add(
            &model, &op, inputs, 2u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(output[0] == 11 && output[1] == 22 && output[2] == 33 &&
               output[3] == 14 && output[4] == 25 && output[5] == 36);

    init_view(&inputs[0], column, CAMPP_DTYPE_FLOAT32, 2u, column_dims);
    init_view(&inputs[1], row, CAMPP_DTYPE_FLOAT32, 1u, row_dims);
    init_model_and_op(&model, &op, CAMPP_OP_MUL, attributes, NULL, 0u);
    CHECK_STATUS(
        campp_reference_mul(
            &model, &op, inputs, 2u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(output[0] == 20 && output[1] == 40 && output[2] == 60 &&
               output[3] == 30 && output[4] == 60 && output[5] == 90);

    init_view(&inputs[0], matrix, CAMPP_DTYPE_FLOAT32, 2u, matrix_dims);
    init_view(&inputs[1], &scalar, CAMPP_DTYPE_FLOAT32, 0u, scalar_dims);
    init_model_and_op(&model, &op, CAMPP_OP_SUB, attributes, NULL, 0u);
    CHECK_STATUS(
        campp_reference_sub(
            &model, &op, inputs, 2u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(output[0] == -1 && output[5] == 4);
    init_model_and_op(&model, &op, CAMPP_OP_DIV, attributes, NULL, 0u);
    CHECK_STATUS(
        campp_reference_div(
            &model, &op, inputs, 2u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_CLOSE(output[0], 0.5f, 1e-7f);
    CHECK_CLOSE(output[5], 3.0f, 1e-7f);

    matrix[0] = 0.0f;
    matrix[1] = 1.0f;
    matrix[2] = 4.0f;
    matrix[3] = 9.0f;
    matrix[4] = 16.0f;
    matrix[5] = 25.0f;
    init_model_and_op(&model, &op, CAMPP_OP_SQRT, attributes, NULL, 0u);
    CHECK_STATUS(
        campp_reference_sqrt(
            &model, &op, inputs, 1u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(output[0] == 0 && output[1] == 1 && output[2] == 2 &&
               output[3] == 3 && output[4] == 4 && output[5] == 5);
    return 0;
}

static int test_reshape_squeeze_unsqueeze(void)
{
    const uint32_t input_dims[2] = {2u, 3u};
    const uint32_t output_dims[2] = {3u, 2u};
    const uint32_t shape_dims[1] = {2u};
    const uint32_t squeezed_input_dims[3] = {1u, 2u, 1u};
    const uint32_t squeezed_dims[1] = {2u};
    const uint32_t unsqueezed_dims[3] = {1u, 2u, 1u};
    float input[6] = {1, 2, 3, 4, 5, 6};
    float output[6] = {0};
    int64_t shape[2] = {3, 2};
    float squeezed_data[2] = {7, 8};
    float squeeze_output[2] = {0};
    CamppTensorView inputs[2];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_buffer[96];
    TestAttribute axes = {CAMPP_ATTR_AXES, CAMPP_ATTR_VALUE_INT, 2u,
                          {0, 2, 0, 0}, {0}};

    init_view(&inputs[0], input, CAMPP_DTYPE_FLOAT32, 2u, input_dims);
    init_view(&inputs[1], shape, CAMPP_DTYPE_INT64, 1u, shape_dims);
    init_view(&outputs[0], output, CAMPP_DTYPE_FLOAT32, 2u, output_dims);
    init_model_and_op(
        &model, &op, CAMPP_OP_RESHAPE, attribute_buffer, NULL, 0u);
    CHECK_STATUS(
        campp_reference_reshape(
            &model, &op, inputs, 2u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(input, output, sizeof(input)) == 0);

    init_view(
        &inputs[0], squeezed_data, CAMPP_DTYPE_FLOAT32, 3u,
        squeezed_input_dims);
    init_view(
        &outputs[0], squeeze_output, CAMPP_DTYPE_FLOAT32, 1u,
        squeezed_dims);
    init_model_and_op(
        &model, &op, CAMPP_OP_SQUEEZE, attribute_buffer, &axes, 1u);
    CHECK_STATUS(
        campp_reference_squeeze(
            &model, &op, inputs, 1u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(squeeze_output[0] == 7 && squeeze_output[1] == 8);

    init_view(
        &inputs[0], squeezed_data, CAMPP_DTYPE_FLOAT32, 1u, squeezed_dims);
    init_view(
        &outputs[0], squeeze_output, CAMPP_DTYPE_FLOAT32, 3u,
        unsqueezed_dims);
    init_model_and_op(
        &model, &op, CAMPP_OP_UNSQUEEZE, attribute_buffer, &axes, 1u);
    CHECK_STATUS(
        campp_reference_unsqueeze(
            &model, &op, inputs, 1u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(squeeze_output[0] == 7 && squeeze_output[1] == 8);
    return 0;
}

static int test_transpose_concat_expand_slice(void)
{
    const uint32_t input_dims[2] = {2u, 3u};
    const uint32_t transpose_dims[2] = {3u, 2u};
    const uint32_t concat_a_dims[2] = {2u, 1u};
    const uint32_t concat_b_dims[2] = {2u, 2u};
    const uint32_t expand_input_dims[2] = {2u, 1u};
    const uint32_t expand_output_dims[2] = {2u, 3u};
    const uint32_t shape_dims[1] = {2u};
    const uint32_t slice_input_dims[2] = {2u, 5u};
    const uint32_t slice_output_dims[2] = {2u, 2u};
    const uint32_t one_dim[1] = {1u};
    const uint32_t reverse_input_dims[1] = {5u};
    const uint32_t reverse_output_dims[1] = {3u};
    float input[6] = {1, 2, 3, 4, 5, 6};
    float transposed[6] = {0};
    float concat_a[2] = {1, 4};
    float concat_b[4] = {2, 3, 5, 6};
    float concatenated[6] = {0};
    float expand_input[2] = {3, 7};
    float expanded[6] = {0};
    int64_t expand_shape[2] = {2, 3};
    float slice_input[10] = {0, 1, 2, 3, 4, 10, 11, 12, 13, 14};
    float sliced[4] = {0};
    float reverse_input[5] = {0, 1, 2, 3, 4};
    float reversed[3] = {0};
    int64_t starts = 1;
    int64_t ends = 5;
    int64_t slice_axis = 1;
    int64_t steps = 2;
    CamppTensorView inputs[5];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_buffer[96];
    TestAttribute attribute;

    init_view(&inputs[0], input, CAMPP_DTYPE_FLOAT32, 2u, input_dims);
    init_view(
        &outputs[0], transposed, CAMPP_DTYPE_FLOAT32, 2u,
        transpose_dims);
    memset(&attribute, 0, sizeof(attribute));
    attribute.key = CAMPP_ATTR_PERM;
    attribute.type = CAMPP_ATTR_VALUE_INT;
    attribute.count = 2u;
    attribute.ints[0] = 1;
    attribute.ints[1] = 0;
    init_model_and_op(
        &model, &op, CAMPP_OP_TRANSPOSE, attribute_buffer, &attribute, 1u);
    CHECK_STATUS(
        campp_reference_transpose(
            &model, &op, inputs, 1u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    {
        const float expected[6] = {1, 4, 2, 5, 3, 6};
        CHECK_TRUE(memcmp(transposed, expected, sizeof(expected)) == 0);
    }

    init_view(&inputs[0], concat_a, CAMPP_DTYPE_FLOAT32, 2u, concat_a_dims);
    init_view(&inputs[1], concat_b, CAMPP_DTYPE_FLOAT32, 2u, concat_b_dims);
    init_view(&outputs[0], concatenated, CAMPP_DTYPE_FLOAT32, 2u, input_dims);
    attribute.key = CAMPP_ATTR_AXIS;
    attribute.count = 1u;
    attribute.ints[0] = 1;
    init_model_and_op(
        &model, &op, CAMPP_OP_CONCAT, attribute_buffer, &attribute, 1u);
    CHECK_STATUS(
        campp_reference_concat(
            &model, &op, inputs, 2u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(concatenated, input, sizeof(input)) == 0);

    init_view(
        &inputs[0], expand_input, CAMPP_DTYPE_FLOAT32, 2u,
        expand_input_dims);
    init_view(&inputs[1], expand_shape, CAMPP_DTYPE_INT64, 1u, shape_dims);
    init_view(
        &outputs[0], expanded, CAMPP_DTYPE_FLOAT32, 2u,
        expand_output_dims);
    init_model_and_op(
        &model, &op, CAMPP_OP_EXPAND, attribute_buffer, NULL, 0u);
    CHECK_STATUS(
        campp_reference_expand(
            &model, &op, inputs, 2u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    {
        const float expected[6] = {3, 3, 3, 7, 7, 7};
        CHECK_TRUE(memcmp(expanded, expected, sizeof(expected)) == 0);
    }

    init_view(
        &inputs[0], slice_input, CAMPP_DTYPE_FLOAT32, 2u,
        slice_input_dims);
    init_view(&inputs[1], &starts, CAMPP_DTYPE_INT64, 1u, one_dim);
    init_view(&inputs[2], &ends, CAMPP_DTYPE_INT64, 1u, one_dim);
    init_view(&inputs[3], &slice_axis, CAMPP_DTYPE_INT64, 1u, one_dim);
    init_view(&inputs[4], &steps, CAMPP_DTYPE_INT64, 1u, one_dim);
    init_view(
        &outputs[0], sliced, CAMPP_DTYPE_FLOAT32, 2u,
        slice_output_dims);
    init_model_and_op(
        &model, &op, CAMPP_OP_SLICE, attribute_buffer, NULL, 0u);
    CHECK_STATUS(
        campp_reference_slice(
            &model, &op, inputs, 5u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(sliced[0] == 1 && sliced[1] == 3 &&
               sliced[2] == 11 && sliced[3] == 13);

    starts = 4;
    ends = INT64_MIN;
    slice_axis = 0;
    steps = -2;
    init_view(
        &inputs[0], reverse_input, CAMPP_DTYPE_FLOAT32, 1u,
        reverse_input_dims);
    init_view(
        &outputs[0], reversed, CAMPP_DTYPE_FLOAT32, 1u,
        reverse_output_dims);
    CHECK_STATUS(
        campp_reference_slice(
            &model, &op, inputs, 5u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(reversed[0] == 4 && reversed[1] == 2 && reversed[2] == 0);
    return 0;
}

static int test_reduction_pool_and_batch_norm(void)
{
    const uint32_t reduce_input_dims[2] = {2u, 3u};
    const uint32_t reduce_output_dims[1] = {2u};
    const uint32_t pool_input_dims[3] = {1u, 1u, 5u};
    const uint32_t pool_output_dims[3] = {1u, 1u, 3u};
    const uint32_t pool_ceil_output_dims[3] = {1u, 1u, 2u};
    const uint32_t batch_dims[3] = {1u, 2u, 2u};
    const uint32_t channel_dims[1] = {2u};
    float reduce_input[6] = {1, 2, 3, 4, 5, 6};
    float reduced[2] = {0};
    float pool_input[5] = {1, 2, 3, 4, 5};
    float pooled[3] = {0};
    float pooled_with_ceil[2] = {0};
    float batch_input[4] = {1, 3, 2, 6};
    float scale[2] = {2, 1};
    float bias[2] = {1, -1};
    float mean[2] = {1, 2};
    float variance[2] = {4, 16};
    float normalized[4] = {0};
    CamppTensorView inputs[5];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_buffer[256];
    TestAttribute attributes[5];

    memset(attributes, 0, sizeof(attributes));
    attributes[0].key = CAMPP_ATTR_AXES;
    attributes[0].type = CAMPP_ATTR_VALUE_INT;
    attributes[0].count = 1u;
    attributes[0].ints[0] = -1;
    attributes[1].key = CAMPP_ATTR_KEEPDIMS;
    attributes[1].type = CAMPP_ATTR_VALUE_INT;
    attributes[1].count = 1u;
    attributes[1].ints[0] = 0;
    init_view(
        &inputs[0], reduce_input, CAMPP_DTYPE_FLOAT32, 2u,
        reduce_input_dims);
    init_view(
        &outputs[0], reduced, CAMPP_DTYPE_FLOAT32, 1u,
        reduce_output_dims);
    init_model_and_op(
        &model, &op, CAMPP_OP_REDUCE_MEAN, attribute_buffer, attributes, 2u);
    CHECK_STATUS(
        campp_reference_reduce_mean(
            &model, &op, inputs, 1u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_CLOSE(reduced[0], 2.0f, 1e-7f);
    CHECK_CLOSE(reduced[1], 5.0f, 1e-7f);

    memset(attributes, 0, sizeof(attributes));
    attributes[0].key = CAMPP_ATTR_KERNEL_SHAPE;
    attributes[0].type = CAMPP_ATTR_VALUE_INT;
    attributes[0].count = 1u;
    attributes[0].ints[0] = 3;
    attributes[1].key = CAMPP_ATTR_PADS;
    attributes[1].type = CAMPP_ATTR_VALUE_INT;
    attributes[1].count = 2u;
    attributes[1].ints[0] = 1;
    attributes[1].ints[1] = 1;
    attributes[2].key = CAMPP_ATTR_STRIDES;
    attributes[2].type = CAMPP_ATTR_VALUE_INT;
    attributes[2].count = 1u;
    attributes[2].ints[0] = 2;
    attributes[3].key = CAMPP_ATTR_CEIL_MODE;
    attributes[3].type = CAMPP_ATTR_VALUE_INT;
    attributes[3].count = 1u;
    attributes[3].ints[0] = 0;
    attributes[4].key = CAMPP_ATTR_COUNT_INCLUDE_PAD;
    attributes[4].type = CAMPP_ATTR_VALUE_INT;
    attributes[4].count = 1u;
    attributes[4].ints[0] = 0;
    init_view(
        &inputs[0], pool_input, CAMPP_DTYPE_FLOAT32, 3u,
        pool_input_dims);
    init_view(
        &outputs[0], pooled, CAMPP_DTYPE_FLOAT32, 3u,
        pool_output_dims);
    init_model_and_op(
        &model, &op, CAMPP_OP_AVERAGE_POOL, attribute_buffer, attributes, 5u);
    CHECK_STATUS(
        campp_reference_average_pool(
            &model, &op, inputs, 1u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_CLOSE(pooled[0], 1.5f, 1e-7f);
    CHECK_CLOSE(pooled[1], 3.0f, 1e-7f);
    CHECK_CLOSE(pooled[2], 4.5f, 1e-7f);

    attributes[0].ints[0] = 4;
    attributes[1].ints[0] = 0;
    attributes[1].ints[1] = 0;
    attributes[2].ints[0] = 4;
    attributes[3].ints[0] = 1;
    attributes[4].ints[0] = 1;
    init_view(
        &outputs[0], pooled_with_ceil, CAMPP_DTYPE_FLOAT32, 3u,
        pool_ceil_output_dims);
    init_model_and_op(
        &model, &op, CAMPP_OP_AVERAGE_POOL, attribute_buffer, attributes, 5u);
    CHECK_STATUS(
        campp_reference_average_pool(
            &model, &op, inputs, 1u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_CLOSE(pooled_with_ceil[0], 2.5f, 1e-7f);
    CHECK_CLOSE(pooled_with_ceil[1], 1.25f, 1e-7f);

    memset(attributes, 0, sizeof(attributes));
    attributes[0].key = CAMPP_ATTR_EPSILON;
    attributes[0].type = CAMPP_ATTR_VALUE_FLOAT;
    attributes[0].count = 1u;
    attributes[0].floats[0] = 0.0;
    init_view(&inputs[0], batch_input, CAMPP_DTYPE_FLOAT32, 3u, batch_dims);
    init_view(&inputs[1], scale, CAMPP_DTYPE_FLOAT32, 1u, channel_dims);
    init_view(&inputs[2], bias, CAMPP_DTYPE_FLOAT32, 1u, channel_dims);
    init_view(&inputs[3], mean, CAMPP_DTYPE_FLOAT32, 1u, channel_dims);
    init_view(&inputs[4], variance, CAMPP_DTYPE_FLOAT32, 1u, channel_dims);
    init_view(&outputs[0], normalized, CAMPP_DTYPE_FLOAT32, 3u, batch_dims);
    init_model_and_op(
        &model, &op, CAMPP_OP_BATCH_NORMALIZATION, attribute_buffer,
        attributes, 1u);
    CHECK_STATUS(
        campp_reference_batch_normalization(
            &model, &op, inputs, 5u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_CLOSE(normalized[0], 1.0f, 1e-7f);
    CHECK_CLOSE(normalized[1], 3.0f, 1e-7f);
    CHECK_CLOSE(normalized[2], -1.0f, 1e-7f);
    CHECK_CLOSE(normalized[3], 0.0f, 1e-7f);
    return 0;
}

int main(void)
{
    if (test_relu_and_sigmoid() != 0) return 1;
    if (test_elementwise_broadcasting() != 0) return 1;
    if (test_reshape_squeeze_unsqueeze() != 0) return 1;
    if (test_transpose_concat_expand_slice() != 0) return 1;
    if (test_reduction_pool_and_batch_norm() != 0) return 1;
    puts("reference operator unit tests: PASS");
    return 0;
}
