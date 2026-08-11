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

typedef struct IntegerAttribute {
    uint16_t key;
    uint8_t count;
    int64_t values[4];
} IntegerAttribute;

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

static uint32_t encode_integer_attributes(
    uint8_t *buffer, const IntegerAttribute *attributes,
    uint32_t attribute_count)
{
    uint32_t cursor = CAMPP_ATTRIBUTE_BLOCK_HEADER_SIZE;
    uint32_t attribute_index;
    for (attribute_index = 0u; attribute_index < attribute_count;
         ++attribute_index) {
        uint8_t value_index;
        write_u16(buffer + cursor, attributes[attribute_index].key);
        buffer[cursor + 2u] = CAMPP_ATTR_VALUE_INT;
        buffer[cursor + 3u] = attributes[attribute_index].count;
        write_u32(buffer + cursor + 4u, 0u);
        cursor += CAMPP_ATTRIBUTE_RECORD_HEADER_SIZE;
        for (value_index = 0u;
             value_index < attributes[attribute_index].count; ++value_index) {
            write_u64(
                buffer + cursor,
                (uint64_t)attributes[attribute_index].values[value_index]);
            cursor += CAMPP_ATTRIBUTE_VALUE_SIZE;
        }
    }
    write_u32(buffer, attribute_count);
    write_u32(buffer + 4u, cursor);
    return cursor;
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

static void init_qconv_model(
    CamppRuntimeModel *model, CamppOperatorDescriptor *op,
    uint8_t *attribute_buffer, const IntegerAttribute *attributes,
    uint32_t attribute_count)
{
    memset(model, 0, sizeof(*model));
    memset(op, 0, sizeof(*op));
    op->opcode = CAMPP_OP_QLINEAR_CONV;
    op->attribute_size = encode_integer_attributes(
        attribute_buffer, attributes, attribute_count);
    model->attribute_section = attribute_buffer;
    model->attribute_section_size = op->attribute_size;
}

static int test_quantize_and_dequantize(void)
{
    const uint32_t dims[1] = {6u};
    float input[6] = {-1.25f, -0.75f, -0.25f, 0.25f, 0.75f, 1.25f};
    float scale = 0.5f;
    uint8_t zero = 127u;
    uint8_t quantized[6] = {0};
    float dequantized[6] = {0};
    const uint8_t expected[6] = {125u, 125u, 127u, 127u, 129u, 129u};
    CamppTensorView inputs[3];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;

    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    op.opcode = CAMPP_OP_QUANTIZE_LINEAR;
    init_view(&inputs[0], input, CAMPP_DTYPE_FLOAT32, 1u, dims);
    init_view(&inputs[1], &scale, CAMPP_DTYPE_FLOAT32, 0u, dims);
    init_view(&inputs[2], &zero, CAMPP_DTYPE_UINT8, 0u, dims);
    init_view(&outputs[0], quantized, CAMPP_DTYPE_UINT8, 1u, dims);
    CHECK_STATUS(
        campp_reference_quantize_linear(
            &model, &op, inputs, 3u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(quantized, expected, sizeof(expected)) == 0);

    op.opcode = CAMPP_OP_DEQUANTIZE_LINEAR;
    inputs[0] = outputs[0];
    init_view(&outputs[0], dequantized, CAMPP_DTYPE_FLOAT32, 1u, dims);
    CHECK_STATUS(
        campp_reference_dequantize_linear(
            &model, &op, inputs, 3u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(dequantized[0] == -1.0f && dequantized[1] == -1.0f &&
               dequantized[2] == 0.0f && dequantized[3] == 0.0f &&
               dequantized[4] == 1.0f && dequantized[5] == 1.0f);
    return 0;
}

static int test_per_axis_quantization(void)
{
    const uint32_t data_dims[3] = {1u, 2u, 2u};
    const uint32_t channel_dims[1] = {2u};
    float input[4] = {1.0f, 2.0f, 1.0f, 2.0f};
    float scales[2] = {1.0f, 2.0f};
    uint8_t zeros[2] = {0u, 0u};
    uint8_t output[4] = {0u, 0u, 0u, 0u};
    const uint8_t expected[4] = {1u, 2u, 0u, 1u};
    CamppTensorView inputs[3];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_buffer[32];
    const IntegerAttribute axis = {CAMPP_ATTR_AXIS, 1u, {1, 0, 0, 0}};

    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    op.opcode = CAMPP_OP_QUANTIZE_LINEAR;
    op.attribute_size = encode_integer_attributes(
        attribute_buffer, &axis, 1u);
    model.attribute_section = attribute_buffer;
    model.attribute_section_size = op.attribute_size;
    init_view(&inputs[0], input, CAMPP_DTYPE_FLOAT32, 3u, data_dims);
    init_view(&inputs[1], scales, CAMPP_DTYPE_FLOAT32, 1u, channel_dims);
    init_view(&inputs[2], zeros, CAMPP_DTYPE_UINT8, 1u, channel_dims);
    init_view(&outputs[0], output, CAMPP_DTYPE_UINT8, 3u, data_dims);
    CHECK_STATUS(
        campp_reference_quantize_linear(
            &model, &op, inputs, 3u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(output, expected, sizeof(expected)) == 0);
    return 0;
}

static int test_qconv_1x1_bias_per_channel(void)
{
    const uint32_t x_dims[3] = {1u, 2u, 3u};
    const uint32_t w_dims[3] = {2u, 2u, 1u};
    const uint32_t channel_dims[1] = {2u};
    uint8_t x[6] = {1, 2, 3, 4, 5, 6};
    int8_t weight[4] = {1, 2, -1, 1};
    float x_scale = 1.0f;
    uint8_t x_zero = 0u;
    float weight_scale[2] = {1.0f, 0.5f};
    int8_t weight_zero[2] = {0, 0};
    float y_scale = 1.0f;
    uint8_t y_zero = 0u;
    int32_t bias[2] = {1, -1};
    uint8_t output[6] = {0};
    const uint8_t expected[6] = {10, 13, 16, 1, 1, 1};
    CamppTensorView inputs[9];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_buffer[256];
    const IntegerAttribute attributes[5] = {
        {CAMPP_ATTR_KERNEL_SHAPE, 1u, {1, 0, 0, 0}},
        {CAMPP_ATTR_PADS, 2u, {0, 0, 0, 0}},
        {CAMPP_ATTR_STRIDES, 1u, {1, 0, 0, 0}},
        {CAMPP_ATTR_DILATIONS, 1u, {1, 0, 0, 0}},
        {CAMPP_ATTR_GROUP, 1u, {1, 0, 0, 0}}
    };

    init_qconv_model(&model, &op, attribute_buffer, attributes, 5u);
    init_view(&inputs[0], x, CAMPP_DTYPE_UINT8, 3u, x_dims);
    init_view(&inputs[1], &x_scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[2], &x_zero, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&inputs[3], weight, CAMPP_DTYPE_INT8, 3u, w_dims);
    init_view(&inputs[4], weight_scale, CAMPP_DTYPE_FLOAT32, 1u, channel_dims);
    init_view(&inputs[5], weight_zero, CAMPP_DTYPE_INT8, 1u, channel_dims);
    init_view(&inputs[6], &y_scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[7], &y_zero, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&inputs[8], bias, CAMPP_DTYPE_INT32, 1u, channel_dims);
    init_view(&outputs[0], output, CAMPP_DTYPE_UINT8, 3u, x_dims);
    CHECK_STATUS(
        campp_reference_qlinear_conv(
            &model, &op, inputs, 9u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(output, expected, sizeof(expected)) == 0);
    return 0;
}

static int test_qconv_padding_stride(void)
{
    const uint32_t x_dims[3] = {1u, 1u, 5u};
    const uint32_t w_dims[3] = {1u, 1u, 3u};
    const uint32_t y_dims[3] = {1u, 1u, 3u};
    uint8_t x[5] = {1, 2, 3, 4, 5};
    int8_t weight[3] = {1, 1, 1};
    float scale = 1.0f;
    uint8_t zero_u8 = 0u;
    int8_t zero_i8 = 0;
    uint8_t output[3] = {0};
    const uint8_t expected[3] = {3, 9, 9};
    CamppTensorView inputs[8];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_buffer[256];
    const IntegerAttribute attributes[5] = {
        {CAMPP_ATTR_KERNEL_SHAPE, 1u, {3, 0, 0, 0}},
        {CAMPP_ATTR_PADS, 2u, {1, 1, 0, 0}},
        {CAMPP_ATTR_STRIDES, 1u, {2, 0, 0, 0}},
        {CAMPP_ATTR_DILATIONS, 1u, {1, 0, 0, 0}},
        {CAMPP_ATTR_GROUP, 1u, {1, 0, 0, 0}}
    };

    init_qconv_model(&model, &op, attribute_buffer, attributes, 5u);
    init_view(&inputs[0], x, CAMPP_DTYPE_UINT8, 3u, x_dims);
    init_view(&inputs[1], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[2], &zero_u8, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&inputs[3], weight, CAMPP_DTYPE_INT8, 3u, w_dims);
    init_view(&inputs[4], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[5], &zero_i8, CAMPP_DTYPE_INT8, 0u, x_dims);
    init_view(&inputs[6], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[7], &zero_u8, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&outputs[0], output, CAMPP_DTYPE_UINT8, 3u, y_dims);
    CHECK_STATUS(
        campp_reference_qlinear_conv(
            &model, &op, inputs, 8u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(output, expected, sizeof(expected)) == 0);
    return 0;
}

static int test_qconv_grouped(void)
{
    const uint32_t x_dims[3] = {1u, 4u, 2u};
    const uint32_t w_dims[3] = {4u, 2u, 1u};
    uint8_t x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int8_t weight[8] = {1, 0, 0, 1, 1, 0, 0, 1};
    float scale = 1.0f;
    uint8_t zero_u8 = 0u;
    int8_t zero_i8 = 0;
    uint8_t output[8] = {0};
    CamppTensorView inputs[8];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_buffer[256];
    const IntegerAttribute attributes[5] = {
        {CAMPP_ATTR_KERNEL_SHAPE, 1u, {1, 0, 0, 0}},
        {CAMPP_ATTR_PADS, 2u, {0, 0, 0, 0}},
        {CAMPP_ATTR_STRIDES, 1u, {1, 0, 0, 0}},
        {CAMPP_ATTR_DILATIONS, 1u, {1, 0, 0, 0}},
        {CAMPP_ATTR_GROUP, 1u, {2, 0, 0, 0}}
    };

    init_qconv_model(&model, &op, attribute_buffer, attributes, 5u);
    init_view(&inputs[0], x, CAMPP_DTYPE_UINT8, 3u, x_dims);
    init_view(&inputs[1], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[2], &zero_u8, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&inputs[3], weight, CAMPP_DTYPE_INT8, 3u, w_dims);
    init_view(&inputs[4], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[5], &zero_i8, CAMPP_DTYPE_INT8, 0u, x_dims);
    init_view(&inputs[6], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[7], &zero_u8, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&outputs[0], output, CAMPP_DTYPE_UINT8, 3u, x_dims);
    CHECK_STATUS(
        campp_reference_qlinear_conv(
            &model, &op, inputs, 8u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(output, x, sizeof(x)) == 0);
    return 0;
}

static int test_qconv_dilation(void)
{
    const uint32_t x_dims[3] = {1u, 1u, 5u};
    const uint32_t w_dims[3] = {1u, 1u, 2u};
    const uint32_t y_dims[3] = {1u, 1u, 3u};
    uint8_t x[5] = {1, 2, 3, 4, 5};
    int8_t weight[2] = {1, 1};
    float scale = 1.0f;
    uint8_t zero_u8 = 0u;
    int8_t zero_i8 = 0;
    uint8_t output[3] = {0};
    const uint8_t expected[3] = {4, 6, 8};
    CamppTensorView inputs[8];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_buffer[256];
    const IntegerAttribute attributes[5] = {
        {CAMPP_ATTR_KERNEL_SHAPE, 1u, {2, 0, 0, 0}},
        {CAMPP_ATTR_PADS, 2u, {0, 0, 0, 0}},
        {CAMPP_ATTR_STRIDES, 1u, {1, 0, 0, 0}},
        {CAMPP_ATTR_DILATIONS, 1u, {2, 0, 0, 0}},
        {CAMPP_ATTR_GROUP, 1u, {1, 0, 0, 0}}
    };

    init_qconv_model(&model, &op, attribute_buffer, attributes, 5u);
    init_view(&inputs[0], x, CAMPP_DTYPE_UINT8, 3u, x_dims);
    init_view(&inputs[1], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[2], &zero_u8, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&inputs[3], weight, CAMPP_DTYPE_INT8, 3u, w_dims);
    init_view(&inputs[4], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[5], &zero_i8, CAMPP_DTYPE_INT8, 0u, x_dims);
    init_view(&inputs[6], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[7], &zero_u8, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&outputs[0], output, CAMPP_DTYPE_UINT8, 3u, y_dims);
    CHECK_STATUS(
        campp_reference_qlinear_conv(
            &model, &op, inputs, 8u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(output, expected, sizeof(expected)) == 0);
    return 0;
}

static int test_qconv_2d(void)
{
    const uint32_t x_dims[4] = {1u, 1u, 3u, 3u};
    const uint32_t w_dims[4] = {1u, 1u, 2u, 2u};
    const uint32_t y_dims[4] = {1u, 1u, 2u, 2u};
    uint8_t x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int8_t weight[4] = {1, 1, 1, 1};
    float scale = 1.0f;
    uint8_t zero_u8 = 0u;
    int8_t zero_i8 = 0;
    uint8_t output[4] = {0};
    const uint8_t expected[4] = {12, 16, 24, 28};
    CamppTensorView inputs[8];
    CamppTensorView outputs[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_buffer[256];
    const IntegerAttribute attributes[5] = {
        {CAMPP_ATTR_KERNEL_SHAPE, 2u, {2, 2, 0, 0}},
        {CAMPP_ATTR_PADS, 4u, {0, 0, 0, 0}},
        {CAMPP_ATTR_STRIDES, 2u, {1, 1, 0, 0}},
        {CAMPP_ATTR_DILATIONS, 2u, {1, 1, 0, 0}},
        {CAMPP_ATTR_GROUP, 1u, {1, 0, 0, 0}}
    };

    init_qconv_model(&model, &op, attribute_buffer, attributes, 5u);
    init_view(&inputs[0], x, CAMPP_DTYPE_UINT8, 4u, x_dims);
    init_view(&inputs[1], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[2], &zero_u8, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&inputs[3], weight, CAMPP_DTYPE_INT8, 4u, w_dims);
    init_view(&inputs[4], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[5], &zero_i8, CAMPP_DTYPE_INT8, 0u, x_dims);
    init_view(&inputs[6], &scale, CAMPP_DTYPE_FLOAT32, 0u, x_dims);
    init_view(&inputs[7], &zero_u8, CAMPP_DTYPE_UINT8, 0u, x_dims);
    init_view(&outputs[0], output, CAMPP_DTYPE_UINT8, 4u, y_dims);
    CHECK_STATUS(
        campp_reference_qlinear_conv(
            &model, &op, inputs, 8u, outputs, 1u, NULL, 0u),
        CAMPP_STATUS_OK);
    CHECK_TRUE(memcmp(output, expected, sizeof(expected)) == 0);
    return 0;
}

int main(void)
{
    if (test_quantize_and_dequantize() != 0) return 1;
    if (test_per_axis_quantization() != 0) return 1;
    if (test_qconv_1x1_bias_per_channel() != 0) return 1;
    if (test_qconv_padding_stride() != 0) return 1;
    if (test_qconv_grouped() != 0) return 1;
    if (test_qconv_dilation() != 0) return 1;
    if (test_qconv_2d() != 0) return 1;
    puts("quantization and QLinearConv unit tests: PASS");
    return 0;
}
