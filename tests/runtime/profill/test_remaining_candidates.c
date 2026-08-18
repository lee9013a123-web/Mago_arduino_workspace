#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_reference/reference_kernels.h"
#include "remaining_candidate.h"
#include "internal/runtime_model.h"

#define CHECK_TRUE(condition)                                           \
    do {                                                                \
        if (!(condition)) {                                             \
            fprintf(stderr, "CHECK failed at line %d: %s\n",          \
                    __LINE__, #condition);                              \
            return 1;                                                   \
        }                                                               \
    } while (0)

#define CHECK_STATUS(expression)                                       \
    do {                                                                \
        const CamppStatus actual = (expression);                        \
        if (actual != CAMPP_STATUS_OK) {                                \
            fprintf(stderr, "STATUS failed at line %d: %s\n",         \
                    __LINE__, campp_status_name(actual));               \
            return 1;                                                   \
        }                                                               \
    } while (0)

typedef struct TestAttribute {
    uint16_t key;
    uint8_t count;
    int64_t values[4];
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
    uint8_t *buffer, const TestAttribute *attributes, uint32_t count)
{
    uint32_t cursor = CAMPP_ATTRIBUTE_BLOCK_HEADER_SIZE;
    uint32_t attribute;
    for (attribute = 0u; attribute < count; ++attribute) {
        uint8_t value;
        write_u16(buffer + cursor, attributes[attribute].key);
        buffer[cursor + 2u] = CAMPP_ATTR_VALUE_INT;
        buffer[cursor + 3u] = attributes[attribute].count;
        write_u32(buffer + cursor + 4u, 0u);
        cursor += CAMPP_ATTRIBUTE_RECORD_HEADER_SIZE;
        for (value = 0u; value < attributes[attribute].count; ++value) {
            write_u64(
                buffer + cursor,
                (uint64_t)attributes[attribute].values[value]);
            cursor += CAMPP_ATTRIBUTE_VALUE_SIZE;
        }
    }
    write_u32(buffer, count);
    write_u32(buffer + 4u, cursor);
    return cursor;
}

static void init_model_op(
    CamppRuntimeModel *model, CamppOperatorDescriptor *op,
    uint16_t opcode, uint16_t kernel_id, uint8_t input_count,
    uint8_t *attribute_buffer, const TestAttribute *attributes,
    uint32_t attribute_count)
{
    memset(model, 0, sizeof(*model));
    memset(op, 0, sizeof(*op));
    op->opcode = opcode;
    op->kernel_id = kernel_id;
    op->input_count = input_count;
    op->output_count = 1u;
    if (attribute_count != 0u) {
        op->attribute_size = encode_attributes(
            attribute_buffer, attributes, attribute_count);
        model->attribute_section = attribute_buffer;
        model->attribute_section_size = op->attribute_size;
    }
}

static void init_contiguous_view(
    CamppTensorView *view, void *data, uint8_t dtype, uint8_t rank,
    const uint32_t *dimensions)
{
    uint32_t stride = campp_dtype_byte_size(dtype);
    uint8_t axis;
    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = dtype;
    view->rank = rank;
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

static void init_packed_view(
    CamppTensorView *view, void *data, uint8_t dtype, uint8_t rank,
    const uint32_t *dimensions)
{
    const uint32_t element = campp_dtype_byte_size(dtype);
    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = dtype;
    view->rank = rank;
    memcpy(view->dimensions, dimensions, rank * sizeof(dimensions[0]));
    view->byte_strides[1] = element;
    if (rank == 2u) {
        view->byte_strides[0] = dimensions[1] * element;
    } else if (rank == 3u) {
        view->byte_strides[2] = dimensions[1] * element;
        view->byte_strides[0] = dimensions[2] * view->byte_strides[2];
    } else {
        view->byte_strides[3] = dimensions[1] * element;
        view->byte_strides[2] = dimensions[3] * view->byte_strides[3];
        view->byte_strides[0] = dimensions[2] * view->byte_strides[2];
    }
    view->logical_byte_size = view->byte_strides[0] * dimensions[0];
    view->storage_span_bytes = view->logical_byte_size;
}

static int compare_kernel(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    CamppKernelRun reference, const CamppTensorView *inputs,
    uint8_t input_count, CamppTensorView *baseline,
    CamppTensorView *candidate, size_t output_bytes)
{
    const CamppKernelEntry *entry = campp_remaining_candidate_entry(
        CAMPP_REMAINING_CANDIDATE_OPTIMIZED, op->opcode, op->kernel_id);
    CHECK_TRUE(entry != NULL);
    memset(baseline->data, 0xA5, output_bytes);
    memset(candidate->data, 0x5A, output_bytes);
    CHECK_STATUS(reference(
        model, op, inputs, input_count, baseline, 1u, NULL, 0u));
    CHECK_STATUS(entry->run(
        model, op, inputs, input_count, candidate, 1u, NULL, 0u));
    CHECK_TRUE(memcmp(baseline->data, candidate->data, output_bytes) == 0);
    return 0;
}

static void fill_f32(float *data, uint32_t count, float offset)
{
    uint32_t index;
    for (index = 0u; index < count; ++index) {
        data[index] = ((float)((int32_t)(index % 17u) - 8) * 0.125f) + offset;
    }
}

static int test_add_and_relu(void)
{
    const uint32_t full_dims[3] = {1u, 8u, 5u};
    const uint32_t bias_dims[3] = {1u, 8u, 1u};
    float bias[8];
    float input[40];
    float baseline[40];
    float candidate[40];
    CamppTensorView add_inputs[2];
    CamppTensorView relu_input;
    CamppTensorView baseline_view;
    CamppTensorView candidate_view;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attributes[8];
    fill_f32(bias, 8u, 0.25f);
    fill_f32(input, 40u, -0.5f);
    init_packed_view(&add_inputs[0], bias, CAMPP_DTYPE_FLOAT32, 3u, bias_dims);
    init_packed_view(&add_inputs[1], input, CAMPP_DTYPE_FLOAT32, 3u, full_dims);
    init_packed_view(
        &baseline_view, baseline, CAMPP_DTYPE_FLOAT32, 3u, full_dims);
    init_packed_view(
        &candidate_view, candidate, CAMPP_DTYPE_FLOAT32, 3u, full_dims);
    init_model_op(
        &model, &op, CAMPP_OP_ADD, CAMPP_AARCH64_PACKED_KERNEL_ID,
        2u, attributes, NULL, 0u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_reference_add, add_inputs, 2u,
        &baseline_view, &candidate_view, sizeof(baseline)) == 0);
    {
        const uint32_t special_bits[4] = {
            UINT32_C(0x80000000), UINT32_C(0x7fc12345),
            UINT32_C(0xffc23456), UINT32_C(0xff800000)
        };
        memcpy(input, special_bits, sizeof(special_bits));
    }
    init_packed_view(&relu_input, input, CAMPP_DTYPE_FLOAT32, 3u, full_dims);
    init_model_op(
        &model, &op, CAMPP_OP_RELU, CAMPP_AARCH64_PACKED_KERNEL_ID,
        1u, attributes, NULL, 0u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_reference_relu, &relu_input, 1u,
        &baseline_view, &candidate_view, sizeof(baseline)) == 0);
    return 0;
}

static int test_quantize(void)
{
    const uint32_t dims[3] = {1u, 8u, 5u};
    float input[40];
    float scale = 0.125f;
    uint8_t zero = 121u;
    uint8_t baseline[40];
    uint8_t candidate[40];
    CamppTensorView inputs[3];
    CamppTensorView baseline_view;
    CamppTensorView candidate_view;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attributes[8];
    fill_f32(input, 40u, 0.0625f);
    init_packed_view(&inputs[0], input, CAMPP_DTYPE_FLOAT32, 3u, dims);
    init_contiguous_view(&inputs[1], &scale, CAMPP_DTYPE_FLOAT32, 0u, NULL);
    init_contiguous_view(&inputs[2], &zero, CAMPP_DTYPE_UINT8, 0u, NULL);
    init_packed_view(&baseline_view, baseline, CAMPP_DTYPE_UINT8, 3u, dims);
    init_packed_view(&candidate_view, candidate, CAMPP_DTYPE_UINT8, 3u, dims);
    init_model_op(
        &model, &op, CAMPP_OP_QUANTIZE_LINEAR,
        CAMPP_AARCH64_PACKED_KERNEL_ID, 3u, attributes, NULL, 0u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_reference_quantize_linear, inputs, 3u,
        &baseline_view, &candidate_view, sizeof(baseline)) == 0);
    return 0;
}

static int test_expand_and_slice(void)
{
    const uint32_t expand_input_dims[4] = {1u, 8u, 1u, 1u};
    const uint32_t expand_output_dims[4] = {1u, 8u, 1u, 7u};
    const uint32_t slice_output_dims[3] = {1u, 8u, 3u};
    const uint32_t scalar_dims[1] = {1u};
    int64_t expand_shape[4] = {1, 8, 1, 7};
    int64_t start = 2;
    int64_t end = 5;
    int64_t axis = 2;
    int64_t step = 1;
    float input[8];
    float expanded[56];
    float baseline[56];
    float candidate[56];
    CamppTensorView expand_inputs[2];
    CamppTensorView slice_inputs[5];
    CamppTensorView baseline_view;
    CamppTensorView candidate_view;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attributes[8];
    fill_f32(input, 8u, 0.0f);
    init_packed_view(
        &expand_inputs[0], input, CAMPP_DTYPE_FLOAT32, 4u,
        expand_input_dims);
    {
        const uint32_t shape_dims[1] = {4u};
        init_contiguous_view(
            &expand_inputs[1], expand_shape, CAMPP_DTYPE_INT64, 1u,
            shape_dims);
    }
    init_packed_view(
        &baseline_view, baseline, CAMPP_DTYPE_FLOAT32, 4u,
        expand_output_dims);
    init_packed_view(
        &candidate_view, candidate, CAMPP_DTYPE_FLOAT32, 4u,
        expand_output_dims);
    init_model_op(
        &model, &op, CAMPP_OP_EXPAND, CAMPP_AARCH64_PACKED_KERNEL_ID,
        2u, attributes, NULL, 0u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_reference_expand, expand_inputs, 2u,
        &baseline_view, &candidate_view, sizeof(baseline)) == 0);
    memcpy(expanded, baseline, sizeof(expanded));
    {
        const uint32_t slice_input_dims[3] = {1u, 8u, 7u};
        init_packed_view(
            &slice_inputs[0], expanded, CAMPP_DTYPE_FLOAT32, 3u,
            slice_input_dims);
    }
    init_contiguous_view(
        &slice_inputs[1], &start, CAMPP_DTYPE_INT64, 1u, scalar_dims);
    init_contiguous_view(
        &slice_inputs[2], &end, CAMPP_DTYPE_INT64, 1u, scalar_dims);
    init_contiguous_view(
        &slice_inputs[3], &axis, CAMPP_DTYPE_INT64, 1u, scalar_dims);
    init_contiguous_view(
        &slice_inputs[4], &step, CAMPP_DTYPE_INT64, 1u, scalar_dims);
    init_packed_view(
        &baseline_view, baseline, CAMPP_DTYPE_FLOAT32, 3u,
        slice_output_dims);
    init_packed_view(
        &candidate_view, candidate, CAMPP_DTYPE_FLOAT32, 3u,
        slice_output_dims);
    init_model_op(
        &model, &op, CAMPP_OP_SLICE, CAMPP_AARCH64_PACKED_KERNEL_ID,
        5u, attributes, NULL, 0u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_reference_slice, slice_inputs, 5u,
        &baseline_view, &candidate_view, 24u * sizeof(float)) == 0);
    return 0;
}

static int test_reductions(void)
{
    const uint32_t input_dims[3] = {1u, 8u, 5u};
    const uint32_t output_dims[3] = {1u, 8u, 1u};
    const TestAttribute reduce_attributes[2] = {
        {CAMPP_ATTR_AXES, 1u, {2, 0, 0, 0}},
        {CAMPP_ATTR_KEEPDIMS, 1u, {1, 0, 0, 0}}
    };
    const TestAttribute pool_attributes[5] = {
        {CAMPP_ATTR_KERNEL_SHAPE, 1u, {5, 0, 0, 0}},
        {CAMPP_ATTR_PADS, 2u, {0, 0, 0, 0}},
        {CAMPP_ATTR_STRIDES, 1u, {1, 0, 0, 0}},
        {CAMPP_ATTR_CEIL_MODE, 1u, {0, 0, 0, 0}},
        {CAMPP_ATTR_COUNT_INCLUDE_PAD, 1u, {0, 0, 0, 0}}
    };
    float input[40];
    float baseline[8];
    float candidate[8];
    CamppTensorView input_view;
    CamppTensorView baseline_view;
    CamppTensorView candidate_view;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attributes[160];
    fill_f32(input, 40u, 0.25f);
    init_packed_view(&input_view, input, CAMPP_DTYPE_FLOAT32, 3u, input_dims);
    init_packed_view(
        &baseline_view, baseline, CAMPP_DTYPE_FLOAT32, 3u, output_dims);
    init_packed_view(
        &candidate_view, candidate, CAMPP_DTYPE_FLOAT32, 3u, output_dims);
    init_model_op(
        &model, &op, CAMPP_OP_REDUCE_MEAN, CAMPP_AARCH64_PACKED_KERNEL_ID,
        1u, attributes, reduce_attributes, 2u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_reference_reduce_mean, &input_view, 1u,
        &baseline_view, &candidate_view, sizeof(baseline)) == 0);
    init_model_op(
        &model, &op, CAMPP_OP_AVERAGE_POOL, CAMPP_AARCH64_PACKED_KERNEL_ID,
        1u, attributes, pool_attributes, 5u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_reference_average_pool, &input_view, 1u,
        &baseline_view, &candidate_view, sizeof(baseline)) == 0);
    return 0;
}

static int test_sigmoid_mul(void)
{
    const uint32_t dims[3] = {1u, 8u, 5u};
    uint8_t quantized[40];
    float scale = 0.0625f;
    uint8_t zero = 119u;
    float other[40];
    float baseline[40];
    float candidate[40];
    CamppTensorView inputs[4];
    CamppTensorView baseline_view;
    CamppTensorView candidate_view;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attributes[8];
    uint32_t index;
    for (index = 0u; index < 40u; ++index) {
        quantized[index] = (uint8_t)(100u + index % 31u);
    }
    fill_f32(other, 40u, 1.25f);
    init_packed_view(&inputs[0], quantized, CAMPP_DTYPE_UINT8, 3u, dims);
    init_contiguous_view(&inputs[1], &scale, CAMPP_DTYPE_FLOAT32, 0u, NULL);
    init_contiguous_view(&inputs[2], &zero, CAMPP_DTYPE_UINT8, 0u, NULL);
    init_packed_view(&inputs[3], other, CAMPP_DTYPE_FLOAT32, 3u, dims);
    init_packed_view(
        &baseline_view, baseline, CAMPP_DTYPE_FLOAT32, 3u, dims);
    init_packed_view(
        &candidate_view, candidate, CAMPP_DTYPE_FLOAT32, 3u, dims);
    init_model_op(
        &model, &op, CAMPP_OP_MUL, CAMPP_FUSION_EPILOGUE_KERNEL_ID,
        4u, attributes, NULL, 0u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_fused_dequant_sigmoid_mul, inputs, 4u,
        &baseline_view, &candidate_view, sizeof(baseline)) == 0);
    return 0;
}

static int test_reshape(void)
{
    const uint32_t input_dims[4] = {1u, 4u, 2u, 3u};
    const uint32_t output_dims[3] = {1u, 8u, 3u};
    const uint32_t shape_dims[1] = {3u};
    int64_t shape[3] = {0, -1, 3};
    float input[24];
    float baseline[24];
    float candidate[24];
    CamppTensorView inputs[2];
    CamppTensorView baseline_view;
    CamppTensorView candidate_view;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attributes[8];
    fill_f32(input, 24u, 0.0f);
    init_packed_view(&inputs[0], input, CAMPP_DTYPE_FLOAT32, 4u, input_dims);
    init_contiguous_view(
        &inputs[1], shape, CAMPP_DTYPE_INT64, 1u, shape_dims);
    init_packed_view(
        &baseline_view, baseline, CAMPP_DTYPE_FLOAT32, 3u, output_dims);
    init_packed_view(
        &candidate_view, candidate, CAMPP_DTYPE_FLOAT32, 3u, output_dims);
    init_model_op(
        &model, &op, CAMPP_OP_RESHAPE, CAMPP_AARCH64_PACKED_KERNEL_ID,
        2u, attributes, NULL, 0u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_reference_reshape, inputs, 2u,
        &baseline_view, &candidate_view, sizeof(baseline)) == 0);
    return 0;
}

static int test_statistics(void)
{
    const uint32_t input_dims[3] = {1u, 8u, 5u};
    const uint32_t output_dims[2] = {1u, 16u};
    float input[40];
    float multiplier = 1.0f;
    float divisor = 1.0f;
    float baseline[16];
    float candidate[16];
    CamppTensorView inputs[3];
    CamppTensorView baseline_view;
    CamppTensorView candidate_view;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attributes[8];
    fill_f32(input, 40u, 0.75f);
    init_packed_view(&inputs[0], input, CAMPP_DTYPE_FLOAT32, 3u, input_dims);
    init_contiguous_view(
        &inputs[1], &multiplier, CAMPP_DTYPE_FLOAT32, 0u, NULL);
    init_contiguous_view(
        &inputs[2], &divisor, CAMPP_DTYPE_FLOAT32, 0u, NULL);
    init_packed_view(
        &baseline_view, baseline, CAMPP_DTYPE_FLOAT32, 2u, output_dims);
    init_packed_view(
        &candidate_view, candidate, CAMPP_DTYPE_FLOAT32, 2u, output_dims);
    init_model_op(
        &model, &op, CAMPP_OP_CONCAT,
        CAMPP_FUSION_STATS_POOLING_KERNEL_ID, 3u,
        attributes, NULL, 0u);
    CHECK_TRUE(compare_kernel(
        &model, &op, campp_fused_statistics_pooling, inputs, 3u,
        &baseline_view, &candidate_view, sizeof(baseline)) == 0);
    return 0;
}

int main(void)
{
    CHECK_TRUE(test_add_and_relu() == 0);
    CHECK_TRUE(test_quantize() == 0);
    CHECK_TRUE(test_expand_and_slice() == 0);
    CHECK_TRUE(test_reductions() == 0);
    CHECK_TRUE(test_sigmoid_mul() == 0);
    CHECK_TRUE(test_reshape() == 0);
    CHECK_TRUE(test_statistics() == 0);
    puts("test_remaining_candidates: PASS");
    return 0;
}
