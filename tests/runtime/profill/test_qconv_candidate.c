#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_aarch64/fused_kernels/fused_quant_qconv_internal.h"
#include "fused_input_quant_neon.h"
#include "fused_quant_qconv_candidate.h"
#include "internal/runtime_model.h"
#include "qconv_candidate.h"

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

static uint32_t encode_attributes(
    uint8_t *buffer, const IntegerAttribute *attributes, uint32_t count)
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

static uint64_t element_count(uint8_t rank, const uint32_t *dimensions)
{
    uint64_t count = 1u;
    uint8_t axis;
    for (axis = 0u; axis < rank; ++axis) count *= dimensions[axis];
    return count;
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
    view->logical_byte_size =
        rank == 0u ? campp_dtype_byte_size(dtype) : stride;
    view->storage_span_bytes = view->logical_byte_size;
}

static void init_channel_packed_view(
    CamppTensorView *view, void *data, uint8_t dtype, uint8_t rank,
    const uint32_t *dimensions)
{
    const uint32_t channels = dimensions[1];
    const uint32_t element_size = campp_dtype_byte_size(dtype);
    uint8_t axis;
    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = dtype;
    view->rank = rank;
    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        view->dimensions[axis] = axis < rank ? dimensions[axis] : 1u;
    }
    view->byte_strides[1] = element_size;
    if (rank == 3u) {
        view->byte_strides[2] = channels * element_size;
        view->byte_strides[0] =
            channels * dimensions[2] * element_size;
    } else {
        view->byte_strides[3] = channels * element_size;
        view->byte_strides[2] =
            channels * dimensions[3] * element_size;
        view->byte_strides[0] =
            channels * dimensions[2] * dimensions[3] * element_size;
    }
    view->logical_byte_size =
        element_count(rank, dimensions) * element_size;
    view->storage_span_bytes = view->logical_byte_size;
}

static size_t pack_o4i4(
    const int8_t *logical, uint32_t outputs, uint32_t inputs,
    uint32_t kernel_elements, const int8_t *zero_points,
    uint8_t *packed)
{
    const uint32_t output_blocks = (outputs + 3u) / 4u;
    const uint32_t input_blocks = (inputs + 3u) / 4u;
    size_t cursor = 0u;
    uint32_t output_block;
    for (output_block = 0u; output_block < output_blocks; ++output_block) {
        uint32_t kernel;
        for (kernel = 0u; kernel < kernel_elements; ++kernel) {
            uint32_t input_block;
            for (input_block = 0u;
                 input_block < input_blocks; ++input_block) {
                uint32_t output_lane;
                for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                    const uint32_t output =
                        output_block * 4u + output_lane;
                    uint32_t input_lane;
                    for (input_lane = 0u; input_lane < 4u; ++input_lane) {
                        const uint32_t input =
                            input_block * 4u + input_lane;
                        int8_t value = output < outputs
                            ? zero_points[output] : 0;
                        if (output < outputs && input < inputs) {
                            value = logical[
                                (output * inputs + input)
                                    * kernel_elements + kernel];
                        }
                        memcpy(packed + cursor, &value, sizeof(value));
                        cursor += 1u;
                    }
                }
            }
        }
    }
    return cursor;
}

static int run_fused_quantize_case(void)
{
    const uint32_t dimensions[3] = {1u, 4u, 5u};
    const float source[20] = {
        -40.0f, -31.875f, -0.625f, -0.5f, -0.375f,
        -0.125f, 0.0f, 0.125f, 0.375f, 0.5f,
        0.625f, 1.0f, 7.875f, 15.5f, 31.875f,
        40.0f, -2.0f, 2.0f, -3.125f, 3.125f
    };
    uint8_t baseline[20];
    uint8_t candidate[20];
    CamppTensorView input;
    CamppTensorView baseline_output;
    CamppTensorView candidate_output;

    memset(baseline, 0, sizeof(baseline));
    memset(candidate, 0, sizeof(candidate));
    init_channel_packed_view(
        &input, (void *)source, CAMPP_DTYPE_FLOAT32, 3u, dimensions);
    init_channel_packed_view(
        &baseline_output, baseline, CAMPP_DTYPE_UINT8, 3u, dimensions);
    init_channel_packed_view(
        &candidate_output, candidate, CAMPP_DTYPE_UINT8, 3u, dimensions);
    CHECK_STATUS(campp_fused_quantize_input_scalar(
        &input, &baseline_output, 0.25f, 127));
    CHECK_STATUS(campp_fused_input_quantize_neon(
        &input, &candidate_output, 0.25f, 127));
    CHECK_TRUE(memcmp(baseline, candidate, sizeof(baseline)) == 0);
    return 0;
}

static int run_case(uint8_t rank)
{
    const uint32_t input_dimensions_1d[3] = {1u, 5u, 7u};
    const uint32_t output_dimensions_1d[3] = {1u, 6u, 7u};
    const uint32_t weight_dimensions_1d[3] = {6u, 5u, 1u};
    const uint32_t input_dimensions_2d[4] = {1u, 4u, 3u, 5u};
    const uint32_t output_dimensions_2d[4] = {1u, 5u, 3u, 5u};
    const uint32_t weight_dimensions_2d[4] = {5u, 4u, 3u, 3u};
    const uint32_t *input_dimensions = rank == 3u
        ? input_dimensions_1d : input_dimensions_2d;
    const uint32_t *output_dimensions = rank == 3u
        ? output_dimensions_1d : output_dimensions_2d;
    const uint32_t *weight_dimensions = rank == 3u
        ? weight_dimensions_1d : weight_dimensions_2d;
    const uint32_t inputs = input_dimensions[1];
    const uint32_t outputs = output_dimensions[1];
    const uint32_t kernel_elements = rank == 3u ? 1u : 9u;
    const uint64_t input_elements = element_count(rank, input_dimensions);
    const uint64_t output_elements = element_count(rank, output_dimensions);
    uint8_t input_data[128];
    float input_float[128];
    int8_t logical_weight[256];
    uint8_t packed_weight[512];
    float input_scale = 0.25f;
    uint8_t input_zero = 127u;
    float weight_scale[8];
    int8_t weight_zero[8];
    float output_scale = 0.5f;
    uint8_t output_zero = 113u;
    int32_t bias[8];
    uint8_t baseline[128];
    uint8_t candidate[128];
    uint8_t scratch[128];
    CamppTensorView inputs_view[9];
    CamppTensorView fused_inputs_view[9];
    CamppTensorView baseline_output[1];
    CamppTensorView candidate_output[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    CamppTensorDescriptor tensor_descriptor;
    uint8_t attributes_buffer[256];
    IntegerAttribute attributes[5];
    size_t packed_size;
    uint64_t index;
    uint32_t output;
    CamppQconvCandidateMode mode;

    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    memset(attributes, 0, sizeof(attributes));
    attributes[0].key = CAMPP_ATTR_KERNEL_SHAPE;
    attributes[0].count = (uint8_t)(rank - 2u);
    attributes[0].values[0] = rank == 3u ? 1 : 3;
    attributes[0].values[1] = 3;
    attributes[1].key = CAMPP_ATTR_PADS;
    attributes[1].count = (uint8_t)((rank - 2u) * 2u);
    attributes[1].values[0] = rank == 3u ? 0 : 1;
    attributes[1].values[1] = rank == 3u ? 0 : 1;
    attributes[1].values[2] = 1;
    attributes[1].values[3] = 1;
    attributes[2].key = CAMPP_ATTR_STRIDES;
    attributes[2].count = (uint8_t)(rank - 2u);
    attributes[2].values[0] = 1;
    attributes[2].values[1] = 1;
    attributes[3].key = CAMPP_ATTR_DILATIONS;
    attributes[3].count = (uint8_t)(rank - 2u);
    attributes[3].values[0] = 1;
    attributes[3].values[1] = 1;
    attributes[4].key = CAMPP_ATTR_GROUP;
    attributes[4].count = 1u;
    attributes[4].values[0] = 1;
    op.opcode = CAMPP_OP_QLINEAR_CONV;
    op.kernel_id = CAMPP_AARCH64_PACKED_KERNEL_ID;
    op.attribute_size = encode_attributes(
        attributes_buffer, attributes, 5u);
    model.attribute_section = attributes_buffer;
    model.attribute_section_size = op.attribute_size;
    memset(&tensor_descriptor, 0, sizeof(tensor_descriptor));
    tensor_descriptor.dtype = CAMPP_DTYPE_FLOAT32;
    tensor_descriptor.storage_span_bytes = input_elements * sizeof(float);
    model.tensors = &tensor_descriptor;
    model.tensor_count = 1u;
    op.input_count = 9u;
    op.input_tensor_ids[0] = 0u;

    for (index = 0u; index < input_elements; ++index) {
        input_data[index] = (uint8_t)(119u + index % 19u);
        input_float[index] =
            ((float)input_data[index] - (float)input_zero) * input_scale;
    }
    for (output = 0u; output < outputs; ++output) {
        uint32_t input;
        weight_scale[output] = 0.125f + (float)output * 0.01f;
        weight_zero[output] = (int8_t)((int32_t)(output % 3u) - 1);
        bias[output] = (int32_t)output * 7 - 11;
        for (input = 0u; input < inputs; ++input) {
            uint32_t kernel;
            for (kernel = 0u; kernel < kernel_elements; ++kernel) {
                logical_weight[
                    (output * inputs + input) * kernel_elements + kernel] =
                    (int8_t)((int32_t)((output * 7u + input * 3u + kernel)
                                      % 13u) - 6);
            }
        }
    }
    packed_size = pack_o4i4(
        logical_weight, outputs, inputs, kernel_elements,
        weight_zero, packed_weight);
    memset(baseline, 0, sizeof(baseline));
    memset(candidate, 0, sizeof(candidate));

    init_channel_packed_view(
        &inputs_view[0], input_data, CAMPP_DTYPE_UINT8,
        rank, input_dimensions);
    init_contiguous_view(
        &inputs_view[1], &input_scale, CAMPP_DTYPE_FLOAT32, 0u,
        input_dimensions);
    init_contiguous_view(
        &inputs_view[2], &input_zero, CAMPP_DTYPE_UINT8, 0u,
        input_dimensions);
    init_contiguous_view(
        &inputs_view[3], packed_weight, CAMPP_DTYPE_INT8,
        rank, weight_dimensions);
    inputs_view[3].flags |= CAMPP_TENSOR_FLAG_PACKED_QCONV_O4I4;
    inputs_view[3].storage_span_bytes = packed_size;
    init_contiguous_view(
        &inputs_view[4], weight_scale, CAMPP_DTYPE_FLOAT32, 1u,
        &output_dimensions[1]);
    init_contiguous_view(
        &inputs_view[5], weight_zero, CAMPP_DTYPE_INT8, 1u,
        &output_dimensions[1]);
    init_contiguous_view(
        &inputs_view[6], &output_scale, CAMPP_DTYPE_FLOAT32, 0u,
        input_dimensions);
    init_contiguous_view(
        &inputs_view[7], &output_zero, CAMPP_DTYPE_UINT8, 0u,
        input_dimensions);
    init_contiguous_view(
        &inputs_view[8], bias, CAMPP_DTYPE_INT32, 1u,
        &output_dimensions[1]);
    init_channel_packed_view(
        &baseline_output[0], baseline, CAMPP_DTYPE_UINT8,
        rank, output_dimensions);
    init_channel_packed_view(
        &candidate_output[0], candidate, CAMPP_DTYPE_UINT8,
        rank, output_dimensions);
    memcpy(fused_inputs_view, inputs_view, sizeof(fused_inputs_view));
    init_channel_packed_view(
        &fused_inputs_view[0], input_float, CAMPP_DTYPE_FLOAT32,
        rank, input_dimensions);

    CHECK_STATUS(campp_aarch64_qlinear_conv_o4i4(
        &model, &op, inputs_view, 9u, baseline_output, 1u, NULL, 0u));
    for (mode = CAMPP_QCONV_CANDIDATE_ADDRESS;
         mode <= CAMPP_QCONV_CANDIDATE_V4;
         mode = (CamppQconvCandidateMode)(mode + 1)) {
        const CamppKernelEntry *entry = campp_qconv_candidate_entry(mode);
        CHECK_TRUE(entry != NULL);
        memset(candidate, 0, sizeof(candidate));
        CHECK_STATUS(entry->run(
            &model, &op, inputs_view, 9u,
            candidate_output, 1u, NULL, 0u));
        CHECK_TRUE(memcmp(
            baseline, candidate, (size_t)output_elements) == 0);
    }
    memset(baseline, 0, sizeof(baseline));
    CHECK_STATUS(campp_fused_quant_qlinear_conv_o4i4(
        &model, &op, fused_inputs_view, 9u, baseline_output, 1u,
        scratch, (size_t)input_elements));
    {
        CamppFusedQconvCandidateMode fused_mode;
        for (fused_mode = CAMPP_FUSED_QCONV_CANDIDATE_MAC;
             fused_mode <= CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_V4;
             fused_mode = (CamppFusedQconvCandidateMode)(fused_mode + 1)) {
            const CamppKernelEntry *entry =
                campp_fused_qconv_candidate_entry(fused_mode);
            CHECK_TRUE(entry != NULL);
            memset(candidate, 0, sizeof(candidate));
            memset(scratch, 0, sizeof(scratch));
            CHECK_STATUS(entry->run(
                &model, &op, fused_inputs_view, 9u,
                candidate_output, 1u, scratch, (size_t)input_elements));
            CHECK_TRUE(memcmp(
                baseline, candidate, (size_t)output_elements) == 0);
        }
    }
    return 0;
}

int main(void)
{
    CHECK_TRUE(run_fused_quantize_case() == 0);
    CHECK_TRUE(run_case(3u) == 0);
    CHECK_TRUE(run_case(4u) == 0);
    CHECK_TRUE(
        strcmp(
            campp_qconv_candidate_mode_name(
                CAMPP_QCONV_CANDIDATE_MAC_FIXED),
            "mac_fixed") == 0);
    CHECK_TRUE(
        strcmp(
            campp_qconv_candidate_mode_name(CAMPP_QCONV_CANDIDATE_V4),
            "v4") == 0);
    CHECK_TRUE(
        strcmp(
            campp_fused_qconv_candidate_mode_name(
                CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED),
            "combined_fixed") == 0);
    CHECK_TRUE(
        strcmp(
            campp_fused_qconv_candidate_mode_name(
                CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_V4),
            "combined_v4") == 0);
    puts("QConv and fused QConv optimization candidates: PASS");
    return 0;
}
