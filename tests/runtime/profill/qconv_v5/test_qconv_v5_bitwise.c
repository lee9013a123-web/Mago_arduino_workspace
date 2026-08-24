/* Bitwise gate for the QLinearConv v5 full-tile MAC paths.
 *
 * test_qconv_candidate.c compares every candidate mode against the packed
 * reference, but its shapes are too small to ever build a full 8-wide spatial
 * tile, so the v5 real 1x1 body, the 3x3 sliding body and the 3x3/Cin32 body
 * all fall through to the shared tail there.  This suite drives shapes that
 * actually enter each v5 body, including the stride variants where the
 * sliding-window address assumption does not hold. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "internal/runtime_model.h"
#include "qconv_candidate.h"

#define CHECK_TRUE(condition)                                           \
    do {                                                                \
        if (!(condition)) {                                             \
            fprintf(stderr, "CHECK failed at line %d: %s\n",            \
                    __LINE__, #condition);                              \
            return 1;                                                   \
        }                                                               \
    } while (0)

#define CHECK_STATUS(expression)                                        \
    do {                                                                \
        const CamppStatus actual = (expression);                        \
        if (actual != CAMPP_STATUS_OK) {                                \
            fprintf(stderr, "STATUS failed at line %d: %s\n",           \
                    __LINE__, campp_status_name(actual));               \
            return 1;                                                   \
        }                                                               \
    } while (0)

#define MAX_INPUT_ELEMENTS 32768u
#define MAX_OUTPUT_ELEMENTS 16384u
#define MAX_LOGICAL_WEIGHTS 16384u
#define MAX_PACKED_WEIGHTS 32768u
#define MAX_CHANNELS 64u

typedef struct IntegerAttribute {
    uint16_t key;
    uint8_t count;
    int64_t values[4];
} IntegerAttribute;

typedef struct ConvCase {
    const char *name;
    uint8_t rank;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t stride_height;
    uint32_t stride_width;
    uint32_t pad_height;
    uint32_t pad_width;
} ConvCase;

static uint8_t input_data[MAX_INPUT_ELEMENTS];
static int8_t logical_weight[MAX_LOGICAL_WEIGHTS];
static uint8_t packed_weight[MAX_PACKED_WEIGHTS];
static uint8_t baseline[MAX_OUTPUT_ELEMENTS];
static uint8_t candidate[MAX_OUTPUT_ELEMENTS];
static float weight_scale[MAX_CHANNELS];
static int8_t weight_zero[MAX_CHANNELS];
static int32_t bias[MAX_CHANNELS];
static uint8_t attributes_buffer[512];

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
        view->byte_strides[0] = channels * dimensions[2] * element_size;
    } else {
        view->byte_strides[3] = channels * element_size;
        view->byte_strides[2] = channels * dimensions[3] * element_size;
        view->byte_strides[0] =
            channels * dimensions[2] * dimensions[3] * element_size;
    }
    view->logical_byte_size =
        element_count(rank, dimensions) * element_size;
    view->storage_span_bytes = view->logical_byte_size;
}

static size_t pack_o4i4(
    const int8_t *logical, uint32_t outputs, uint32_t inputs,
    uint32_t kernel_elements, const int8_t *zero_points, uint8_t *packed)
{
    const uint32_t output_blocks = (outputs + 3u) / 4u;
    const uint32_t input_blocks = (inputs + 3u) / 4u;
    size_t cursor = 0u;
    uint32_t output_block;
    for (output_block = 0u; output_block < output_blocks; ++output_block) {
        uint32_t kernel;
        for (kernel = 0u; kernel < kernel_elements; ++kernel) {
            uint32_t input_block;
            for (input_block = 0u; input_block < input_blocks;
                 ++input_block) {
                uint32_t output_lane;
                for (output_lane = 0u; output_lane < 4u; ++output_lane) {
                    const uint32_t output = output_block * 4u + output_lane;
                    uint32_t input_lane;
                    for (input_lane = 0u; input_lane < 4u; ++input_lane) {
                        const uint32_t input = input_block * 4u + input_lane;
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

/* Compares the v5 candidate against the packed aarch64 reference and against
 * v4, which shares planning, requant and store with v5. */
static int run_conv_case(const ConvCase *test_case, int zero_weight_zero_point)
{
    const uint8_t rank = test_case->rank;
    const uint32_t kernel_elements = test_case->kernel_height *
        (rank == 3u ? 1u : test_case->kernel_width);
    const uint32_t output_height =
        (test_case->input_height + 2u * test_case->pad_height -
         test_case->kernel_height) / test_case->stride_height + 1u;
    const uint32_t output_width = rank == 3u ? 1u :
        (test_case->input_width + 2u * test_case->pad_width -
         test_case->kernel_width) / test_case->stride_width + 1u;
    uint32_t input_dimensions[4];
    uint32_t output_dimensions[4];
    uint32_t weight_dimensions[4];
    float input_scale = 0.25f;
    uint8_t input_zero = 127u;
    float output_scale = 0.5f;
    uint8_t output_zero = 113u;
    uint64_t input_elements;
    uint64_t output_elements;
    CamppTensorView inputs_view[9];
    CamppTensorView baseline_output[1];
    CamppTensorView candidate_output[1];
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    CamppTensorDescriptor tensor_descriptor;
    IntegerAttribute attributes[5];
    size_t packed_size;
    uint64_t index;
    uint32_t output;
    CamppQconvCandidateMode mode;

    input_dimensions[0] = 1u;
    input_dimensions[1] = test_case->input_channels;
    input_dimensions[2] = test_case->input_height;
    input_dimensions[3] = test_case->input_width;
    output_dimensions[0] = 1u;
    output_dimensions[1] = test_case->output_channels;
    output_dimensions[2] = output_height;
    output_dimensions[3] = output_width;
    weight_dimensions[0] = test_case->output_channels;
    weight_dimensions[1] = test_case->input_channels;
    weight_dimensions[2] = test_case->kernel_height;
    weight_dimensions[3] = test_case->kernel_width;

    input_elements = element_count(rank, input_dimensions);
    output_elements = element_count(rank, output_dimensions);
    CHECK_TRUE(input_elements <= MAX_INPUT_ELEMENTS);
    CHECK_TRUE(output_elements <= MAX_OUTPUT_ELEMENTS);
    CHECK_TRUE(test_case->output_channels <= MAX_CHANNELS);
    CHECK_TRUE((uint64_t)test_case->output_channels *
        test_case->input_channels * kernel_elements <= MAX_LOGICAL_WEIGHTS);

    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    memset(attributes, 0, sizeof(attributes));
    attributes[0].key = CAMPP_ATTR_KERNEL_SHAPE;
    attributes[0].count = (uint8_t)(rank - 2u);
    attributes[0].values[0] = (int64_t)test_case->kernel_height;
    attributes[0].values[1] = (int64_t)test_case->kernel_width;
    attributes[1].key = CAMPP_ATTR_PADS;
    attributes[1].count = (uint8_t)((rank - 2u) * 2u);
    if (rank == 3u) {
        attributes[1].values[0] = (int64_t)test_case->pad_height;
        attributes[1].values[1] = (int64_t)test_case->pad_height;
    } else {
        attributes[1].values[0] = (int64_t)test_case->pad_height;
        attributes[1].values[1] = (int64_t)test_case->pad_width;
        attributes[1].values[2] = (int64_t)test_case->pad_height;
        attributes[1].values[3] = (int64_t)test_case->pad_width;
    }
    attributes[2].key = CAMPP_ATTR_STRIDES;
    attributes[2].count = (uint8_t)(rank - 2u);
    attributes[2].values[0] = (int64_t)test_case->stride_height;
    attributes[2].values[1] = (int64_t)test_case->stride_width;
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
        input_data[index] = (uint8_t)((index * 37u + 11u) % 251u);
    }
    for (output = 0u; output < test_case->output_channels; ++output) {
        uint32_t input;
        weight_scale[output] = 0.125f + (float)output * 0.003f;
        weight_zero[output] = zero_weight_zero_point
            ? (int8_t)0
            : (int8_t)((int32_t)(output % 5u) - 2);
        bias[output] = (int32_t)output * 13 - 41;
        for (input = 0u; input < test_case->input_channels; ++input) {
            uint32_t kernel;
            for (kernel = 0u; kernel < kernel_elements; ++kernel) {
                logical_weight[
                    (output * test_case->input_channels + input)
                        * kernel_elements + kernel] =
                    (int8_t)((int32_t)((output * 11u + input * 5u
                        + kernel * 3u) % 17u) - 8);
            }
        }
    }
    packed_size = pack_o4i4(
        logical_weight, test_case->output_channels,
        test_case->input_channels, kernel_elements,
        weight_zero, packed_weight);
    CHECK_TRUE(packed_size <= MAX_PACKED_WEIGHTS);
    memset(baseline, 0, sizeof(baseline));
    memset(candidate, 0, sizeof(candidate));

    init_channel_packed_view(
        &inputs_view[0], input_data, CAMPP_DTYPE_UINT8, rank,
        input_dimensions);
    init_contiguous_view(
        &inputs_view[1], &input_scale, CAMPP_DTYPE_FLOAT32, 0u,
        input_dimensions);
    init_contiguous_view(
        &inputs_view[2], &input_zero, CAMPP_DTYPE_UINT8, 0u,
        input_dimensions);
    init_contiguous_view(
        &inputs_view[3], packed_weight, CAMPP_DTYPE_INT8, rank,
        weight_dimensions);
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
        &baseline_output[0], baseline, CAMPP_DTYPE_UINT8, rank,
        output_dimensions);
    init_channel_packed_view(
        &candidate_output[0], candidate, CAMPP_DTYPE_UINT8, rank,
        output_dimensions);

    CHECK_STATUS(campp_aarch64_qlinear_conv_o4i4(
        &model, &op, inputs_view, 9u, baseline_output, 1u, NULL, 0u));

    for (mode = CAMPP_QCONV_CANDIDATE_V4;
         mode <= CAMPP_QCONV_CANDIDATE_V5;
         mode = (CamppQconvCandidateMode)(mode + 1)) {
        const CamppKernelEntry *entry = campp_qconv_candidate_entry(mode);
        CHECK_TRUE(entry != NULL);
        memset(candidate, 0, sizeof(candidate));
        CHECK_STATUS(entry->run(
            &model, &op, inputs_view, 9u, candidate_output, 1u, NULL, 0u));
        if (memcmp(baseline, candidate, (size_t)output_elements) != 0) {
            uint64_t position;
            fprintf(stderr,
                "BITWISE MISMATCH case=%s mode=%s weight_zero=%s\n",
                test_case->name, campp_qconv_candidate_mode_name(mode),
                zero_weight_zero_point ? "zero" : "nonzero");
            for (position = 0u; position < output_elements; ++position) {
                if (baseline[position] != candidate[position]) {
                    fprintf(stderr,
                        "  first difference at %llu: reference=%u v=%u\n",
                        (unsigned long long)position,
                        (unsigned)baseline[position],
                        (unsigned)candidate[position]);
                    break;
                }
            }
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    /* Shapes chosen so the v5 full-tile bodies are actually entered:
     * output spatial >= 8 with 8-aligned interior tiles. */
    static const ConvCase cases[] = {
        /* Real 1x1 8 spatial x 8 output body. */
        {"1x1_c32_o8_w16",      3u, 32u,  8u, 16u, 1u, 1u, 1u, 1u, 1u, 0u, 0u},
        {"1x1_c32_o16_w24",     3u, 32u, 16u, 24u, 1u, 1u, 1u, 1u, 1u, 0u, 0u},
        {"1x1_c64_o8_w32",      3u, 64u,  8u, 32u, 1u, 1u, 1u, 1u, 1u, 0u, 0u},
        /* 1x1 with a non-unit stride: pointer-table path, not sliding. */
        {"1x1_c32_o8_w16_s2",   3u, 32u,  8u, 16u, 1u, 1u, 1u, 2u, 1u, 0u, 0u},
        /* 3x3 interior sliding-window reuse (stride 1). */
        {"3x3_c32_o8_20x6",     4u, 32u,  8u,  6u, 20u, 3u, 3u, 1u, 1u, 1u, 1u},
        {"3x3_c64_o16_24x8",    4u, 64u, 16u,  8u, 24u, 3u, 3u, 1u, 1u, 1u, 1u},
        {"3x3_c32_o8_20x6_p0",  4u, 32u,  8u,  6u, 20u, 3u, 3u, 1u, 1u, 0u, 0u},
        /* Row stride 2 with column stride 1: the shape CAM++ actually uses.
         * Sliding walks columns only, so this must KEEP the sliding body. */
        {"3x3_c32_o8_24x9_s21", 4u, 32u,  8u,  9u, 24u, 3u, 3u, 2u, 1u, 1u, 1u},
        {"3x3_c64_o8_20x9_s21", 4u, 64u,  8u,  9u, 20u, 3u, 3u, 2u, 1u, 1u, 1u},
        /* Column stride 2: sliding must stay off; Cin32 raw fallback runs. */
        {"3x3_c32_o8_24x8_s2",  4u, 32u,  8u,  8u, 24u, 3u, 3u, 2u, 2u, 1u, 1u},
        {"3x3_c32_o8_24x8_s12", 4u, 32u,  8u,  8u, 24u, 3u, 3u, 1u, 2u, 1u, 1u},
        /* 3x3 stride 2 without Cin32: shared tail fallback. */
        {"3x3_c16_o8_24x8_s2",  4u, 16u,  8u,  8u, 24u, 3u, 3u, 2u, 2u, 1u, 1u},
    };
    const uint32_t case_count = (uint32_t)(sizeof(cases) / sizeof(cases[0]));
    uint32_t index;
    uint32_t failures = 0u;

    for (index = 0u; index < case_count; ++index) {
        int variant;
        for (variant = 0; variant < 2; ++variant) {
            if (run_conv_case(&cases[index], variant) != 0) {
                failures += 1u;
            } else {
                printf("PASS %-22s weight_zero=%s\n",
                       cases[index].name, variant ? "zero" : "nonzero");
            }
        }
    }
    if (failures != 0u) {
        fprintf(stderr, "%u v5 bitwise case(s) failed\n", failures);
        return 1;
    }
    printf("qconv v5 bitwise suite passed (%u cases)\n", case_count * 2u);
    return 0;
}
