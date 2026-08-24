#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "bn_candidate.h"
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

static void write_f64(uint8_t *target, double value)
{
    uint64_t bits;
    uint8_t index;
    memcpy(&bits, &value, sizeof(bits));
    for (index = 0u; index < 8u; ++index) {
        target[index] = (uint8_t)(bits >> (index * 8u));
    }
}

static void encode_epsilon(uint8_t buffer[24], double epsilon)
{
    memset(buffer, 0, 24u);
    write_u32(buffer, 1u);
    write_u32(buffer + 4u, 24u);
    write_u16(buffer + 8u, CAMPP_ATTR_EPSILON);
    buffer[10] = CAMPP_ATTR_VALUE_FLOAT;
    buffer[11] = 1u;
    write_f64(buffer + 16u, epsilon);
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

static void init_channel_packed_view(
    CamppTensorView *view, void *data, uint8_t dtype,
    const uint32_t dimensions[3])
{
    const uint32_t element_size = campp_dtype_byte_size(dtype);
    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = dtype;
    view->rank = 3u;
    memcpy(view->dimensions, dimensions, 3u * sizeof(dimensions[0]));
    view->dimensions[3] = 1u;
    view->byte_strides[1] = element_size;
    view->byte_strides[2] = dimensions[1] * element_size;
    view->byte_strides[0] =
        dimensions[1] * dimensions[2] * element_size;
    view->logical_byte_size = view->byte_strides[0] * dimensions[0];
    view->storage_span_bytes = view->logical_byte_size;
}

static int run_case(uint8_t output_dtype)
{
    const uint32_t tensor_dimensions[3] = {1u, 5u, 7u};
    const uint32_t parameter_dimensions[1] = {5u};
    float input_data[35];
    float scale[5] = {0.75f, 1.25f, -0.5f, 2.0f, 0.125f};
    float bias[5] = {0.1f, -0.2f, 0.3f, 0.0f, 1.0f};
    float mean[5] = {-0.5f, 0.25f, 1.0f, -1.0f, 0.0f};
    float variance[5] = {0.5f, 1.5f, 2.5f, 0.25f, 4.0f};
    float quant_scale = 0.0625f;
    uint8_t zero_u8 = 117u;
    int8_t zero_i8 = -11;
    uint8_t baseline[35];
    uint8_t candidate[35];
    CamppTensorView inputs[7];
    CamppTensorView output;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_section[24];
    uint32_t index;
    CamppBnCandidateMode mode;

    for (index = 0u; index < 35u; ++index) {
        input_data[index] = ((float)((int32_t)(index % 13u) - 6)) * 0.17f;
    }
    init_channel_packed_view(
        &inputs[0], input_data, CAMPP_DTYPE_FLOAT32, tensor_dimensions);
    init_contiguous_view(
        &inputs[1], scale, CAMPP_DTYPE_FLOAT32, 1u, parameter_dimensions);
    init_contiguous_view(
        &inputs[2], bias, CAMPP_DTYPE_FLOAT32, 1u, parameter_dimensions);
    init_contiguous_view(
        &inputs[3], mean, CAMPP_DTYPE_FLOAT32, 1u, parameter_dimensions);
    init_contiguous_view(
        &inputs[4], variance, CAMPP_DTYPE_FLOAT32, 1u,
        parameter_dimensions);
    init_contiguous_view(
        &inputs[5], &quant_scale, CAMPP_DTYPE_FLOAT32, 0u, NULL);
    init_contiguous_view(
        &inputs[6], output_dtype == CAMPP_DTYPE_UINT8
            ? (void *)&zero_u8 : (void *)&zero_i8,
        output_dtype, 0u, NULL);
    init_channel_packed_view(
        &output, baseline, output_dtype, tensor_dimensions);

    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    encode_epsilon(attribute_section, 1.0e-5);
    model.attribute_section = attribute_section;
    model.attribute_section_size = sizeof(attribute_section);
    op.opcode = CAMPP_OP_BATCH_NORMALIZATION;
    op.input_count = 7u;
    op.output_count = 1u;
    op.kernel_id = CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID;
    op.attribute_size = sizeof(attribute_section);

    memset(baseline, 0xA5, sizeof(baseline));
    CHECK_STATUS(campp_fused_bn_relu_quant(
        &model, &op, inputs, 7u, &output, 1u, NULL, 0u));
    for (mode = CAMPP_BN_CANDIDATE_ADDRESS;
         mode <= CAMPP_BN_CANDIDATE_COMBINED;
         mode = (CamppBnCandidateMode)(mode + 1)) {
        const CamppKernelEntry *entry = campp_bn_candidate_entry(mode);
        CHECK_TRUE(entry != NULL);
        memset(candidate, 0x5A, sizeof(candidate));
        init_channel_packed_view(
            &output, candidate, output_dtype, tensor_dimensions);
        CHECK_STATUS(entry->run(
            &model, &op, inputs, 7u, &output, 1u, NULL, 0u));
        if (memcmp(baseline, candidate, sizeof(baseline)) != 0) {
            fprintf(
                stderr, "BN %s output differs for dtype %u\n",
                campp_bn_candidate_mode_name(mode),
                (unsigned int)output_dtype);
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    CHECK_TRUE(run_case(CAMPP_DTYPE_UINT8) == 0);
    CHECK_TRUE(run_case(CAMPP_DTYPE_INT8) == 0);
    puts("test_bn_candidate: PASS");
    return 0;
}
