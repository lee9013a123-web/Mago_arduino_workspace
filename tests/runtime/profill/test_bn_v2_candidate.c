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

#define MAX_CHANNELS 32u
#define SPATIAL 5u

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

static int run_case(uint32_t channels, uint8_t output_dtype)
{
    const uint32_t tensor_dimensions[3] = {1u, channels, SPATIAL};
    const uint32_t parameter_dimensions[1] = {channels};
    float input_data[MAX_CHANNELS * SPATIAL];
    float scale[MAX_CHANNELS];
    float bias[MAX_CHANNELS];
    float mean[MAX_CHANNELS];
    float variance[MAX_CHANNELS];
    float quant_scale = 0.125f;
    uint8_t zero_u8 = 101u;
    int8_t zero_i8 = -9;
    uint8_t baseline[MAX_CHANNELS * SPATIAL];
    uint8_t candidate[MAX_CHANNELS * SPATIAL];
    CamppTensorView inputs[7];
    CamppTensorView output;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint8_t attribute_section[24];
    uint32_t index;
    CamppBnCandidateMode mode;
    const CamppBnCandidateMode modes[] = {
        CAMPP_BN_CANDIDATE_V2_EXACT16,
        CAMPP_BN_CANDIDATE_V2_SPATIAL2,
        CAMPP_BN_CANDIDATE_V2_PRESCALED
    };

    for (index = 0u; index < channels * SPATIAL; ++index) {
        input_data[index] =
            ((float)((int32_t)(index % 29u) - 14)) * 0.09375f;
    }
    for (index = 0u; index < channels; ++index) {
        scale[index] = 0.25f + (float)(index % 7u) * 0.125f;
        if (index % 5u == 0u) scale[index] = -scale[index];
        bias[index] = ((float)((int32_t)(index % 9u) - 4)) * 0.0625f;
        mean[index] = ((float)((int32_t)(index % 11u) - 5)) * 0.125f;
        variance[index] = 0.5f + (float)(index % 13u) * 0.25f;
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
    for (index = 0u; index < sizeof(modes) / sizeof(modes[0]); ++index) {
        const CamppKernelEntry *entry;
        mode = modes[index];
        entry = campp_bn_candidate_entry(mode);
        CHECK_TRUE(entry != NULL);
        memset(candidate, 0x5A, sizeof(candidate));
        init_channel_packed_view(
            &output, candidate, output_dtype, tensor_dimensions);
        CHECK_STATUS(entry->run(
            &model, &op, inputs, 7u, &output, 1u, NULL, 0u));
        if (memcmp(baseline, candidate, channels * SPATIAL) != 0) {
            fprintf(
                stderr, "BN %s differs for channels=%u dtype=%u\n",
                campp_bn_candidate_mode_name(mode), (unsigned int)channels,
                (unsigned int)output_dtype);
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    CHECK_TRUE(run_case(19u, CAMPP_DTYPE_UINT8) == 0);
    CHECK_TRUE(run_case(32u, CAMPP_DTYPE_UINT8) == 0);
    CHECK_TRUE(run_case(19u, CAMPP_DTYPE_INT8) == 0);
    CHECK_TRUE(run_case(32u, CAMPP_DTYPE_INT8) == 0);
    puts("test_bn_v2_candidate: PASS");
    return 0;
}
