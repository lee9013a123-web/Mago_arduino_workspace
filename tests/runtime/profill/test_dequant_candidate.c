#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_reference/reference_kernels.h"
#include "dequant_candidate.h"
#include "internal/runtime_model.h"

#define CHECK_TRUE(condition)                                          \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "CHECK failed at line %d: %s\n",        \
                    __LINE__, #condition);                             \
            return 1;                                                  \
        }                                                              \
    } while (0)

#define CHECK_STATUS(expression)                                      \
    do {                                                               \
        const CamppStatus actual = (expression);                       \
        if (actual != CAMPP_STATUS_OK) {                               \
            fprintf(stderr, "STATUS failed at line %d: %s\n",       \
                    __LINE__, campp_status_name(actual));              \
            return 1;                                                  \
        }                                                              \
    } while (0)

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

static void init_channel_packed_view4(
    CamppTensorView *view, void *data, uint8_t dtype,
    const uint32_t dimensions[4])
{
    const uint32_t element_size = campp_dtype_byte_size(dtype);
    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = dtype;
    view->rank = 4u;
    memcpy(view->dimensions, dimensions, 4u * sizeof(dimensions[0]));
    view->byte_strides[1] = element_size;
    view->byte_strides[3] = dimensions[1] * element_size;
    view->byte_strides[2] = dimensions[3] * view->byte_strides[3];
    view->byte_strides[0] = dimensions[2] * view->byte_strides[2];
    view->logical_byte_size = view->byte_strides[0] * dimensions[0];
    view->storage_span_bytes = view->logical_byte_size;
}

static int run_channel_packed_case(
    uint8_t input_dtype, int per_axis, int has_zero)
{
    const uint32_t dimensions[3] = {1u, 17u, 3u};
    const uint32_t parameter_dimensions[1] = {17u};
    uint8_t input_data[51];
    uint8_t zero_data[17];
    float scales[17];
    float baseline[51];
    float candidate[51];
    CamppTensorView inputs[3];
    CamppTensorView output;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    CamppDequantCandidateMode mode;
    uint32_t index;

    for (index = 0u; index < 51u; ++index) {
        input_data[index] = input_dtype == CAMPP_DTYPE_UINT8
            ? (uint8_t)(101u + index % 31u)
            : (uint8_t)(int8_t)((int32_t)(index % 29u) - 14);
    }
    for (index = 0u; index < 17u; ++index) {
        scales[index] = 0.03125f * (float)(1u + index % 5u);
        zero_data[index] = input_dtype == CAMPP_DTYPE_UINT8
            ? (uint8_t)(105u + index % 7u)
            : (uint8_t)(int8_t)((int32_t)(index % 9u) - 4);
    }
    init_channel_packed_view(
        &inputs[0], input_data, input_dtype, dimensions);
    init_contiguous_view(
        &inputs[1], scales, CAMPP_DTYPE_FLOAT32,
        per_axis ? 1u : 0u,
        per_axis ? parameter_dimensions : NULL);
    if (has_zero) {
        init_contiguous_view(
            &inputs[2], zero_data, input_dtype,
            per_axis ? 1u : 0u,
            per_axis ? parameter_dimensions : NULL);
    }
    init_channel_packed_view(
        &output, baseline, CAMPP_DTYPE_FLOAT32, dimensions);

    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    op.opcode = CAMPP_OP_DEQUANTIZE_LINEAR;
    op.input_count = has_zero ? 3u : 2u;
    op.output_count = 1u;
    op.kernel_id = CAMPP_AARCH64_PACKED_KERNEL_ID;

    memset(baseline, 0xA5, sizeof(baseline));
    CHECK_STATUS(campp_reference_dequantize_linear(
        &model, &op, inputs, op.input_count, &output, 1u, NULL, 0u));
    for (mode = CAMPP_DEQUANT_CANDIDATE_ADDRESS;
         mode <= CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED;
         mode = (CamppDequantCandidateMode)(mode + 1)) {
        const CamppKernelEntry *entry = campp_dequant_candidate_entry(mode);
        CHECK_TRUE(entry != NULL);
        memset(candidate, 0x5A, sizeof(candidate));
        init_channel_packed_view(
            &output, candidate, CAMPP_DTYPE_FLOAT32, dimensions);
        CHECK_STATUS(entry->run(
            &model, &op, inputs, op.input_count,
            &output, 1u, NULL, 0u));
        if (memcmp(baseline, candidate, sizeof(baseline)) != 0) {
            fprintf(
                stderr,
                "Dequant %s differs for dtype=%u per_axis=%d zero=%d\n",
                campp_dequant_candidate_mode_name(mode),
                (unsigned int)input_dtype, per_axis, has_zero);
            return 1;
        }
    }
    return 0;
}

static int run_fallback_case(void)
{
    const uint32_t dimensions[3] = {1u, 3u, 5u};
    uint8_t input_data[15];
    uint8_t zero = 119u;
    float scale = 0.125f;
    float baseline[15];
    float candidate[15];
    CamppTensorView inputs[3];
    CamppTensorView output;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    uint32_t index;
    const CamppKernelEntry *entry = campp_dequant_candidate_entry(
        CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED);

    for (index = 0u; index < 15u; ++index) {
        input_data[index] = (uint8_t)(110u + index);
    }
    init_contiguous_view(
        &inputs[0], input_data, CAMPP_DTYPE_UINT8, 3u, dimensions);
    init_contiguous_view(
        &inputs[1], &scale, CAMPP_DTYPE_FLOAT32, 0u, NULL);
    init_contiguous_view(
        &inputs[2], &zero, CAMPP_DTYPE_UINT8, 0u, NULL);
    init_contiguous_view(
        &output, baseline, CAMPP_DTYPE_FLOAT32, 3u, dimensions);
    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    op.opcode = CAMPP_OP_DEQUANTIZE_LINEAR;
    op.input_count = 3u;
    op.output_count = 1u;
    op.kernel_id = CAMPP_AARCH64_PACKED_KERNEL_ID;
    CHECK_TRUE(entry != NULL);
    CHECK_STATUS(campp_reference_dequantize_linear(
        &model, &op, inputs, 3u, &output, 1u, NULL, 0u));
    init_contiguous_view(
        &output, candidate, CAMPP_DTYPE_FLOAT32, 3u, dimensions);
    CHECK_STATUS(entry->run(
        &model, &op, inputs, 3u, &output, 1u, NULL, 0u));
    CHECK_TRUE(memcmp(baseline, candidate, sizeof(baseline)) == 0);
    return 0;
}

static int run_rank4_case(void)
{
    const uint32_t dimensions[4] = {1u, 17u, 2u, 3u};
    uint8_t input_data[102];
    int8_t zero = -3;
    float scale = 0.0625f;
    float baseline[102];
    float candidate[102];
    CamppTensorView inputs[3];
    CamppTensorView output;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    const CamppKernelEntry *entry = campp_dequant_candidate_entry(
        CAMPP_DEQUANT_CANDIDATE_NEON_COMBINED);
    uint32_t index;

    for (index = 0u; index < 102u; ++index) {
        input_data[index] = (uint8_t)(int8_t)((int32_t)(index % 41u) - 20);
    }
    init_channel_packed_view4(
        &inputs[0], input_data, CAMPP_DTYPE_INT8, dimensions);
    init_contiguous_view(
        &inputs[1], &scale, CAMPP_DTYPE_FLOAT32, 0u, NULL);
    init_contiguous_view(
        &inputs[2], &zero, CAMPP_DTYPE_INT8, 0u, NULL);
    init_channel_packed_view4(
        &output, baseline, CAMPP_DTYPE_FLOAT32, dimensions);
    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    op.opcode = CAMPP_OP_DEQUANTIZE_LINEAR;
    op.input_count = 3u;
    op.output_count = 1u;
    op.kernel_id = CAMPP_AARCH64_PACKED_KERNEL_ID;
    CHECK_TRUE(entry != NULL);
    CHECK_STATUS(campp_reference_dequantize_linear(
        &model, &op, inputs, 3u, &output, 1u, NULL, 0u));
    init_channel_packed_view4(
        &output, candidate, CAMPP_DTYPE_FLOAT32, dimensions);
    CHECK_STATUS(entry->run(
        &model, &op, inputs, 3u, &output, 1u, NULL, 0u));
    CHECK_TRUE(memcmp(baseline, candidate, sizeof(baseline)) == 0);
    return 0;
}

int main(void)
{
    CHECK_TRUE(run_channel_packed_case(CAMPP_DTYPE_UINT8, 0, 1) == 0);
    CHECK_TRUE(run_channel_packed_case(CAMPP_DTYPE_INT8, 0, 0) == 0);
    CHECK_TRUE(run_channel_packed_case(CAMPP_DTYPE_UINT8, 1, 1) == 0);
    CHECK_TRUE(run_channel_packed_case(CAMPP_DTYPE_INT8, 1, 1) == 0);
    CHECK_TRUE(run_rank4_case() == 0);
    CHECK_TRUE(run_fallback_case() == 0);
    puts("test_dequant_candidate: PASS");
    return 0;
}
