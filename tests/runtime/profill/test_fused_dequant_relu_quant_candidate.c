#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "fused_dequant_relu_quant_candidate.h"
#include "internal/runtime_model.h"

#define CHECK_TRUE(condition)                                          \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "CHECK failed at line %d: %s\n",       \
                    __LINE__, #condition);                             \
            return 1;                                                  \
        }                                                              \
    } while (0)

#define CHECK_STATUS(expression)                                      \
    do {                                                               \
        const CamppStatus actual = (expression);                       \
        if (actual != CAMPP_STATUS_OK) {                               \
            fprintf(stderr, "STATUS failed at line %d: %s\n",      \
                    __LINE__, campp_status_name(actual));              \
            return 1;                                                  \
        }                                                              \
    } while (0)

static void init_scalar_view(
    CamppTensorView *view, void *data, uint8_t dtype)
{
    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = dtype;
    view->rank = 0u;
    view->logical_byte_size = campp_dtype_byte_size(dtype);
    view->storage_span_bytes = view->logical_byte_size;
}

static void init_channel_packed_view(
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

static void init_contiguous_view(
    CamppTensorView *view, void *data, uint8_t dtype,
    const uint32_t dimensions[4])
{
    uint32_t stride = campp_dtype_byte_size(dtype);
    uint8_t axis;
    memset(view, 0, sizeof(*view));
    view->data = data;
    view->dtype = dtype;
    view->rank = 4u;
    memcpy(view->dimensions, dimensions, 4u * sizeof(dimensions[0]));
    for (axis = 4u; axis > 0u; --axis) {
        view->byte_strides[axis - 1u] = stride;
        stride *= dimensions[axis - 1u];
    }
    view->logical_byte_size = stride;
    view->storage_span_bytes = stride;
}

static int run_case(uint8_t input_dtype, uint8_t output_dtype)
{
    const uint32_t dimensions[4] = {1u, 19u, 2u, 3u};
    uint8_t input_data[114];
    uint8_t baseline[114];
    uint8_t candidate[114];
    float old_scale = 0.125f;
    float new_scale = 0.0625f;
    uint8_t old_zero_storage = input_dtype == CAMPP_DTYPE_UINT8
        ? 121u : (uint8_t)(int8_t)-3;
    uint8_t new_zero_storage = output_dtype == CAMPP_DTYPE_UINT8
        ? 117u : (uint8_t)(int8_t)5;
    CamppTensorView inputs[5];
    CamppTensorView output;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    CamppFusedDqRqCandidateMode mode;
    uint32_t index;

    for (index = 0u; index < sizeof(input_data); ++index) {
        input_data[index] = input_dtype == CAMPP_DTYPE_UINT8
            ? (uint8_t)(91u + index % 71u)
            : (uint8_t)(int8_t)((int32_t)(index % 97u) - 48);
    }
    init_channel_packed_view(
        &inputs[0], input_data, input_dtype, dimensions);
    init_scalar_view(&inputs[1], &old_scale, CAMPP_DTYPE_FLOAT32);
    init_scalar_view(&inputs[2], &old_zero_storage, input_dtype);
    init_scalar_view(&inputs[3], &new_scale, CAMPP_DTYPE_FLOAT32);
    init_scalar_view(&inputs[4], &new_zero_storage, output_dtype);
    init_channel_packed_view(
        &output, baseline, output_dtype, dimensions);
    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    op.opcode = CAMPP_OP_DEQUANTIZE_LINEAR;
    op.input_count = 5u;
    op.output_count = 1u;
    op.kernel_id = CAMPP_FUSION_EPILOGUE_KERNEL_ID;

    memset(baseline, 0xA5, sizeof(baseline));
    CHECK_STATUS(campp_fused_dequant_relu_quant(
        &model, &op, inputs, 5u, &output, 1u, NULL, 0u));
    for (mode = CAMPP_FUSED_DQRQ_CANDIDATE_SCALAR;
         mode <= CAMPP_FUSED_DQRQ_CANDIDATE_NEON;
         mode = (CamppFusedDqRqCandidateMode)(mode + 1)) {
        const CamppKernelEntry *entry =
            campp_fused_dqrq_candidate_entry(mode);
        CHECK_TRUE(entry != NULL);
        memset(candidate, 0x5A, sizeof(candidate));
        init_channel_packed_view(
            &output, candidate, output_dtype, dimensions);
        CHECK_STATUS(entry->run(
            &model, &op, inputs, 5u, &output, 1u, NULL, 0u));
        if (memcmp(baseline, candidate, sizeof(baseline)) != 0) {
            fprintf(
                stderr, "fused DQRQ %s differs for dtype %u -> %u\n",
                campp_fused_dqrq_candidate_mode_name(mode),
                (unsigned int)input_dtype, (unsigned int)output_dtype);
            return 1;
        }
    }
    return 0;
}

static int run_fallback_case(void)
{
    const uint32_t dimensions[4] = {1u, 3u, 2u, 5u};
    uint8_t input_data[30];
    uint8_t baseline[30];
    uint8_t candidate[30];
    float old_scale = 0.25f;
    float new_scale = 0.125f;
    uint8_t old_zero = 120u;
    uint8_t new_zero = 113u;
    CamppTensorView inputs[5];
    CamppTensorView output;
    CamppRuntimeModel model;
    CamppOperatorDescriptor op;
    const CamppKernelEntry *entry = campp_fused_dqrq_candidate_entry(
        CAMPP_FUSED_DQRQ_CANDIDATE_NEON);
    uint32_t index;

    for (index = 0u; index < sizeof(input_data); ++index) {
        input_data[index] = (uint8_t)(101u + index);
    }
    init_contiguous_view(
        &inputs[0], input_data, CAMPP_DTYPE_UINT8, dimensions);
    init_scalar_view(&inputs[1], &old_scale, CAMPP_DTYPE_FLOAT32);
    init_scalar_view(&inputs[2], &old_zero, CAMPP_DTYPE_UINT8);
    init_scalar_view(&inputs[3], &new_scale, CAMPP_DTYPE_FLOAT32);
    init_scalar_view(&inputs[4], &new_zero, CAMPP_DTYPE_UINT8);
    init_contiguous_view(
        &output, baseline, CAMPP_DTYPE_UINT8, dimensions);
    memset(&model, 0, sizeof(model));
    memset(&op, 0, sizeof(op));
    op.opcode = CAMPP_OP_DEQUANTIZE_LINEAR;
    op.input_count = 5u;
    op.output_count = 1u;
    op.kernel_id = CAMPP_FUSION_EPILOGUE_KERNEL_ID;
    CHECK_TRUE(entry != NULL);
    CHECK_STATUS(campp_fused_dequant_relu_quant(
        &model, &op, inputs, 5u, &output, 1u, NULL, 0u));
    init_contiguous_view(
        &output, candidate, CAMPP_DTYPE_UINT8, dimensions);
    CHECK_STATUS(entry->run(
        &model, &op, inputs, 5u, &output, 1u, NULL, 0u));
    CHECK_TRUE(memcmp(baseline, candidate, sizeof(baseline)) == 0);
    return 0;
}

int main(void)
{
    CamppFusedDqRqCandidateMode mode;
    CHECK_TRUE(campp_fused_dqrq_candidate_mode_parse("neon", &mode) == 0);
    CHECK_TRUE(mode == CAMPP_FUSED_DQRQ_CANDIDATE_NEON);
    CHECK_TRUE(run_case(CAMPP_DTYPE_UINT8, CAMPP_DTYPE_UINT8) == 0);
    CHECK_TRUE(run_case(CAMPP_DTYPE_UINT8, CAMPP_DTYPE_INT8) == 0);
    CHECK_TRUE(run_case(CAMPP_DTYPE_INT8, CAMPP_DTYPE_UINT8) == 0);
    CHECK_TRUE(run_case(CAMPP_DTYPE_INT8, CAMPP_DTYPE_INT8) == 0);
    CHECK_TRUE(run_fallback_case() == 0);
    puts("test_fused_dequant_relu_quant_candidate: PASS");
    return 0;
}
