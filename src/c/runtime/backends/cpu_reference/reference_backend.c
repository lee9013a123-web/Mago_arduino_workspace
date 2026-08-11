/* Scalar CPU Reference backend의 opcode -> C 함수 정적 등록표다. */

#include "internal/kernel_registry.h"

#include <stddef.h>
#include <stdint.h>

#include "internal/runtime_model.h"

/*
 * NULL 함수로 등록하면 Context 생성이 MISSING_KERNEL에서 끝난다. 모든 opcode에
 * 실행 가능한 스텁을 등록하고 실제 호출 지점에서 NOT_IMPLEMENTED를 반환한다.
 */
#define CAMPP_DEFINE_REFERENCE_STUB(function_name)                             \
    static CamppStatus function_name(                                          \
        const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,      \
        const CamppTensorView *inputs, uint8_t input_count,                     \
        CamppTensorView *outputs, uint8_t output_count, void *scratch,          \
        size_t scratch_size)                                                    \
    {                                                                           \
        (void)model;                                                            \
        (void)op;                                                               \
        (void)inputs;                                                           \
        (void)input_count;                                                      \
        (void)outputs;                                                          \
        (void)output_count;                                                     \
        (void)scratch;                                                          \
        (void)scratch_size;                                                     \
        return CAMPP_STATUS_NOT_IMPLEMENTED;                                   \
    }

CAMPP_DEFINE_REFERENCE_STUB(campp_reference_qlinear_conv)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_quantize_linear)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_dequantize_linear)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_batch_normalization)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_relu)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_sigmoid)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_average_pool)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_reduce_mean)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_add)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_mul)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_sub)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_div)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_sqrt)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_concat)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_expand)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_slice)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_reshape)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_transpose)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_squeeze)
CAMPP_DEFINE_REFERENCE_STUB(campp_reference_unsqueeze)

#define CAMPP_REFERENCE_ENTRY(opcode_value, function_name, display_name)       \
    {                                                                          \
        (opcode_value), CAMPP_DEFAULT_KERNEL_ID, (function_name), NULL,         \
        (display_name)                                                         \
    }

static const CamppKernelEntry CAMPP_CPU_REFERENCE_ENTRIES[] = {
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_QLINEAR_CONV, campp_reference_qlinear_conv,
        "qlinear_conv_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_QUANTIZE_LINEAR, campp_reference_quantize_linear,
        "quantize_linear_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_DEQUANTIZE_LINEAR, campp_reference_dequantize_linear,
        "dequantize_linear_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_BATCH_NORMALIZATION, campp_reference_batch_normalization,
        "batch_normalization_reference"),
    CAMPP_REFERENCE_ENTRY(CAMPP_OP_RELU, campp_reference_relu, "relu_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_SIGMOID, campp_reference_sigmoid, "sigmoid_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_AVERAGE_POOL, campp_reference_average_pool,
        "average_pool_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_REDUCE_MEAN, campp_reference_reduce_mean,
        "reduce_mean_reference"),
    CAMPP_REFERENCE_ENTRY(CAMPP_OP_ADD, campp_reference_add, "add_reference"),
    CAMPP_REFERENCE_ENTRY(CAMPP_OP_MUL, campp_reference_mul, "mul_reference"),
    CAMPP_REFERENCE_ENTRY(CAMPP_OP_SUB, campp_reference_sub, "sub_reference"),
    CAMPP_REFERENCE_ENTRY(CAMPP_OP_DIV, campp_reference_div, "div_reference"),
    CAMPP_REFERENCE_ENTRY(CAMPP_OP_SQRT, campp_reference_sqrt, "sqrt_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_CONCAT, campp_reference_concat, "concat_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_EXPAND, campp_reference_expand, "expand_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_SLICE, campp_reference_slice, "slice_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_RESHAPE, campp_reference_reshape, "reshape_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_TRANSPOSE, campp_reference_transpose, "transpose_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_SQUEEZE, campp_reference_squeeze, "squeeze_reference"),
    CAMPP_REFERENCE_ENTRY(
        CAMPP_OP_UNSQUEEZE, campp_reference_unsqueeze,
        "unsqueeze_reference")
};

static const CamppKernelRegistry CAMPP_CPU_REFERENCE_REGISTRY = {
    CAMPP_BACKEND_CPU_REFERENCE,
    "cpu_reference",
    CAMPP_CPU_REFERENCE_ENTRIES,
    sizeof(CAMPP_CPU_REFERENCE_ENTRIES) /
        sizeof(CAMPP_CPU_REFERENCE_ENTRIES[0])
};

#undef CAMPP_REFERENCE_ENTRY
#undef CAMPP_DEFINE_REFERENCE_STUB

const CamppKernelRegistry *campp_cpu_reference_registry(void)
{
    return &CAMPP_CPU_REFERENCE_REGISTRY;
}
