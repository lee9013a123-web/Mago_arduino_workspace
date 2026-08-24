/* Scalar CPU Reference backend의 opcode -> C 함수 정적 등록표다. */

#include "internal/kernel_registry.h"

#include <stddef.h>
#include <stdint.h>

#include "internal/runtime_model.h"
#include "reference_kernels.h"

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

const CamppKernelRegistry *campp_cpu_reference_registry(void)
{
    return &CAMPP_CPU_REFERENCE_REGISTRY;
}
