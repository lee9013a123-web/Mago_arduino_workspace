/* AArch64 optimized registry. Non-QConv operators keep the exact reference path. */

#include "backends/cpu_aarch64/aarch64_kernels.h"

#include "backends/cpu_reference/reference_kernels.h"

#define AARCH64_ENTRY(opcode_value, function_name, display_name) \
    { (opcode_value), CAMPP_AARCH64_PACKED_KERNEL_ID, (function_name), NULL, \
      (display_name) }

static const CamppKernelEntry CAMPP_AARCH64_ENTRIES[] = {
    AARCH64_ENTRY(CAMPP_OP_QLINEAR_CONV,
                  campp_aarch64_qlinear_conv_o4i4, "qlinear_conv_o4i4_neon"),
    AARCH64_ENTRY(CAMPP_OP_QUANTIZE_LINEAR, campp_reference_quantize_linear, "quantize_linear_stride"),
    AARCH64_ENTRY(CAMPP_OP_DEQUANTIZE_LINEAR, campp_reference_dequantize_linear, "dequantize_linear_stride"),
    AARCH64_ENTRY(CAMPP_OP_BATCH_NORMALIZATION, campp_reference_batch_normalization, "batch_normalization_stride"),
    AARCH64_ENTRY(CAMPP_OP_RELU, campp_reference_relu, "relu_stride"),
    AARCH64_ENTRY(CAMPP_OP_SIGMOID, campp_reference_sigmoid, "sigmoid_stride"),
    AARCH64_ENTRY(CAMPP_OP_AVERAGE_POOL, campp_reference_average_pool, "average_pool_stride"),
    AARCH64_ENTRY(CAMPP_OP_REDUCE_MEAN, campp_reference_reduce_mean, "reduce_mean_stride"),
    AARCH64_ENTRY(CAMPP_OP_ADD, campp_reference_add, "add_stride"),
    AARCH64_ENTRY(CAMPP_OP_MUL, campp_reference_mul, "mul_stride"),
    AARCH64_ENTRY(CAMPP_OP_SUB, campp_reference_sub, "sub_stride"),
    AARCH64_ENTRY(CAMPP_OP_DIV, campp_reference_div, "div_stride"),
    AARCH64_ENTRY(CAMPP_OP_SQRT, campp_reference_sqrt, "sqrt_stride"),
    AARCH64_ENTRY(CAMPP_OP_CONCAT, campp_reference_concat, "concat_stride"),
    AARCH64_ENTRY(CAMPP_OP_EXPAND, campp_reference_expand, "expand_stride"),
    AARCH64_ENTRY(CAMPP_OP_SLICE, campp_reference_slice, "slice_stride"),
    AARCH64_ENTRY(CAMPP_OP_RESHAPE, campp_reference_reshape, "reshape_stride"),
    AARCH64_ENTRY(CAMPP_OP_TRANSPOSE, campp_reference_transpose, "transpose_stride"),
    AARCH64_ENTRY(CAMPP_OP_SQUEEZE, campp_reference_squeeze, "squeeze_stride"),
    AARCH64_ENTRY(CAMPP_OP_UNSQUEEZE, campp_reference_unsqueeze, "unsqueeze_stride")
};

static const CamppKernelRegistry CAMPP_AARCH64_REGISTRY = {
    CAMPP_BACKEND_CPU_AARCH64,
    "cpu_aarch64_o4i4",
    CAMPP_AARCH64_ENTRIES,
    sizeof(CAMPP_AARCH64_ENTRIES) / sizeof(CAMPP_AARCH64_ENTRIES[0])
};

const CamppKernelRegistry *campp_cpu_aarch64_registry(void)
{
    return &CAMPP_AARCH64_REGISTRY;
}

#undef AARCH64_ENTRY
