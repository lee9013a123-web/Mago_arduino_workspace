#ifndef CAMPP_RUNTIME_FUSED_QUANT_QCONV_INTERNAL_H
#define CAMPP_RUNTIME_FUSED_QUANT_QCONV_INTERNAL_H

#include "internal/kernel_registry.h"

typedef CamppStatus (*CamppFusedInputQuantizeRun)(
    const CamppTensorView *input, CamppTensorView *output,
    float scale, int32_t zero_point);

CamppStatus campp_fused_quantize_input_scalar(
    const CamppTensorView *input, CamppTensorView *output,
    float scale, int32_t zero_point);

/*
 * Candidate hook that keeps fused validation, scratch construction and stage
 * accounting identical while independently replacing input quantization and
 * the packed QConv runner.
 */
CamppStatus campp_fused_quant_qlinear_conv_with_components(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size,
    CamppFusedInputQuantizeRun quantize_run, CamppKernelRun qconv_run);

/*
 * Shared QuantizeLinear -> QLinearConv driver. Production and profiling
 * candidates use the same validation, scratch layout and quantization pass;
 * only the quantized QConv runner is injected.
 */
CamppStatus campp_fused_quant_qlinear_conv_with_runner(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size, CamppKernelRun qconv_run);

#endif /* CAMPP_RUNTIME_FUSED_QUANT_QCONV_INTERNAL_H */
