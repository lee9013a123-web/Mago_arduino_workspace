#ifndef CAMPP_RUNTIME_FUSED_QUANT_QCONV_INTERNAL_H
#define CAMPP_RUNTIME_FUSED_QUANT_QCONV_INTERNAL_H

#include "internal/kernel_registry.h"

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
