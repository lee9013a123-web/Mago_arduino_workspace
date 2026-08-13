#ifndef CAMPP_RUNTIME_AARCH64_KERNELS_H
#define CAMPP_RUNTIME_AARCH64_KERNELS_H

#include "internal/kernel_registry.h"

#define CAMPP_AARCH64_PACKED_KERNEL_ID 1u

CamppStatus campp_aarch64_qlinear_conv_o4i4(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

#endif
