#ifndef CAMPP_RUNTIME_CPU_REFERENCE_KERNELS_H
#define CAMPP_RUNTIME_CPU_REFERENCE_KERNELS_H

#include "internal/kernel_registry.h"
#include "internal/runtime_model.h"

#define CAMPP_DECLARE_REFERENCE_KERNEL(name)                                   \
    CamppStatus name(                                                          \
        const CamppRuntimeModel *model,                                        \
        const CamppOperatorDescriptor *op, const CamppTensorView *inputs,      \
        uint8_t input_count, CamppTensorView *outputs, uint8_t output_count,   \
        void *scratch, size_t scratch_size)

CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_qlinear_conv);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_quantize_linear);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_dequantize_linear);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_batch_normalization);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_relu);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_sigmoid);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_average_pool);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_reduce_mean);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_add);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_mul);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_sub);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_div);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_sqrt);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_concat);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_expand);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_slice);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_reshape);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_transpose);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_squeeze);
CAMPP_DECLARE_REFERENCE_KERNEL(campp_reference_unsqueeze);

#undef CAMPP_DECLARE_REFERENCE_KERNEL

#endif /* CAMPP_RUNTIME_CPU_REFERENCE_KERNELS_H */
