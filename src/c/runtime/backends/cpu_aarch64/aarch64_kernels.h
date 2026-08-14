#ifndef CAMPP_RUNTIME_AARCH64_KERNELS_H
#define CAMPP_RUNTIME_AARCH64_KERNELS_H

#include "internal/kernel_registry.h"

#define CAMPP_AARCH64_PACKED_KERNEL_ID 1u
#define CAMPP_FUSION_BN_RELU_QUANT_KERNEL_ID 2u
#define CAMPP_FUSION_QUANT_QCONV_KERNEL_ID 3u
#define CAMPP_FUSION_EPILOGUE_KERNEL_ID 4u
#define CAMPP_FUSION_STATS_POOLING_KERNEL_ID 5u
#define CAMPP_FUSION_QDQ_ELEMENTWISE_KERNEL_ID 6u

CamppStatus campp_aarch64_qlinear_conv_o4i4(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

CamppStatus campp_fused_bn_relu_quant(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

CamppStatus campp_fused_quant_qlinear_conv_o4i4(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

CamppStatus campp_fused_quant_qlinear_conv_scratch_bytes(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op, size_t *out_bytes);

CamppStatus campp_fused_dequant_relu_quant(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

CamppStatus campp_fused_dequant_sigmoid_mul(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

CamppStatus campp_fused_qdq_elementwise_quant(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

CamppStatus campp_fused_statistics_pooling(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

#endif
