#ifndef CAMPP_PROFILL_QCONV_V4_DISPATCH_H
#define CAMPP_PROFILL_QCONV_V4_DISPATCH_H

#include "internal/kernel_registry.h"
#include "internal/runtime_model.h"

CamppStatus campp_qconv_v4_run(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

#endif /* CAMPP_PROFILL_QCONV_V4_DISPATCH_H */
