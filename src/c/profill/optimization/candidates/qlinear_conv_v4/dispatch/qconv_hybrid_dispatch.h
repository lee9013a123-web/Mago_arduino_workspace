#ifndef CAMPP_PROFILL_QCONV_HYBRID_DISPATCH_H
#define CAMPP_PROFILL_QCONV_HYBRID_DISPATCH_H

#include "internal/kernel_registry.h"
#include "internal/runtime_model.h"

typedef enum CamppQconvHybridPath {
    CAMPP_QCONV_HYBRID_PATH_MAC_FIXED = 0,
    CAMPP_QCONV_HYBRID_PATH_V4 = 1
} CamppQconvHybridPath;

CamppQconvHybridPath campp_qconv_hybrid_select_path(
    const CamppTensorView *inputs, uint8_t input_count);

CamppStatus campp_qconv_hybrid_run(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

#endif /* CAMPP_PROFILL_QCONV_HYBRID_DISPATCH_H */
