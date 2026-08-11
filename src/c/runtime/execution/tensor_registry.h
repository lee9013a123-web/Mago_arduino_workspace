#ifndef CAMPP_RUNTIME_EXECUTION_TENSOR_REGISTRY_H
#define CAMPP_RUNTIME_EXECUTION_TENSOR_REGISTRY_H

#include <stdint.h>

#include "campp_runtime/status_code.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"
#include "internal/tensor_view.h"

/* Descriptor 표를 pointer가 연결된 TensorView 표로 바꾼다. */
CamppStatus campp_tensor_registry_create(
    const CamppRuntimeModel *model,
    const CamppActivationStorage *activation_storage,
    CamppTensorView **out_tensors, uint32_t *out_tensor_count);

void campp_tensor_registry_release(
    CamppTensorView **tensors, uint32_t *tensor_count);

#endif /* CAMPP_RUNTIME_EXECUTION_TENSOR_REGISTRY_H */
