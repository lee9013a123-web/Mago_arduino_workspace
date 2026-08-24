#ifndef CAMPP_RUNTIME_MEMORY_MANAGEMENT_MEMORY_BOUNDS_CHECKER_H
#define CAMPP_RUNTIME_MEMORY_MANAGEMENT_MEMORY_BOUNDS_CHECKER_H

#include <stdint.h>

#include "campp_runtime/operator_descriptor.h"
#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"
#include "internal/tensor_view.h"

/* Descriptor와 실제 view의 shape, stride, byte 범위를 비교한다. */
CamppStatus campp_memory_bounds_check_view(
    const CamppTensorDescriptor *descriptor, const CamppTensorView *view);

/* CONSTANT의 data_offset + span이 weights.bin 안에 있는지 확인한다. */
CamppStatus campp_memory_bounds_check_constant(
    const CamppRuntimeModel *model,
    const CamppTensorDescriptor *descriptor);

/* Context에 연결된 Tensor 하나 또는 전체를 검사한다. */
CamppStatus campp_memory_bounds_check_tensor(
    const CamppRuntimeContext *context, uint32_t tensor_id);

CamppStatus campp_memory_bounds_check_all(
    const CamppRuntimeContext *context);

/* Kernel 호출 직전에 사용할 입출력 Tensor 검사다. */
CamppStatus campp_memory_bounds_check_operator(
    const CamppRuntimeContext *context,
    const CamppOperatorDescriptor *operator_descriptor);

/* Reference storage의 모든 canary를 확인한다. */
CamppStatus campp_memory_bounds_check_guards(
    const CamppRuntimeContext *context);

#endif /* CAMPP_RUNTIME_MEMORY_MANAGEMENT_MEMORY_BOUNDS_CHECKER_H */
