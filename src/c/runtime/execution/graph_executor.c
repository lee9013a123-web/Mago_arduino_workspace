/* execution plan의 Operator table을 순서대로 실행한다. */

#include "graph_executor.h"

#include <stdint.h>

#include "campp_runtime/operator_descriptor.h"
#include "execution/tensor_registry.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_model.h"
#include "internal/tensor_view.h"
#include "memory_management/memory_bounds_checker.h"

static CamppStatus campp_graph_fail(
    CamppRuntimeContext *context, CamppStatus status)
{
    context->last_status = status;
    return status;
}

static CamppStatus campp_graph_validate_context(
    const CamppRuntimeContext *context)
{
    if (context == NULL || context->model == NULL || context->tensors == NULL ||
        context->registry == NULL || context->resolved_kernels == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (context->model->operators == NULL ||
        context->model->operator_count == 0u ||
        context->tensor_count != context->model->tensor_count ||
        context->resolved_kernel_count != context->model->operator_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_graph_execute_operator(
    CamppRuntimeContext *context, uint32_t operator_id)
{
    const CamppRuntimeModel *model;
    const CamppOperatorDescriptor *operator_descriptor;
    const CamppKernelEntry *kernel;
    CamppTensorView input_views[CAMPP_OPERATOR_INPUT_CAPACITY];
    CamppTensorView output_views[CAMPP_OPERATOR_OUTPUT_CAPACITY];
    uint8_t slot;
    CamppStatus status;
    CamppStatus kernel_status;
    CamppStatus post_check_status;

    status = campp_graph_validate_context(context);
    if (status != CAMPP_STATUS_OK) {
        if (context != NULL) {
            return campp_graph_fail(context, status);
        }
        return status;
    }
    model = context->model;
    if (operator_id >= model->operator_count) {
        return campp_graph_fail(context, CAMPP_STATUS_INVALID_ARGUMENT);
    }

    context->diagnostics.current_operator_id = operator_id;
    operator_descriptor = &model->operators[operator_id];
    if (operator_descriptor->operator_id != operator_id ||
        operator_descriptor->input_count > CAMPP_OPERATOR_INPUT_CAPACITY ||
        operator_descriptor->output_count > CAMPP_OPERATOR_OUTPUT_CAPACITY) {
        return campp_graph_fail(context, CAMPP_STATUS_CORRUPT_PLAN);
    }

    kernel = context->resolved_kernels[operator_id];
    if (kernel == NULL || kernel->run == NULL ||
        kernel->opcode != operator_descriptor->opcode ||
        kernel->kernel_id != operator_descriptor->kernel_id) {
        return campp_graph_fail(context, CAMPP_STATUS_MISSING_KERNEL);
    }

    for (slot = 0u; slot < operator_descriptor->input_count; ++slot) {
        CamppTensorView *view = NULL;

        status = campp_runtime_context_tensor(
            context, operator_descriptor->input_tensor_ids[slot], &view);
        if (status != CAMPP_STATUS_OK) {
            return campp_graph_fail(context, status);
        }
        if (view == NULL) {
            return campp_graph_fail(context, CAMPP_STATUS_CORRUPT_PLAN);
        }
        input_views[slot] = *view;
    }
    for (slot = 0u; slot < operator_descriptor->output_count; ++slot) {
        CamppTensorView *view = NULL;

        status = campp_runtime_context_tensor(
            context, operator_descriptor->output_tensor_ids[slot], &view);
        if (status != CAMPP_STATUS_OK) {
            return campp_graph_fail(context, status);
        }
        if (view == NULL) {
            return campp_graph_fail(context, CAMPP_STATUS_CORRUPT_PLAN);
        }
        output_views[slot] = *view;
    }

    status = campp_memory_bounds_check_operator(context, operator_descriptor);
    if (status != CAMPP_STATUS_OK) {
        return campp_graph_fail(context, status);
    }

    kernel_status = kernel->run(
        model, operator_descriptor, input_views,
        operator_descriptor->input_count, output_views,
        operator_descriptor->output_count, context->scratch,
        context->scratch_size);

    /* Kernel 실패 시에도 guard 손상을 먼저 보고해 실제 원인을 보존한다. */
    post_check_status = campp_memory_bounds_check_operator(
        context, operator_descriptor);
    if (post_check_status != CAMPP_STATUS_OK) {
        return campp_graph_fail(context, post_check_status);
    }
    if (kernel_status != CAMPP_STATUS_OK) {
        return campp_graph_fail(context, kernel_status);
    }

    /*
     * Arena에서는 이 출력 주소가 이후 Tensor에 재사용될 수 있다. 진단 도구는
     * 값이 살아 있는 지금 복사해야 하므로 Operator 완료 직후 callback을 부른다.
     */
    if (context->diagnostics.tensor_ready != NULL) {
        for (slot = 0u; slot < operator_descriptor->output_count; ++slot) {
            const uint32_t tensor_id =
                operator_descriptor->output_tensor_ids[slot];
            status = context->diagnostics.tensor_ready(
                context->diagnostics.tensor_ready_user_data,
                operator_id, tensor_id, &context->tensors[tensor_id]);
            if (status != CAMPP_STATUS_OK) {
                return campp_graph_fail(context, status);
            }
        }
    }

    context->diagnostics.executed_operator_count += 1u;
    context->last_status = CAMPP_STATUS_OK;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_graph_execute(CamppRuntimeContext *context)
{
    const CamppRuntimeModel *model;
    uint32_t operator_id;
    CamppStatus status;

    status = campp_graph_validate_context(context);
    if (status != CAMPP_STATUS_OK) {
        if (context != NULL) {
            return campp_graph_fail(context, status);
        }
        return status;
    }
    model = context->model;

    status = campp_runtime_context_reset(context);
    if (status != CAMPP_STATUS_OK) {
        return campp_graph_fail(context, status);
    }

    for (operator_id = 0u; operator_id < model->operator_count; ++operator_id) {
        status = campp_graph_execute_operator(context, operator_id);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }

    context->last_status = CAMPP_STATUS_OK;
    return CAMPP_STATUS_OK;
}
