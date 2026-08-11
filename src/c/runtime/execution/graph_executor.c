/* execution plan의 Operator table을 0번부터 순서대로 실행한다. */

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

CamppStatus campp_graph_execute(CamppRuntimeContext *context)
{
    const CamppRuntimeModel *model;
    uint32_t operator_id;
    CamppStatus status;

    if (context == NULL || context->model == NULL || context->tensors == NULL ||
        context->registry == NULL || context->resolved_kernels == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    model = context->model;
    if (model->operators == NULL || model->operator_count == 0u ||
        context->tensor_count != model->tensor_count ||
        context->resolved_kernel_count != model->operator_count) {
        return campp_graph_fail(context, CAMPP_STATUS_INVALID_ARGUMENT);
    }

    status = campp_runtime_context_reset(context);
    if (status != CAMPP_STATUS_OK) {
        return campp_graph_fail(context, status);
    }

    for (operator_id = 0u; operator_id < model->operator_count; ++operator_id) {
        const CamppOperatorDescriptor *operator_descriptor =
            &model->operators[operator_id];
        const CamppKernelEntry *kernel;
        CamppTensorView input_views[CAMPP_OPERATOR_INPUT_CAPACITY];
        CamppTensorView output_views[CAMPP_OPERATOR_OUTPUT_CAPACITY];
        uint8_t slot;
        CamppStatus kernel_status;
        CamppStatus post_check_status;

        context->diagnostics.current_operator_id = operator_id;
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
                return campp_graph_fail(
                    context, CAMPP_STATUS_CORRUPT_PLAN);
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
                return campp_graph_fail(
                    context, CAMPP_STATUS_CORRUPT_PLAN);
            }
            output_views[slot] = *view;
        }

        status = campp_memory_bounds_check_operator(
            context, operator_descriptor);
        if (status != CAMPP_STATUS_OK) {
            return campp_graph_fail(context, status);
        }

        kernel_status = kernel->run(
            model, operator_descriptor, input_views,
            operator_descriptor->input_count, output_views,
            operator_descriptor->output_count, context->scratch,
            context->scratch_size);

        /* 실패한 Kernel도 메모리를 건드렸을 수 있으므로 guard를 먼저 확인한다. */
        post_check_status = campp_memory_bounds_check_operator(
            context, operator_descriptor);
        if (post_check_status != CAMPP_STATUS_OK) {
            return campp_graph_fail(context, post_check_status);
        }
        if (kernel_status != CAMPP_STATUS_OK) {
            return campp_graph_fail(context, kernel_status);
        }

        context->diagnostics.executed_operator_count += 1u;
    }

    context->last_status = CAMPP_STATUS_OK;
    return CAMPP_STATUS_OK;
}
