/* opcode와 kernel_id를 backend의 정적 함수 표에서 찾는 초기화 경로다. */

#include "internal/kernel_registry.h"

#include <stddef.h>
#include <stdint.h>

#include "internal/runtime_model.h"

CamppStatus campp_kernel_registry_lookup(
    const CamppKernelRegistry *registry, uint16_t opcode, uint16_t kernel_id,
    const CamppKernelEntry **out_entry)
{
    size_t index;

    if (registry == NULL || out_entry == NULL ||
        (registry->entry_count != 0u && registry->entries == NULL)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    *out_entry = NULL;
    for (index = 0u; index < registry->entry_count; ++index) {
        const CamppKernelEntry *entry = &registry->entries[index];

        if (entry->opcode == opcode && entry->kernel_id == kernel_id) {
            if (entry->run == NULL) {
                return CAMPP_STATUS_MISSING_KERNEL;
            }
            *out_entry = entry;
            return CAMPP_STATUS_OK;
        }
    }
    return CAMPP_STATUS_MISSING_KERNEL;
}

CamppStatus campp_kernel_registry_validate(const CamppKernelRegistry *registry)
{
    size_t first;
    size_t second;

    if (registry == NULL || registry->backend_id == CAMPP_BACKEND_AUTO ||
        registry->entries == NULL || registry->entry_count == 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    for (first = 0u; first < registry->entry_count; ++first) {
        const CamppKernelEntry *entry = &registry->entries[first];

        if (entry->opcode <= CAMPP_OP_INVALID ||
            entry->opcode > CAMPP_OP_UNSQUEEZE || entry->run == NULL) {
            return CAMPP_STATUS_MISSING_KERNEL;
        }
        for (second = first + 1u; second < registry->entry_count; ++second) {
            if (entry->opcode == registry->entries[second].opcode &&
                entry->kernel_id == registry->entries[second].kernel_id) {
                return CAMPP_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_kernel_registry_covers_model(
    const CamppKernelRegistry *registry, const CamppRuntimeModel *model,
    uint32_t *out_failing_operator_id, uint16_t *out_missing_opcode)
{
    uint32_t operator_id;
    CamppStatus status;

    if (registry == NULL || model == NULL ||
        (model->operator_count != 0u && model->operators == NULL)) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (out_failing_operator_id != NULL) {
        *out_failing_operator_id = CAMPP_INVALID_OPERATOR_INDEX;
    }
    if (out_missing_opcode != NULL) {
        *out_missing_opcode = CAMPP_OP_INVALID;
    }

    status = campp_kernel_registry_validate(registry);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    for (operator_id = 0u; operator_id < model->operator_count; ++operator_id) {
        const CamppOperatorDescriptor *operator_descriptor =
            &model->operators[operator_id];
        const CamppKernelEntry *entry = NULL;

        if (operator_descriptor->operator_id != operator_id) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        if (operator_descriptor->backend_id != CAMPP_BACKEND_AUTO &&
            operator_descriptor->backend_id != registry->backend_id) {
            if (out_failing_operator_id != NULL) {
                *out_failing_operator_id = operator_id;
            }
            if (out_missing_opcode != NULL) {
                *out_missing_opcode = operator_descriptor->opcode;
            }
            return CAMPP_STATUS_BACKEND_MISMATCH;
        }
        status = campp_kernel_registry_lookup(
            registry, operator_descriptor->opcode,
            operator_descriptor->kernel_id, &entry);
        if (status != CAMPP_STATUS_OK) {
            if (out_failing_operator_id != NULL) {
                *out_failing_operator_id = operator_id;
            }
            if (out_missing_opcode != NULL) {
                *out_missing_opcode = operator_descriptor->opcode;
            }
            return status;
        }
        (void)entry;
    }
    return CAMPP_STATUS_OK;
}
