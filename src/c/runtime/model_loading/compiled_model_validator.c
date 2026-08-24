/*
 * plan 검증 구현. 규칙과 의도는 compiled_model_validator.h를 본다.
 *
 * SHA-256은 integrity/sha256.c에 둔다. plan, self-contained package, 외부
 * deployment asset이 같은 checksum 구현을 공유한다.
 */

#include "compiled_model_validator.h"

#include <stdlib.h>
#include <string.h>

#include "internal/runtime_model.h"
#include "internal/tensor_view.h"
#include "memory_management/tensor_arena.h"

/* --------------------------------------------------------------------- */
/* header와 구역                                                          */
/* --------------------------------------------------------------------- */

static CamppStatus campp_read_plan_header(
    const CamppByteSpan *plan, CamppPlanHeader *out_header)
{
    CamppStatus status;

    status = campp_span_read_bytes(
        plan, CAMPP_PLAN_MAGIC_OFFSET, out_header->magic,
        CAMPP_PLAN_MAGIC_SIZE);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (memcmp(out_header->magic, CAMPP_PLAN_MAGIC, CAMPP_PLAN_MAGIC_SIZE) != 0) {
        return CAMPP_STATUS_INVALID_MAGIC;
    }

    status = campp_span_read_u32(
        plan, CAMPP_PLAN_FORMAT_VERSION_OFFSET, &out_header->format_version);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (out_header->format_version != CAMPP_PLAN_FORMAT_VERSION) {
        return CAMPP_STATUS_UNSUPPORTED_VERSION;
    }

    status = campp_span_read_u32(
        plan, CAMPP_PLAN_BUCKET_FRAMES_OFFSET, &out_header->bucket_frames);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u32(
        plan, CAMPP_PLAN_TENSOR_COUNT_OFFSET, &out_header->tensor_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u32(
        plan, CAMPP_PLAN_OPERATOR_COUNT_OFFSET, &out_header->operator_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u64(
        plan, CAMPP_PLAN_TENSOR_TABLE_OFFSET_OFFSET,
        &out_header->tensor_table_offset);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u64(
        plan, CAMPP_PLAN_OPERATOR_TABLE_OFFSET_OFFSET,
        &out_header->operator_table_offset);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u64(
        plan, CAMPP_PLAN_ATTRIBUTE_SECTION_OFFSET_OFFSET,
        &out_header->attribute_section_offset);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return campp_span_read_bytes(
        plan, CAMPP_PLAN_CHECKSUM_OFFSET, out_header->checksum,
        CAMPP_PLAN_CHECKSUM_SIZE);
}

static CamppStatus campp_validate_plan_checksum(
    const CamppByteSpan *plan, const CamppPlanHeader *header)
{
    CamppStatus status;
    CamppByteSpan payload;

    status = campp_span_slice(
        plan, CAMPP_PLAN_HEADER_SIZE, plan->size - CAMPP_PLAN_HEADER_SIZE,
        &payload);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    return campp_validate_sha256(
        payload.data, payload.size, header->checksum);
}

CamppStatus campp_validate_plan(
    const CamppByteSpan *plan, CamppPlanLayout *out_layout)
{
    CamppStatus status;
    CamppPlanHeader header;
    uint64_t tensor_bytes;
    uint64_t operator_bytes;
    uint64_t tensor_end;
    uint64_t operator_end;

    if (plan == NULL || out_layout == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (plan->size < CAMPP_PLAN_HEADER_SIZE) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    memset(&header, 0, sizeof(header));
    status = campp_read_plan_header(plan, &header);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    if (header.bucket_frames == 0u || header.tensor_count == 0u
        || header.operator_count == 0u) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    status = campp_checked_mul_u64(
        (uint64_t)header.tensor_count, CAMPP_TENSOR_DESCRIPTOR_SIZE,
        &tensor_bytes);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_checked_mul_u64(
        (uint64_t)header.operator_count, CAMPP_OPERATOR_DESCRIPTOR_SIZE,
        &operator_bytes);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    /* 세 구역은 header 뒤에서 겹치지 않고 이 순서로 놓인다. */
    if (header.tensor_table_offset < CAMPP_PLAN_HEADER_SIZE) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_span_check_aligned(
        plan, header.tensor_table_offset, tensor_bytes,
        CAMPP_PLAN_SECTION_ALIGNMENT);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_checked_add_u64(
        header.tensor_table_offset, tensor_bytes, &tensor_end);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (header.operator_table_offset < tensor_end) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_span_check_aligned(
        plan, header.operator_table_offset, operator_bytes,
        CAMPP_PLAN_SECTION_ALIGNMENT);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_checked_add_u64(
        header.operator_table_offset, operator_bytes, &operator_end);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (header.attribute_section_offset < operator_end) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_span_check_aligned(
        plan, header.attribute_section_offset, 0u,
        CAMPP_PLAN_SECTION_ALIGNMENT);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    status = campp_validate_plan_checksum(plan, &header);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    out_layout->header = header;
    status = campp_span_slice(
        plan, header.tensor_table_offset, tensor_bytes,
        &out_layout->tensor_table);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_slice(
        plan, header.operator_table_offset, operator_bytes,
        &out_layout->operator_table);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return campp_span_slice(
        plan, header.attribute_section_offset,
        plan->size - header.attribute_section_offset,
        &out_layout->attribute_section);
}

/* --------------------------------------------------------------------- */
/* descriptor                                                            */
/* --------------------------------------------------------------------- */

static bool campp_dtype_supported(uint8_t dtype)
{
    return campp_dtype_byte_size(dtype) != 0u;
}

static bool campp_storage_supported(uint8_t storage_type)
{
    return storage_type >= CAMPP_TENSOR_STORAGE_INPUT
        && storage_type <= CAMPP_TENSOR_STORAGE_VIEW;
}

static bool campp_opcode_supported(uint16_t opcode)
{
    return opcode >= CAMPP_OP_QLINEAR_CONV && opcode <= CAMPP_OP_UNSQUEEZE;
}

CamppStatus campp_validate_tensor_descriptor(
    const CamppTensorDescriptor *descriptor, uint32_t expected_id,
    uint32_t tensor_count, uint32_t operator_count)
{
    static const uint8_t known_flags = (uint8_t)(
        CAMPP_TENSOR_FLAG_READ_ONLY | CAMPP_TENSOR_FLAG_CONTIGUOUS
        | CAMPP_TENSOR_FLAG_EXTERNAL | CAMPP_TENSOR_FLAG_ALIASED
        | CAMPP_TENSOR_FLAG_DENSE_SLAB
        | CAMPP_TENSOR_FLAG_PACKED_QCONV_O4I4);
    uint64_t elements = 1u;
    uint8_t axis;
    CamppStatus status;

    if (descriptor == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    /* ID는 표에서의 자리와 같아야 한다. 그래야 index로 바로 찾을 수 있다. */
    if (descriptor->tensor_id != expected_id) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (!campp_dtype_supported(descriptor->dtype)) {
        return CAMPP_STATUS_UNSUPPORTED_DTYPE;
    }
    if (descriptor->rank > CAMPP_TENSOR_MAX_RANK) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (!campp_storage_supported(descriptor->storage_type)) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if ((descriptor->flags & (uint8_t)~known_flags) != 0u) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        if (axis < descriptor->rank) {
            if (descriptor->dimensions[axis] == 0u) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
            status = campp_checked_mul_u64(
                elements, (uint64_t)descriptor->dimensions[axis], &elements);
            if (status != CAMPP_STATUS_OK) {
                return status;
            }
        } else {
            if (descriptor->dimensions[axis] != 1u
                || descriptor->byte_strides[axis] != 0u) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
        }
    }

    /* shape와 dtype이 실제로 그 크기를 만드는지 대조한다. */
    {
        uint64_t expected_bytes;

        status = campp_checked_mul_u64(
            elements, (uint64_t)campp_dtype_byte_size(descriptor->dtype),
            &expected_bytes);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        if (expected_bytes == 0u
            || descriptor->logical_byte_size != expected_bytes) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    if (descriptor->storage_span_bytes < descriptor->logical_byte_size) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if ((descriptor->flags & CAMPP_TENSOR_FLAG_PACKED_QCONV_O4I4) != 0u) {
        if (descriptor->storage_type != CAMPP_TENSOR_STORAGE_CONSTANT ||
            (descriptor->dtype != CAMPP_DTYPE_INT8 &&
             descriptor->dtype != CAMPP_DTYPE_UINT8) ||
            (descriptor->rank != 3u && descriptor->rank != 4u) ||
            descriptor->storage_span_bytes % 16u != 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    } else if (descriptor->storage_type == CAMPP_TENSOR_STORAGE_CONSTANT &&
               descriptor->storage_span_bytes != descriptor->logical_byte_size) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    /* data_offset의 기준과 DENSE_SLAB 표시는 storage_type과 일치해야 한다. */
    switch (descriptor->storage_type) {
    case CAMPP_TENSOR_STORAGE_INPUT:
        if (descriptor->data_offset != CAMPP_INVALID_DATA_OFFSET ||
            (descriptor->flags & CAMPP_TENSOR_FLAG_DENSE_SLAB) != 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        break;
    case CAMPP_TENSOR_STORAGE_CONSTANT:
        if (descriptor->data_offset == CAMPP_INVALID_DATA_OFFSET ||
            (descriptor->flags & CAMPP_TENSOR_FLAG_DENSE_SLAB) != 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        break;
    case CAMPP_TENSOR_STORAGE_ACTIVATION:
    case CAMPP_TENSOR_STORAGE_OUTPUT: {
        const bool dense =
            (descriptor->flags & CAMPP_TENSOR_FLAG_DENSE_SLAB) != 0u;
        const bool has_offset =
            descriptor->data_offset != CAMPP_INVALID_DATA_OFFSET;
        if (dense != has_offset) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        break;
    }
    case CAMPP_TENSOR_STORAGE_VIEW:
        if ((descriptor->flags & CAMPP_TENSOR_FLAG_DENSE_SLAB) != 0u ||
            descriptor->data_offset == CAMPP_INVALID_DATA_OFFSET) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        break;
    default:
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    if (descriptor->storage_type == CAMPP_TENSOR_STORAGE_VIEW) {
        if (descriptor->alias_of_tensor_id >= tensor_count ||
            descriptor->alias_of_tensor_id == descriptor->tensor_id) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        if ((descriptor->flags & CAMPP_TENSOR_FLAG_ALIASED) == 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    } else if (descriptor->alias_of_tensor_id != CAMPP_INVALID_TENSOR_ID) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    if (descriptor->first_use != CAMPP_INVALID_OPERATOR_INDEX) {
        if (descriptor->first_use >= operator_count) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    if (descriptor->last_use != CAMPP_INVALID_OPERATOR_INDEX) {
        if (descriptor->last_use >= operator_count) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    if (descriptor->first_use != CAMPP_INVALID_OPERATOR_INDEX
        && descriptor->last_use != CAMPP_INVALID_OPERATOR_INDEX
        && descriptor->first_use > descriptor->last_use) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_validate_operator_descriptor(
    const CamppOperatorDescriptor *descriptor, uint32_t expected_id,
    uint32_t tensor_count, uint64_t attribute_section_size)
{
    uint8_t slot;
    CamppStatus status;

    if (descriptor == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor->operator_id != expected_id) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (!campp_opcode_supported(descriptor->opcode)) {
        return CAMPP_STATUS_UNSUPPORTED_OPCODE;
    }
    if (descriptor->input_count == 0u
        || descriptor->input_count > CAMPP_OPERATOR_INPUT_CAPACITY) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (descriptor->output_count != CAMPP_OPERATOR_OUTPUT_CAPACITY) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (descriptor->backend_id > CAMPP_BACKEND_GPU_VULKAN) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    for (slot = 0u; slot < CAMPP_OPERATOR_INPUT_CAPACITY; ++slot) {
        if (slot < descriptor->input_count) {
            if (descriptor->input_tensor_ids[slot] >= tensor_count) {
                return CAMPP_STATUS_CORRUPT_PLAN;
            }
        } else if (descriptor->input_tensor_ids[slot]
                   != CAMPP_INVALID_TENSOR_ID) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    if (descriptor->output_tensor_ids[0] >= tensor_count) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    if (descriptor->attribute_size == 0u) {
        if (descriptor->attribute_offset != 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    } else {
        uint64_t end;

        if ((descriptor->attribute_offset
             & (uint64_t)(CAMPP_PLAN_SECTION_ALIGNMENT - 1u)) != 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        status = campp_checked_add_u64(
            descriptor->attribute_offset, (uint64_t)descriptor->attribute_size,
            &end);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        if (end > attribute_section_size) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }
    return CAMPP_STATUS_OK;
}

/* --------------------------------------------------------------------- */
/* 표 전체                                                                */
/* --------------------------------------------------------------------- */

CamppStatus campp_validate_model(const struct CamppRuntimeModel *model)
{
    const CamppRuntimeModel *target = model;
    uint32_t index;
    uint8_t slot;
    CamppStatus status;
    uint8_t *produced;
    uint8_t *has_producer;
    bool uses_arena;

    if (target == NULL || target->tensors == NULL
        || target->operators == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    status = campp_tensor_arena_model_uses_arena(target, &uses_arena);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (uses_arena) {
        size_t arena_size;
        status = campp_tensor_arena_validate_layout(target, &arena_size);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        (void)arena_size;
    }

    /* VIEW는 arena-owned backing 또는 외부 INPUT을 직접 가리킨다. */
    for (index = 0u; index < target->tensor_count; ++index) {
        const CamppTensorDescriptor *view = &target->tensors[index];
        const CamppTensorDescriptor *base;
        uint64_t remaining;

        if (view->storage_type != CAMPP_TENSOR_STORAGE_VIEW) {
            continue;
        }
        if (view->alias_of_tensor_id >= target->tensor_count ||
            view->alias_of_tensor_id == index) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        base = &target->tensors[view->alias_of_tensor_id];
        if ((base->storage_type != CAMPP_TENSOR_STORAGE_INPUT &&
             base->storage_type != CAMPP_TENSOR_STORAGE_ACTIVATION &&
             base->storage_type != CAMPP_TENSOR_STORAGE_OUTPUT) ||
            base->dtype != view->dtype ||
            view->data_offset > base->storage_span_bytes) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        remaining = base->storage_span_bytes - view->data_offset;
        if (view->storage_span_bytes > remaining) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
        /* INPUT은 caller가 inference 전체 동안 소유하므로 정적 수명 포위가 없다. */
        if (base->storage_type != CAMPP_TENSOR_STORAGE_INPUT &&
            (
            base->first_use == CAMPP_INVALID_OPERATOR_INDEX ||
            base->last_use == CAMPP_INVALID_OPERATOR_INDEX ||
            view->first_use == CAMPP_INVALID_OPERATOR_INDEX ||
            view->last_use == CAMPP_INVALID_OPERATOR_INDEX ||
            base->first_use > view->first_use ||
            base->last_use < view->last_use)) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }
    }

    /* CONSTANT는 weights.bin 안을 가리켜야 한다. */
    for (index = 0u; index < target->tensor_count; ++index) {
        const CamppTensorDescriptor *descriptor = &target->tensors[index];
        uint64_t end;

        if (descriptor->storage_type != CAMPP_TENSOR_STORAGE_CONSTANT) {
            continue;
        }
        status = campp_checked_add_u64(
            descriptor->data_offset, descriptor->storage_span_bytes, &end);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        if (end > (uint64_t)target->weights_size) {
            return CAMPP_STATUS_CORRUPT_WEIGHTS;
        }
    }

    /*
     * 표를 위에서 아래로 읽었을 때 실행이 성립하는지 본다. exporter가 이미
     * 보장하지만, 여기서 다시 보는 이유는 파일이 도중에 바뀌었을 수 있기
     * 때문이다. 순서가 뒤집힌 plan은 초기화되지 않은 버퍼를 읽으면서도
     * 조용히 끝나므로 실행 전에 잡아야 한다.
     */
    produced = (uint8_t *)calloc((size_t)target->tensor_count, sizeof(uint8_t));
    has_producer =
        (uint8_t *)calloc((size_t)target->tensor_count, sizeof(uint8_t));
    if (produced == NULL || has_producer == NULL) {
        free(produced);
        free(has_producer);
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    status = CAMPP_STATUS_OK;
    for (index = 0u; index < target->operator_count; ++index) {
        uint32_t output_id = target->operators[index].output_tensor_ids[0];

        if (has_producer[output_id] != 0u) {
            status = CAMPP_STATUS_CORRUPT_PLAN;
            break;
        }
        has_producer[output_id] = 1u;
    }
    for (index = 0u; index < target->tensor_count; ++index) {
        uint8_t storage = target->tensors[index].storage_type;

        if (storage == CAMPP_TENSOR_STORAGE_INPUT
            || storage == CAMPP_TENSOR_STORAGE_CONSTANT
            || (storage == CAMPP_TENSOR_STORAGE_VIEW
                && has_producer[index] == 0u)) {
            produced[index] = 1u;
        }
    }

    for (index = 0u; index < target->operator_count && status == CAMPP_STATUS_OK;
         ++index) {
        const CamppOperatorDescriptor *descriptor = &target->operators[index];

        for (slot = 0u; slot < descriptor->input_count; ++slot) {
            if (produced[descriptor->input_tensor_ids[slot]] == 0u) {
                status = CAMPP_STATUS_CORRUPT_PLAN;
                break;
            }
        }
        if (status == CAMPP_STATUS_OK) {
            uint32_t output_id = descriptor->output_tensor_ids[0];

            if (produced[output_id] != 0u) {
                /* 같은 Tensor를 두 번 생산하면 어느 값이 맞는지 알 수 없다. */
                status = CAMPP_STATUS_CORRUPT_PLAN;
            } else {
                produced[output_id] = 1u;
            }
        }
    }

    if (status == CAMPP_STATUS_OK) {
        for (index = 0u; index < target->tensor_count; ++index) {
            if (produced[index] == 0u) {
                status = CAMPP_STATUS_CORRUPT_PLAN;
                break;
            }
        }
    }

    free(produced);
    free(has_producer);
    return status;
}
