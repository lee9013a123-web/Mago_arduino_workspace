/*
 * plan_*.bin과 weights.bin을 읽어 CamppRuntimeModel을 만든다.
 *
 * 흐름은 한 방향이다.
 *
 *   파일 열기 -> header 읽기 -> validator -> Tensor table -> Operator table
 *   -> weights 연결 -> 표 전체 검증
 *
 * 어느 단계에서 실패하든 그때까지 잡은 자원을 스스로 정리하고, model은 손대지
 * 않은 상태로 남긴다. 부분적으로 채워진 model이 밖으로 나가면 호출자가 성공과
 * 실패를 구분할 수 없게 된다.
 *
 * 디스크 바이트를 구조체로 cast하지 않는다. descriptor는 little-endian reader로
 * 한 필드씩 읽어 model이 소유하는 배열에 넣는다. plan 파일은 attribute section
 * 때문에 계속 살아 있어야 하므로 model이 함께 들고 있는다.
 *
 * graph의 입출력 Tensor ID는 plan에 따로 적혀 있지 않다. storage_type이 INPUT과
 * OUTPUT인 Tensor를 표에서 골라내며, 이는 exporter가 그 두 storage를 정의한
 * 방식과 같다. 덕분에 형식에 목록을 하나 더 둘 필요가 없다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "binary_section_reader.h"
#include "compiled_model_validator.h"
#include "internal/runtime_model.h"

/* Tensor descriptor 안의 필드 위치. tensor_descriptor.h의 static assert와 같다. */
#define CAMPP_TD_TENSOR_ID 0u
#define CAMPP_TD_DTYPE 4u
#define CAMPP_TD_RANK 5u
#define CAMPP_TD_STORAGE_TYPE 6u
#define CAMPP_TD_FLAGS 7u
#define CAMPP_TD_DIMENSIONS 8u
#define CAMPP_TD_BYTE_STRIDES 24u
#define CAMPP_TD_DATA_OFFSET 40u
#define CAMPP_TD_LOGICAL_BYTE_SIZE 48u
#define CAMPP_TD_STORAGE_SPAN_BYTES 56u
#define CAMPP_TD_ALIAS_OF_TENSOR_ID 64u
#define CAMPP_TD_QUANTIZATION_INDEX 68u
#define CAMPP_TD_FIRST_USE 72u
#define CAMPP_TD_LAST_USE 76u

/* Operator descriptor 안의 필드 위치. operator_descriptor.h와 같다. */
#define CAMPP_OD_OPERATOR_ID 0u
#define CAMPP_OD_OPCODE 4u
#define CAMPP_OD_INPUT_COUNT 6u
#define CAMPP_OD_OUTPUT_COUNT 7u
#define CAMPP_OD_INPUT_TENSOR_IDS 8u
#define CAMPP_OD_OUTPUT_TENSOR_IDS 44u
#define CAMPP_OD_ATTRIBUTE_OFFSET 48u
#define CAMPP_OD_ATTRIBUTE_SIZE 56u
#define CAMPP_OD_BACKEND_ID 60u
#define CAMPP_OD_KERNEL_ID 62u

/* attribute block과 record 안의 필드 위치. */
#define CAMPP_AB_RECORD_COUNT 0u
#define CAMPP_AB_TOTAL_SIZE 4u
#define CAMPP_AR_KEY 0u
#define CAMPP_AR_VALUE_TYPE 2u
#define CAMPP_AR_VALUE_COUNT 3u
#define CAMPP_AR_RESERVED 4u

/* --------------------------------------------------------------------- */
/* 파일 읽기                                                              */
/* --------------------------------------------------------------------- */

static CamppStatus campp_read_whole_file(
    const char *path, uint8_t **out_bytes, size_t *out_size)
{
    FILE *handle;
    long length;
    size_t size;
    uint8_t *buffer;
    size_t read_bytes;

    if (path == NULL || out_bytes == NULL || out_size == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    handle = fopen(path, "rb");
    if (handle == NULL) {
        return CAMPP_STATUS_FILE_NOT_FOUND;
    }
    if (fseek(handle, 0L, SEEK_END) != 0) {
        fclose(handle);
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    length = ftell(handle);
    if (length < 0L) {
        fclose(handle);
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    if (fseek(handle, 0L, SEEK_SET) != 0) {
        fclose(handle);
        return CAMPP_STATUS_FILE_READ_FAILED;
    }

    size = (size_t)length;
    /* 크기 0인 파일에도 NULL이 아닌 pointer를 돌려준다. */
    buffer = (uint8_t *)malloc(size == 0u ? 1u : size);
    if (buffer == NULL) {
        fclose(handle);
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    read_bytes = (size == 0u) ? 0u : fread(buffer, 1u, size, handle);
    fclose(handle);

    if (read_bytes != size) {
        free(buffer);
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    *out_bytes = buffer;
    *out_size = size;
    return CAMPP_STATUS_OK;
}

/* --------------------------------------------------------------------- */
/* descriptor 디코드                                                      */
/* --------------------------------------------------------------------- */

static CamppStatus campp_decode_tensor_descriptor(
    const CamppByteSpan *record, CamppTensorDescriptor *out_descriptor)
{
    CamppStatus status;
    uint8_t axis;

    status = campp_span_read_u32(
        record, CAMPP_TD_TENSOR_ID, &out_descriptor->tensor_id);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u8(record, CAMPP_TD_DTYPE, &out_descriptor->dtype);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u8(record, CAMPP_TD_RANK, &out_descriptor->rank);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u8(
        record, CAMPP_TD_STORAGE_TYPE, &out_descriptor->storage_type);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u8(record, CAMPP_TD_FLAGS, &out_descriptor->flags);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        status = campp_span_read_u32(
            record, CAMPP_TD_DIMENSIONS + (uint64_t)axis * 4u,
            &out_descriptor->dimensions[axis]);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        status = campp_span_read_u32(
            record, CAMPP_TD_BYTE_STRIDES + (uint64_t)axis * 4u,
            &out_descriptor->byte_strides[axis]);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    status = campp_span_read_u64(
        record, CAMPP_TD_DATA_OFFSET, &out_descriptor->data_offset);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u64(
        record, CAMPP_TD_LOGICAL_BYTE_SIZE, &out_descriptor->logical_byte_size);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u64(
        record, CAMPP_TD_STORAGE_SPAN_BYTES,
        &out_descriptor->storage_span_bytes);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u32(
        record, CAMPP_TD_ALIAS_OF_TENSOR_ID,
        &out_descriptor->alias_of_tensor_id);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u32(
        record, CAMPP_TD_QUANTIZATION_INDEX,
        &out_descriptor->quantization_index);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u32(
        record, CAMPP_TD_FIRST_USE, &out_descriptor->first_use);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return campp_span_read_u32(
        record, CAMPP_TD_LAST_USE, &out_descriptor->last_use);
}

static CamppStatus campp_decode_operator_descriptor(
    const CamppByteSpan *record, CamppOperatorDescriptor *out_descriptor)
{
    CamppStatus status;
    uint8_t slot;

    status = campp_span_read_u32(
        record, CAMPP_OD_OPERATOR_ID, &out_descriptor->operator_id);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u16(
        record, CAMPP_OD_OPCODE, &out_descriptor->opcode);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u8(
        record, CAMPP_OD_INPUT_COUNT, &out_descriptor->input_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u8(
        record, CAMPP_OD_OUTPUT_COUNT, &out_descriptor->output_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    for (slot = 0u; slot < CAMPP_OPERATOR_INPUT_CAPACITY; ++slot) {
        status = campp_span_read_u32(
            record, CAMPP_OD_INPUT_TENSOR_IDS + (uint64_t)slot * 4u,
            &out_descriptor->input_tensor_ids[slot]);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    for (slot = 0u; slot < CAMPP_OPERATOR_OUTPUT_CAPACITY; ++slot) {
        status = campp_span_read_u32(
            record, CAMPP_OD_OUTPUT_TENSOR_IDS + (uint64_t)slot * 4u,
            &out_descriptor->output_tensor_ids[slot]);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    status = campp_span_read_u64(
        record, CAMPP_OD_ATTRIBUTE_OFFSET, &out_descriptor->attribute_offset);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u32(
        record, CAMPP_OD_ATTRIBUTE_SIZE, &out_descriptor->attribute_size);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u16(
        record, CAMPP_OD_BACKEND_ID, &out_descriptor->backend_id);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return campp_span_read_u16(
        record, CAMPP_OD_KERNEL_ID, &out_descriptor->kernel_id);
}

static CamppStatus campp_load_tensor_table(
    const CamppPlanLayout *layout, CamppRuntimeModel *model)
{
    uint32_t index;
    CamppStatus status;

    model->tensors = (CamppTensorDescriptor *)calloc(
        (size_t)layout->header.tensor_count, sizeof(CamppTensorDescriptor));
    if (model->tensors == NULL) {
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    model->tensor_count = layout->header.tensor_count;

    for (index = 0u; index < model->tensor_count; ++index) {
        CamppByteSpan record;

        status = campp_span_slice(
            &layout->tensor_table,
            (uint64_t)index * CAMPP_TENSOR_DESCRIPTOR_SIZE,
            CAMPP_TENSOR_DESCRIPTOR_SIZE, &record);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        status = campp_decode_tensor_descriptor(&record, &model->tensors[index]);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        status = campp_validate_tensor_descriptor(
            &model->tensors[index], index, model->tensor_count,
            layout->header.operator_count);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    return CAMPP_STATUS_OK;
}

static CamppStatus campp_load_operator_table(
    const CamppPlanLayout *layout, CamppRuntimeModel *model)
{
    uint32_t index;
    CamppStatus status;

    model->operators = (CamppOperatorDescriptor *)calloc(
        (size_t)layout->header.operator_count, sizeof(CamppOperatorDescriptor));
    if (model->operators == NULL) {
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    model->operator_count = layout->header.operator_count;

    for (index = 0u; index < model->operator_count; ++index) {
        CamppByteSpan record;

        status = campp_span_slice(
            &layout->operator_table,
            (uint64_t)index * CAMPP_OPERATOR_DESCRIPTOR_SIZE,
            CAMPP_OPERATOR_DESCRIPTOR_SIZE, &record);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        status = campp_decode_operator_descriptor(
            &record, &model->operators[index]);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        status = campp_validate_operator_descriptor(
            &model->operators[index], index, model->tensor_count,
            layout->attribute_section.size);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    return CAMPP_STATUS_OK;
}

/* storage_type이 INPUT과 OUTPUT인 Tensor를 골라 graph 경계를 만든다. */
static CamppStatus campp_collect_graph_io(CamppRuntimeModel *model)
{
    uint32_t index;
    uint32_t input_index = 0u;
    uint32_t output_index = 0u;

    model->input_count = 0u;
    model->output_count = 0u;
    for (index = 0u; index < model->tensor_count; ++index) {
        uint8_t storage = model->tensors[index].storage_type;

        if (storage == CAMPP_TENSOR_STORAGE_INPUT) {
            model->input_count += 1u;
        } else if (storage == CAMPP_TENSOR_STORAGE_OUTPUT) {
            model->output_count += 1u;
        }
    }
    if (model->input_count == 0u || model->output_count == 0u) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    model->input_tensor_ids =
        (uint32_t *)calloc((size_t)model->input_count, sizeof(uint32_t));
    model->output_tensor_ids =
        (uint32_t *)calloc((size_t)model->output_count, sizeof(uint32_t));
    if (model->input_tensor_ids == NULL || model->output_tensor_ids == NULL) {
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }

    for (index = 0u; index < model->tensor_count; ++index) {
        uint8_t storage = model->tensors[index].storage_type;

        if (storage == CAMPP_TENSOR_STORAGE_INPUT) {
            model->input_tensor_ids[input_index] = index;
            input_index += 1u;
        } else if (storage == CAMPP_TENSOR_STORAGE_OUTPUT) {
            model->output_tensor_ids[output_index] = index;
            output_index += 1u;
        }
    }
    return CAMPP_STATUS_OK;
}

/* --------------------------------------------------------------------- */
/* 공개 진입점                                                            */
/* --------------------------------------------------------------------- */

void campp_runtime_model_release(CamppRuntimeModel *model)
{
    if (model == NULL) {
        return;
    }
    free(model->tensors);
    free(model->operators);
    free(model->input_tensor_ids);
    free(model->output_tensor_ids);
    if (model->owns_plan_bytes) {
        free(model->plan_bytes);
    }
    if (model->owns_weights) {
        /* malloc으로 얻은 메모리라 const를 떼고 반납해도 안전하다. */
        void *owned;

        memcpy(&owned, &model->weights, sizeof(owned));
        free(owned);
    }
    memset(model, 0, sizeof(*model));
}

CamppStatus campp_runtime_model_adopt(
    uint8_t *plan_bytes, size_t plan_size, bool take_plan_ownership,
    const uint8_t *weights, size_t weights_size, bool take_weights_ownership,
    CamppRuntimeModel *model)
{
    CamppRuntimeModel staging;
    CamppByteSpan plan;
    CamppPlanLayout layout;
    CamppStatus status;

    if (model == NULL || plan_bytes == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (weights == NULL && weights_size != 0u) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    /*
     * 성공할 때까지 호출자의 model을 건드리지 않는다. 중간에 실패하면 staging만
     * 정리하면 되고, 호출자는 이전 model을 그대로 쓸 수 있다.
     */
    memset(&staging, 0, sizeof(staging));
    staging.plan_bytes = plan_bytes;
    staging.plan_size = plan_size;
    staging.owns_plan_bytes = take_plan_ownership;
    staging.weights = weights;
    staging.weights_size = weights_size;
    staging.owns_weights = take_weights_ownership;

    status = campp_span_make(plan_bytes, (uint64_t)plan_size, &plan);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    memset(&layout, 0, sizeof(layout));
    status = campp_validate_plan(&plan, &layout);
    if (status != CAMPP_STATUS_OK) {
        goto failed;
    }

    staging.header = layout.header;
    staging.bucket_frames = layout.header.bucket_frames;
    staging.attribute_section = layout.attribute_section.data;
    staging.attribute_section_size = (size_t)layout.attribute_section.size;

    status = campp_load_tensor_table(&layout, &staging);
    if (status != CAMPP_STATUS_OK) {
        goto failed;
    }
    status = campp_load_operator_table(&layout, &staging);
    if (status != CAMPP_STATUS_OK) {
        goto failed;
    }
    status = campp_collect_graph_io(&staging);
    if (status != CAMPP_STATUS_OK) {
        goto failed;
    }
    status = campp_validate_model(&staging);
    if (status != CAMPP_STATUS_OK) {
        goto failed;
    }

    *model = staging;
    return CAMPP_STATUS_OK;

failed:
    /* 넘겨받은 버퍼의 소유권은 아직 옮기지 않았으므로 여기서 해제하지 않는다. */
    staging.owns_plan_bytes = false;
    staging.owns_weights = false;
    campp_runtime_model_release(&staging);
    return status;
}

CamppStatus campp_runtime_model_load(
    const char *plan_path, const char *weights_path, CamppRuntimeModel *model)
{
    uint8_t *plan_bytes = NULL;
    uint8_t *weight_bytes = NULL;
    size_t plan_size = 0u;
    size_t weights_size = 0u;
    CamppStatus status;

    if (plan_path == NULL || weights_path == NULL || model == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }

    status = campp_read_whole_file(plan_path, &plan_bytes, &plan_size);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_read_whole_file(weights_path, &weight_bytes, &weights_size);
    if (status != CAMPP_STATUS_OK) {
        free(plan_bytes);
        return status;
    }

    status = campp_runtime_model_adopt(
        plan_bytes, plan_size, true, weight_bytes, weights_size, true, model);
    if (status != CAMPP_STATUS_OK) {
        free(plan_bytes);
        free(weight_bytes);
    }
    return status;
}

CamppStatus campp_runtime_model_tensor(
    const CamppRuntimeModel *model, uint32_t tensor_id,
    const CamppTensorDescriptor **out_descriptor)
{
    if (model == NULL || out_descriptor == NULL || model->tensors == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (tensor_id >= model->tensor_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    *out_descriptor = &model->tensors[tensor_id];
    return CAMPP_STATUS_OK;
}

CamppStatus campp_runtime_model_operator(
    const CamppRuntimeModel *model, uint32_t operator_id,
    const CamppOperatorDescriptor **out_descriptor)
{
    if (model == NULL || out_descriptor == NULL || model->operators == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (operator_id >= model->operator_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    *out_descriptor = &model->operators[operator_id];
    return CAMPP_STATUS_OK;
}

CamppStatus campp_runtime_model_constant_data(
    const CamppRuntimeModel *model, const CamppTensorDescriptor *descriptor,
    const void **out_data)
{
    uint64_t end;
    CamppStatus status;

    if (model == NULL || descriptor == NULL || out_data == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor->storage_type != CAMPP_TENSOR_STORAGE_CONSTANT) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_checked_add_u64(
        descriptor->data_offset, descriptor->storage_span_bytes, &end);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (end > (uint64_t)model->weights_size) {
        return CAMPP_STATUS_CORRUPT_WEIGHTS;
    }
    *out_data = model->weights + descriptor->data_offset;
    return CAMPP_STATUS_OK;
}

/* --------------------------------------------------------------------- */
/* attribute                                                             */
/* --------------------------------------------------------------------- */

CamppStatus campp_runtime_model_attribute(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, CamppAttributeValues *out_values)
{
    CamppByteSpan section;
    CamppByteSpan block;
    CamppStatus status;
    uint32_t record_count;
    uint32_t total_size;
    uint32_t index;
    uint64_t cursor;

    if (model == NULL || op == NULL || out_values == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (op->attribute_size == 0u) {
        return CAMPP_STATUS_MISSING_ATTRIBUTE;
    }

    status = campp_span_make(
        model->attribute_section, (uint64_t)model->attribute_section_size,
        &section);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_slice(
        &section, op->attribute_offset, (uint64_t)op->attribute_size, &block);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }

    status = campp_span_read_u32(&block, CAMPP_AB_RECORD_COUNT, &record_count);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    status = campp_span_read_u32(&block, CAMPP_AB_TOTAL_SIZE, &total_size);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if ((uint64_t)total_size != block.size) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }

    cursor = CAMPP_ATTRIBUTE_BLOCK_HEADER_SIZE;
    for (index = 0u; index < record_count; ++index) {
        uint16_t record_key;
        uint8_t value_type;
        uint8_t value_count;
        uint32_t reserved;
        uint64_t values_offset;
        uint64_t values_bytes;

        status = campp_span_read_u16(&block, cursor + CAMPP_AR_KEY, &record_key);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        status = campp_span_read_u8(
            &block, cursor + CAMPP_AR_VALUE_TYPE, &value_type);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        status = campp_span_read_u8(
            &block, cursor + CAMPP_AR_VALUE_COUNT, &value_count);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        status = campp_span_read_u32(
            &block, cursor + CAMPP_AR_RESERVED, &reserved);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
        if (reserved != 0u || value_count == 0u) {
            return CAMPP_STATUS_CORRUPT_PLAN;
        }

        values_offset = cursor + CAMPP_ATTRIBUTE_RECORD_HEADER_SIZE;
        values_bytes = (uint64_t)value_count * CAMPP_ATTRIBUTE_VALUE_SIZE;
        status = campp_span_check(&block, values_offset, values_bytes);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }

        if (record_key == key) {
            out_values->key = record_key;
            out_values->value_type = value_type;
            out_values->count = value_count;
            out_values->raw = block.data + values_offset;
            return CAMPP_STATUS_OK;
        }
        cursor = values_offset + values_bytes;
    }
    return CAMPP_STATUS_MISSING_ATTRIBUTE;
}

CamppStatus campp_runtime_model_attribute_ints(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, int64_t *values, uint8_t capacity, uint8_t *out_count)
{
    CamppAttributeValues found;
    CamppByteSpan raw;
    CamppStatus status;
    uint8_t index;

    if (values == NULL || out_count == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_runtime_model_attribute(model, op, key, &found);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (found.value_type != CAMPP_ATTR_VALUE_INT) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    if (found.count > capacity) {
        return CAMPP_STATUS_BUFFER_OVERFLOW;
    }

    status = campp_span_make(
        found.raw, (uint64_t)found.count * CAMPP_ATTRIBUTE_VALUE_SIZE, &raw);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    for (index = 0u; index < found.count; ++index) {
        status = campp_span_read_i64(
            &raw, (uint64_t)index * CAMPP_ATTRIBUTE_VALUE_SIZE, &values[index]);
        if (status != CAMPP_STATUS_OK) {
            return status;
        }
    }
    *out_count = found.count;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_runtime_model_attribute_double(
    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op,
    uint16_t key, double *out_value)
{
    CamppAttributeValues found;
    CamppByteSpan raw;
    CamppStatus status;

    if (out_value == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_runtime_model_attribute(model, op, key, &found);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    if (found.value_type != CAMPP_ATTR_VALUE_FLOAT || found.count != 1u) {
        return CAMPP_STATUS_CORRUPT_PLAN;
    }
    status = campp_span_make(
        found.raw, CAMPP_ATTRIBUTE_VALUE_SIZE, &raw);
    if (status != CAMPP_STATUS_OK) {
        return status;
    }
    return campp_span_read_f64(&raw, 0u, out_value);
}
