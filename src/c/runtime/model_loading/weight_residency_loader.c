/* Load and execute the compact page-window schedule emitted by Python. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "internal/runtime_model.h"
#include "model_loading/binary_section_reader.h"
#include "platform_linux/mapped_file.h"

#define CAMPP_WS_HEADER_SIZE 64u
#define CAMPP_WS_RECORD_SIZE 40u
#define CAMPP_WS_VERSION 2u
#define CAMPP_WS_USED 1u
#define CAMPP_WS_PREFETCH_AFTER 2u

#define CAMPP_WS_VERSION_OFFSET 8u
#define CAMPP_WS_HEADER_SIZE_OFFSET 12u
#define CAMPP_WS_BUCKET_OFFSET 16u
#define CAMPP_WS_PAGE_SIZE_OFFSET 20u
#define CAMPP_WS_BLOCK_COUNT_OFFSET 24u
#define CAMPP_WS_OPERATOR_COUNT_OFFSET 28u
#define CAMPP_WS_WEIGHTS_SIZE_OFFSET 32u
#define CAMPP_WS_RECORDS_OFFSET_OFFSET 40u
#define CAMPP_WS_RECORDS_SIZE_OFFSET 48u
#define CAMPP_WS_RESERVED_OFFSET 56u

#define CAMPP_WB_ID_OFFSET 0u
#define CAMPP_WB_FIRST_OPERATOR_OFFSET 4u
#define CAMPP_WB_LAST_OPERATOR_OFFSET 8u
#define CAMPP_WB_PREFETCH_OPERATOR_OFFSET 12u
#define CAMPP_WB_FLAGS_OFFSET 16u
#define CAMPP_WB_RESERVED_OFFSET 20u
#define CAMPP_WB_FILE_OFFSET 24u
#define CAMPP_WB_BYTE_SIZE_OFFSET 32u

static const uint8_t CAMPP_WS_MAGIC[8] = {
    'C', 'A', 'M', 'P', 'P', 'W', 'S', '1'
};

static CamppStatus campp_weight_schedule_read_file(
    const char *path, uint8_t **out_bytes, size_t *out_size)
{
    FILE *file;
    long length;
    uint8_t *bytes;

    if (path == NULL || out_bytes == NULL || out_size == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return CAMPP_STATUS_FILE_NOT_FOUND;
    }
    if (fseek(file, 0L, SEEK_END) != 0 || (length = ftell(file)) < 0L ||
        fseek(file, 0L, SEEK_SET) != 0) {
        fclose(file);
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    bytes = (uint8_t *)malloc((size_t)length == 0u ? 1u : (size_t)length);
    if (bytes == NULL) {
        fclose(file);
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    if ((size_t)length != 0u &&
        fread(bytes, 1u, (size_t)length, file) != (size_t)length) {
        free(bytes);
        fclose(file);
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    fclose(file);
    *out_bytes = bytes;
    *out_size = (size_t)length;
    return CAMPP_STATUS_OK;
}

static bool campp_is_power_of_two_u32(uint32_t value)
{
    return value != 0u && (value & (value - 1u)) == 0u;
}

static CamppStatus campp_build_event_table(
    const CamppWeightResidencyBlock *blocks,
    uint32_t block_count,
    uint32_t operator_count,
    CamppWeightResidencyEvent event_type,
    uint32_t **out_offsets,
    uint32_t **out_indices)
{
    uint32_t *offsets;
    uint32_t *indices;
    uint32_t *cursor;
    uint32_t block_id;
    uint32_t operator_id;

    offsets = (uint32_t *)calloc(
        (size_t)operator_count + 1u, sizeof(*offsets));
    indices = (uint32_t *)calloc((size_t)block_count, sizeof(*indices));
    cursor = (uint32_t *)calloc((size_t)operator_count, sizeof(*cursor));
    if (offsets == NULL || indices == NULL || cursor == NULL) {
        free(offsets);
        free(indices);
        free(cursor);
        return CAMPP_STATUS_OUT_OF_MEMORY;
    }
    for (block_id = 0u; block_id < block_count; ++block_id) {
        bool selected = event_type == CAMPP_WEIGHT_EVENT_EVICT ||
            (event_type == CAMPP_WEIGHT_EVENT_PREFETCH_BEFORE &&
             !blocks[block_id].prefetch_after_operator) ||
            (event_type == CAMPP_WEIGHT_EVENT_PREFETCH_AFTER &&
             blocks[block_id].prefetch_after_operator);
        if (!selected) continue;
        operator_id = event_type == CAMPP_WEIGHT_EVENT_EVICT
            ? blocks[block_id].last_operator
            : blocks[block_id].prefetch_operator;
        offsets[operator_id + 1u] += 1u;
    }
    for (operator_id = 0u; operator_id < operator_count; ++operator_id) {
        offsets[operator_id + 1u] += offsets[operator_id];
        cursor[operator_id] = offsets[operator_id];
    }
    for (block_id = 0u; block_id < block_count; ++block_id) {
        bool selected = event_type == CAMPP_WEIGHT_EVENT_EVICT ||
            (event_type == CAMPP_WEIGHT_EVENT_PREFETCH_BEFORE &&
             !blocks[block_id].prefetch_after_operator) ||
            (event_type == CAMPP_WEIGHT_EVENT_PREFETCH_AFTER &&
             blocks[block_id].prefetch_after_operator);
        if (!selected) continue;
        operator_id = event_type == CAMPP_WEIGHT_EVENT_EVICT
            ? blocks[block_id].last_operator
            : blocks[block_id].prefetch_operator;
        indices[cursor[operator_id]] = block_id;
        cursor[operator_id] += 1u;
    }
    free(cursor);
    *out_offsets = offsets;
    *out_indices = indices;
    return CAMPP_STATUS_OK;
}

CamppStatus campp_runtime_model_enable_weight_window(
    CamppRuntimeModel *model, const char *schedule_path)
{
    uint8_t *bytes = NULL;
    size_t byte_size = 0u;
    CamppByteSpan schedule;
    CamppByteSpan record;
    CamppWeightResidencyBlock *blocks = NULL;
    uint32_t version;
    uint32_t header_size;
    uint32_t bucket_frames;
    uint32_t page_size;
    uint32_t block_count;
    uint32_t operator_count;
    uint64_t weights_size;
    uint64_t records_offset;
    uint64_t records_size;
    uint64_t records_end = 0u;
    uint64_t reserved;
    uint64_t expected_records_size;
    uint64_t prior_end = 0u;
    uint32_t index;
    long system_page_size;
    CamppStatus status;

    if (model == NULL || schedule_path == NULL ||
        !model->weight_mapping.mapped || model->weight_windowing_enabled) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    status = campp_weight_schedule_read_file(
        schedule_path, &bytes, &byte_size);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_span_make(bytes, byte_size, &schedule);
    if (status != CAMPP_STATUS_OK || byte_size < CAMPP_WS_HEADER_SIZE ||
        memcmp(bytes, CAMPP_WS_MAGIC, sizeof(CAMPP_WS_MAGIC)) != 0) {
        status = CAMPP_STATUS_INVALID_MAGIC;
        goto failed;
    }
#define CAMPP_WS_READ_U32(field, target)                                      \
    do {                                                                       \
        status = campp_span_read_u32(&schedule, (field), &(target));           \
        if (status != CAMPP_STATUS_OK) goto failed;                            \
    } while (0)
#define CAMPP_WS_READ_U64(field, target)                                      \
    do {                                                                       \
        status = campp_span_read_u64(&schedule, (field), &(target));           \
        if (status != CAMPP_STATUS_OK) goto failed;                            \
    } while (0)
    CAMPP_WS_READ_U32(CAMPP_WS_VERSION_OFFSET, version);
    CAMPP_WS_READ_U32(CAMPP_WS_HEADER_SIZE_OFFSET, header_size);
    CAMPP_WS_READ_U32(CAMPP_WS_BUCKET_OFFSET, bucket_frames);
    CAMPP_WS_READ_U32(CAMPP_WS_PAGE_SIZE_OFFSET, page_size);
    CAMPP_WS_READ_U32(CAMPP_WS_BLOCK_COUNT_OFFSET, block_count);
    CAMPP_WS_READ_U32(CAMPP_WS_OPERATOR_COUNT_OFFSET, operator_count);
    CAMPP_WS_READ_U64(CAMPP_WS_WEIGHTS_SIZE_OFFSET, weights_size);
    CAMPP_WS_READ_U64(CAMPP_WS_RECORDS_OFFSET_OFFSET, records_offset);
    CAMPP_WS_READ_U64(CAMPP_WS_RECORDS_SIZE_OFFSET, records_size);
    CAMPP_WS_READ_U64(CAMPP_WS_RESERVED_OFFSET, reserved);
#undef CAMPP_WS_READ_U32
#undef CAMPP_WS_READ_U64
    system_page_size = sysconf(_SC_PAGESIZE);
    status = campp_checked_mul_u64(
        block_count, CAMPP_WS_RECORD_SIZE, &expected_records_size);
    if (status != CAMPP_STATUS_OK) goto failed;
    if (version != CAMPP_WS_VERSION || header_size != CAMPP_WS_HEADER_SIZE ||
        bucket_frames != model->bucket_frames ||
        operator_count != model->operator_count || block_count == 0u ||
        !campp_is_power_of_two_u32(page_size) ||
        system_page_size <= 0 || page_size != (uint32_t)system_page_size ||
        weights_size != model->weights_size ||
        records_offset != CAMPP_WS_HEADER_SIZE ||
        records_size != expected_records_size || reserved != 0u) {
        status = CAMPP_STATUS_CORRUPT_PLAN;
        goto failed;
    }
    status = campp_span_check(&schedule, records_offset, records_size);
    if (status == CAMPP_STATUS_OK) {
        status = campp_checked_add_u64(
            records_offset, records_size, &records_end);
    }
    if (status != CAMPP_STATUS_OK || records_end != byte_size) {
        status = CAMPP_STATUS_CORRUPT_PLAN;
        goto failed;
    }
    blocks = (CamppWeightResidencyBlock *)calloc(
        block_count, sizeof(*blocks));
    if (blocks == NULL) {
        status = CAMPP_STATUS_OUT_OF_MEMORY;
        goto failed;
    }
    for (index = 0u; index < block_count; ++index) {
        uint32_t flags;
        uint32_t record_reserved;
        uint64_t end;

        status = campp_span_slice(
            &schedule,
            records_offset + (uint64_t)index * CAMPP_WS_RECORD_SIZE,
            CAMPP_WS_RECORD_SIZE,
            &record);
        if (status != CAMPP_STATUS_OK) goto failed;
        status = campp_span_read_u32(
            &record, CAMPP_WB_ID_OFFSET, &blocks[index].block_id);
        if (status != CAMPP_STATUS_OK) goto failed;
        status = campp_span_read_u32(
            &record, CAMPP_WB_FIRST_OPERATOR_OFFSET,
            &blocks[index].first_operator);
        if (status != CAMPP_STATUS_OK) goto failed;
        status = campp_span_read_u32(
            &record, CAMPP_WB_LAST_OPERATOR_OFFSET,
            &blocks[index].last_operator);
        if (status != CAMPP_STATUS_OK) goto failed;
        status = campp_span_read_u32(
            &record, CAMPP_WB_PREFETCH_OPERATOR_OFFSET,
            &blocks[index].prefetch_operator);
        if (status != CAMPP_STATUS_OK) goto failed;
        status = campp_span_read_u32(&record, CAMPP_WB_FLAGS_OFFSET, &flags);
        if (status != CAMPP_STATUS_OK) goto failed;
        status = campp_span_read_u32(
            &record, CAMPP_WB_RESERVED_OFFSET, &record_reserved);
        if (status != CAMPP_STATUS_OK) goto failed;
        status = campp_span_read_u64(
            &record, CAMPP_WB_FILE_OFFSET, &blocks[index].file_offset);
        if (status != CAMPP_STATUS_OK) goto failed;
        status = campp_span_read_u64(
            &record, CAMPP_WB_BYTE_SIZE_OFFSET, &blocks[index].byte_size);
        if (status != CAMPP_STATUS_OK) goto failed;
        status = campp_checked_add_u64(
            blocks[index].file_offset, blocks[index].byte_size, &end);
        if (status != CAMPP_STATUS_OK) goto failed;
        blocks[index].prefetch_after_operator =
            (flags & CAMPP_WS_PREFETCH_AFTER) != 0u;
        if (blocks[index].block_id != index ||
            (flags & CAMPP_WS_USED) == 0u ||
            (flags & ~(CAMPP_WS_USED | CAMPP_WS_PREFETCH_AFTER)) != 0u ||
            record_reserved != 0u ||
            blocks[index].first_operator > blocks[index].last_operator ||
            blocks[index].last_operator >= operator_count ||
            blocks[index].prefetch_operator >= operator_count ||
            blocks[index].prefetch_operator >
                blocks[index].first_operator ||
            (blocks[index].prefetch_after_operator &&
             blocks[index].prefetch_operator >=
                blocks[index].first_operator) ||
            blocks[index].file_offset % page_size != 0u ||
            blocks[index].byte_size == 0u ||
            blocks[index].byte_size % page_size != 0u ||
            blocks[index].file_offset < prior_end || end > weights_size) {
            status = CAMPP_STATUS_CORRUPT_PLAN;
            goto failed;
        }
        prior_end = end;
    }
    status = campp_build_event_table(
        blocks, block_count, operator_count,
        CAMPP_WEIGHT_EVENT_PREFETCH_BEFORE,
        &model->weight_prefetch_offsets, &model->weight_prefetch_indices);
    if (status != CAMPP_STATUS_OK) goto failed;
    status = campp_build_event_table(
        blocks, block_count, operator_count,
        CAMPP_WEIGHT_EVENT_PREFETCH_AFTER,
        &model->weight_postfetch_offsets, &model->weight_postfetch_indices);
    if (status != CAMPP_STATUS_OK) goto failed;
    status = campp_build_event_table(
        blocks, block_count, operator_count, CAMPP_WEIGHT_EVENT_EVICT,
        &model->weight_evict_offsets, &model->weight_evict_indices);
    if (status != CAMPP_STATUS_OK) goto failed;

    model->weight_blocks = blocks;
    model->weight_block_count = block_count;
    model->weight_page_size = page_size;
    model->weight_windowing_enabled = true;
    blocks = NULL;
    /* huge page backing이면 부분 반납이 불가능하다.  windowing 전제 조건이다. */
    status = campp_mapped_file_advise_no_huge_page(&model->weight_mapping);
    if (status != CAMPP_STATUS_OK) return status;
    status = campp_mapped_file_advise_sequential(&model->weight_mapping);
    if (status != CAMPP_STATUS_OK) goto failed_enabled;
    for (index = 0u; index < block_count; ++index) {
        if (model->weight_blocks[index].first_operator == 0u) {
            status = campp_mapped_file_prefetch(
                &model->weight_mapping,
                model->weight_blocks[index].file_offset,
                model->weight_blocks[index].byte_size);
            if (status != CAMPP_STATUS_OK) goto failed_enabled;
            if (model->weight_event_hook != NULL) {
                model->weight_event_hook(
                    model->weight_event_hook_user_data,
                    0u,
                    CAMPP_WEIGHT_EVENT_PREFETCH_BEFORE,
                    &model->weight_blocks[index]);
            }
        }
    }
    free(bytes);
    return CAMPP_STATUS_OK;

failed_enabled:
    model->weight_windowing_enabled = false;
    free(model->weight_blocks);
    free(model->weight_prefetch_offsets);
    free(model->weight_prefetch_indices);
    free(model->weight_postfetch_offsets);
    free(model->weight_postfetch_indices);
    free(model->weight_evict_offsets);
    free(model->weight_evict_indices);
    model->weight_blocks = NULL;
    model->weight_prefetch_offsets = NULL;
    model->weight_prefetch_indices = NULL;
    model->weight_postfetch_offsets = NULL;
    model->weight_postfetch_indices = NULL;
    model->weight_evict_offsets = NULL;
    model->weight_evict_indices = NULL;
    model->weight_block_count = 0u;
    model->weight_page_size = 0u;
failed:
    free(model->weight_prefetch_offsets);
    free(model->weight_prefetch_indices);
    free(model->weight_postfetch_offsets);
    free(model->weight_postfetch_indices);
    free(model->weight_evict_offsets);
    free(model->weight_evict_indices);
    model->weight_prefetch_offsets = NULL;
    model->weight_prefetch_indices = NULL;
    model->weight_postfetch_offsets = NULL;
    model->weight_postfetch_indices = NULL;
    model->weight_evict_offsets = NULL;
    model->weight_evict_indices = NULL;
    free(blocks);
    free(bytes);
    return status;
}

CamppStatus campp_runtime_model_weight_before_operator(
    const CamppRuntimeModel *model, uint32_t operator_id)
{
    uint32_t begin;
    uint32_t end;
    uint32_t event;

    if (model == NULL || operator_id >= model->operator_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (!model->weight_windowing_enabled) {
        return CAMPP_STATUS_OK;
    }
    begin = model->weight_prefetch_offsets[operator_id];
    end = model->weight_prefetch_offsets[operator_id + 1u];
    for (event = begin; event < end; ++event) {
        const CamppWeightResidencyBlock *block =
            &model->weight_blocks[model->weight_prefetch_indices[event]];
        CamppStatus status = campp_mapped_file_prefetch(
            &model->weight_mapping, block->file_offset, block->byte_size);
        if (status != CAMPP_STATUS_OK) return status;
        if (model->weight_event_hook != NULL) {
            model->weight_event_hook(
                model->weight_event_hook_user_data,
                operator_id,
                CAMPP_WEIGHT_EVENT_PREFETCH_BEFORE,
                block);
        }
    }
    return CAMPP_STATUS_OK;
}

CamppStatus campp_runtime_model_weight_after_operator(
    const CamppRuntimeModel *model, uint32_t operator_id)
{
    uint32_t begin;
    uint32_t end;
    uint32_t event;

    if (model == NULL || operator_id >= model->operator_count) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    if (!model->weight_windowing_enabled) {
        return CAMPP_STATUS_OK;
    }
    begin = model->weight_evict_offsets[operator_id];
    end = model->weight_evict_offsets[operator_id + 1u];
    for (event = begin; event < end; ++event) {
        const CamppWeightResidencyBlock *block =
            &model->weight_blocks[model->weight_evict_indices[event]];
        CamppStatus status = campp_mapped_file_discard(
            &model->weight_mapping, block->file_offset, block->byte_size);
        if (status != CAMPP_STATUS_OK) return status;
        if (model->weight_event_hook != NULL) {
            model->weight_event_hook(
                model->weight_event_hook_user_data,
                operator_id,
                CAMPP_WEIGHT_EVENT_EVICT,
                block);
        }
    }
    begin = model->weight_postfetch_offsets[operator_id];
    end = model->weight_postfetch_offsets[operator_id + 1u];
    for (event = begin; event < end; ++event) {
        const CamppWeightResidencyBlock *block =
            &model->weight_blocks[model->weight_postfetch_indices[event]];
        CamppStatus status = campp_mapped_file_prefetch(
            &model->weight_mapping, block->file_offset, block->byte_size);
        if (status != CAMPP_STATUS_OK) return status;
        if (model->weight_event_hook != NULL) {
            model->weight_event_hook(
                model->weight_event_hook_user_data,
                operator_id,
                CAMPP_WEIGHT_EVENT_PREFETCH_AFTER,
                block);
        }
    }
    if (operator_id + 1u == model->operator_count) {
        uint32_t block_id;
        for (block_id = 0u; block_id < model->weight_block_count; ++block_id) {
            const CamppWeightResidencyBlock *block =
                &model->weight_blocks[block_id];
            if (block->first_operator == 0u) {
                CamppStatus status = campp_mapped_file_prefetch(
                    &model->weight_mapping,
                    block->file_offset,
                    block->byte_size);
                if (status != CAMPP_STATUS_OK) return status;
                if (model->weight_event_hook != NULL) {
                    model->weight_event_hook(
                        model->weight_event_hook_user_data,
                        operator_id,
                        CAMPP_WEIGHT_EVENT_CYCLE_RESET,
                        block);
                }
            }
        }
    }
    return CAMPP_STATUS_OK;
}

void campp_runtime_model_set_weight_event_hook(
    CamppRuntimeModel *model,
    CamppWeightResidencyEventHook hook,
    void *user_data)
{
    if (model == NULL) return;
    model->weight_event_hook = hook;
    model->weight_event_hook_user_data = user_data;
}
