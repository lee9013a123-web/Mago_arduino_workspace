/*
 * ORT가 저장한 Tensor를 각 Operator의 입력으로 다시 주입해 C Kernel 하나씩
 * 독립 실행한다.
 *
 * usage:
 *   campp_operator_replay <plan.bin> <weights.bin> <reference.rpl> <out_prefix>
 *
 * reference.rpl은 test_operator_replay.py가 만드는 little-endian 파일이다.
 * Tensor payload는 한 번만 저장되고 fixed-size index가 tensor_id로 위치를
 * 찾는다. 파일 전체를 RAM에 올리지 않고 필요한 입력만 seek/read한다.
 */

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backends/cpu_reference/reference_kernel_utils.h"
#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "execution/graph_executor.h"
#include "execution/tensor_registry.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"
#include "model_loading/binary_section_reader.h"

#define CAMPP_REPLAY_STORE_VERSION 1u
#define CAMPP_REPLAY_STORE_HEADER_SIZE 64u
#define CAMPP_REPLAY_STORE_ENTRY_SIZE 48u
#define CAMPP_REPLAY_ENTRY_PRESENT 1u

static const uint8_t CAMPP_REPLAY_STORE_MAGIC[8] = {
    'C', 'M', 'P', 'P', 'R', 'P', 'L', '1'
};

typedef struct CamppReplayEntry {
    uint32_t tensor_id;
    uint8_t dtype;
    uint8_t rank;
    uint16_t flags;
    uint32_t dimensions[CAMPP_TENSOR_MAX_RANK];
    uint64_t payload_offset;
    uint64_t byte_size;
} CamppReplayEntry;

typedef struct CamppReplayStore {
    FILE *file;
    uint64_t file_size;
    uint32_t bucket_frames;
    uint32_t tensor_count;
    CamppReplayEntry *entries;
} CamppReplayStore;

static int campp_replay_seek(FILE *file, uint64_t offset)
{
    if (file == NULL || offset > (uint64_t)LONG_MAX) {
        return 1;
    }
    return fseek(file, (long)offset, SEEK_SET) == 0 ? 0 : 1;
}

static int campp_replay_read_at(
    FILE *file, uint64_t offset, void *destination, size_t size)
{
    if ((destination == NULL && size != 0u) ||
        campp_replay_seek(file, offset) != 0) {
        return 1;
    }
    return size == 0u || fread(destination, 1u, size, file) == size ? 0 : 1;
}

static int campp_replay_file_size(FILE *file, uint64_t *out_size)
{
    long length;

    if (file == NULL || out_size == NULL ||
        fseek(file, 0, SEEK_END) != 0) {
        return 1;
    }
    length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        return 1;
    }
    *out_size = (uint64_t)length;
    return 0;
}

static void campp_replay_store_close(CamppReplayStore *store)
{
    if (store == NULL) {
        return;
    }
    if (store->file != NULL) {
        fclose(store->file);
    }
    free(store->entries);
    memset(store, 0, sizeof(*store));
}

static int campp_replay_entry_matches_descriptor(
    const CamppReplayEntry *entry, const CamppTensorDescriptor *descriptor)
{
    uint8_t axis;

    if (entry->tensor_id != descriptor->tensor_id ||
        entry->dtype != descriptor->dtype || entry->rank != descriptor->rank) {
        return 0;
    }
    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        if (entry->dimensions[axis] != descriptor->dimensions[axis]) {
            return 0;
        }
    }
    return 1;
}

static int campp_replay_store_open(
    const char *path, const CamppRuntimeModel *model, CamppReplayStore *store)
{
    CamppReplayStore staging;
    uint8_t header[CAMPP_REPLAY_STORE_HEADER_SIZE];
    uint64_t table_offset;
    uint64_t data_offset;
    uint64_t declared_file_size;
    uint64_t table_bytes;
    uint32_t version;
    uint32_t header_size;
    uint32_t entry_size;
    uint32_t reserved;
    uint32_t tensor_id;

    if (path == NULL || model == NULL || store == NULL) {
        return 1;
    }
    memset(&staging, 0, sizeof(staging));
    staging.file = fopen(path, "rb");
    if (staging.file == NULL ||
        campp_replay_file_size(staging.file, &staging.file_size) != 0 ||
        campp_replay_read_at(
            staging.file, 0u, header, sizeof(header)) != 0) {
        campp_replay_store_close(&staging);
        return 1;
    }

    if (memcmp(header, CAMPP_REPLAY_STORE_MAGIC, 8u) != 0) {
        campp_replay_store_close(&staging);
        return 1;
    }
    version = campp_read_u32_le(header + 8u);
    header_size = campp_read_u32_le(header + 12u);
    staging.bucket_frames = campp_read_u32_le(header + 16u);
    staging.tensor_count = campp_read_u32_le(header + 20u);
    entry_size = campp_read_u32_le(header + 24u);
    reserved = campp_read_u32_le(header + 28u);
    table_offset = campp_read_u64_le(header + 32u);
    data_offset = campp_read_u64_le(header + 40u);
    declared_file_size = campp_read_u64_le(header + 48u);
    if (version != CAMPP_REPLAY_STORE_VERSION ||
        header_size != CAMPP_REPLAY_STORE_HEADER_SIZE ||
        entry_size != CAMPP_REPLAY_STORE_ENTRY_SIZE || reserved != 0u ||
        campp_read_u64_le(header + 56u) != 0u ||
        staging.bucket_frames != model->bucket_frames ||
        staging.tensor_count != model->tensor_count ||
        table_offset != CAMPP_REPLAY_STORE_HEADER_SIZE ||
        declared_file_size != staging.file_size ||
        staging.file_size > (uint64_t)LONG_MAX) {
        campp_replay_store_close(&staging);
        return 1;
    }
    if (staging.tensor_count > SIZE_MAX / sizeof(*staging.entries)) {
        campp_replay_store_close(&staging);
        return 1;
    }
    table_bytes = (uint64_t)staging.tensor_count *
        CAMPP_REPLAY_STORE_ENTRY_SIZE;
    if (table_bytes > staging.file_size - table_offset ||
        data_offset < table_offset + table_bytes ||
        data_offset > staging.file_size || (data_offset & 7u) != 0u) {
        campp_replay_store_close(&staging);
        return 1;
    }

    staging.entries = (CamppReplayEntry *)calloc(
        (size_t)staging.tensor_count, sizeof(*staging.entries));
    if (staging.entries == NULL) {
        campp_replay_store_close(&staging);
        return 1;
    }

    for (tensor_id = 0u; tensor_id < staging.tensor_count; ++tensor_id) {
        uint8_t raw[CAMPP_REPLAY_STORE_ENTRY_SIZE];
        CamppReplayEntry *entry = &staging.entries[tensor_id];
        uint64_t offset = table_offset +
            (uint64_t)tensor_id * CAMPP_REPLAY_STORE_ENTRY_SIZE;
        uint8_t axis;

        if (campp_replay_read_at(
                staging.file, offset, raw, sizeof(raw)) != 0) {
            campp_replay_store_close(&staging);
            return 1;
        }
        entry->tensor_id = campp_read_u32_le(raw);
        entry->dtype = raw[4];
        entry->rank = raw[5];
        entry->flags = campp_read_u16_le(raw + 6u);
        for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
            entry->dimensions[axis] = campp_read_u32_le(
                raw + 8u + (uint64_t)axis * 4u);
        }
        entry->payload_offset = campp_read_u64_le(raw + 24u);
        entry->byte_size = campp_read_u64_le(raw + 32u);
        if (campp_read_u64_le(raw + 40u) != 0u ||
            (entry->flags & UINT16_C(0xFFFE)) != 0u ||
            !campp_replay_entry_matches_descriptor(
                entry, &model->tensors[tensor_id])) {
            campp_replay_store_close(&staging);
            return 1;
        }
        if ((entry->flags & CAMPP_REPLAY_ENTRY_PRESENT) != 0u) {
            if (entry->byte_size != model->tensors[tensor_id].logical_byte_size ||
                entry->payload_offset < data_offset ||
                entry->payload_offset > staging.file_size ||
                entry->byte_size > staging.file_size - entry->payload_offset) {
                campp_replay_store_close(&staging);
                return 1;
            }
        } else if (entry->payload_offset != UINT64_MAX ||
                   entry->byte_size != 0u) {
            campp_replay_store_close(&staging);
            return 1;
        }
    }

    *store = staging;
    return 0;
}

static int campp_replay_copy_dense_to_view(
    const uint8_t *dense, uint64_t dense_size, CamppTensorView *view)
{
    uint32_t element_size;
    uint64_t element_count;
    uint64_t index;

    if (dense == NULL || view == NULL || view->data == NULL ||
        dense_size != view->logical_byte_size) {
        return 1;
    }
    element_size = campp_dtype_byte_size(view->dtype);
    element_count = campp_tensor_view_element_count(view);
    if (element_size == 0u ||
        element_count > UINT64_MAX / element_size ||
        element_count * element_size != dense_size) {
        return 1;
    }
    if (campp_tensor_view_is_contiguous(view)) {
        memcpy(view->data, dense, (size_t)dense_size);
        return 0;
    }
    for (index = 0u; index < element_count; ++index) {
        uint64_t destination_offset =
            campp_reference_offset_for_linear(view, index);
        uint64_t source_offset = index * element_size;
        if (destination_offset > view->storage_span_bytes ||
            element_size > view->storage_span_bytes - destination_offset) {
            return 1;
        }
        memcpy(
            (uint8_t *)view->data + destination_offset,
            dense + source_offset, element_size);
    }
    return 0;
}

static int campp_replay_restore_tensor(
    CamppReplayStore *store, uint32_t tensor_id, uint8_t *dense_buffer,
    size_t dense_capacity, CamppTensorView *view)
{
    const CamppReplayEntry *entry;

    if (store == NULL || tensor_id >= store->tensor_count ||
        dense_buffer == NULL || view == NULL) {
        return 1;
    }
    entry = &store->entries[tensor_id];
    if ((entry->flags & CAMPP_REPLAY_ENTRY_PRESENT) == 0u ||
        entry->byte_size > dense_capacity ||
        campp_replay_read_at(
            store->file, entry->payload_offset, dense_buffer,
            (size_t)entry->byte_size) != 0) {
        return 1;
    }
    return campp_replay_copy_dense_to_view(
        dense_buffer, entry->byte_size, view);
}

static int campp_replay_write_dense(
    FILE *sink, const CamppTensorView *view)
{
    uint32_t element_size;
    uint64_t element_count;
    uint64_t index;

    if (sink == NULL || view == NULL || view->data == NULL) {
        return 1;
    }
    element_size = campp_dtype_byte_size(view->dtype);
    element_count = campp_tensor_view_element_count(view);
    if (element_size == 0u || element_count > SIZE_MAX / element_size) {
        return 1;
    }
    if (campp_tensor_view_is_contiguous(view)) {
        size_t total = (size_t)element_count * element_size;
        return fwrite(view->data, 1u, total, sink) == total ? 0 : 1;
    }
    for (index = 0u; index < element_count; ++index) {
        uint64_t offset = campp_reference_offset_for_linear(view, index);
        if (fwrite(
                (const uint8_t *)view->data + offset, 1u,
                element_size, sink) != element_size) {
            return 1;
        }
    }
    return 0;
}

static void campp_replay_free_inputs(
    void **external_inputs, uint32_t tensor_count)
{
    uint32_t tensor_id;

    if (external_inputs == NULL) {
        return;
    }
    for (tensor_id = 0u; tensor_id < tensor_count; ++tensor_id) {
        free(external_inputs[tensor_id]);
    }
    free(external_inputs);
}

int main(int argc, char **argv)
{
    CamppRuntimeModel model;
    CamppRuntimeContext context;
    CamppReplayStore store;
    const CamppKernelRegistry *registry = campp_cpu_reference_registry();
    void **external_inputs = NULL;
    uint8_t *dense_buffer = NULL;
    size_t dense_capacity = 0u;
    char path[4096];
    FILE *payload_sink = NULL;
    FILE *index_sink = NULL;
    uint64_t running_offset = 0u;
    uint32_t tensor_id;
    uint32_t operator_id;
    int first_entry = 1;
    int exit_code = 1;
    CamppStatus status;

    if (argc != 5) {
        fprintf(stderr,
            "usage: %s <plan.bin> <weights.bin> <reference.rpl> "
            "<out_prefix>\n", argv[0]);
        return 2;
    }

    memset(&model, 0, sizeof(model));
    memset(&context, 0, sizeof(context));
    memset(&store, 0, sizeof(store));
    status = campp_runtime_model_load(argv[1], argv[2], &model);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    status = campp_runtime_context_create(&model, registry, &context);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "context create failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    if (campp_replay_store_open(argv[3], &model, &store) != 0) {
        fprintf(stderr, "invalid replay reference store: %s\n", argv[3]);
        goto cleanup;
    }

    for (tensor_id = 0u; tensor_id < model.tensor_count; ++tensor_id) {
        uint64_t logical = model.tensors[tensor_id].logical_byte_size;
        if (logical > SIZE_MAX) {
            fprintf(stderr, "tensor too large: %" PRIu32 "\n", tensor_id);
            goto cleanup;
        }
        if ((size_t)logical > dense_capacity) {
            dense_capacity = (size_t)logical;
        }
    }
    dense_buffer = (uint8_t *)malloc(dense_capacity == 0u ? 1u : dense_capacity);
    external_inputs = (void **)calloc(
        (size_t)model.tensor_count, sizeof(*external_inputs));
    if (dense_buffer == NULL || external_inputs == NULL) {
        fprintf(stderr, "out of memory while preparing replay\n");
        goto cleanup;
    }

    for (tensor_id = 0u; tensor_id < model.tensor_count; ++tensor_id) {
        const CamppTensorDescriptor *descriptor = &model.tensors[tensor_id];
        CamppTensorView candidate;

        if (descriptor->storage_type != CAMPP_TENSOR_STORAGE_INPUT) {
            continue;
        }
        if (descriptor->storage_span_bytes > SIZE_MAX) {
            fprintf(stderr, "input too large: %" PRIu32 "\n", tensor_id);
            goto cleanup;
        }
        external_inputs[tensor_id] = calloc(
            1u, (size_t)descriptor->storage_span_bytes);
        if (external_inputs[tensor_id] == NULL) {
            fprintf(stderr, "cannot allocate input tensor %" PRIu32 "\n", tensor_id);
            goto cleanup;
        }
        candidate = context.tensors[tensor_id];
        candidate.data = external_inputs[tensor_id];
        candidate.storage_span_bytes = descriptor->storage_span_bytes;
        if (campp_replay_restore_tensor(
                &store, tensor_id, dense_buffer, dense_capacity,
                &candidate) != 0) {
            fprintf(stderr, "missing graph input tensor %" PRIu32 "\n", tensor_id);
            goto cleanup;
        }
        status = campp_runtime_context_bind_input(
            &context, tensor_id, external_inputs[tensor_id],
            (size_t)descriptor->storage_span_bytes, descriptor->dtype,
            descriptor->rank, descriptor->dimensions);
        if (status != CAMPP_STATUS_OK) {
            fprintf(stderr, "input bind failed: %s\n", campp_status_name(status));
            goto cleanup;
        }
    }

    if (snprintf(path, sizeof(path), "%s.bin", argv[4]) < 0) {
        goto cleanup;
    }
    payload_sink = fopen(path, "wb");
    if (snprintf(path, sizeof(path), "%s.json", argv[4]) < 0) {
        goto cleanup;
    }
    index_sink = fopen(path, "wb");
    if (payload_sink == NULL || index_sink == NULL) {
        fprintf(stderr, "cannot open replay outputs for %s\n", argv[4]);
        goto cleanup;
    }

    status = campp_runtime_context_reset(&context);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "context reset failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    fprintf(index_sink,
        "{\"mode\": \"operator_replay\", \"bucket_frames\": %" PRIu32
        ", \"operator_count\": %" PRIu32 ", \"executed\": %" PRIu32
        ", \"tensors\": [",
        model.bucket_frames, model.operator_count, model.operator_count);

    for (operator_id = 0u; operator_id < model.operator_count; ++operator_id) {
        const CamppOperatorDescriptor *op = &model.operators[operator_id];
        uint8_t slot;

        for (slot = 0u; slot < op->input_count; ++slot) {
            uint32_t input_id = op->input_tensor_ids[slot];
            const CamppTensorDescriptor *descriptor = &model.tensors[input_id];
            CamppTensorView *view = &context.tensors[input_id];

            if (descriptor->storage_type == CAMPP_TENSOR_STORAGE_CONSTANT) {
                continue;
            }
            if (campp_replay_restore_tensor(
                    &store, input_id, dense_buffer, dense_capacity, view) != 0) {
                fprintf(stderr,
                    "operator %" PRIu32 " missing ORT input tensor %" PRIu32 "\n",
                    operator_id, input_id);
                goto cleanup;
            }
        }
        for (slot = 0u; slot < op->output_count; ++slot) {
            uint32_t output_id = op->output_tensor_ids[slot];
            CamppTensorView *view = &context.tensors[output_id];
            memset(view->data, 0xA5, (size_t)view->storage_span_bytes);
        }

        status = campp_graph_execute_operator(&context, operator_id);
        if (status != CAMPP_STATUS_OK) {
            fprintf(stderr,
                "operator %" PRIu32 " replay failed: %s\n",
                operator_id, campp_status_name(status));
            goto cleanup;
        }

        for (slot = 0u; slot < op->output_count; ++slot) {
            uint32_t output_id = op->output_tensor_ids[slot];
            const CamppTensorView *view = &context.tensors[output_id];
            uint64_t byte_size = view->logical_byte_size;
            uint8_t axis;

            if (campp_replay_write_dense(payload_sink, view) != 0) {
                fprintf(stderr,
                    "operator %" PRIu32 " output write failed\n", operator_id);
                goto cleanup;
            }
            fprintf(index_sink,
                "%s{\"operator_id\": %" PRIu32 ", \"tensor_id\": %" PRIu32
                ", \"dtype\": %u, \"rank\": %u, \"shape\": [",
                first_entry ? "" : ", ", operator_id, output_id,
                view->dtype, view->rank);
            for (axis = 0u; axis < view->rank; ++axis) {
                fprintf(index_sink, "%s%" PRIu32,
                    axis == 0u ? "" : ", ", view->dimensions[axis]);
            }
            fprintf(index_sink,
                "], \"offset\": %" PRIu64 ", \"byte_size\": %" PRIu64 "}",
                running_offset, byte_size);
            running_offset += byte_size;
            first_entry = 0;
        }
    }
    fprintf(index_sink, "]}\n");

    printf(
        "replayed bucket=%" PRIu32 " operators=%" PRIu32
        " payload=%" PRIu64 " bytes\n",
        model.bucket_frames, context.diagnostics.executed_operator_count,
        running_offset);
    exit_code = 0;

cleanup:
    if (payload_sink != NULL) {
        fclose(payload_sink);
    }
    if (index_sink != NULL) {
        fclose(index_sink);
    }
    free(dense_buffer);
    campp_replay_free_inputs(external_inputs, model.tensor_count);
    campp_replay_store_close(&store);
    campp_runtime_context_release(&context);
    campp_runtime_model_release(&model);
    return exit_code;
}
