/*
 * plan 하나를 실제 입력으로 실행하고 각 Operator 출력을 dense 순서로 덤프한다.
 * Arena에서 중간값이 다음 Tensor에 덮어써지기 전에 즉시 기록한다.
 *
 * usage: campp_reference_dump <plan.bin> <weights.bin> <input.f32> <out_prefix>
 *        [--weight-mode malloc|mmap|windowed]
 *        [--weight-schedule schedule.bin]
 *        campp_reference_dump --model <model.camppmodel> <input.f32> <out_prefix>
 *
 * 출력:
 *   <out_prefix>.bin   각 Tensor의 payload를 C 순서로 이어 붙인 것
 *   <out_prefix>.json  tensor_id, dtype, shape, offset, byte_size 색인
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "backends/cpu_reference/reference_kernel_utils.h"
#include "execution/graph_executor.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"

#if defined(CAMPP_ENABLE_FINAL_CANDIDATE_SUITE)
#include "campp_profill/optimization/final_candidate_suite.h"
#endif

static int read_entire_file(const char *path, uint8_t **out_data, size_t *out_size)
{
    FILE *file = fopen(path, "rb");
    long length;
    uint8_t *buffer;

    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0) {
        fclose(file);
        return 1;
    }
    rewind(file);
    buffer = (uint8_t *)malloc((size_t)length);
    if (buffer == NULL) {
        fclose(file);
        return 1;
    }
    if (fread(buffer, 1u, (size_t)length, file) != (size_t)length) {
        free(buffer);
        fclose(file);
        return 1;
    }
    fclose(file);
    *out_data = buffer;
    *out_size = (size_t)length;
    return 0;
}

/*
 * Raw feature files do not carry shape metadata.  When their byte size differs
 * from the selected plan, derive the provided frame count from the one axis
 * whose descriptor dimension equals bucket_frames.  bind_input can then return
 * BUCKET_MISMATCH instead of the command-line harness hiding it behind a generic
 * file-size error.
 */
static int infer_input_dimensions(
    const CamppRuntimeModel *model,
    const CamppTensorDescriptor *descriptor,
    size_t input_size,
    uint32_t dimensions[CAMPP_TENSOR_MAX_RANK])
{
    const uint32_t element_size = campp_dtype_byte_size(descriptor->dtype);
    uint64_t fixed_elements = 1u;
    uint64_t bytes_per_frame;
    uint64_t provided_frames;
    uint8_t time_axis = 0u;
    uint8_t time_axis_count = 0u;
    uint8_t axis;

    if (model == NULL || descriptor == NULL || dimensions == NULL ||
        element_size == 0u) {
        return 1;
    }
    for (axis = 0u; axis < CAMPP_TENSOR_MAX_RANK; ++axis) {
        dimensions[axis] = descriptor->dimensions[axis];
    }
    if ((uint64_t)input_size == descriptor->storage_span_bytes) {
        return 0;
    }

    for (axis = 0u; axis < descriptor->rank; ++axis) {
        const uint32_t dimension = descriptor->dimensions[axis];
        if (dimension == model->bucket_frames) {
            time_axis = axis;
            time_axis_count += 1u;
            continue;
        }
        if (dimension == 0u || fixed_elements > UINT64_MAX / dimension) {
            return 1;
        }
        fixed_elements *= dimension;
    }
    if (time_axis_count != 1u ||
        fixed_elements > UINT64_MAX / element_size) {
        return 1;
    }
    bytes_per_frame = fixed_elements * element_size;
    if (bytes_per_frame == 0u || (uint64_t)input_size % bytes_per_frame != 0u) {
        return 1;
    }
    provided_frames = (uint64_t)input_size / bytes_per_frame;
    if (provided_frames == 0u || provided_frames > UINT32_MAX) {
        return 1;
    }
    dimensions[time_axis] = (uint32_t)provided_frames;
    return 0;
}

/* view를 논리적 C 순서로 모아 file에 쓴다. 비연속 VIEW도 dense로 펴진다. */
static int write_dense_payload(FILE *sink, const CamppTensorView *view)
{
    const uint32_t element_size = campp_dtype_byte_size(view->dtype);
    const uint64_t element_count = campp_tensor_view_element_count(view);
    uint64_t index;

    if (element_size == 0u) {
        return 1;
    }
    if (campp_tensor_view_is_contiguous(view)) {
        const size_t total = (size_t)element_count * element_size;
        return fwrite(view->data, 1u, total, sink) == total ? 0 : 1;
    }
    for (index = 0u; index < element_count; ++index) {
        const uint64_t offset = campp_reference_offset_for_linear(view, index);
        if (fwrite((const uint8_t *)view->data + offset, 1u, element_size, sink) !=
            element_size) {
            return 1;
        }
    }
    return 0;
}

typedef struct CamppDumpWriter {
    FILE *payload_sink;
    FILE *index_sink;
    uint64_t running_offset;
    int first_entry;
} CamppDumpWriter;

static CamppStatus dump_tensor_ready(
    void *user_data, uint32_t operator_id, uint32_t tensor_id,
    const CamppTensorView *view)
{
    CamppDumpWriter *writer = (CamppDumpWriter *)user_data;
    uint64_t byte_size;
    uint8_t axis;

    if (writer == NULL || writer->payload_sink == NULL ||
        writer->index_sink == NULL || view == NULL || view->data == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    byte_size = campp_tensor_view_element_count(view) *
                campp_dtype_byte_size(view->dtype);
    if (write_dense_payload(writer->payload_sink, view) != 0) {
        return CAMPP_STATUS_FILE_READ_FAILED;
    }

    fprintf(
        writer->index_sink,
        "%s{\"operator_id\": %" PRIu32 ", \"tensor_id\": %" PRIu32
        ", \"dtype\": %u, \"rank\": %u, \"shape\": [",
        writer->first_entry ? "" : ", ", operator_id, tensor_id,
        view->dtype, view->rank);
    for (axis = 0u; axis < view->rank; ++axis) {
        fprintf(
            writer->index_sink, "%s%" PRIu32,
            axis == 0u ? "" : ", ", view->dimensions[axis]);
    }
    fprintf(
        writer->index_sink,
        "], \"offset\": %" PRIu64 ", \"byte_size\": %" PRIu64 "}",
        writer->running_offset, byte_size);
    if (ferror(writer->index_sink)) {
        return CAMPP_STATUS_FILE_READ_FAILED;
    }
    writer->running_offset += byte_size;
    writer->first_entry = 0;
    return CAMPP_STATUS_OK;
}

int main(int argc, char **argv)
{
    CamppRuntimeModel model;
    CamppRuntimeContext context;
    const CamppKernelRegistry *registry = NULL;
#if defined(CAMPP_ENABLE_FINAL_CANDIDATE_SUITE)
    CamppFinalCandidateSuite final_candidate_suite;
#endif
    const CamppTensorDescriptor *input_descriptor;
    uint32_t input_dimensions[CAMPP_TENSOR_MAX_RANK];
    uint32_t input_tensor_id;
    uint8_t *input_data = NULL;
    size_t input_size = 0u;
    CamppStatus status;
    const char *plan_path = NULL;
    const char *weights_path = NULL;
    const char *input_path = NULL;
    const char *output_prefix = NULL;
    const char *weight_mode = "malloc";
    const char *weight_schedule_path = NULL;
    int package_mode = 0;
    int option_index;
    char path[4096];
    FILE *payload_sink;
    FILE *index_sink;
    CamppDumpWriter writer;

    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <plan.bin> <weights.bin> <input.f32> <out_prefix> "
                "[--weight-mode malloc|mmap|windowed] "
                "[--weight-schedule schedule.bin]\n"
                "       %s --model <model.camppmodel> <input.f32> <out_prefix>\n",
                argv[0],
                argv[0]);
        return 2;
    }

    package_mode = strcmp(argv[1], "--model") == 0;
    if (package_mode) {
        if (argc != 5) {
            fprintf(stderr, "model package mode does not accept weight options\n");
            return 2;
        }
        input_path = argv[3];
        output_prefix = argv[4];
    } else {
        plan_path = argv[1];
        weights_path = argv[2];
        input_path = argv[3];
        output_prefix = argv[4];
        for (option_index = 5; option_index < argc; option_index += 2) {
            if (option_index + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", argv[option_index]);
                return 2;
            }
            if (strcmp(argv[option_index], "--weight-mode") == 0) {
                weight_mode = argv[option_index + 1];
            } else if (strcmp(
                    argv[option_index], "--weight-schedule") == 0) {
                weight_schedule_path = argv[option_index + 1];
            } else {
                fprintf(stderr, "unknown option: %s\n", argv[option_index]);
                return 2;
            }
        }
    }
    if (strcmp(weight_mode, "malloc") != 0 &&
        strcmp(weight_mode, "mmap") != 0 &&
        strcmp(weight_mode, "windowed") != 0) {
        fprintf(stderr, "invalid weight mode: %s\n", weight_mode);
        return 2;
    }
    if ((strcmp(weight_mode, "windowed") == 0) !=
        (weight_schedule_path != NULL)) {
        fprintf(stderr, "windowed mode requires exactly one schedule\n");
        return 2;
    }
#if !defined(CAMPP_ENABLE_WEIGHT_STREAMING)
    if (strcmp(weight_mode, "malloc") != 0) {
        fprintf(stderr, "binary was built without weight streaming\n");
        return 2;
    }
#endif

    memset(&model, 0, sizeof(model));
    if (package_mode) {
        status = campp_runtime_model_load_package(argv[2], &model);
    } else if (strcmp(weight_mode, "malloc") == 0) {
        status = campp_runtime_model_load(plan_path, weights_path, &model);
    } else {
        status = campp_runtime_model_load_mapped(
            plan_path, weights_path, &model);
        if (status == CAMPP_STATUS_OK &&
            strcmp(weight_mode, "windowed") == 0) {
            status = campp_runtime_model_enable_weight_window(
                &model, weight_schedule_path);
        }
    }
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", campp_status_name(status));
        return 1;
    }
    registry =
        model.operator_count != 0u && model.operators[0].kernel_id != 0u
            ? campp_cpu_aarch64_registry()
            : campp_cpu_reference_registry();
#if defined(CAMPP_ENABLE_FINAL_CANDIDATE_SUITE)
    memset(&final_candidate_suite, 0, sizeof(final_candidate_suite));
    status = campp_final_candidate_suite_create(registry, &final_candidate_suite);
    if (status != CAMPP_STATUS_OK) {
        fprintf(
            stderr, "final candidate suite create failed: %s\n",
            campp_status_name(status));
        campp_runtime_model_release(&model);
        return 1;
    }
    registry = &final_candidate_suite.registry;
#endif

    memset(&context, 0, sizeof(context));
    status = campp_runtime_context_create(&model, registry, &context);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "context create failed: %s\n", campp_status_name(status));
        campp_runtime_model_release(&model);
        return 1;
    }

    if (model.input_count != 1u) {
        fprintf(stderr, "expected exactly one graph input, got %" PRIu32 "\n",
                model.input_count);
        campp_runtime_context_release(&context);
        campp_runtime_model_release(&model);
        return 1;
    }
    input_tensor_id = model.input_tensor_ids[0];
    input_descriptor = &model.tensors[input_tensor_id];

    if (read_entire_file(input_path, &input_data, &input_size) != 0) {
        campp_runtime_context_release(&context);
        campp_runtime_model_release(&model);
        return 1;
    }
    if (infer_input_dimensions(
            &model, input_descriptor, input_size, input_dimensions) != 0) {
        fprintf(stderr,
                "cannot derive input shape: file %zu bytes, plan expects %" PRIu64
                "\n",
                input_size, input_descriptor->storage_span_bytes);
        free(input_data);
        campp_runtime_context_release(&context);
        campp_runtime_model_release(&model);
        return 1;
    }

    status = campp_runtime_context_bind_input(
        &context, input_tensor_id, input_data, input_size,
        input_descriptor->dtype, input_descriptor->rank,
        input_dimensions);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "input bind failed: %s\n", campp_status_name(status));
        free(input_data);
        campp_runtime_context_release(&context);
        campp_runtime_model_release(&model);
        return 1;
    }
    snprintf(path, sizeof(path), "%s.bin", output_prefix);
    payload_sink = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s.json", output_prefix);
    index_sink = fopen(path, "wb");
    if (payload_sink == NULL || index_sink == NULL) {
        fprintf(stderr, "cannot open output files for prefix %s\n", output_prefix);
        if (payload_sink != NULL) { fclose(payload_sink); }
        if (index_sink != NULL) { fclose(index_sink); }
        free(input_data);
        campp_runtime_context_release(&context);
        campp_runtime_model_release(&model);
        return 1;
    }
    memset(&writer, 0, sizeof(writer));
    writer.payload_sink = payload_sink;
    writer.index_sink = index_sink;
    writer.first_entry = 1;
    fprintf(
        index_sink,
        "{\"mode\": \"full_graph\", \"memory_layout\": \"%s\", "
        "\"activation_bytes\": %zu, \"bucket_frames\": %" PRIu32
        ", \"operator_count\": %" PRIu32 ", \"tensors\": [",
        context.activations.mode == CAMPP_ACTIVATION_STORAGE_ARENA
            ? "tensor_arena" : "reference",
        context.activations.total_bytes, model.bucket_frames,
        model.operator_count);
    context.diagnostics.tensor_ready = dump_tensor_ready;
    context.diagnostics.tensor_ready_user_data = &writer;

    status = campp_graph_execute(&context);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr,
                "execute failed at operator %" PRIu32 ": %s\n",
                context.diagnostics.current_operator_id,
                campp_status_name(status));
        fclose(payload_sink);
        fclose(index_sink);
        free(input_data);
        campp_runtime_context_release(&context);
        campp_runtime_model_release(&model);
        return 1;
    }
    fprintf(
        index_sink, "], \"executed\": %" PRIu32 "}\n",
        context.diagnostics.executed_operator_count);

    fclose(payload_sink);
    fclose(index_sink);

    printf("dumped bucket=%" PRIu32 " operators=%" PRIu32 " payload=%" PRIu64
           " bytes\n",
           model.bucket_frames, context.diagnostics.executed_operator_count,
           writer.running_offset);

    free(input_data);
    campp_runtime_context_release(&context);
    campp_runtime_model_release(&model);
    return 0;
}
