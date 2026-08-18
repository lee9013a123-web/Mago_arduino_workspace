#define _POSIX_C_SOURCE 200809L

/* E7 graph를 실행하고 Operator ID별 kernel exclusive time만 JSON으로 출력한다. */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "campp_profill/operator_profiler.h"
#include "campp_runtime/operator_descriptor.h"
#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "execution/graph_executor.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"

typedef struct ProfilerOptions {
    const char *plan_path;
    const char *weights_path;
    const char *input_path;
    uint32_t warmup;
    uint32_t repeat;
    uint32_t requested_threads;
} ProfilerOptions;

static int parse_u32(const char *text, uint32_t *out_value)
{
    char *end = NULL;
    unsigned long value;

    if (text == NULL || out_value == NULL || *text == '\0') {
        return 1;
    }
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return 1;
    }
    *out_value = (uint32_t)value;
    return 0;
}

static void usage(const char *program)
{
    fprintf(
        stderr,
        "usage: %s --plan plan.bin --weights weights.bin --input feature.f32 "
        "--warmup N --repeat N --threads 1\n",
        program);
}

static int parse_options(
    int argc, char **argv, ProfilerOptions *options)
{
    int index;

    if (options == NULL) {
        return 1;
    }
    memset(options, 0, sizeof(*options));
    options->warmup = 20u;
    options->repeat = 100u;
    options->requested_threads = 1u;

    for (index = 1; index < argc; ++index) {
        const char *name = argv[index];
        const char *value;
        if (strcmp(name, "--help") == 0 || strcmp(name, "-h") == 0) {
            usage(argv[0]);
            return 2;
        }
        if (index + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", name);
            return 1;
        }
        value = argv[++index];
        if (strcmp(name, "--plan") == 0) {
            options->plan_path = value;
        } else if (strcmp(name, "--weights") == 0) {
            options->weights_path = value;
        } else if (strcmp(name, "--input") == 0) {
            options->input_path = value;
        } else if (strcmp(name, "--warmup") == 0) {
            if (parse_u32(value, &options->warmup) != 0) {
                fprintf(stderr, "invalid --warmup: %s\n", value);
                return 1;
            }
        } else if (strcmp(name, "--repeat") == 0) {
            if (parse_u32(value, &options->repeat) != 0 ||
                options->repeat == 0u) {
                fprintf(stderr, "invalid --repeat: %s\n", value);
                return 1;
            }
        } else if (strcmp(name, "--threads") == 0) {
            if (parse_u32(value, &options->requested_threads) != 0 ||
                options->requested_threads != 1u) {
                fprintf(stderr, "profiler requires --threads 1\n");
                return 1;
            }
        } else {
            fprintf(stderr, "unknown option: %s\n", name);
            return 1;
        }
    }

    if (options->plan_path == NULL || options->weights_path == NULL ||
        options->input_path == NULL) {
        usage(argv[0]);
        return 1;
    }
    return 0;
}

static int read_entire_file(
    const char *path, uint8_t **out_data, size_t *out_size)
{
    FILE *file;
    long length;
    uint8_t *buffer;

    if (path == NULL || out_data == NULL || out_size == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
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

static int infer_input_dimensions(
    const CamppRuntimeModel *model,
    const CamppTensorDescriptor *descriptor, size_t input_size,
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
    if (time_axis_count != 1u || fixed_elements > UINT64_MAX / element_size) {
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

static void print_json_string(const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;
    putchar('"');
    if (cursor != NULL) {
        while (*cursor != '\0') {
            switch (*cursor) {
            case '"': fputs("\\\"", stdout); break;
            case '\\': fputs("\\\\", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (*cursor < 0x20u) {
                    printf("\\u%04x", (unsigned int)*cursor);
                } else {
                    putchar((int)*cursor);
                }
            }
            cursor += 1;
        }
    }
    putchar('"');
}

static const CamppKernelRegistry *registry_for_model(
    const CamppRuntimeModel *model)
{
    uint32_t operator_id;

    for (operator_id = 0u; operator_id < model->operator_count; ++operator_id) {
        if (model->operators[operator_id].kernel_id !=
            CAMPP_DEFAULT_KERNEL_ID) {
            return campp_cpu_aarch64_registry();
        }
    }
    return campp_cpu_reference_registry();
}

static void print_result(
    const ProfilerOptions *options, const CamppRuntimeModel *model,
    const CamppRuntimeContext *context,
    const CamppOperatorProfiler *profiler, const uint64_t *end_to_end_ns,
    size_t input_size)
{
    uint32_t index;

    fputs("{\"schema_version\":1,\"runtime\":\"campp-c-runtime\",", stdout);
    fputs("\"backend\":", stdout);
    print_json_string(context->registry->name);
    fputs(",\"clock\":", stdout);
    print_json_string(campp_operator_profiler_clock_name());
    printf(
        ",\"configuration\":{\"requested_threads\":%" PRIu32
        ",\"effective_threads\":1,\"warmup\":%" PRIu32
        ",\"repeat\":%" PRIu32 "},",
        options->requested_threads, options->warmup, options->repeat);
    fputs("\"model\":{\"plan_path\":", stdout);
    print_json_string(options->plan_path);
    fputs(",\"weights_path\":", stdout);
    print_json_string(options->weights_path);
    printf(
        ",\"bucket_frames\":%" PRIu32
        ",\"operator_count\":%" PRIu32 "},",
        model->bucket_frames, model->operator_count);
    fputs("\"input\":{\"path\":", stdout);
    print_json_string(options->input_path);
    printf(",\"byte_size\":%zu},", input_size);
    fputs("\"measurement_scope\":\"kernel_run_exclusive\",", stdout);
    fputs("\"end_to_end_ns\":[", stdout);
    for (index = 0u; index < options->repeat; ++index) {
        printf(
            "%s%" PRIu64, index == 0u ? "" : ",",
            end_to_end_ns[index]);
    }
    fputs("],\"operators\":[", stdout);
    for (index = 0u; index < model->operator_count; ++index) {
        const CamppOperatorDescriptor *descriptor = &model->operators[index];
        const CamppKernelEntry *kernel = context->resolved_kernels[index];
        const uint32_t call_count =
            campp_operator_profiler_call_count(profiler, index);
        const uint64_t *samples =
            campp_operator_profiler_samples(profiler, index);
        uint32_t call;

        printf(
            "%s{\"operator_id\":%" PRIu32
            ",\"kernel_id\":%u,\"kernel_name\":",
            index == 0u ? "" : ",", descriptor->operator_id,
            (unsigned int)descriptor->kernel_id);
        print_json_string(kernel->name);
        printf(",\"call_count\":%" PRIu32 ",\"samples_ns\":[", call_count);
        for (call = 0u; call < call_count; ++call) {
            printf(
                "%s%" PRIu64, call == 0u ? "" : ",",
                samples[call]);
        }
        fputs("]}", stdout);
    }
    fputs("]}\n", stdout);
}

int main(int argc, char **argv)
{
    ProfilerOptions options;
    CamppRuntimeModel model;
    CamppRuntimeContext context;
    CamppOperatorProfiler profiler;
    const CamppKernelRegistry *registry;
    const CamppTensorDescriptor *input_descriptor;
    uint32_t input_dimensions[CAMPP_TENSOR_MAX_RANK];
    uint32_t input_tensor_id;
    uint8_t *input_data = NULL;
    size_t input_size = 0u;
    uint64_t *end_to_end_ns = NULL;
    uint32_t iteration;
    CamppStatus status;
    int parse_result;
    int exit_code = 1;

    if (argc == 2 && strcmp(argv[1], "--capabilities") == 0) {
        fputs(
            "{\"runtime\":\"campp-c-runtime-profiler\","
            "\"effective_threads\":1,"
            "\"measurement_scope\":\"kernel_run_exclusive\","
            "\"clock\":",
            stdout);
        print_json_string(campp_operator_profiler_clock_name());
        fputs("}\n", stdout);
        return 0;
    }

    parse_result = parse_options(argc, argv, &options);
    if (parse_result != 0) {
        return parse_result == 2 ? 0 : 2;
    }
    memset(&model, 0, sizeof(model));
    memset(&context, 0, sizeof(context));
    memset(&profiler, 0, sizeof(profiler));

    status = campp_runtime_model_load(
        options.plan_path, options.weights_path, &model);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    registry = registry_for_model(&model);
    status = campp_runtime_context_create(&model, registry, &context);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "context create failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    if (model.input_count != 1u || model.output_count != 1u) {
        fprintf(stderr, "profiler requires exactly one input and output\n");
        goto cleanup;
    }
    input_tensor_id = model.input_tensor_ids[0];
    input_descriptor = &model.tensors[input_tensor_id];
    if (read_entire_file(options.input_path, &input_data, &input_size) != 0) {
        goto cleanup;
    }
    if (infer_input_dimensions(
            &model, input_descriptor, input_size, input_dimensions) != 0) {
        fprintf(stderr, "cannot derive input dimensions from %zu bytes\n", input_size);
        goto cleanup;
    }
    status = campp_runtime_context_bind_input(
        &context, input_tensor_id, input_data, input_size,
        input_descriptor->dtype, input_descriptor->rank, input_dimensions);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "input bind failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    status = campp_operator_profiler_create(
        model.operator_count, options.repeat, &profiler);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "profiler create failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    end_to_end_ns = (uint64_t *)calloc(
        options.repeat, sizeof(*end_to_end_ns));
    if (end_to_end_ns == NULL) {
        fprintf(stderr, "cannot allocate end-to-end samples\n");
        goto cleanup;
    }
    context.diagnostics.operator_profiler = &profiler;

    for (iteration = 0u; iteration < options.warmup; ++iteration) {
        status = campp_graph_execute(&context);
        if (status != CAMPP_STATUS_OK) {
            fprintf(
                stderr, "warmup failed at operator %" PRIu32 ": %s\n",
                context.diagnostics.current_operator_id,
                campp_status_name(status));
            goto cleanup;
        }
    }

    campp_operator_profiler_set_enabled(&profiler, true);
    for (iteration = 0u; iteration < options.repeat; ++iteration) {
        uint64_t started_ns;
        uint64_t finished_ns;

        status = campp_operator_profiler_clock_now_ns(&started_ns);
        if (status != CAMPP_STATUS_OK) {
            goto cleanup;
        }
        status = campp_graph_execute(&context);
        if (status != CAMPP_STATUS_OK) {
            fprintf(
                stderr, "measurement failed at operator %" PRIu32 ": %s\n",
                context.diagnostics.current_operator_id,
                campp_status_name(status));
            goto cleanup;
        }
        status = campp_operator_profiler_clock_now_ns(&finished_ns);
        if (status != CAMPP_STATUS_OK || finished_ns < started_ns) {
            goto cleanup;
        }
        end_to_end_ns[iteration] = finished_ns - started_ns;
    }
    campp_operator_profiler_set_enabled(&profiler, false);

    for (iteration = 0u; iteration < model.operator_count; ++iteration) {
        if (campp_operator_profiler_call_count(&profiler, iteration) !=
            options.repeat) {
            fprintf(
                stderr, "operator %" PRIu32 " has incomplete samples\n",
                iteration);
            goto cleanup;
        }
    }
    print_result(
        &options, &model, &context, &profiler, end_to_end_ns, input_size);
    if (ferror(stdout)) {
        fprintf(stderr, "cannot write profiler JSON\n");
        goto cleanup;
    }
    exit_code = 0;

cleanup:
    context.diagnostics.operator_profiler = NULL;
    free(end_to_end_ns);
    free(input_data);
    campp_operator_profiler_release(&profiler);
    campp_runtime_context_release(&context);
    campp_runtime_model_release(&model);
    return exit_code;
}
