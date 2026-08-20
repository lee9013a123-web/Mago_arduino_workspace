#define _POSIX_C_SOURCE 200809L

/*
 * CAM++ C Runtime의 cold/warm latency와 프로세스 RSS를 측정한다.
 *
 * 이 실행 파일은 Tensor dump callback을 설치하지 않는다. 모델과 context를 한
 * 번 만든 뒤 같은 입력을 반복 실행하므로 측정 구간에서 파일 I/O나 allocation이
 * 발생하지 않는다. JSON은 stdout, 오류와 진행 메시지는 stderr로만 출력한다.
 */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "execution/graph_executor.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"

#if defined(CAMPP_ENABLE_FINAL_CANDIDATE_SUITE)
#include "campp_profill/optimization/final_candidate_suite.h"
#endif

typedef struct BenchmarkOptions {
    const char *model_path;
    const char *plan_path;
    const char *weights_path;
    const char *input_path;
    const char *embedding_path;
    uint32_t warmup;
    uint32_t repeat;
    uint32_t requested_threads;
    double audio_seconds;
} BenchmarkOptions;

typedef struct MemorySnapshot {
    uint64_t current_rss_bytes;
    uint64_t peak_rss_bytes;
    int available;
} MemorySnapshot;

static uint64_t monotonic_ns(void)
{
    struct timespec value;
#ifdef _WIN32
    if (timespec_get(&value, TIME_UTC) != TIME_UTC) {
        return 0u;
    }
#else
#ifdef CLOCK_MONOTONIC_RAW
    const clockid_t clock_id = CLOCK_MONOTONIC_RAW;
#else
    const clockid_t clock_id = CLOCK_MONOTONIC;
#endif
    if (clock_gettime(clock_id, &value) != 0) {
        return 0u;
    }
#endif
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value.tv_nsec;
}

static double elapsed_ms(uint64_t start, uint64_t end)
{
    return end >= start ? (double)(end - start) / 1000000.0 : 0.0;
}

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

static int parse_positive_double(const char *text, double *out_value)
{
    char *end = NULL;
    double value;

    if (text == NULL || out_value == NULL || *text == '\0') {
        return 1;
    }
    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !(value > 0.0)) {
        return 1;
    }
    *out_value = value;
    return 0;
}

static void usage(const char *program)
{
    fprintf(
        stderr,
        "usage: %s (--model model.camppmodel | "
        "--plan plan.bin --weights weights.bin) --input feature.f32 "
        "--audio-seconds N --warmup N --repeat N --threads N "
        "[--embedding-output embedding.f32]\n",
        program);
}

static int parse_options(int argc, char **argv, BenchmarkOptions *options)
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
        if (strcmp(name, "--model") == 0) {
            options->model_path = value;
        } else if (strcmp(name, "--plan") == 0) {
            options->plan_path = value;
        } else if (strcmp(name, "--weights") == 0) {
            options->weights_path = value;
        } else if (strcmp(name, "--input") == 0) {
            options->input_path = value;
        } else if (strcmp(name, "--embedding-output") == 0) {
            options->embedding_path = value;
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
                options->requested_threads == 0u) {
                fprintf(stderr, "invalid --threads: %s\n", value);
                return 1;
            }
        } else if (strcmp(name, "--audio-seconds") == 0) {
            if (parse_positive_double(value, &options->audio_seconds) != 0) {
                fprintf(stderr, "invalid --audio-seconds: %s\n", value);
                return 1;
            }
        } else {
            fprintf(stderr, "unknown option: %s\n", name);
            return 1;
        }
    }

    if (options->input_path == NULL || !(options->audio_seconds > 0.0) ||
        ((options->model_path != NULL) ==
         (options->plan_path != NULL || options->weights_path != NULL)) ||
        (options->model_path == NULL &&
         (options->plan_path == NULL || options->weights_path == NULL))) {
        usage(argv[0]);
        return 1;
    }
    /* cpu_reference backend에는 아직 worker pool이 없다. */
    if (options->requested_threads != 1u) {
        fprintf(
            stderr,
            "requested %" PRIu32
            " threads, but cpu_reference effective_threads is 1\n",
            options->requested_threads);
        return 3;
    }
    return 0;
}

static int read_entire_file(const char *path, uint8_t **out_data, size_t *out_size)
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

static MemorySnapshot read_memory_snapshot(void)
{
    MemorySnapshot result;
    FILE *file;
    char line[256];

    memset(&result, 0, sizeof(result));
    file = fopen("/proc/self/status", "rb");
    if (file == NULL) {
        return result;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        unsigned long long kibibytes;
        if (sscanf(line, "VmRSS: %llu kB", &kibibytes) == 1) {
            result.current_rss_bytes = (uint64_t)kibibytes * 1024u;
            result.available = 1;
        } else if (sscanf(line, "VmHWM: %llu kB", &kibibytes) == 1) {
            result.peak_rss_bytes = (uint64_t)kibibytes * 1024u;
            result.available = 1;
        }
    }
    fclose(file);
    return result;
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
            case '\b': fputs("\\b", stdout); break;
            case '\f': fputs("\\f", stdout); break;
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

static void print_memory(const MemorySnapshot *snapshot)
{
    if (snapshot == NULL || !snapshot->available) {
        fputs("{\"current_rss_bytes\":null,\"peak_rss_bytes\":null}", stdout);
        return;
    }
    printf(
        "{\"current_rss_bytes\":%" PRIu64
        ",\"peak_rss_bytes\":%" PRIu64 "}",
        snapshot->current_rss_bytes, snapshot->peak_rss_bytes);
}

static int write_dense_payload(const char *path, const CamppTensorView *view)
{
    FILE *sink;
    const uint32_t element_size = campp_dtype_byte_size(view->dtype);
    const uint64_t element_count = campp_tensor_view_element_count(view);
    uint64_t index;

    if (path == NULL) {
        return 0;
    }
    if (view == NULL || view->data == NULL || element_size == 0u) {
        return 1;
    }
    sink = fopen(path, "wb");
    if (sink == NULL) {
        fprintf(stderr, "cannot open embedding output %s\n", path);
        return 1;
    }
    if (campp_tensor_view_is_contiguous(view)) {
        const size_t total = (size_t)element_count * element_size;
        if (fwrite(view->data, 1u, total, sink) != total) {
            fclose(sink);
            return 1;
        }
    } else {
        for (index = 0u; index < element_count; ++index) {
            uint64_t remaining = index;
            uint64_t offset = 0u;
            uint8_t axis;
            for (axis = view->rank; axis > 0u; --axis) {
                const uint8_t position = axis - 1u;
                const uint32_t coordinate =
                    (uint32_t)(remaining % view->dimensions[position]);
                remaining /= view->dimensions[position];
                offset += (uint64_t)coordinate * view->byte_strides[position];
            }
            if (fwrite(
                    (const uint8_t *)view->data + offset,
                    1u, element_size, sink) != element_size) {
                fclose(sink);
                return 1;
            }
        }
    }
    return fclose(sink) == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    BenchmarkOptions options;
    CamppRuntimeModel model;
    CamppRuntimeContext context;
    const CamppKernelRegistry *registry = NULL;
    const CamppTensorDescriptor *input_descriptor;
    const CamppTensorView *output_view = NULL;
    uint32_t input_dimensions[CAMPP_TENSOR_MAX_RANK];
    uint32_t input_tensor_id;
    uint8_t *input_data = NULL;
    size_t input_size = 0u;
    double *timings = NULL;
    CamppStatus status;
    MemorySnapshot memory_start;
    MemorySnapshot memory_after_model;
    MemorySnapshot memory_after_context;
    MemorySnapshot memory_after_warmup;
    MemorySnapshot memory_after_measurement;
    uint64_t process_started;
    uint64_t model_started;
    uint64_t model_finished;
    uint64_t context_started;
    uint64_t context_finished;
    uint64_t input_started;
    uint64_t input_finished;
    double first_inference_ms = 0.0;
    uint32_t iteration;
    int parse_result;
    int exit_code = 1;
#if defined(CAMPP_ENABLE_FINAL_CANDIDATE_SUITE)
    CamppFinalCandidateSuite final_candidate_suite;
#endif

    if (argc == 2 && strcmp(argv[1], "--capabilities") == 0) {
        fputs(
            "{\"runtime\":\"campp-c-runtime\",\"effective_threads\":1,"
            "\"threading\":\"single-thread\","
            "\"backends\":[\"cpu_reference\",\"cpu_aarch64_o4i4\"],"
            "\"model_package_format\":\"camppmodel-v1\"",
            stdout);
#if defined(CAMPP_ENABLE_FINAL_CANDIDATE_SUITE)
        fputs(",\"optimization_suite\":\"final\","
              "\"optimization_suite_config\":", stdout);
        print_json_string(campp_final_candidate_suite_name());
#else
        fputs(",\"optimization_suite\":\"stock\","
              "\"optimization_suite_config\":null", stdout);
#endif
        fputs("}\n", stdout);
        return 0;
    }

    process_started = monotonic_ns();
    parse_result = parse_options(argc, argv, &options);
    if (parse_result != 0) {
        return parse_result == 2 ? 0 : parse_result;
    }
    memory_start = read_memory_snapshot();

    memset(&model, 0, sizeof(model));
    memset(&context, 0, sizeof(context));
    model_started = monotonic_ns();
    status = options.model_path != NULL
        ? campp_runtime_model_load_package(options.model_path, &model)
        : campp_runtime_model_load(
            options.plan_path, options.weights_path, &model);
    model_finished = monotonic_ns();
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    registry =
        model.operator_count != 0u && model.operators[0].kernel_id != 0u
            ? campp_cpu_aarch64_registry()
            : campp_cpu_reference_registry();
#if defined(CAMPP_ENABLE_FINAL_CANDIDATE_SUITE)
    status = campp_final_candidate_suite_create(
        registry, &final_candidate_suite);
    if (status != CAMPP_STATUS_OK) {
        fprintf(
            stderr, "final candidate suite create failed: %s\n",
            campp_status_name(status));
        goto cleanup;
    }
    registry = &final_candidate_suite.registry;
#endif
    memory_after_model = read_memory_snapshot();

    context_started = monotonic_ns();
    status = campp_runtime_context_create(&model, registry, &context);
    context_finished = monotonic_ns();
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "context create failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    memory_after_context = read_memory_snapshot();

    if (model.input_count != 1u || model.output_count != 1u) {
        fprintf(stderr, "benchmark requires exactly one input and one output\n");
        goto cleanup;
    }
    input_tensor_id = model.input_tensor_ids[0];
    input_descriptor = &model.tensors[input_tensor_id];

    input_started = monotonic_ns();
    if (read_entire_file(options.input_path, &input_data, &input_size) != 0) {
        goto cleanup;
    }
    input_finished = monotonic_ns();
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

    timings = (double *)calloc(options.repeat, sizeof(*timings));
    if (timings == NULL) {
        fprintf(stderr, "cannot allocate timing array\n");
        goto cleanup;
    }

    for (iteration = 0u; iteration < options.warmup; ++iteration) {
        const uint64_t started = monotonic_ns();
        status = campp_graph_execute(&context);
        if (status != CAMPP_STATUS_OK) {
            fprintf(
                stderr, "warmup failed at operator %" PRIu32 ": %s\n",
                context.diagnostics.current_operator_id, campp_status_name(status));
            goto cleanup;
        }
        if (iteration == 0u) {
            first_inference_ms = elapsed_ms(started, monotonic_ns());
        }
    }
    memory_after_warmup = read_memory_snapshot();

    for (iteration = 0u; iteration < options.repeat; ++iteration) {
        const uint64_t started = monotonic_ns();
        status = campp_graph_execute(&context);
        if (status != CAMPP_STATUS_OK) {
            fprintf(
                stderr, "measurement failed at operator %" PRIu32 ": %s\n",
                context.diagnostics.current_operator_id, campp_status_name(status));
            goto cleanup;
        }
        timings[iteration] = elapsed_ms(started, monotonic_ns());
    }
    if (options.warmup == 0u) {
        first_inference_ms = timings[0];
    }
    memory_after_measurement = read_memory_snapshot();

    status = campp_runtime_context_output(
        &context, model.output_tensor_ids[0], &output_view);
    if (status != CAMPP_STATUS_OK || output_view == NULL) {
        fprintf(stderr, "output lookup failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    if (write_dense_payload(options.embedding_path, output_view) != 0) {
        goto cleanup;
    }

    fputs("{\"schema_version\":1,\"runtime\":\"campp-c-runtime\",", stdout);
    fputs("\"backend\":", stdout);
    print_json_string(registry->name);
    fputs(",\"optimization_suite\":", stdout);
#if defined(CAMPP_ENABLE_FINAL_CANDIDATE_SUITE)
    print_json_string("final");
    fputs(",\"optimization_suite_config\":", stdout);
    print_json_string(campp_final_candidate_suite_name());
#else
    print_json_string("stock");
#endif
    fputs(",", stdout);
    fputs("\"configuration\":{", stdout);
    printf(
        "\"requested_threads\":%" PRIu32
        ",\"effective_threads\":1,\"warmup\":%" PRIu32
        ",\"repeat\":%" PRIu32 ",\"audio_seconds\":%.9g},",
        options.requested_threads, options.warmup, options.repeat,
        options.audio_seconds);
    fputs("\"model\":{\"package_path\":", stdout);
    if (options.model_path != NULL) {
        print_json_string(options.model_path);
    } else {
        fputs("null", stdout);
    }
    fputs(",\"plan_path\":", stdout);
    if (options.plan_path != NULL) {
        print_json_string(options.plan_path);
    } else {
        fputs("null", stdout);
    }
    fputs(",\"weights_path\":", stdout);
    if (options.weights_path != NULL) {
        print_json_string(options.weights_path);
    } else {
        fputs("null", stdout);
    }
    printf(
        ",\"bucket_frames\":%" PRIu32
        ",\"plan_bytes\":%zu,\"weight_bytes\":%zu,"
        "\"operator_count\":%" PRIu32 "},",
        model.bucket_frames, model.plan_size, model.weights_size,
        model.operator_count);
    fputs("\"input\":{\"path\":", stdout);
    print_json_string(options.input_path);
    printf(",\"byte_size\":%zu},", input_size);
    printf(
        "\"lifecycle\":{\"model_load_ms\":%.9g,"
        "\"context_create_ms\":%.9g,\"input_load_ms\":%.9g,"
        "\"first_inference_ms\":%.9g,\"process_elapsed_ms\":%.9g},",
        elapsed_ms(model_started, model_finished),
        elapsed_ms(context_started, context_finished),
        elapsed_ms(input_started, input_finished), first_inference_ms,
        elapsed_ms(process_started, monotonic_ns()));
    fputs("\"warm\":{\"timings_ms\":[", stdout);
    for (iteration = 0u; iteration < options.repeat; ++iteration) {
        printf("%s%.9g", iteration == 0u ? "" : ",", timings[iteration]);
    }
    fputs("]},\"memory\":{\"start\":", stdout);
    print_memory(&memory_start);
    fputs(",\"after_model_load\":", stdout);
    print_memory(&memory_after_model);
    fputs(",\"after_context_create\":", stdout);
    print_memory(&memory_after_context);
    fputs(",\"after_warmup\":", stdout);
    print_memory(&memory_after_warmup);
    fputs(",\"after_measurement\":", stdout);
    print_memory(&memory_after_measurement);
    printf(
        ",\"activation_bytes\":%zu,\"arena_bytes\":%zu,"
        "\"scratch_bytes\":%zu},",
        context.activations.total_bytes, context.activations.arena_size,
        context.scratch_size);
    printf(
        "\"embedding\":{\"tensor_id\":%" PRIu32
        ",\"dtype\":%u,\"shape\":[",
        model.output_tensor_ids[0], output_view->dtype);
    for (iteration = 0u; iteration < output_view->rank; ++iteration) {
        printf(
            "%s%" PRIu32, iteration == 0u ? "" : ",",
            output_view->dimensions[iteration]);
    }
    fputs("],\"output_path\":", stdout);
    if (options.embedding_path == NULL) {
        fputs("null", stdout);
    } else {
        print_json_string(options.embedding_path);
    }
    fputs("}}\n", stdout);
    if (ferror(stdout)) {
        fprintf(stderr, "cannot write JSON result\n");
        goto cleanup;
    }
    exit_code = 0;

cleanup:
    free(timings);
    free(input_data);
    campp_runtime_context_release(&context);
    campp_runtime_model_release(&model);
    return exit_code;
}
