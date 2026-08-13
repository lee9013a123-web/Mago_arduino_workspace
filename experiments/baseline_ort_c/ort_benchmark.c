/*
 * Python 없이 ONNX Runtime C API로 직접 추론해 latency / RTF / RSS를 측정한다.
 *
 * scripts/1_benchmark/benchmark_onnx.py와 같은 일을 하지만 Python 인터프리터와
 * numpy가 프로세스에 없다. 그래서 RSS가 ORT 자체의 사용량에 가깝다.
 *
 * 출력 JSON은 src/c/runtime/command_line/campp_runtime_benchmark.c의 스키마를
 * 따른다. experiments/rtf/measure_rtf.py가 두 backend를 같은 코드로 읽는다.
 *
 * usage: campp_ort_benchmark --model m.onnx --input feature.f32
 *            --audio-seconds N --warmup N --repeat N --threads N
 *            [--graph-opt disable|basic|extended|all]
 *            [--embedding-output embedding.f32]
 */

#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "onnxruntime_c_api.h"

#define FEATURE_DIM 80

static const OrtApi *g_ort = NULL;

typedef struct MemorySnapshot {
    uint64_t current_rss_bytes;
    uint64_t peak_rss_bytes;
    int available;
} MemorySnapshot;

static uint64_t monotonic_ns(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0u;
    }
    return (uint64_t)value.tv_sec * 1000000000u + (uint64_t)value.tv_nsec;
}

static double ns_to_ms(uint64_t nanoseconds)
{
    return (double)nanoseconds / 1000000.0;
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

static void print_memory(const MemorySnapshot *snapshot)
{
    if (snapshot == NULL || !snapshot->available) {
        fputs("{\"current_rss_bytes\":null,\"peak_rss_bytes\":null}", stdout);
        return;
    }
    printf(
        "{\"current_rss_bytes\":%" PRIu64 ",\"peak_rss_bytes\":%" PRIu64 "}",
        snapshot->current_rss_bytes, snapshot->peak_rss_bytes);
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

/* OrtStatus가 NULL이 아니면 메시지를 찍고 종료한다. */
static void check(OrtStatus *status, const char *context)
{
    if (status != NULL) {
        fprintf(stderr, "%s failed: %s\n", context, g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

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

static GraphOptimizationLevel parse_graph_opt(const char *name)
{
    if (strcmp(name, "disable") == 0) { return ORT_DISABLE_ALL; }
    if (strcmp(name, "basic") == 0)   { return ORT_ENABLE_BASIC; }
    if (strcmp(name, "extended") == 0){ return ORT_ENABLE_EXTENDED; }
    if (strcmp(name, "all") == 0)     { return ORT_ENABLE_ALL; }
    fprintf(stderr, "unknown --graph-opt %s\n", name);
    exit(2);
}

static int compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *input_path = NULL;
    const char *embedding_output = NULL;
    const char *graph_opt_name = "all";
    double audio_seconds = 0.0;
    int warmup = 0, repeat = 1, threads = 1;
    int index;

    uint8_t *input_data = NULL;
    size_t input_size = 0u;
    int64_t frames;
    int64_t shape[3];

    OrtEnv *env = NULL;
    OrtSessionOptions *options = NULL;
    OrtSession *session = NULL;
    OrtMemoryInfo *memory_info = NULL;
    OrtAllocator *allocator = NULL;
    OrtValue *input_tensor = NULL;
    char *input_name = NULL;
    char *output_name = NULL;
    const char *input_names[1];
    const char *output_names[1];

    MemorySnapshot memory_start, memory_after_session, memory_after_input;
    MemorySnapshot memory_after_warmup, memory_after_measure;
    uint64_t process_started, session_started, session_finished;
    uint64_t input_started, input_finished;
    double first_inference_ms = 0.0;
    double *timings = NULL;
    double *sorted = NULL;
    const float *embedding = NULL;
    size_t embedding_count = 0u;

    for (index = 1; index < argc; ++index) {
        const char *flag = argv[index];
        const char *value = (index + 1 < argc) ? argv[index + 1] : NULL;
        if (strcmp(flag, "--model") == 0 && value)                 { model_path = value; index++; }
        else if (strcmp(flag, "--input") == 0 && value)            { input_path = value; index++; }
        else if (strcmp(flag, "--audio-seconds") == 0 && value)    { audio_seconds = atof(value); index++; }
        else if (strcmp(flag, "--warmup") == 0 && value)           { warmup = atoi(value); index++; }
        else if (strcmp(flag, "--repeat") == 0 && value)           { repeat = atoi(value); index++; }
        else if (strcmp(flag, "--threads") == 0 && value)          { threads = atoi(value); index++; }
        else if (strcmp(flag, "--graph-opt") == 0 && value)        { graph_opt_name = value; index++; }
        else if (strcmp(flag, "--embedding-output") == 0 && value) { embedding_output = value; index++; }
        else {
            fprintf(stderr,
                    "usage: %s --model m.onnx --input feature.f32 "
                    "--audio-seconds N --warmup N --repeat N --threads N "
                    "[--graph-opt disable|basic|extended|all] "
                    "[--embedding-output out.f32]\n",
                    argv[0]);
            return 2;
        }
    }
    if (model_path == NULL || input_path == NULL || audio_seconds <= 0.0 ||
        repeat < 1 || warmup < 0 || threads < 1) {
        fprintf(stderr, "missing or invalid arguments\n");
        return 2;
    }

    process_started = monotonic_ns();
    memory_start = read_memory_snapshot();

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (g_ort == NULL) {
        fprintf(stderr, "ORT API version %d is not available\n", ORT_API_VERSION);
        return 1;
    }

    session_started = monotonic_ns();
    check(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "campp_ort_benchmark", &env),
          "CreateEnv");
    check(g_ort->CreateSessionOptions(&options), "CreateSessionOptions");
    check(g_ort->SetIntraOpNumThreads(options, threads), "SetIntraOpNumThreads");
    check(g_ort->SetInterOpNumThreads(options, 1), "SetInterOpNumThreads");
    check(g_ort->SetSessionExecutionMode(options, ORT_SEQUENTIAL),
          "SetSessionExecutionMode");
    check(g_ort->SetSessionGraphOptimizationLevel(
              options, parse_graph_opt(graph_opt_name)),
          "SetSessionGraphOptimizationLevel");
    check(g_ort->CreateSession(env, model_path, options, &session), "CreateSession");
    session_finished = monotonic_ns();
    memory_after_session = read_memory_snapshot();

    check(g_ort->GetAllocatorWithDefaultOptions(&allocator), "GetAllocator");
    check(g_ort->SessionGetInputName(session, 0, allocator, &input_name),
          "SessionGetInputName");
    check(g_ort->SessionGetOutputName(session, 0, allocator, &output_name),
          "SessionGetOutputName");
    input_names[0] = input_name;
    output_names[0] = output_name;

    input_started = monotonic_ns();
    if (read_entire_file(input_path, &input_data, &input_size) != 0) {
        return 1;
    }
    if (input_size % (FEATURE_DIM * sizeof(float)) != 0u) {
        fprintf(stderr, "input size %zu is not a multiple of %zu\n",
                input_size, FEATURE_DIM * sizeof(float));
        return 1;
    }
    frames = (int64_t)(input_size / (FEATURE_DIM * sizeof(float)));
    shape[0] = 1; shape[1] = frames; shape[2] = FEATURE_DIM;
    check(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info),
          "CreateCpuMemoryInfo");
    check(g_ort->CreateTensorWithDataAsOrtValue(
              memory_info, input_data, input_size, shape, 3,
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor),
          "CreateTensorWithDataAsOrtValue");
    input_finished = monotonic_ns();
    memory_after_input = read_memory_snapshot();

    timings = (double *)malloc(sizeof(double) * (size_t)repeat);
    sorted = (double *)malloc(sizeof(double) * (size_t)repeat);
    if (timings == NULL || sorted == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    for (index = 0; index < warmup + repeat; ++index) {
        OrtValue *output_tensor = NULL;
        const uint64_t started = monotonic_ns();
        check(g_ort->Run(session, NULL, input_names,
                         (const OrtValue *const *)&input_tensor, 1,
                         output_names, 1, &output_tensor),
              "Run");
        {
            const double elapsed = ns_to_ms(monotonic_ns() - started);
            if (index == 0) {
                first_inference_ms = elapsed;
            }
            if (index >= warmup) {
                timings[index - warmup] = elapsed;
            }
        }
        if (index == warmup + repeat - 1) {
            /* 마지막 출력만 embedding 보고용으로 남긴다. */
            OrtTensorTypeAndShapeInfo *info = NULL;
            float *values = NULL;
            check(g_ort->GetTensorTypeAndShape(output_tensor, &info),
                  "GetTensorTypeAndShape");
            check(g_ort->GetTensorShapeElementCount(info, &embedding_count),
                  "GetTensorShapeElementCount");
            g_ort->ReleaseTensorTypeAndShapeInfo(info);
            check(g_ort->GetTensorMutableData(output_tensor, (void **)&values),
                  "GetTensorMutableData");
            embedding = (const float *)malloc(sizeof(float) * embedding_count);
            if (embedding != NULL) {
                memcpy((void *)embedding, values, sizeof(float) * embedding_count);
            }
        }
        if (index == warmup - 1) {
            memory_after_warmup = read_memory_snapshot();
        }
        g_ort->ReleaseValue(output_tensor);
    }
    if (warmup == 0) {
        memory_after_warmup = memory_after_input;
    }
    memory_after_measure = read_memory_snapshot();

    if (embedding_output != NULL && embedding != NULL) {
        FILE *sink = fopen(embedding_output, "wb");
        if (sink != NULL) {
            fwrite(embedding, sizeof(float), embedding_count, sink);
            fclose(sink);
        }
    }

    memcpy(sorted, timings, sizeof(double) * (size_t)repeat);
    qsort(sorted, (size_t)repeat, sizeof(double), compare_double);

    printf("{\"schema_version\":1,\"runtime\":\"onnxruntime-c-api\"");
    printf(",\"ort_version\":");
    print_json_string(OrtGetApiBase()->GetVersionString());
    printf(",\"configuration\":{\"requested_threads\":%d,\"effective_threads\":%d"
           ",\"warmup\":%d,\"repeat\":%d,\"audio_seconds\":%g"
           ",\"graph_optimization_level\":", threads, threads, warmup, repeat,
           audio_seconds);
    print_json_string(graph_opt_name);
    printf(",\"execution_mode\":\"ORT_SEQUENTIAL\"}");

    printf(",\"model\":{\"path\":");
    print_json_string(model_path);
    printf(",\"input_name\":");
    print_json_string(input_name);
    printf(",\"output_name\":");
    print_json_string(output_name);
    printf(",\"frames\":%" PRId64 "}", frames);

    printf(",\"input\":{\"path\":");
    print_json_string(input_path);
    printf(",\"byte_size\":%zu}", input_size);

    printf(",\"lifecycle\":{\"model_load_ms\":%.6f,\"context_create_ms\":0.0"
           ",\"input_load_ms\":%.6f,\"first_inference_ms\":%.6f"
           ",\"process_elapsed_ms\":%.6f}",
           ns_to_ms(session_finished - session_started),
           ns_to_ms(input_finished - input_started),
           first_inference_ms,
           ns_to_ms(monotonic_ns() - process_started));

    printf(",\"warm\":{\"timings_ms\":[");
    for (index = 0; index < repeat; ++index) {
        printf("%s%.6f", index == 0 ? "" : ",", timings[index]);
    }
    printf("],\"p50_ms\":%.6f}", sorted[repeat / 2]);

    printf(",\"memory\":{\"start\":");
    print_memory(&memory_start);
    printf(",\"after_model_load\":");
    print_memory(&memory_after_session);
    printf(",\"after_context_create\":");
    print_memory(&memory_after_input);
    printf(",\"after_warmup\":");
    print_memory(&memory_after_warmup);
    printf(",\"after_measurement\":");
    print_memory(&memory_after_measure);
    printf(",\"activation_bytes\":null,\"arena_bytes\":null,\"scratch_bytes\":null}");

    printf(",\"embedding\":{\"element_count\":%zu}}\n", embedding_count);

    free((void *)embedding);
    free(timings);
    free(sorted);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    if (allocator != NULL) {
        allocator->Free(allocator, input_name);
        allocator->Free(allocator, output_name);
    }
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(options);
    g_ort->ReleaseEnv(env);
    free(input_data);
    return 0;
}
