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
static void check(OrtStatus *status, const char *context);

typedef struct MemorySnapshot {
    uint64_t current_rss_bytes;
    uint64_t peak_rss_bytes;
    uint64_t virtual_memory_bytes;
    uint64_t rss_anon_bytes;
    uint64_t rss_file_bytes;
    uint64_t rss_shmem_bytes;
    uint64_t vm_data_bytes;
    uint64_t vm_stack_bytes;
    uint64_t vm_executable_bytes;
    uint64_t vm_library_bytes;
    uint64_t pss_bytes;
    uint64_t private_clean_bytes;
    uint64_t private_dirty_bytes;
    uint64_t shared_clean_bytes;
    uint64_t shared_dirty_bytes;
    uint64_t anonymous_bytes;
    uint64_t swap_bytes;
    int status_available;
    int smaps_available;
} MemorySnapshot;

typedef struct AllocatorStats {
    uint64_t in_use;
    uint64_t total_allocated;
    uint64_t max_in_use;
    uint64_t num_allocs;
    uint64_t num_reserves;
    uint64_t num_arena_extensions;
    uint64_t max_alloc_size;
    int available;
} AllocatorStats;

typedef struct InferenceObservation {
    double elapsed_ms;
    MemorySnapshot memory_before;
    MemorySnapshot memory_after_run;
    MemorySnapshot memory_after_release;
    AllocatorStats allocator_before;
    AllocatorStats allocator_after_run;
    AllocatorStats allocator_after_release;
    uint64_t output_hash;
    size_t output_bytes;
} InferenceObservation;

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
            result.status_available = 1;
        } else if (sscanf(line, "VmHWM: %llu kB", &kibibytes) == 1) {
            result.peak_rss_bytes = (uint64_t)kibibytes * 1024u;
            result.status_available = 1;
        } else if (sscanf(line, "VmSize: %llu kB", &kibibytes) == 1) {
            result.virtual_memory_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "RssAnon: %llu kB", &kibibytes) == 1) {
            result.rss_anon_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "RssFile: %llu kB", &kibibytes) == 1) {
            result.rss_file_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "RssShmem: %llu kB", &kibibytes) == 1) {
            result.rss_shmem_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "VmData: %llu kB", &kibibytes) == 1) {
            result.vm_data_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "VmStk: %llu kB", &kibibytes) == 1) {
            result.vm_stack_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "VmExe: %llu kB", &kibibytes) == 1) {
            result.vm_executable_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "VmLib: %llu kB", &kibibytes) == 1) {
            result.vm_library_bytes = (uint64_t)kibibytes * 1024u;
        }
    }
    fclose(file);

    file = fopen("/proc/self/smaps_rollup", "rb");
    if (file == NULL) {
        return result;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        unsigned long long kibibytes;
        if (sscanf(line, "Pss: %llu kB", &kibibytes) == 1) {
            result.pss_bytes = (uint64_t)kibibytes * 1024u;
            result.smaps_available = 1;
        } else if (sscanf(line, "Private_Clean: %llu kB", &kibibytes) == 1) {
            result.private_clean_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "Private_Dirty: %llu kB", &kibibytes) == 1) {
            result.private_dirty_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "Shared_Clean: %llu kB", &kibibytes) == 1) {
            result.shared_clean_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "Shared_Dirty: %llu kB", &kibibytes) == 1) {
            result.shared_dirty_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "Anonymous: %llu kB", &kibibytes) == 1) {
            result.anonymous_bytes = (uint64_t)kibibytes * 1024u;
        } else if (sscanf(line, "Swap: %llu kB", &kibibytes) == 1) {
            result.swap_bytes = (uint64_t)kibibytes * 1024u;
        }
    }
    fclose(file);
    return result;
}

static void print_optional_u64(int available, uint64_t value)
{
    if (available) {
        printf("%" PRIu64, value);
    } else {
        fputs("null", stdout);
    }
}

static void print_memory(const MemorySnapshot *snapshot)
{
    if (snapshot == NULL) {
        fputs("null", stdout);
        return;
    }
    fputs("{\"current_rss_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->current_rss_bytes);
    fputs(",\"peak_rss_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->peak_rss_bytes);
    fputs(",\"virtual_memory_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->virtual_memory_bytes);
    fputs(",\"rss_anon_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->rss_anon_bytes);
    fputs(",\"rss_file_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->rss_file_bytes);
    fputs(",\"rss_shmem_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->rss_shmem_bytes);
    fputs(",\"vm_data_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->vm_data_bytes);
    fputs(",\"vm_stack_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->vm_stack_bytes);
    fputs(",\"vm_executable_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->vm_executable_bytes);
    fputs(",\"vm_library_bytes\":", stdout);
    print_optional_u64(snapshot->status_available, snapshot->vm_library_bytes);
    fputs(",\"pss_bytes\":", stdout);
    print_optional_u64(snapshot->smaps_available, snapshot->pss_bytes);
    fputs(",\"private_clean_bytes\":", stdout);
    print_optional_u64(snapshot->smaps_available, snapshot->private_clean_bytes);
    fputs(",\"private_dirty_bytes\":", stdout);
    print_optional_u64(snapshot->smaps_available, snapshot->private_dirty_bytes);
    fputs(",\"shared_clean_bytes\":", stdout);
    print_optional_u64(snapshot->smaps_available, snapshot->shared_clean_bytes);
    fputs(",\"shared_dirty_bytes\":", stdout);
    print_optional_u64(snapshot->smaps_available, snapshot->shared_dirty_bytes);
    fputs(",\"anonymous_bytes\":", stdout);
    print_optional_u64(snapshot->smaps_available, snapshot->anonymous_bytes);
    fputs(",\"swap_bytes\":", stdout);
    print_optional_u64(snapshot->smaps_available, snapshot->swap_bytes);
    putchar('}');
}

static uint64_t parse_allocator_stat(const OrtKeyValuePairs *pairs, const char *key)
{
    const char *value = g_ort->GetKeyValue(pairs, key);
    char *end = NULL;
    unsigned long long parsed;
    if (value == NULL || *value == '\0') {
        return 0u;
    }
    parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t)parsed : 0u;
}

static AllocatorStats read_allocator_stats(OrtAllocator *allocator)
{
    AllocatorStats result;
    OrtKeyValuePairs *pairs = NULL;
    OrtStatus *status;

    memset(&result, 0, sizeof(result));
    if (allocator == NULL || g_ort->AllocatorGetStats == NULL) {
        return result;
    }
    status = g_ort->AllocatorGetStats(allocator, &pairs);
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        return result;
    }
    if (pairs == NULL || g_ort->GetKeyValue(pairs, "InUse") == NULL) {
        g_ort->ReleaseKeyValuePairs(pairs);
        return result;
    }
    result.in_use = parse_allocator_stat(pairs, "InUse");
    result.total_allocated = parse_allocator_stat(pairs, "TotalAllocated");
    result.max_in_use = parse_allocator_stat(pairs, "MaxInUse");
    result.num_allocs = parse_allocator_stat(pairs, "NumAllocs");
    result.num_reserves = parse_allocator_stat(pairs, "NumReserves");
    result.num_arena_extensions = parse_allocator_stat(pairs, "NumArenaExtensions");
    result.max_alloc_size = parse_allocator_stat(pairs, "MaxAllocSize");
    result.available = 1;
    g_ort->ReleaseKeyValuePairs(pairs);
    return result;
}

static void print_allocator_stats(const AllocatorStats *stats)
{
    if (stats == NULL || !stats->available) {
        fputs("null", stdout);
        return;
    }
    printf("{\"in_use_bytes\":%" PRIu64
           ",\"total_allocated_bytes\":%" PRIu64
           ",\"max_in_use_bytes\":%" PRIu64
           ",\"num_allocs\":%" PRIu64
           ",\"num_reserves\":%" PRIu64
           ",\"num_arena_extensions\":%" PRIu64
           ",\"max_alloc_size_bytes\":%" PRIu64 "}",
           stats->in_use, stats->total_allocated, stats->max_in_use,
           stats->num_allocs, stats->num_reserves,
           stats->num_arena_extensions, stats->max_alloc_size);
}

static uint64_t nonnegative_delta(uint64_t after, uint64_t before)
{
    return after >= before ? after - before : 0u;
}

static void print_allocator_delta(
    const AllocatorStats *before, const AllocatorStats *after)
{
    if (before == NULL || after == NULL ||
        !before->available || !after->available) {
        fputs("null", stdout);
        return;
    }
    printf("{\"total_allocated_bytes\":%" PRIu64
           ",\"num_allocs\":%" PRIu64
           ",\"num_reserves\":%" PRIu64
           ",\"num_arena_extensions\":%" PRIu64
           ",\"in_use_bytes\":%" PRIu64 "}",
           nonnegative_delta(after->total_allocated, before->total_allocated),
           nonnegative_delta(after->num_allocs, before->num_allocs),
           nonnegative_delta(after->num_reserves, before->num_reserves),
           nonnegative_delta(after->num_arena_extensions,
                             before->num_arena_extensions),
           after->in_use);
}

static uint64_t fnv1a64(const void *data, size_t byte_size)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t value = UINT64_C(14695981039346656037);
    size_t index;
    for (index = 0; index < byte_size; ++index) {
        value ^= bytes[index];
        value *= UINT64_C(1099511628211);
    }
    return value;
}

static void inspect_output(
    OrtValue *output, uint64_t *out_hash, size_t *out_bytes)
{
    void *data = NULL;
    size_t byte_size = 0u;
    check(g_ort->GetTensorSizeInBytes(output, &byte_size),
          "GetTensorSizeInBytes");
    check(g_ort->GetTensorMutableData(output, &data), "GetTensorMutableData");
    *out_hash = fnv1a64(data, byte_size);
    *out_bytes = byte_size;
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
    const char *memory_pattern_name = "on";
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
    OrtAllocator *session_allocator = NULL;
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
    InferenceObservation *observations = NULL;
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
        else if (strcmp(flag, "--memory-pattern") == 0 && value)   { memory_pattern_name = value; index++; }
        else if (strcmp(flag, "--embedding-output") == 0 && value) { embedding_output = value; index++; }
        else {
            fprintf(stderr,
                    "usage: %s --model m.onnx --input feature.f32 "
                    "--audio-seconds N --warmup N --repeat N --threads N "
                    "[--graph-opt disable|basic|extended|all] "
                    "[--memory-pattern on|off] "
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
    if (strcmp(memory_pattern_name, "on") != 0 &&
        strcmp(memory_pattern_name, "off") != 0) {
        fprintf(stderr, "--memory-pattern must be on or off\n");
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
    if (strcmp(memory_pattern_name, "on") == 0) {
        check(g_ort->EnableMemPattern(options), "EnableMemPattern");
    } else {
        check(g_ort->DisableMemPattern(options), "DisableMemPattern");
    }
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
    check(g_ort->CreateAllocator(session, memory_info, &session_allocator),
          "CreateAllocator");
    input_finished = monotonic_ns();
    memory_after_input = read_memory_snapshot();

    timings = (double *)malloc(sizeof(double) * (size_t)repeat);
    sorted = (double *)malloc(sizeof(double) * (size_t)repeat);
    observations = (InferenceObservation *)calloc(
        (size_t)(warmup + repeat), sizeof(*observations));
    if (timings == NULL || sorted == NULL || observations == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    for (index = 0; index < warmup + repeat; ++index) {
        OrtValue *output_tensor = NULL;
        InferenceObservation *observation = &observations[index];
        observation->memory_before = read_memory_snapshot();
        observation->allocator_before = read_allocator_stats(session_allocator);
        const uint64_t started = monotonic_ns();
        check(g_ort->Run(session, NULL, input_names,
                         (const OrtValue *const *)&input_tensor, 1,
                         output_names, 1, &output_tensor),
              "Run");
        {
            const double elapsed = ns_to_ms(monotonic_ns() - started);
            observation->elapsed_ms = elapsed;
            if (index == 0) {
                first_inference_ms = elapsed;
            }
            if (index >= warmup) {
                timings[index - warmup] = elapsed;
            }
        }
        observation->memory_after_run = read_memory_snapshot();
        observation->allocator_after_run = read_allocator_stats(session_allocator);
        inspect_output(output_tensor, &observation->output_hash,
                       &observation->output_bytes);
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
        observation->memory_after_release = read_memory_snapshot();
        observation->allocator_after_release =
            read_allocator_stats(session_allocator);
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
    printf(",\"memory_pattern\":");
    print_json_string(memory_pattern_name);
    printf(",\"cpu_memory_arena\":true"
           ",\"execution_mode\":\"ORT_SEQUENTIAL\"}");

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

    printf(",\"inferences\":[");
    for (index = 0; index < warmup + repeat; ++index) {
        const InferenceObservation *observation = &observations[index];
        printf("%s{\"session_run\":%d,\"phase\":\"%s\""
               ",\"latency_ms\":%.6f,\"output_hash_fnv1a64\":\"%016" PRIx64 "\""
               ",\"output_bytes\":%zu,\"memory_before\":",
               index == 0 ? "" : ",", index + 1,
               index < warmup ? "warmup" : "measurement",
               observation->elapsed_ms, observation->output_hash,
               observation->output_bytes);
        print_memory(&observation->memory_before);
        printf(",\"memory_after_run\":");
        print_memory(&observation->memory_after_run);
        printf(",\"memory_after_output_release\":");
        print_memory(&observation->memory_after_release);
        printf(",\"allocator_before\":");
        print_allocator_stats(&observation->allocator_before);
        printf(",\"allocator_after_run\":");
        print_allocator_stats(&observation->allocator_after_run);
        printf(",\"allocator_after_output_release\":");
        print_allocator_stats(&observation->allocator_after_release);
        printf(",\"allocator_run_delta\":");
        print_allocator_delta(&observation->allocator_before,
                              &observation->allocator_after_release);
        putchar('}');
    }
    putchar(']');

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
    free(observations);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseAllocator(session_allocator);
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
