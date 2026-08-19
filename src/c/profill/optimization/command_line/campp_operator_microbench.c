#define _POSIX_C_SOURCE 200809L

/* E7 graph의 실제 중간 Tensor를 만든 뒤 target Operator만 독립 반복한다. */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backends/cpu_reference/reference_kernel_utils.h"
#include "campp_profill/operator_profiler.h"
#include "campp_profill/optimization/linux_pmu.h"
#include "campp_profill/optimization/perf_sample_window.h"
#include "campp_profill/optimization/stage_probe.h"
#include "campp_profill/runtime_fixture.h"
#include "campp_runtime/operator_descriptor.h"
#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "execution/graph_executor.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"
#include "memory_management/memory_bounds_checker.h"
#include "bn_candidate.h"
#include "dequant_candidate.h"
#include "fused_quant_qconv_candidate.h"
#include "qconv_candidate.h"
#include "remaining_candidate.h"

typedef struct MicrobenchOptions {
    const char *plan_path;
    const char *weights_path;
    const char *input_path;
    uint32_t operator_id;
    uint32_t warmup;
    uint32_t repeat;
    uint32_t requested_threads;
    uint64_t expected_output_hash;
    CamppQconvCandidateMode qconv_candidate;
    CamppFusedQconvCandidateMode fused_qconv_candidate;
    CamppBnCandidateMode bn_candidate;
    CamppDequantCandidateMode dequant_candidate;
    CamppRemainingCandidateMode remaining_candidate;
    int has_expected_output_hash;
    int perf_window;
    const char *perf_control_path;
    const char *perf_ack_path;
} MicrobenchOptions;

typedef struct TensorSnapshot {
    uint32_t tensor_id;
    uint8_t *data;
    size_t byte_size;
} TensorSnapshot;

typedef struct TargetInvocation {
    const CamppRuntimeModel *model;
    const CamppOperatorDescriptor *op;
    const CamppKernelEntry *kernel;
    CamppTensorView input_views[CAMPP_OPERATOR_INPUT_CAPACITY];
    CamppTensorView output_views[CAMPP_OPERATOR_OUTPUT_CAPACITY];
    void *scratch;
    size_t scratch_size;
} TargetInvocation;

static int parse_u32(const char *text, uint32_t *out_value)
{
    char *end = NULL;
    unsigned long value;
    if (text == NULL || out_value == NULL || *text == '\0') return 1;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return 1;
    }
    *out_value = (uint32_t)value;
    return 0;
}

static int parse_u64_hex(const char *text, uint64_t *out_value)
{
    char *end = NULL;
    unsigned long long value;
    if (text == NULL || out_value == NULL || *text == '\0') return 1;
    errno = 0;
    value = strtoull(text, &end, 16);
    if (errno != 0 || end == text || *end != '\0') return 1;
    *out_value = (uint64_t)value;
    return 0;
}

static void usage(const char *program)
{
    fprintf(
        stderr,
        "usage: %s --plan plan.bin --weights weights.bin --input feature.f32 "
        "--operator-id N [--warmup 5] [--repeat 20] [--threads 1] "
        "[--expected-output-hash HEX] [--perf-window] "
        "[--perf-control CTL_FIFO --perf-ack ACK_FIFO] "
        "[--qconv-candidate baseline|address|mac|combined|mac_fixed|mac_asm] "
        "[--fused-qconv-candidate baseline|mac|combined|mac_fixed|"
        "quant_neon|combined_fixed] "
        "[--bn-candidate baseline|address|affine|quant|combined] "
        "[--dequant-candidate baseline|address|parameter|scalar_combined|"
        "neon_combined] [--remaining-candidate baseline|optimized]\n",
        program);
}

static int parse_options(
    int argc, char **argv, MicrobenchOptions *options)
{
    int index;
    int has_operator_id = 0;

    if (options == NULL) return 1;
    memset(options, 0, sizeof(*options));
    options->warmup = 5u;
    options->repeat = 20u;
    options->requested_threads = 1u;
    options->qconv_candidate = CAMPP_QCONV_CANDIDATE_BASELINE;
    options->fused_qconv_candidate =
        CAMPP_FUSED_QCONV_CANDIDATE_BASELINE;
    options->bn_candidate = CAMPP_BN_CANDIDATE_BASELINE;
    options->dequant_candidate = CAMPP_DEQUANT_CANDIDATE_BASELINE;
    options->remaining_candidate = CAMPP_REMAINING_CANDIDATE_BASELINE;

    for (index = 1; index < argc; ++index) {
        const char *name = argv[index];
        const char *value;
        if (strcmp(name, "--help") == 0 || strcmp(name, "-h") == 0) {
            usage(argv[0]);
            return 2;
        }
        if (strcmp(name, "--perf-window") == 0) {
            options->perf_window = 1;
            continue;
        }
        if (index + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", name);
            return 1;
        }
        value = argv[++index];
        if (strcmp(name, "--plan") == 0) {
            options->plan_path = value;
        } else if (strcmp(name, "--perf-control") == 0) {
            options->perf_control_path = value;
        } else if (strcmp(name, "--perf-ack") == 0) {
            options->perf_ack_path = value;
        } else if (strcmp(name, "--weights") == 0) {
            options->weights_path = value;
        } else if (strcmp(name, "--input") == 0) {
            options->input_path = value;
        } else if (strcmp(name, "--operator-id") == 0) {
            if (parse_u32(value, &options->operator_id) != 0) return 1;
            has_operator_id = 1;
        } else if (strcmp(name, "--warmup") == 0) {
            if (parse_u32(value, &options->warmup) != 0) return 1;
        } else if (strcmp(name, "--repeat") == 0) {
            if (parse_u32(value, &options->repeat) != 0 ||
                options->repeat == 0u) return 1;
        } else if (strcmp(name, "--threads") == 0) {
            if (parse_u32(value, &options->requested_threads) != 0 ||
                options->requested_threads != 1u) return 1;
        } else if (strcmp(name, "--expected-output-hash") == 0) {
            if (parse_u64_hex(value, &options->expected_output_hash) != 0) {
                return 1;
            }
            options->has_expected_output_hash = 1;
        } else if (strcmp(name, "--qconv-candidate") == 0) {
            if (campp_qconv_candidate_mode_parse(
                    value, &options->qconv_candidate) != 0) {
                fprintf(stderr, "invalid QConv candidate: %s\n", value);
                return 1;
            }
        } else if (strcmp(name, "--fused-qconv-candidate") == 0) {
            if (campp_fused_qconv_candidate_mode_parse(
                    value, &options->fused_qconv_candidate) != 0) {
                fprintf(
                    stderr, "invalid fused QConv candidate: %s\n", value);
                return 1;
            }
        } else if (strcmp(name, "--bn-candidate") == 0) {
            if (campp_bn_candidate_mode_parse(
                    value, &options->bn_candidate) != 0) {
                fprintf(stderr, "invalid BN candidate: %s\n", value);
                return 1;
            }
        } else if (strcmp(name, "--dequant-candidate") == 0) {
            if (campp_dequant_candidate_mode_parse(
                    value, &options->dequant_candidate) != 0) {
                fprintf(stderr, "invalid Dequant candidate: %s\n", value);
                return 1;
            }
        } else if (strcmp(name, "--remaining-candidate") == 0) {
            if (campp_remaining_candidate_mode_parse(
                    value, &options->remaining_candidate) != 0) {
                fprintf(stderr, "invalid remaining candidate: %s\n", value);
                return 1;
            }
        } else {
            fprintf(stderr, "unknown option: %s\n", name);
            return 1;
        }
    }
    if (options->plan_path == NULL || options->weights_path == NULL ||
        options->input_path == NULL || !has_operator_id) {
        usage(argv[0]);
        return 1;
    }
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

static void release_snapshots(TensorSnapshot *snapshots, uint8_t count)
{
    uint8_t index;
    if (snapshots == NULL) return;
    for (index = 0u; index < count; ++index) free(snapshots[index].data);
    free(snapshots);
}

static int create_input_snapshots(
    const CamppRuntimeModel *model, const CamppRuntimeContext *context,
    const CamppOperatorDescriptor *op, TensorSnapshot **out_snapshots,
    uint8_t *out_count)
{
    TensorSnapshot *snapshots;
    uint8_t count = 0u;
    uint8_t slot;

    if (model == NULL || context == NULL || op == NULL ||
        out_snapshots == NULL || out_count == NULL) return 1;
    snapshots = (TensorSnapshot *)calloc(
        op->input_count == 0u ? 1u : op->input_count, sizeof(*snapshots));
    if (snapshots == NULL) return 1;
    for (slot = 0u; slot < op->input_count; ++slot) {
        const uint32_t tensor_id = op->input_tensor_ids[slot];
        const CamppTensorDescriptor *descriptor = &model->tensors[tensor_id];
        const CamppTensorView *view = &context->tensors[tensor_id];
        TensorSnapshot *snapshot;
        if (descriptor->storage_type == CAMPP_TENSOR_STORAGE_CONSTANT) continue;
        if (view->data == NULL || view->storage_span_bytes > SIZE_MAX) {
            release_snapshots(snapshots, count);
            return 1;
        }
        snapshot = &snapshots[count];
        snapshot->tensor_id = tensor_id;
        snapshot->byte_size = (size_t)view->storage_span_bytes;
        snapshot->data = (uint8_t *)malloc(
            snapshot->byte_size == 0u ? 1u : snapshot->byte_size);
        if (snapshot->data == NULL) {
            release_snapshots(snapshots, count);
            return 1;
        }
        memcpy(snapshot->data, view->data, snapshot->byte_size);
        count += 1u;
    }
    *out_snapshots = snapshots;
    *out_count = count;
    return 0;
}

static int restore_input_snapshots(
    CamppRuntimeContext *context, const TensorSnapshot *snapshots,
    uint8_t count)
{
    uint8_t index;
    for (index = 0u; index < count; ++index) {
        CamppTensorView *view = &context->tensors[snapshots[index].tensor_id];
        if (view->data == NULL || view->storage_span_bytes !=
            snapshots[index].byte_size) return 1;
        memcpy(view->data, snapshots[index].data, snapshots[index].byte_size);
    }
    return 0;
}

static int prepare_target_invocation(
    CamppRuntimeContext *context, const CamppOperatorDescriptor *op,
    TargetInvocation *invocation)
{
    uint8_t slot;

    if (context == NULL || op == NULL || invocation == NULL ||
        op->operator_id >= context->resolved_kernel_count) return 1;
    memset(invocation, 0, sizeof(*invocation));
    invocation->model = context->model;
    invocation->op = op;
    invocation->kernel = context->resolved_kernels[op->operator_id];
    invocation->scratch = context->scratch;
    invocation->scratch_size = context->scratch_size;
    if (invocation->kernel == NULL || invocation->kernel->run == NULL ||
        invocation->kernel->opcode != op->opcode ||
        invocation->kernel->kernel_id != op->kernel_id) return 1;
    for (slot = 0u; slot < op->input_count; ++slot) {
        invocation->input_views[slot] =
            context->tensors[op->input_tensor_ids[slot]];
    }
    for (slot = 0u; slot < op->output_count; ++slot) {
        invocation->output_views[slot] =
            context->tensors[op->output_tensor_ids[slot]];
    }
    return campp_memory_bounds_check_operator(context, op) == CAMPP_STATUS_OK
        ? 0 : 1;
}

static CamppStatus run_target_kernel(TargetInvocation *invocation)
{
    if (invocation == NULL || invocation->kernel == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    return invocation->kernel->run(
        invocation->model, invocation->op, invocation->input_views,
        invocation->op->input_count, invocation->output_views,
        invocation->op->output_count, invocation->scratch,
        invocation->scratch_size);
}

static void initialize_empty_pmu(CamppOptimizationPmu *pmu)
{
    uint32_t event;

    memset(pmu, 0, sizeof(*pmu));
    pmu->leader_fd = -1;
    for (event = 0u; event < CAMPP_OPT_PMU_EVENT_COUNT; ++event) {
        pmu->file_descriptors[event] = -1;
    }
}

static void clear_outputs(
    CamppRuntimeContext *context, const CamppOperatorDescriptor *op)
{
    uint8_t slot;
    for (slot = 0u; slot < op->output_count; ++slot) {
        CamppTensorView *view = &context->tensors[op->output_tensor_ids[slot]];
        if (view->data != NULL && view->storage_span_bytes <= SIZE_MAX) {
            memset(view->data, 0xA5, (size_t)view->storage_span_bytes);
        }
    }
}

static uint64_t fnv1a_update(
    uint64_t hash, const uint8_t *data, size_t byte_size)
{
    size_t index;
    for (index = 0u; index < byte_size; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int output_hash(
    const CamppRuntimeContext *context, const CamppOperatorDescriptor *op,
    uint64_t *out_hash)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    uint8_t slot;

    if (context == NULL || op == NULL || out_hash == NULL) return 1;
    for (slot = 0u; slot < op->output_count; ++slot) {
        const uint32_t tensor_id = op->output_tensor_ids[slot];
        const CamppTensorView *view = &context->tensors[tensor_id];
        const uint32_t element_size = campp_dtype_byte_size(view->dtype);
        const uint64_t element_count = campp_tensor_view_element_count(view);
        uint64_t index;
        if (view->data == NULL || element_size == 0u) return 1;
        hash = fnv1a_update(
            hash, (const uint8_t *)&tensor_id, sizeof(tensor_id));
        if (campp_tensor_view_is_contiguous(view)) {
            if (element_count > SIZE_MAX / element_size) return 1;
            hash = fnv1a_update(
                hash, (const uint8_t *)view->data,
                (size_t)element_count * element_size);
            continue;
        }
        for (index = 0u; index < element_count; ++index) {
            const uint64_t offset =
                campp_reference_offset_for_linear(view, index);
            if (offset > view->storage_span_bytes ||
                element_size > view->storage_span_bytes - offset) return 1;
            hash = fnv1a_update(
                hash, (const uint8_t *)view->data + offset, element_size);
        }
    }
    *out_hash = hash;
    return 0;
}

static void print_u64_samples(const uint64_t *samples, uint32_t count)
{
    uint32_t index;
    putchar('[');
    for (index = 0u; index < count; ++index) {
        printf(
            "%s%" PRIu64, index == 0u ? "" : ",",
            samples == NULL ? 0u : samples[index]);
    }
    putchar(']');
}

static void print_result(
    const MicrobenchOptions *options, const CamppRuntimeModel *model,
    const CamppRuntimeContext *context, const CamppOperatorDescriptor *op,
    const uint64_t *samples_ns, const CamppOptimizationProbe *probe,
    const CamppOptimizationPmu *pmu,
    const CamppPerfSampleWindow *perf_window, uint64_t hash,
    int hash_matches)
{
    uint32_t index;

    fputs("{\"schema_version\":1,\"mode\":\"operator_microbench\",", stdout);
    fputs("\"clock\":", stdout);
    print_json_string(campp_operator_profiler_clock_name());
    printf(
        ",\"configuration\":{\"requested_threads\":%" PRIu32
        ",\"effective_threads\":1,\"warmup\":%" PRIu32
        ",\"repeat\":%" PRIu32 "},",
        options->requested_threads, options->warmup, options->repeat);
    printf(
        "\"model\":{\"bucket_frames\":%" PRIu32
        ",\"operator_count\":%" PRIu32 "},",
        model->bucket_frames, model->operator_count);
    fputs("\"qconv_candidate\":", stdout);
    print_json_string(campp_qconv_candidate_mode_name(
        options->qconv_candidate));
    putchar(',');
    fputs("\"fused_qconv_candidate\":", stdout);
    print_json_string(campp_fused_qconv_candidate_mode_name(
        options->fused_qconv_candidate));
    putchar(',');
    fputs("\"bn_candidate\":", stdout);
    print_json_string(campp_bn_candidate_mode_name(options->bn_candidate));
    putchar(',');
    fputs("\"dequant_candidate\":", stdout);
    print_json_string(campp_dequant_candidate_mode_name(
        options->dequant_candidate));
    putchar(',');
    fputs("\"remaining_candidate\":", stdout);
    print_json_string(campp_remaining_candidate_mode_name(
        options->remaining_candidate));
    putchar(',');
    printf(
        "\"operator\":{\"operator_id\":%" PRIu32
        ",\"opcode\":%u,\"kernel_id\":%u,\"kernel_name\":",
        op->operator_id, (unsigned int)op->opcode,
        (unsigned int)op->kernel_id);
    print_json_string(context->resolved_kernels[op->operator_id]->name);
    fputs("},\"measurement_scope\":\"single_kernel_run\",", stdout);
    fputs("\"samples_ns\":", stdout);
    print_u64_samples(samples_ns, options->repeat);
    printf(
        ",\"output_hash\":\"%016" PRIx64
        "\",\"expected_output_hash_provided\":%s,"
        "\"output_hash_matches\":%s,",
        hash, options->has_expected_output_hash ? "true" : "false",
        hash_matches ? "true" : "false");
    fputs("\"stages\":[", stdout);
    for (index = 0u; index < CAMPP_OPT_STAGE_COUNT; ++index) {
        const uint64_t *stage_samples = campp_optimization_stage_samples(
            probe, (CamppOptimizationStage)index);
        printf("%s{\"name\":", index == 0u ? "" : ",");
        print_json_string(campp_optimization_stage_name(
            (CamppOptimizationStage)index));
        fputs(",\"samples_ns\":", stdout);
        print_u64_samples(stage_samples, options->repeat);
        putchar('}');
    }
    printf(
        "],\"pmu\":{\"available\":%s,\"unavailable_errno\":%d,"
        "\"events\":[",
        pmu->available ? "true" : "false", pmu->unavailable_errno);
    for (index = 0u; index < CAMPP_OPT_PMU_EVENT_COUNT; ++index) {
        const uint64_t *event_samples = campp_optimization_pmu_samples(
            pmu, (CamppOptimizationPmuEvent)index);
        printf("%s{\"name\":", index == 0u ? "" : ",");
        print_json_string(campp_optimization_pmu_event_name(
            (CamppOptimizationPmuEvent)index));
        fputs(",\"samples\":", stdout);
        print_u64_samples(event_samples, options->repeat);
        putchar('}');
    }
    printf(
        "]},\"perf_window\":{\"requested\":%s,\"supported\":%s,"
        "\"enabled_at_exit\":%s,\"error_number\":%d}}\n",
        perf_window->requested ? "true" : "false",
        perf_window->supported ? "true" : "false",
        perf_window->enabled ? "true" : "false",
        perf_window->error_number);
}

int main(int argc, char **argv)
{
    MicrobenchOptions options;
    CamppRuntimeModel model;
    CamppRuntimeContext context;
    CamppOptimizationProbe probe;
    CamppOptimizationPmu pmu;
    CamppPerfSampleWindow perf_window;
    TargetInvocation invocation;
    const CamppKernelRegistry *registry;
    const CamppTensorDescriptor *input_descriptor;
    const CamppOperatorDescriptor *op;
    TensorSnapshot *input_snapshots = NULL;
    uint8_t input_snapshot_count = 0u;
    uint32_t input_dimensions[CAMPP_TENSOR_MAX_RANK];
    uint8_t *input_data = NULL;
    size_t input_size = 0u;
    uint64_t *samples_ns = NULL;
    uint64_t hash = 0u;
    uint32_t input_tensor_id;
    uint32_t iteration;
    CamppStatus status;
    int parse_result;
    int hash_matches = 1;
    int exit_code = 1;

    if (argc == 2 && strcmp(argv[1], "--capabilities") == 0) {
        fputs(
            "{\"runtime\":\"campp-operator-microbench\","
            "\"effective_threads\":1,\"bucket\":98,"
            "\"measurement_scope\":\"single_kernel_run\","
            "\"pmu\":true,"
            "\"qconv_candidates\":[\"baseline\",\"address\","
            "\"mac\",\"combined\",\"mac_fixed\",\"mac_asm\"],"
            "\"fused_qconv_candidates\":[\"baseline\",\"mac\","
            "\"combined\",\"mac_fixed\",\"quant_neon\","
            "\"combined_fixed\"],"
            "\"bn_candidates\":[\"baseline\",\"address\","
            "\"affine\",\"quant\",\"combined\"],"
            "\"dequant_candidates\":[\"baseline\",\"address\","
            "\"parameter\",\"scalar_combined\",\"neon_combined\"],"
            "\"remaining_candidates\":[\"baseline\",\"optimized\"],"
#if defined(CAMPP_ENABLE_OPTIMIZATION_DIAGNOSTICS)
            "\"stage_probe\":true,"
#else
            "\"stage_probe\":false,"
#endif
#if defined(__linux__)
            "\"perf_sample_window\":true}\n",
#else
            "\"perf_sample_window\":false}\n",
#endif
            stdout);
        return 0;
    }
    parse_result = parse_options(argc, argv, &options);
    if (parse_result != 0) return parse_result == 2 ? 0 : 2;

    memset(&model, 0, sizeof(model));
    memset(&context, 0, sizeof(context));
    memset(&probe, 0, sizeof(probe));
    initialize_empty_pmu(&pmu);
    memset(&invocation, 0, sizeof(invocation));
    campp_perf_sample_window_initialize(
        &perf_window, options.perf_window != 0);
    if (options.perf_window && options.perf_control_path != NULL &&
        options.perf_ack_path != NULL) {
        /*
         * perf was started with -D -1, so its events are already disabled and
         * the prelude cannot leak into the profile. Attaching here only wires
         * up the control FIFOs; sampling stays off until the measurement loop.
         */
        if (campp_perf_sample_window_attach_control(
                &perf_window, options.perf_control_path,
                options.perf_ack_path) != 0) {
            fprintf(stderr, "cannot attach perf control FIFOs\n");
            goto cleanup;
        }
    } else if (campp_perf_sample_window_disable(&perf_window) != 0) {
        fprintf(stderr, "cannot disable perf events during prelude\n");
        goto cleanup;
    }
    status = campp_runtime_model_load(
        options.plan_path, options.weights_path, &model);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    if (model.bucket_frames != 98u || options.operator_id >= model.operator_count ||
        model.input_count != 1u) {
        fprintf(stderr, "microbench requires E7 bucket 98 and a valid operator ID\n");
        goto cleanup;
    }
    op = &model.operators[options.operator_id];
    registry = campp_profill_registry_for_model(&model);
    status = campp_runtime_context_create(&model, registry, &context);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "context create failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    input_tensor_id = model.input_tensor_ids[0];
    input_descriptor = &model.tensors[input_tensor_id];
    if (campp_profill_read_entire_file(
            options.input_path, &input_data, &input_size) != 0 ||
        campp_profill_infer_input_dimensions(
            &model, input_descriptor, input_size, input_dimensions) != 0) {
        fprintf(stderr, "cannot load input feature\n");
        goto cleanup;
    }
    status = campp_runtime_context_bind_input(
        &context, input_tensor_id, input_data, input_size,
        input_descriptor->dtype, input_descriptor->rank, input_dimensions);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "input bind failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    status = campp_runtime_context_reset(&context);
    if (status != CAMPP_STATUS_OK) goto cleanup;
    for (iteration = 0u; iteration < options.operator_id; ++iteration) {
        status = campp_graph_execute_operator(&context, iteration);
        if (status != CAMPP_STATUS_OK) {
            fprintf(
                stderr, "prelude failed at operator %" PRIu32 ": %s\n",
                iteration, campp_status_name(status));
            goto cleanup;
        }
    }
    if (create_input_snapshots(
            &model, &context, op, &input_snapshots,
            &input_snapshot_count) != 0) {
        fprintf(stderr, "cannot snapshot target inputs\n");
        goto cleanup;
    }
    if (prepare_target_invocation(&context, op, &invocation) != 0) {
        fprintf(stderr, "cannot prepare target kernel invocation\n");
        goto cleanup;
    }
    if (options.qconv_candidate != CAMPP_QCONV_CANDIDATE_BASELINE) {
        const CamppKernelEntry *candidate =
            campp_qconv_candidate_entry(options.qconv_candidate);
        if (options.fused_qconv_candidate !=
                CAMPP_FUSED_QCONV_CANDIDATE_BASELINE ||
            options.dequant_candidate != CAMPP_DEQUANT_CANDIDATE_BASELINE ||
            options.remaining_candidate !=
                CAMPP_REMAINING_CANDIDATE_BASELINE ||
            candidate == NULL || op->opcode != CAMPP_OP_QLINEAR_CONV ||
            op->kernel_id != candidate->kernel_id) {
            fprintf(stderr, "QConv candidate requires packed QLinearConv\n");
            goto cleanup;
        }
        invocation.kernel = candidate;
    }
    if (options.fused_qconv_candidate !=
        CAMPP_FUSED_QCONV_CANDIDATE_BASELINE) {
        const CamppKernelEntry *candidate =
            campp_fused_qconv_candidate_entry(
                options.fused_qconv_candidate);
        if (options.qconv_candidate != CAMPP_QCONV_CANDIDATE_BASELINE ||
            options.bn_candidate != CAMPP_BN_CANDIDATE_BASELINE ||
            options.dequant_candidate != CAMPP_DEQUANT_CANDIDATE_BASELINE ||
            options.remaining_candidate !=
                CAMPP_REMAINING_CANDIDATE_BASELINE ||
            candidate == NULL || op->opcode != CAMPP_OP_QLINEAR_CONV ||
            op->kernel_id != candidate->kernel_id) {
            fprintf(
                stderr,
                "fused QConv candidate requires Quantize/QLinearConv\n");
            goto cleanup;
        }
        invocation.kernel = candidate;
    }
    if (options.bn_candidate != CAMPP_BN_CANDIDATE_BASELINE) {
        const CamppKernelEntry *candidate =
            campp_bn_candidate_entry(options.bn_candidate);
        if (options.qconv_candidate != CAMPP_QCONV_CANDIDATE_BASELINE ||
            options.fused_qconv_candidate !=
                CAMPP_FUSED_QCONV_CANDIDATE_BASELINE ||
            options.dequant_candidate != CAMPP_DEQUANT_CANDIDATE_BASELINE ||
            options.remaining_candidate !=
                CAMPP_REMAINING_CANDIDATE_BASELINE ||
            candidate == NULL || op->opcode != CAMPP_OP_BATCH_NORMALIZATION ||
            op->kernel_id != candidate->kernel_id) {
            fprintf(stderr, "BN candidate requires fused BN/ReLU/Quantize\n");
            goto cleanup;
        }
        invocation.kernel = candidate;
    }
    if (options.dequant_candidate != CAMPP_DEQUANT_CANDIDATE_BASELINE) {
        const CamppKernelEntry *candidate =
            campp_dequant_candidate_entry(options.dequant_candidate);
        if (options.qconv_candidate != CAMPP_QCONV_CANDIDATE_BASELINE ||
            options.fused_qconv_candidate !=
                CAMPP_FUSED_QCONV_CANDIDATE_BASELINE ||
            options.bn_candidate != CAMPP_BN_CANDIDATE_BASELINE ||
            options.remaining_candidate !=
                CAMPP_REMAINING_CANDIDATE_BASELINE ||
            candidate == NULL || op->opcode != CAMPP_OP_DEQUANTIZE_LINEAR ||
            op->kernel_id != candidate->kernel_id) {
            fprintf(
                stderr,
                "Dequant candidate requires DequantizeLinear stride kernel\n");
            goto cleanup;
        }
        invocation.kernel = candidate;
    }
    if (options.remaining_candidate != CAMPP_REMAINING_CANDIDATE_BASELINE) {
        const CamppKernelEntry *candidate = campp_remaining_candidate_entry(
            options.remaining_candidate, op->opcode, op->kernel_id);
        if (options.qconv_candidate != CAMPP_QCONV_CANDIDATE_BASELINE ||
            options.fused_qconv_candidate !=
                CAMPP_FUSED_QCONV_CANDIDATE_BASELINE ||
            options.bn_candidate != CAMPP_BN_CANDIDATE_BASELINE ||
            options.dequant_candidate != CAMPP_DEQUANT_CANDIDATE_BASELINE ||
            candidate == NULL) {
            fprintf(stderr, "remaining candidate does not support target\n");
            goto cleanup;
        }
        invocation.kernel = candidate;
    }

    if (restore_input_snapshots(
            &context, input_snapshots, input_snapshot_count) != 0) goto cleanup;
    clear_outputs(&context, op);
    status = run_target_kernel(&invocation);
    if (status != CAMPP_STATUS_OK || output_hash(&context, op, &hash) != 0) {
        fprintf(stderr, "target reference execution failed\n");
        goto cleanup;
    }
    if (options.has_expected_output_hash &&
        hash != options.expected_output_hash) {
        hash_matches = 0;
    }

    for (iteration = 0u; iteration < options.warmup; ++iteration) {
        if (restore_input_snapshots(
                &context, input_snapshots, input_snapshot_count) != 0) {
            goto cleanup;
        }
        clear_outputs(&context, op);
        status = run_target_kernel(&invocation);
        if (status != CAMPP_STATUS_OK) {
            fprintf(stderr, "warmup failed: %s\n", campp_status_name(status));
            goto cleanup;
        }
    }

    samples_ns = (uint64_t *)calloc(options.repeat, sizeof(*samples_ns));
    if (samples_ns == NULL ||
        campp_optimization_probe_create(options.repeat, &probe) !=
            CAMPP_STATUS_OK) {
        fprintf(stderr, "cannot allocate diagnostic samples\n");
        goto cleanup;
    }
    if (options.perf_window) {
        pmu.sample_capacity = options.repeat;
    } else if (campp_optimization_pmu_create(options.repeat, &pmu) != 0) {
        fprintf(stderr, "cannot allocate PMU samples\n");
        goto cleanup;
    }
    campp_optimization_probe_set_active(&probe);
    /*
     * Sampling covers the measurement loop only. Enabling per iteration would
     * put a FIFO round-trip inside the timed region, so the loop overhead
     * (snapshot restore, output hash) is excluded by symbol instead.
     */
    if (perf_window.control_active &&
        campp_perf_sample_window_enable(&perf_window) != 0) {
        fprintf(stderr, "cannot enable target perf window\n");
        goto cleanup;
    }
    for (iteration = 0u; iteration < options.repeat; ++iteration) {
        uint64_t started_ns;
        uint64_t finished_ns;
        uint64_t iteration_hash;
        if (restore_input_snapshots(
                &context, input_snapshots, input_snapshot_count) != 0) {
            goto cleanup;
        }
        clear_outputs(&context, op);
        if (campp_optimization_probe_begin_iteration(&probe, iteration) !=
                CAMPP_STATUS_OK ||
            campp_optimization_pmu_begin(&pmu) != 0 ||
            campp_operator_profiler_clock_now_ns(&started_ns) !=
                CAMPP_STATUS_OK) {
            goto cleanup;
        }
        if (!perf_window.control_active &&
            campp_perf_sample_window_enable(&perf_window) != 0) {
            fprintf(stderr, "cannot enable target perf window\n");
            goto cleanup;
        }
        status = run_target_kernel(&invocation);
        if (!perf_window.control_active &&
            campp_perf_sample_window_disable(&perf_window) != 0) {
            fprintf(stderr, "cannot disable target perf window\n");
            goto cleanup;
        }
        if (campp_operator_profiler_clock_now_ns(&finished_ns) !=
                CAMPP_STATUS_OK ||
            campp_optimization_pmu_end(&pmu, iteration) != 0 ||
            campp_optimization_probe_finish_iteration(&probe) !=
                CAMPP_STATUS_OK ||
            status != CAMPP_STATUS_OK || finished_ns < started_ns) {
            fprintf(stderr, "measurement failed\n");
            goto cleanup;
        }
        samples_ns[iteration] = finished_ns - started_ns;
        if (output_hash(&context, op, &iteration_hash) != 0 ||
            iteration_hash != hash) {
            fprintf(stderr, "non-deterministic target output\n");
            goto cleanup;
        }
    }
    if (perf_window.control_active &&
        campp_perf_sample_window_disable(&perf_window) != 0) {
        fprintf(stderr, "cannot disable target perf window\n");
        goto cleanup;
    }
    if (campp_memory_bounds_check_operator(&context, op) != CAMPP_STATUS_OK) {
        fprintf(stderr, "target kernel violated Tensor bounds\n");
        goto cleanup;
    }
    campp_optimization_probe_set_active(NULL);
    print_result(
        &options, &model, &context, op, samples_ns, &probe, &pmu,
        &perf_window, hash, hash_matches);
    if (ferror(stdout)) goto cleanup;
    exit_code = hash_matches ? 0 : 3;

cleanup:
    (void)campp_perf_sample_window_disable(&perf_window);
    campp_perf_sample_window_release(&perf_window);
    campp_optimization_probe_set_active(NULL);
    free(samples_ns);
    release_snapshots(input_snapshots, input_snapshot_count);
    free(input_data);
    campp_optimization_pmu_release(&pmu);
    campp_optimization_probe_release(&probe);
    campp_runtime_context_release(&context);
    campp_runtime_model_release(&model);
    return exit_code;
}
