#define _POSIX_C_SOURCE 200809L

/*
 * E7 graph를 입력당 한 번만 순회하면서 모든 fused Quant-QConv 후보를
 * 같은 중간 Tensor로 측정한다. Operator별 process/prelude 반복을 피한다.
 */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backends/cpu_aarch64/aarch64_kernels.h"
#include "backends/cpu_reference/reference_kernel_utils.h"
#include "campp_profill/operator_profiler.h"
#include "campp_profill/runtime_fixture.h"
#include "campp_runtime/operator_descriptor.h"
#include "campp_runtime/status_code.h"
#include "campp_runtime/tensor_descriptor.h"
#include "execution/graph_executor.h"
#include "fused_quant_qconv_candidate.h"
#include "internal/kernel_registry.h"
#include "internal/runtime_context.h"
#include "internal/runtime_model.h"
#include "memory_management/memory_bounds_checker.h"

#define CAMPP_FAMILY_MODE_CAPACITY 8u

typedef struct FamilyOptions {
    const char *plan_path;
    const char *weights_path;
    const char *input_path;
    uint32_t warmup;
    uint32_t repeat;
    uint32_t requested_threads;
    CamppFusedQconvCandidateMode modes[CAMPP_FAMILY_MODE_CAPACITY];
    uint8_t mode_count;
} FamilyOptions;

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

typedef struct ModeResult {
    CamppFusedQconvCandidateMode mode;
    uint64_t *samples_ns;
    uint64_t output_hash;
    int matches_baseline;
} ModeResult;

typedef struct OperatorResult {
    uint32_t operator_id;
    uint16_t kernel_id;
    const char *kernel_name;
    ModeResult modes[CAMPP_FAMILY_MODE_CAPACITY];
    uint8_t mode_count;
} OperatorResult;

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

static int add_mode(
    FamilyOptions *options, CamppFusedQconvCandidateMode mode)
{
    uint8_t index;

    for (index = 0u; index < options->mode_count; ++index) {
        if (options->modes[index] == mode) return 0;
    }
    if (options->mode_count >= CAMPP_FAMILY_MODE_CAPACITY) return 1;
    options->modes[options->mode_count++] = mode;
    return 0;
}

static void usage(const char *program)
{
    fprintf(
        stderr,
        "usage: %s --plan plan.bin --weights weights.bin --input feature.f32 "
        "[--warmup 5] [--repeat 20] [--threads 1] "
        "[--mode baseline] [--mode mac_fixed] [--mode quant_neon] "
        "[--mode combined_fixed] [--mode combined_v4] "
        "[--mode combined_hybrid] [--mode combined_v5]\n",
        program);
}

static int parse_options(int argc, char **argv, FamilyOptions *options)
{
    int index;
    int has_baseline = 0;

    if (options == NULL) return 1;
    memset(options, 0, sizeof(*options));
    options->warmup = 5u;
    options->repeat = 20u;
    options->requested_threads = 1u;

    for (index = 1; index < argc; ++index) {
        const char *name = argv[index];
        const char *value;
        CamppFusedQconvCandidateMode mode;

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
            if (parse_u32(value, &options->warmup) != 0) return 1;
        } else if (strcmp(name, "--repeat") == 0) {
            if (parse_u32(value, &options->repeat) != 0 ||
                options->repeat == 0u) return 1;
        } else if (strcmp(name, "--threads") == 0) {
            if (parse_u32(value, &options->requested_threads) != 0 ||
                options->requested_threads != 1u) return 1;
        } else if (strcmp(name, "--mode") == 0) {
            if (campp_fused_qconv_candidate_mode_parse(value, &mode) != 0 ||
                add_mode(options, mode) != 0) {
                fprintf(stderr, "invalid or excessive mode: %s\n", value);
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
    if (options->mode_count == 0u) {
        if (add_mode(options, CAMPP_FUSED_QCONV_CANDIDATE_BASELINE) != 0 ||
            add_mode(options, CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED) !=
                0) return 1;
    }
    for (index = 0; index < options->mode_count; ++index) {
        if (options->modes[index] ==
            CAMPP_FUSED_QCONV_CANDIDATE_BASELINE) has_baseline = 1;
    }
    if (!has_baseline &&
        add_mode(options, CAMPP_FUSED_QCONV_CANDIDATE_BASELINE) != 0) {
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
        if (view->data == NULL ||
            view->storage_span_bytes != snapshots[index].byte_size) return 1;
        memcpy(view->data, snapshots[index].data, snapshots[index].byte_size);
    }
    return 0;
}

static int prepare_invocation(
    CamppRuntimeContext *context, const CamppOperatorDescriptor *op,
    TargetInvocation *invocation)
{
    uint8_t slot;

    if (context == NULL || op == NULL || invocation == NULL ||
        op->operator_id >= context->resolved_kernel_count) return 1;
    memset(invocation, 0, sizeof(*invocation));
    invocation->model = context->model;
    invocation->op = op;
    invocation->scratch = context->scratch;
    invocation->scratch_size = context->scratch_size;
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

static CamppStatus run_invocation(TargetInvocation *invocation)
{
    if (invocation == NULL || invocation->kernel == NULL ||
        invocation->kernel->run == NULL) {
        return CAMPP_STATUS_INVALID_ARGUMENT;
    }
    return invocation->kernel->run(
        invocation->model, invocation->op, invocation->input_views,
        invocation->op->input_count, invocation->output_views,
        invocation->op->output_count, invocation->scratch,
        invocation->scratch_size);
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

static int reset_target_buffers(
    CamppRuntimeContext *context, const CamppOperatorDescriptor *op,
    const TensorSnapshot *snapshots, uint8_t snapshot_count)
{
    /* Output/Input이 alias인 plan에서도 input 복원이 마지막이어야 한다. */
    clear_outputs(context, op);
    return restore_input_snapshots(context, snapshots, snapshot_count);
}

static int benchmark_mode(
    CamppRuntimeContext *context, TargetInvocation *invocation,
    const TensorSnapshot *snapshots, uint8_t snapshot_count,
    const CamppKernelEntry *kernel, uint32_t warmup, uint32_t repeat,
    ModeResult *result)
{
    uint32_t iteration;
    CamppStatus status;

    invocation->kernel = kernel;
    result->samples_ns = (uint64_t *)calloc(repeat, sizeof(uint64_t));
    if (result->samples_ns == NULL) return 1;

    if (reset_target_buffers(
            context, invocation->op, snapshots, snapshot_count) != 0) return 1;
    status = run_invocation(invocation);
    if (status != CAMPP_STATUS_OK ||
        output_hash(context, invocation->op, &result->output_hash) != 0) {
        return 1;
    }
    for (iteration = 0u; iteration < warmup; ++iteration) {
        if (reset_target_buffers(
                context, invocation->op, snapshots, snapshot_count) != 0) {
            return 1;
        }
        status = run_invocation(invocation);
        if (status != CAMPP_STATUS_OK) return 1;
    }
    for (iteration = 0u; iteration < repeat; ++iteration) {
        uint64_t started_ns;
        uint64_t finished_ns;
        uint64_t hash;

        if (reset_target_buffers(
                context, invocation->op, snapshots, snapshot_count) != 0 ||
            campp_operator_profiler_clock_now_ns(&started_ns) !=
                CAMPP_STATUS_OK) return 1;
        status = run_invocation(invocation);
        if (campp_operator_profiler_clock_now_ns(&finished_ns) !=
                CAMPP_STATUS_OK ||
            status != CAMPP_STATUS_OK || finished_ns <= started_ns ||
            output_hash(context, invocation->op, &hash) != 0 ||
            hash != result->output_hash) return 1;
        result->samples_ns[iteration] = finished_ns - started_ns;
    }
    return campp_memory_bounds_check_operator(context, invocation->op) ==
        CAMPP_STATUS_OK ? 0 : 1;
}

static int is_fused_qconv_target(
    const CamppOperatorDescriptor *op, const CamppKernelEntry *kernel)
{
    return op != NULL && kernel != NULL && kernel->name != NULL &&
        op->opcode == CAMPP_OP_QLINEAR_CONV &&
        op->kernel_id == CAMPP_FUSION_QUANT_QCONV_KERNEL_ID &&
        strcmp(kernel->name, "fused_quant_qlinear_conv_o4i4") == 0;
}

static const CamppKernelEntry *kernel_for_mode(
    const CamppKernelEntry *baseline, CamppFusedQconvCandidateMode mode)
{
    return mode == CAMPP_FUSED_QCONV_CANDIDATE_BASELINE
        ? baseline : campp_fused_qconv_candidate_entry(mode);
}

static int benchmark_operator(
    const FamilyOptions *options, CamppRuntimeContext *context,
    uint32_t operator_id, OperatorResult *result)
{
    const CamppOperatorDescriptor *op = &context->model->operators[operator_id];
    const CamppKernelEntry *baseline = context->resolved_kernels[operator_id];
    TensorSnapshot *snapshots = NULL;
    uint8_t snapshot_count = 0u;
    TargetInvocation invocation;
    uint64_t baseline_hash = 0u;
    uint8_t result_index;
    int pass;
    int baseline_measured = 0;
    int failed = 1;

    memset(result, 0, sizeof(*result));
    result->operator_id = operator_id;
    result->kernel_id = op->kernel_id;
    result->kernel_name = baseline->name;
    result->mode_count = options->mode_count;
    for (result_index = 0u; result_index < options->mode_count; ++result_index) {
        result->modes[result_index].mode = options->modes[result_index];
    }
    if (create_input_snapshots(
            context->model, context, op, &snapshots, &snapshot_count) != 0 ||
        prepare_invocation(context, op, &invocation) != 0) goto cleanup;

    /* Candidate를 먼저 측정하고 baseline을 마지막에 남겨 graph를 계속한다. */
    for (pass = 0; pass < 2; ++pass) {
        for (result_index = 0u;
             result_index < options->mode_count; ++result_index) {
            ModeResult *mode_result = &result->modes[result_index];
            const int is_baseline = mode_result->mode ==
                CAMPP_FUSED_QCONV_CANDIDATE_BASELINE;
            const CamppKernelEntry *kernel;

            if ((pass == 0 && is_baseline) || (pass == 1 && !is_baseline)) {
                continue;
            }
            kernel = kernel_for_mode(baseline, mode_result->mode);
            if (kernel == NULL || kernel->opcode != op->opcode ||
                kernel->kernel_id != op->kernel_id ||
                benchmark_mode(
                    context, &invocation, snapshots, snapshot_count, kernel,
                    options->warmup, options->repeat, mode_result) != 0) {
                goto cleanup;
            }
            if (is_baseline) {
                baseline_hash = mode_result->output_hash;
                baseline_measured = 1;
            }
        }
    }
    if (!baseline_measured) goto cleanup;
    for (result_index = 0u; result_index < options->mode_count; ++result_index) {
        result->modes[result_index].matches_baseline =
            result->modes[result_index].output_hash == baseline_hash;
    }
    /* 마지막 baseline repeat의 출력이 실제 graph의 다음 입력이다. */
    context->diagnostics.current_operator_id = operator_id;
    context->diagnostics.executed_operator_count += 1u;
    context->last_status = CAMPP_STATUS_OK;
    failed = 0;

cleanup:
    release_snapshots(snapshots, snapshot_count);
    return failed;
}

static void release_results(OperatorResult *results, uint32_t count)
{
    uint32_t case_index;

    if (results == NULL) return;
    for (case_index = 0u; case_index < count; ++case_index) {
        uint8_t mode_index;
        for (mode_index = 0u;
             mode_index < results[case_index].mode_count; ++mode_index) {
            free(results[case_index].modes[mode_index].samples_ns);
        }
    }
    free(results);
}

static void print_u64_samples(const uint64_t *samples, uint32_t count)
{
    uint32_t index;

    putchar('[');
    for (index = 0u; index < count; ++index) {
        printf("%s%" PRIu64, index == 0u ? "" : ",", samples[index]);
    }
    putchar(']');
}

static void print_result(
    const FamilyOptions *options, const CamppRuntimeModel *model,
    const OperatorResult *results, uint32_t result_count,
    int all_bitwise)
{
    uint32_t case_index;

    fputs(
        "{\"schema_version\":1,\"mode\":\"fused_qconv_family_batch\"," 
        "\"clock\":", stdout);
    print_json_string(campp_operator_profiler_clock_name());
    printf(
        ",\"configuration\":{\"requested_threads\":%" PRIu32
        ",\"effective_threads\":1,\"warmup\":%" PRIu32
        ",\"repeat\":%" PRIu32 ",\"graph_traversals\":1},"
        "\"model\":{\"bucket_frames\":%" PRIu32
        ",\"operator_count\":%" PRIu32 "},\"measurement_scope\":"
        "\"single_kernel_run_batch\",\"cases\":[",
        options->requested_threads, options->warmup, options->repeat,
        model->bucket_frames, model->operator_count);
    for (case_index = 0u; case_index < result_count; ++case_index) {
        const OperatorResult *result = &results[case_index];
        uint8_t mode_index;

        printf(
            "%s{\"operator_id\":%" PRIu32
            ",\"kernel_id\":%u,\"kernel_name\":",
            case_index == 0u ? "" : ",", result->operator_id,
            (unsigned int)result->kernel_id);
        print_json_string(result->kernel_name);
        fputs(",\"modes\":[", stdout);
        for (mode_index = 0u; mode_index < result->mode_count; ++mode_index) {
            const ModeResult *mode = &result->modes[mode_index];
            printf("%s{\"name\":", mode_index == 0u ? "" : ",");
            print_json_string(campp_fused_qconv_candidate_mode_name(mode->mode));
            fputs(",\"samples_ns\":", stdout);
            print_u64_samples(mode->samples_ns, options->repeat);
            printf(
                ",\"output_hash\":\"%016" PRIx64
                "\",\"matches_baseline\":%s}",
                mode->output_hash,
                mode->matches_baseline ? "true" : "false");
        }
        fputs("]}", stdout);
    }
    printf(
        "],\"all_output_hashes_bitwise_identical\":%s}\n",
        all_bitwise ? "true" : "false");
}

int main(int argc, char **argv)
{
    FamilyOptions options;
    CamppRuntimeModel model;
    CamppRuntimeContext context;
    const CamppKernelRegistry *registry;
    const CamppTensorDescriptor *input_descriptor;
    OperatorResult *results = NULL;
    uint32_t result_capacity = 0u;
    uint32_t result_count = 0u;
    uint32_t input_dimensions[CAMPP_TENSOR_MAX_RANK];
    uint8_t *input_data = NULL;
    size_t input_size = 0u;
    uint32_t input_tensor_id;
    uint32_t operator_id;
    CamppStatus status;
    int parse_result;
    int all_bitwise = 1;
    int exit_code = 1;

    if (argc == 2 && strcmp(argv[1], "--capabilities") == 0) {
        fputs(
            "{\"runtime\":\"campp-fused-qconv-family-bench\"," 
            "\"effective_threads\":1,\"bucket\":98,"
            "\"measurement_scope\":\"single_kernel_run_batch\"," 
            "\"batch_graph_traversal\":true,"
            "\"fused_qconv_candidates\":[\"baseline\",\"mac\"," 
            "\"combined\",\"mac_fixed\",\"quant_neon\"," 
            "\"combined_fixed\",\"combined_v4\","
            "\"combined_hybrid\",\"combined_v5\"]}\n",
            stdout);
        return 0;
    }
    parse_result = parse_options(argc, argv, &options);
    if (parse_result != 0) return parse_result == 2 ? 0 : 2;

    memset(&model, 0, sizeof(model));
    memset(&context, 0, sizeof(context));
    status = campp_runtime_model_load(
        options.plan_path, options.weights_path, &model);
    if (status != CAMPP_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    if (model.bucket_frames != 98u || model.input_count != 1u) {
        fprintf(stderr, "batch benchmark requires E7 bucket 98\n");
        goto cleanup;
    }
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
    if (status != CAMPP_STATUS_OK ||
        campp_runtime_context_reset(&context) != CAMPP_STATUS_OK) {
        fprintf(stderr, "input bind/reset failed: %s\n", campp_status_name(status));
        goto cleanup;
    }
    for (operator_id = 0u; operator_id < model.operator_count; ++operator_id) {
        if (is_fused_qconv_target(
                &model.operators[operator_id],
                context.resolved_kernels[operator_id])) {
            result_capacity += 1u;
        }
    }
    if (result_capacity == 0u) {
        fprintf(stderr, "model has no fused Quant-QConv operators\n");
        goto cleanup;
    }
    results = (OperatorResult *)calloc(result_capacity, sizeof(*results));
    if (results == NULL) goto cleanup;

    for (operator_id = 0u; operator_id < model.operator_count; ++operator_id) {
        const CamppOperatorDescriptor *op = &model.operators[operator_id];
        const CamppKernelEntry *kernel = context.resolved_kernels[operator_id];

        if (is_fused_qconv_target(op, kernel)) {
            uint8_t mode_index;
            if (benchmark_operator(
                    &options, &context, operator_id,
                    &results[result_count]) != 0) {
                fprintf(
                    stderr, "family benchmark failed at operator %" PRIu32
                    "\n", operator_id);
                goto cleanup;
            }
            for (mode_index = 0u;
                 mode_index < results[result_count].mode_count; ++mode_index) {
                if (!results[result_count].modes[mode_index].matches_baseline) {
                    all_bitwise = 0;
                }
            }
            result_count += 1u;
        } else {
            status = campp_graph_execute_operator(&context, operator_id);
            if (status != CAMPP_STATUS_OK) {
                fprintf(
                    stderr, "graph traversal failed at operator %" PRIu32
                    ": %s\n", operator_id, campp_status_name(status));
                goto cleanup;
            }
        }
    }
    if (result_count != result_capacity) goto cleanup;
    print_result(&options, &model, results, result_count, all_bitwise);
    if (ferror(stdout)) goto cleanup;
    exit_code = all_bitwise ? 0 : 3;

cleanup:
    release_results(results, result_capacity);
    free(input_data);
    campp_runtime_context_release(&context);
    campp_runtime_model_release(&model);
    return exit_code;
}
