/*
 * Internal Runtime header compile smoke test.
 *
 * 이 파일은 실행 테스트가 아니라 공개된 내부 타입과 함수 포인터 규격을 실제 C
 * compiler가 함께 해석할 수 있는지 확인하는 translation unit이다.
 */

#include "internal/runtime_context.h"

static CamppStatus smoke_kernel_run(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs,
    uint8_t input_count,
    CamppTensorView *outputs,
    uint8_t output_count,
    void *scratch,
    size_t scratch_size)
{
    (void)model;
    (void)op;
    (void)inputs;
    (void)input_count;
    (void)outputs;
    (void)output_count;
    (void)scratch;
    (void)scratch_size;
    return CAMPP_STATUS_OK;
}

static CamppStatus smoke_scratch_query(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    size_t *out_bytes)
{
    (void)model;
    (void)op;
    *out_bytes = 0u;
    return CAMPP_STATUS_OK;
}

int campp_internal_headers_compile_smoke(void)
{
    CamppTensorView view = {0};
    CamppKernelEntry entry = {0};
    CamppKernelRegistry registry = {0};
    CamppRuntimeModel model = {0};
    CamppRuntimeContext context = {0};
    const CamppKernelEntry *resolved[1] = {&entry};

    view.logical_byte_size = 4u;
    view.storage_span_bytes = 4u;

    entry.run = smoke_kernel_run;
    entry.scratch_bytes = smoke_scratch_query;

    registry.backend_id = CAMPP_BACKEND_CPU_REFERENCE;
    registry.entries = &entry;
    registry.entry_count = 1u;

    context.model = &model;
    context.backend_id = registry.backend_id;
    context.registry = &registry;
    context.resolved_kernels = resolved;
    context.resolved_kernel_count = 1u;

    return (context.resolved_kernels[0]->run == smoke_kernel_run &&
            campp_status_name(CAMPP_STATUS_BACKEND_MISMATCH) != NULL &&
            view.storage_span_bytes == view.logical_byte_size)
        ? 0
        : 1;
}
