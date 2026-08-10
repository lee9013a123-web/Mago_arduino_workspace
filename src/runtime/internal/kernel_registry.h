#ifndef CAMPP_RUNTIME_INTERNAL_KERNEL_REGISTRY_H
#define CAMPP_RUNTIME_INTERNAL_KERNEL_REGISTRY_H

/*
 * opcode를 C 함수에 연결하는 표다.
 *
 *   CAMPP_OP_QLINEAR_CONV -> campp_qlinear_convolution_reference()
 *   CAMPP_OP_RELU         -> campp_relu_reference()
 *
 * 정적 그래프에는 node가 1,438개 있지만 opcode는 20종뿐이다. dispatcher는
 * operator마다 이 표를 한 번 찾고 같은 함수를 반복 호출한다. node마다 함수를
 * 만들지 않는 이유가 여기 있다. QLinearConv 225개는 kernel 함수 하나를 225번
 * 부르는 것이지 225개의 코드가 아니다.
 *
 * backend는 자기 표 하나를 내놓는다. 표를 바꿔 끼우는 것이 backend 교체이며,
 * executor는 어느 backend가 붙었는지 알 필요가 없다.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "campp_runtime/operator_descriptor.h"
#include "campp_runtime/status_code.h"
#include "tensor_view.h"

/* 현재 동결된 opcode 수. CAMPP_OP_INVALID는 세지 않는다. */
#define CAMPP_KERNEL_OPCODE_COUNT 20u

struct CamppRuntimeContext;
struct CamppRuntimeModel;

/*
 * kernel 하나의 실행 함수.
 *
 * kernel은 view만 본다. Tensor ID, storage_type, weights base는 executor가
 * 이미 풀어 두었다. inputs는 descriptor의 input_count만큼 순서대로 채워지며,
 * ONNX 입력 순서를 그대로 따른다. 예를 들어 QLinearConv는 항상
 * (x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [B]) 순이다.
 *
 * scratch는 campp_kernel_scratch_bytes가 요구한 크기만큼 확보된 임시 버퍼이며,
 * 호출마다 내용이 보장되지 않는다. 필요 없으면 NULL이다.
 */
typedef CamppStatus (*CamppKernelRun)(
    struct CamppRuntimeContext *context,
    const CamppOperatorDescriptor *op,
    const CamppTensorView *inputs, uint8_t input_count,
    CamppTensorView *outputs, uint8_t output_count,
    void *scratch, size_t scratch_size);

/*
 * 이 operator를 실행하는 데 필요한 scratch byte 수를 보고한다.
 *
 * 모델이 고정이라 필요한 크기도 고정이다. context는 로드 직후 모든 operator에
 * 대해 한 번씩 물어보고 그중 최댓값만 잡아 둔다. NULL이면 scratch가 필요 없다.
 */
typedef CamppStatus (*CamppKernelScratchQuery)(
    const struct CamppRuntimeModel *model,
    const CamppOperatorDescriptor *op,
    size_t *out_bytes);

/* opcode 하나에 대응하는 구현. */
typedef struct CamppKernelEntry {
    uint16_t opcode;
    /* 같은 opcode의 변종을 고를 때 쓴다. plan이 0을 쓰면 기본 구현이다. */
    uint16_t kernel_id;
    CamppKernelRun run;
    CamppKernelScratchQuery scratch_bytes;
    /* trace와 profile 출력에 쓰는 사람이 읽는 이름. */
    const char *name;
} CamppKernelEntry;

/* 한 backend가 제공하는 kernel 전체. entries는 backend가 소유하는 정적 배열이다. */
typedef struct CamppKernelRegistry {
    uint16_t backend_id;
    const char *name;
    const CamppKernelEntry *entries;
    size_t entry_count;
} CamppKernelRegistry;

/*
 * opcode와 kernel_id로 구현을 찾는다.
 *
 * 찾지 못하면 CAMPP_STATUS_MISSING_KERNEL이다. 실행 도중이 아니라 로드 직후에
 * campp_kernel_registry_validate로 미리 확인하는 것을 원칙으로 한다. 1,000번째
 * operator에서 kernel이 없다는 사실을 알게 되는 것은 너무 늦다.
 */
CamppStatus campp_kernel_registry_lookup(
    const CamppKernelRegistry *registry, uint16_t opcode, uint16_t kernel_id,
    const CamppKernelEntry **out_entry);

/* 표 자체가 성립하는지 본다. 중복된 (opcode, kernel_id)와 NULL run을 거른다. */
CamppStatus campp_kernel_registry_validate(const CamppKernelRegistry *registry);

/*
 * model이 쓰는 모든 opcode에 구현이 있는지 확인한다.
 *
 * 없으면 CAMPP_STATUS_MISSING_KERNEL을 돌려주고, out_missing_opcode에 처음
 * 빠진 opcode를 적어 어떤 kernel을 만들어야 하는지 바로 알 수 있게 한다.
 */
CamppStatus campp_kernel_registry_covers_model(
    const CamppKernelRegistry *registry, const struct CamppRuntimeModel *model,
    uint16_t *out_missing_opcode);

/*
 * CPU Reference backend의 표.
 *
 * Phase 3의 유일한 backend이며 정확도 기준이다. NEON backend가 생기면 같은
 * 모양의 함수를 하나 더 내놓고, 어느 표를 쓸지는 context가 정한다.
 */
const CamppKernelRegistry *campp_cpu_reference_registry(void);

#endif /* CAMPP_RUNTIME_INTERNAL_KERNEL_REGISTRY_H */
