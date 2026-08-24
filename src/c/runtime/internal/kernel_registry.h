#ifndef CAMPP_RUNTIME_INTERNAL_KERNEL_REGISTRY_H
#define CAMPP_RUNTIME_INTERNAL_KERNEL_REGISTRY_H

/*
 * opcode를 C 함수에 연결하는 표다.
 *
 *   CAMPP_OP_QLINEAR_CONV -> campp_qlinear_convolution_reference()
 *   CAMPP_OP_RELU         -> campp_relu_reference()
 *
 * 정적 그래프에는 node가 1,438개 있지만 opcode는 20종뿐이다. context를 만들 때
 * operator마다 이 표를 한 번 조회하여 resolved_kernels 배열에 실행 함수 주소를
 * 저장한다. 추론 중에는 표를 다시 검색하지 않고 저장된 주소를 바로 호출한다.
 * QLinearConv 225개는 kernel 함수 하나를 225번 부르는 것이지 225개의 코드가
 * 아니다.
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

/*
 * kernel은 실행 상태를 보지 않는다. 그래서 이 헤더는 CamppRuntimeContext를
 * 전방 선언조차 하지 않는다.
 */
struct CamppRuntimeModel;

/*
 * kernel 하나의 실행 함수.
 *
 * kernel은 Tensor data에 접근할 때 view만 본다. Tensor ID, storage_type,
 * weights base를 이용한 주소 해석은 executor가 이미 끝냈다. model과 op는
 * attribute를 읽는 데만 사용하는 읽기 전용 정보이며 실행 상태는 노출하지
 * 않는다. inputs는 descriptor의 input_count만큼 순서대로 채워지며, ONNX 입력
 * 순서를 그대로 따른다. 예를 들어 QLinearConv는 항상
 * (x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [B]) 순이다.
 *
 * scratch는 campp_kernel_scratch_bytes가 요구한 크기만큼 확보된 임시 버퍼이며,
 * 호출마다 내용이 보장되지 않는다. 필요 없으면 NULL이다.
 */
typedef CamppStatus (*CamppKernelRun)(
    const struct CamppRuntimeModel *model,
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
 * backend 선택 규칙.
 *
 * Operator가 CAMPP_BACKEND_AUTO이면 전달된 registry를 사용할 수 있다. 그 외에는
 * operator.backend_id와 registry.backend_id가 반드시 같아야 한다. 현재 생성되는
 * plan은 AUTO이므로 CPU Reference 또는 이후의 AArch64 registry를 context 생성 때
 * 선택할 수 있다. 미래에 특정 backend로 고정한 plan은 같은 backend만 허용한다.
 */

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
 * backend가 맞지 않으면 CAMPP_STATUS_BACKEND_MISMATCH, 구현이 없으면
 * CAMPP_STATUS_MISSING_KERNEL을 돌려준다. out_failing_operator_id와
 * out_missing_opcode에는 처음 실패한 Operator와 opcode를 적는다.
 */
CamppStatus campp_kernel_registry_covers_model(
    const CamppKernelRegistry *registry, const struct CamppRuntimeModel *model,
    uint32_t *out_failing_operator_id, uint16_t *out_missing_opcode);

/*
 * CPU Reference backend의 표.
 *
 * Phase 3의 유일한 backend이며 정확도 기준이다. NEON backend가 생기면 같은
 * 모양의 함수를 하나 더 내놓고, 어느 표를 쓸지는 context가 정한다.
 */
const CamppKernelRegistry *campp_cpu_reference_registry(void);

/* O4I4 packed QLinearConv와 stride-aware fallback을 제공한다. */
const CamppKernelRegistry *campp_cpu_aarch64_registry(void);

#endif /* CAMPP_RUNTIME_INTERNAL_KERNEL_REGISTRY_H */
