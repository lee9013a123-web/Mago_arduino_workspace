#ifndef CAMPP_RUNTIME_INTERNAL_RUNTIME_CONTEXT_H
#define CAMPP_RUNTIME_INTERNAL_RUNTIME_CONTEXT_H

/*
 * 한 번의 inference가 쓰는 실행 상태다.
 *
 * CamppRuntimeModel이 "무엇을 계산하는가"라면 context는 "지금 어디까지
 * 계산했고 값이 어디에 있는가"다. model은 로드 후 불변이고 context만 바뀐다.
 * 그래서 같은 model에 context를 여러 개 붙여 동시에 돌릴 수 있고, context를
 * 초기화해도 model을 다시 읽을 필요가 없다.
 *
 * context는 model을 소유하지 않는다. 수명은 호출자가 관리하며, model이 먼저
 * 사라지면 context는 무효다.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "campp_runtime/status_code.h"
#include "kernel_registry.h"
#include "runtime_model.h"
#include "tensor_view.h"

/*
 * activation 저장소.
 *
 * Phase 3은 Tensor마다 독립 버퍼를 잡는다. 느리고 메모리를 많이 쓰지만, 어떤
 * Tensor가 언제 덮어써졌는지 헷갈릴 일이 없어 ORT와 값을 맞출 때 유리하다.
 * Tensor Arena로 옮길 때는 buffers를 slab 하나와 offset 배열로 바꾸면 되고,
 * 그 판단에 필요한 수명 정보(first_use/last_use)는 이미 plan에 들어 있다.
 */
typedef struct CamppActivationStorage {
    /* tensor_count 길이. CONSTANT와 INPUT 자리는 NULL이다. */
    void **buffers;
    size_t *buffer_sizes;
    uint32_t buffer_count;
    size_t total_bytes;
} CamppActivationStorage;

/*
 * 실행 중 관찰 상태.
 *
 * 기본값은 전부 꺼져 있다. ORT와 값이 갈리는 지점을 찾을 때만 켠다.
 */
typedef struct CamppDiagnosticsState {
    bool trace_operators;
    bool dump_tensors;
    bool check_bounds;

    /* 마지막으로 실행을 시작한 operator. 실패 지점을 그대로 가리킨다. */
    uint32_t current_operator_id;
    uint32_t executed_operator_count;

    /* dump 파일을 쓸 디렉터리. NULL이면 dump하지 않는다. */
    const char *dump_directory;
} CamppDiagnosticsState;

/* 한 번의 inference 상태 전체. */
typedef struct CamppRuntimeContext {
    /* 소유하지 않는다. 여러 context가 같은 model을 공유할 수 있다. */
    const CamppRuntimeModel *model;

    /*
     * Tensor pointer table.
     *
     * tensor_id로 바로 찾는 배열이며 model->tensor_count 길이다. 로드 직후
     * CONSTANT는 weights를, ACTIVATION은 자기 버퍼를 가리키도록 채워 둔다.
     * INPUT은 bind 전까지 data가 NULL이다.
     */
    CamppTensorView *tensors;
    uint32_t tensor_count;

    CamppActivationStorage activations;

    /*
     * kernel이 임시로 쓰는 버퍼. 모든 operator가 요구한 크기의 최댓값이며
     * 한 번만 잡아 재사용한다.
     */
    void *scratch;
    size_t scratch_size;

    /* 실제로 실행에 쓰는 backend와 그 kernel 표. */
    uint16_t backend_id;
    const CamppKernelRegistry *registry;

    /*
     * operator_id로 바로 찾는 실행 함수 표. model->operator_count 길이이며 context가
     * 배열을 소유한다. create 때 registry lookup을 모두 끝내므로 inference 중에는
     * opcode 검색이 발생하지 않는다. 각 entry 자체는 registry가 소유한다.
     */
    const CamppKernelEntry **resolved_kernels;
    uint32_t resolved_kernel_count;

    CamppDiagnosticsState diagnostics;

    /* 마지막으로 실패한 원인. 성공하면 CAMPP_STATUS_OK다. */
    CamppStatus last_status;
} CamppRuntimeContext;

/*
 * 생성과 해제.
 *
 * create는 model을 훑어 Tensor view를 채우고, activation 버퍼와 scratch를
 * 미리 잡고, registry가 model의 모든 backend/opcode/kernel_id를 덮는지 확인한
 * 뒤 resolved_kernels를 채운다. 즉 실행 중에 새로 할당하거나 registry를
 * 조회하는 일이 없도록 여기서 다 끝낸다. registry는 context보다 오래 살아야
 * 하며 보통 backend가 제공하는 정적 객체다.
 */
CamppStatus campp_runtime_context_create(
    const CamppRuntimeModel *model, const CamppKernelRegistry *registry,
    CamppRuntimeContext *context);

void campp_runtime_context_release(CamppRuntimeContext *context);

/*
 * 다음 inference를 위해 실행 상태만 되돌린다.
 *
 * 잡아 둔 버퍼는 그대로 두고 진단 counter와 last_status만 초기화한다.
 * 연속 추론에서 재할당을 피하기 위한 경로다.
 */
CamppStatus campp_runtime_context_reset(CamppRuntimeContext *context);

/*
 * 입출력 연결.
 *
 * bind_input은 외부 버퍼를 INPUT Tensor에 그대로 연결한다. 복사하지 않으므로
 * 호출자가 실행이 끝날 때까지 버퍼를 살려 두어야 한다. dtype, rank,
 * dimensions, byte_size를 descriptor와 모두 비교한다. 시간축 길이만 현재 plan의
 * bucket과 다르면 CAMPP_STATUS_BUCKET_MISMATCH, 그 밖의 형식 차이는
 * CAMPP_STATUS_SHAPE_MISMATCH로 거절한다.
 */
CamppStatus campp_runtime_context_bind_input(
    CamppRuntimeContext *context, uint32_t tensor_id, void *data,
    size_t byte_size, uint8_t dtype, uint8_t rank,
    const uint32_t dimensions[CAMPP_TENSOR_MAX_RANK]);

CamppStatus campp_runtime_context_output(
    const CamppRuntimeContext *context, uint32_t tensor_id,
    const CamppTensorView **out_view);

/* tensor_id로 view를 얻는다. 범위를 벗어나면 CAMPP_STATUS_INVALID_ARGUMENT다. */
CamppStatus campp_runtime_context_tensor(
    const CamppRuntimeContext *context, uint32_t tensor_id,
    CamppTensorView **out_view);

#endif /* CAMPP_RUNTIME_INTERNAL_RUNTIME_CONTEXT_H */
