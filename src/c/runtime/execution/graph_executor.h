#ifndef CAMPP_RUNTIME_EXECUTION_GRAPH_EXECUTOR_H
#define CAMPP_RUNTIME_EXECUTION_GRAPH_EXECUTOR_H

#include "campp_runtime/status_code.h"
#include "internal/runtime_context.h"

/*
 * Context에 미리 연결된 Tensor와 Kernel 표를 사용해 Operator 0부터 순서대로
 * 실행한다. 이 함수는 메모리를 할당하거나 해제하지 않는다.
 */
CamppStatus campp_graph_execute(CamppRuntimeContext *context);

/*
 * context를 reset하거나 앞 Operator를 실행하지 않고 operator_id 하나만
 * 실행한다. ORT 입력을 주입한 독립 Operator replay에서 전체 graph와 같은
 * bounds 검사와 dispatch 경로를 재사용하기 위한 내부 API다.
 */
CamppStatus campp_graph_execute_operator(
    CamppRuntimeContext *context, uint32_t operator_id);

#endif /* CAMPP_RUNTIME_EXECUTION_GRAPH_EXECUTOR_H */
