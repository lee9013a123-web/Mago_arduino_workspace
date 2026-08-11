#ifndef CAMPP_RUNTIME_EXECUTION_GRAPH_EXECUTOR_H
#define CAMPP_RUNTIME_EXECUTION_GRAPH_EXECUTOR_H

#include "campp_runtime/status_code.h"
#include "internal/runtime_context.h"

/*
 * Context에 미리 연결된 Tensor와 Kernel 표를 사용해 Operator 0부터 순서대로
 * 실행한다. 이 함수는 메모리를 할당하거나 해제하지 않는다.
 */
CamppStatus campp_graph_execute(CamppRuntimeContext *context);

#endif /* CAMPP_RUNTIME_EXECUTION_GRAPH_EXECUTOR_H */
