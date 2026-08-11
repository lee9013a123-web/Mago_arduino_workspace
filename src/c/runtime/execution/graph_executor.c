/*
 * execution plan의 Operator table을 0번부터 순서대로 실행할 파일이다.
 *
 * 각 단계에서:
 * - input/output Tensor ID를 pointer로 해석
 * - opcode에 맞는 kernel을 dispatcher에 요청
 * - kernel 오류 확인
 * - 필요 시 중간 Tensor와 trace 기록
 *
 * Phase 3에서는 최적화 없이 plan의 모든 runtime node를 그대로 실행한다.
 */
