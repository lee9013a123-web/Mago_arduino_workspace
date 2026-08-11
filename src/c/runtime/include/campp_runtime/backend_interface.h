/*
 * CPU Reference, AArch64 NEON 등 모든 backend가 따를 함수 규격을 선언한다.
 *
 * Backend 책임:
 * - 지원 opcode 조회
 * - kernel 함수 등록
 * - Operator descriptor와 Tensor pointer를 받아 계산 수행
 * - 필요한 scratch byte 수 보고
 *
 * Graph executor는 backend 내부 계산 방식을 알지 못하도록 분리한다.
 */
