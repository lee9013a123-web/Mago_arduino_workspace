/*
 * 로드한 compiled model이 현재 Runtime에서 안전하게 실행 가능한지 검증한다.
 *
 * 검증 항목:
 * - magic, format version, byte order
 * - model checksum과 target
 * - section offset과 파일 크기 범위
 * - Tensor ID와 Operator 입출력 참조
 * - 입력 bucket과 feature shape
 * - 지원하지 않는 dtype/opcode 존재 여부
 */
