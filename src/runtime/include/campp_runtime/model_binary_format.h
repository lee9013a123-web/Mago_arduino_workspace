/*
 * weights.bin과 plan_*.bin의 디스크 형식을 정의할 파일이다.
 *
 * binary_format_schema.py와 동일하게 유지할 항목:
 * - magic과 format version
 * - target, dtype, bucket frame 수
 * - Tensor/Operator/attribute section 위치
 * - weights section 위치와 크기
 * - checksum 필드
 *
 * 컴파일러 padding 차이를 피하기 위해 고정 폭 정수 타입만 사용한다.
 */
