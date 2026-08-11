/*
 * Operator opcode와 선택된 backend에 맞는 kernel 함수를 반환할 파일이다.
 * 초기에는 cpu_reference만 등록하고 이후 AArch64 NEON backend를 추가한다.
 * 지원 kernel이 없으면 Reference fallback 또는 명시적 오류를 반환한다.
 */
