/*
 * Runtime 함수가 반환할 상태 코드와 오류 분류를 선언할 파일이다.
 *
 * 구분할 오류:
 * - 파일 열기와 읽기 실패
 * - binary magic, version, checksum 불일치
 * - 지원하지 않는 opcode 또는 dtype
 * - 입력 shape와 bucket 불일치
 * - 메모리 부족과 buffer 범위 초과
 * - kernel 실행 실패
 */
