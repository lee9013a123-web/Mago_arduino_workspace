/*
 * execution plan에 저장되는 Operator 명령 구조체를 선언할 파일이다.
 *
 * 포함할 필드:
 * - opcode
 * - input/output Tensor ID 목록
 * - attribute block 위치와 크기
 * - 선택된 backend와 kernel ID
 *
 * padding, stride, dilation, axis 같은 가변 정보는 별도 attribute 영역에 저장한다.
 */
