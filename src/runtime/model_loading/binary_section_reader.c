/*
 * compiled model 내부 section을 안전하게 읽는 저수준 도우미를 구현할 파일이다.
 *
 * 역할:
 * - 고정 폭 정수와 descriptor 읽기
 * - offset + size의 overflow 검사
 * - alignment 검사
 * - weights의 file offset을 read-only pointer로 연결
 *
 * 임의의 C 구조체 포인터 캐스팅보다 명시적인 field 읽기를 우선한다.
 */
