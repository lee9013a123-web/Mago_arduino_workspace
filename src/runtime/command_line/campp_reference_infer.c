/*
 * 터미널에서 Reference Runtime을 실행하는 command-line 프로그램을 구현할 파일이다.
 *
 * 받을 인자:
 * - weights.bin, plan_*.bin 경로
 * - 입력 feature binary 경로
 * - embedding 출력 경로
 * - 선택적 Tensor dump와 trace 설정
 *
 * 이 파일에는 operator 계산을 넣지 않고 Runtime 공개 API만 호출한다.
 */
