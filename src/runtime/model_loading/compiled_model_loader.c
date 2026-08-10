/*
 * weights.bin과 선택한 plan_*.bin을 읽어 Runtime Model을 구성할 파일이다.
 *
 * 처리 순서:
 * 1. 파일 header 읽기
 * 2. validator 호출
 * 3. Tensor와 Operator table 로드
 * 4. weights section 연결
 * 5. Runtime Context에 Model 전달
 *
 * 이 파일은 operator 계산이나 메모리 offset 계획을 수행하지 않는다.
 */
