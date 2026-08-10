/*
 * 정적 Tensor ID를 inference 중 사용할 실제 data pointer와 연결한다.
 * 입력 feature, read-only weight, activation buffer, 최종 embedding 주소를 관리한다.
 * Tensor의 shape와 dtype은 바꾸지 않고 실행 시 주소만 연결한다.
 */
