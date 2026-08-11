/*
 * Scalar C로 작성한 Reference kernel을 opcode별 함수 table에 등록할 파일이다.
 * 이 backend는 속도가 아니라 읽기 쉬운 계산 순서와 ORT 수치 일치를 우선한다.
 * 이후 NEON과 fused kernel의 정확도 oracle로 계속 유지한다.
 */
