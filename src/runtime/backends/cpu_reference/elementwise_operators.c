/*
 * Add, Mul, Sub, Div, Sqrt의 elementwise 기준 구현을 둘 파일이다.
 * 동일 shape뿐 아니라 ONNX broadcasting과 axis 해석을 포함한다.
 * 통계 pooling의 분산·표준편차 경로에서 누적 순서를 추적할 수 있게 작성한다.
 */
