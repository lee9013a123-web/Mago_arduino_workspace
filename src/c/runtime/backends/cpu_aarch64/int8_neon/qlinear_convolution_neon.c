/*
 * Phase 7에서 사용할 INT8 QLinearConv NEON 구현을 둘 파일이다.
 * INT8 입력과 weight, INT32 accumulation, bias, requantization epilogue를 처리한다.
 * QRB2210이 지원하는 실제 ISA만 사용하고 SDOT 지원을 가정하지 않는다.
 */
