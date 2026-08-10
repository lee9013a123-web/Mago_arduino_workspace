/*
 * QuantizeLinear, DequantizeLinear과 필요한 Requantize 기준 계산을 구현할 파일이다.
 * scale과 zero point의 scalar/per-channel 형태, rounding, dtype clamp 규칙을 검증한다.
 * 정수 출력이 있는 경로는 가능한 경우 ORT와 완전 일치를 목표로 한다.
 */
