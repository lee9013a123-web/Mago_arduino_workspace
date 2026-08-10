/*
 * Tensor buffer의 앞뒤 guard 영역과 descriptor byte 크기를 검증할 파일이다.
 * 잘못된 Tensor ID, overflow offset, undersized output buffer를 Runtime이 거부하는지 확인한다.
 * Phase 4에서는 수명이 겹치는 Tensor의 arena alias 손상 테스트를 추가한다.
 */
