/*
 * Tensor pointer, size, arena offset이 허용 범위 안에 있는지 검사할 파일이다.
 * Reference 개발 빌드에서는 모든 kernel 호출 전후에 guard 영역 손상을 확인한다.
 * Release 빌드에서는 비용이 큰 검사를 선택적으로 비활성화할 수 있게 설계한다.
 */
