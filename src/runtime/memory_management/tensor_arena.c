/*
 * Phase 4에서 사용할 정적 offset 기반 단일 Tensor Arena를 구현할 파일이다.
 * exporter가 계산한 arena 크기와 Tensor offset을 사용해 arena + offset으로 주소를 만든다.
 * Reference storage 검증이 끝나기 전에는 실행 경로에 연결하지 않는다.
 */
