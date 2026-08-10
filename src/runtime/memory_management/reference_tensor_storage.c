/*
 * Phase 3 Reference Runtime의 단순한 activation 저장 방식을 구현할 파일이다.
 * Runtime 초기화 때 activation Tensor마다 독립 buffer를 한 번 할당한다.
 * inference 도중에는 malloc/free하지 않으며, 정확도 검증을 쉽게 하는 것이 목적이다.
 */
