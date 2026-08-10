/*
 * Conv와 이후 NEON kernel이 요구하는 임시 작업 공간을 관리할 파일이다.
 * 필요한 byte 수를 모델 로딩 때 계산하고 inference 시작 전에 한 번 확보한다.
 * 다중 thread 단계에서는 코어별 scratch가 같은 cache line을 공유하지 않게 정렬한다.
 */
