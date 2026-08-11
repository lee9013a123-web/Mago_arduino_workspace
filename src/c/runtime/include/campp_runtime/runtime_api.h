/*
 * CAM++ Runtime의 외부 공개 API를 선언할 파일이다.
 *
 * 예정 API:
 * - Runtime 생성 및 compiled model 로드
 * - 입력 feature Tensor 연결
 * - 단일 inference 실행
 * - embedding 출력 조회
 * - Runtime 자원 해제
 *
 * 호출자는 내부 Tensor table이나 kernel 구현을 직접 다루지 않도록 한다.
 */
