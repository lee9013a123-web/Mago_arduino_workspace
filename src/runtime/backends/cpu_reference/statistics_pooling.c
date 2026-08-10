/*
 * CAM++ 최종 통계 pooling의 평균과 표준편차 계산을 구현할 파일이다.
 * frame 축과 channel 축을 descriptor에서 확인하고 분산의 epsilon과 sqrt 순서를 보존한다.
 * 최종 embedding 오차가 커질 경우 별도로 중간 mean/std Tensor를 dump할 수 있게 한다.
 */
