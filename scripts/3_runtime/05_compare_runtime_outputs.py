# Phase 3의 다섯 번째 실행 스크립트다.
#
# 역할:
# - ORT와 C Runtime Tensor의 dtype과 shape를 먼저 확인한다.
# - max absolute error, relative error, cosine similarity를 계산한다.
# - topological order상 처음 허용 오차를 넘은 operator를 보고한다.
# - 네 bucket의 최종 embedding 검증 결과를 results/runtime에 저장한다.
#
# INT8 정수 출력과 FP32 출력은 서로 다른 허용 오차 정책을 사용한다.
