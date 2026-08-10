# 기존 graph 분석 결과인 ir_*.json을 읽는 모듈이다.
#
# 입력:
# - results/graph/ir_{1,3,5,10}s.json
#
# 사용할 정보:
# - bucket별 Tensor shape와 byte 크기
# - producer와 consumer 관계
# - static/runtime node 구분
# - topological execution order
#
# ONNX에서 읽은 정보와 IR의 shape가 다르면 변환을 중단하도록 검증한다.
