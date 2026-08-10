# bucket별 정적 ONNX를 읽는 모듈이다.
#
# 입력:
# - results/static/campp_static_{98,298,498,998}.onnx
#
# 추출할 정보:
# - runtime node의 op_type, input, output, attribute
# - initializer의 dtype, shape, raw data
# - QLinearConv의 scale, zero point, bias
#
# 이 모듈은 최적화를 수행하지 않고 ONNX 정보를 내부 자료구조로 옮기는 역할만 한다.
