# ONNX initializer를 weights.bin에 직렬화하는 모듈이다.
#
# 저장 대상:
# - convolution weight와 bias
# - BatchNormalization parameter
# - quantization scale과 zero point
# - runtime에서 필요한 작은 상수 Tensor
#
# Phase 3에서는 ONNX 원본 layout을 유지한다.
# NEON용 channel padding과 block packing은 이후 단계에서 별도 적용한다.
