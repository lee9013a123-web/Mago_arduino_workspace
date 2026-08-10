# Phase 3의 두 번째 실행 스크립트다.
#
# 역할:
# - 고정 feature 입력을 ONNX Runtime에 넣는다.
# - 지정 operator의 실제 입력과 출력을 저장한다.
# - Dense block 경계와 최종 embedding을 저장한다.
# - Tensor ID, dtype, shape, checksum을 manifest에 기록한다.
#
# 원시 출력은 runs/runtime/ort_reference에 저장한다.
