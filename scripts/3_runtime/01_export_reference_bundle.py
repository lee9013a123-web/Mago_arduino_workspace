# Phase 3의 첫 번째 실행 스크립트다.
#
# 역할:
# - src/runtime_bundle_exporter 패키지를 호출한다.
# - canonical model, 정적 ONNX, graph IR 경로를 인자로 받는다.
# - weights.bin, bucket별 plan_*.bin, manifest.json을 생성한다.
# - 생성 직후 크기, node 수, Tensor 수와 SHA-256을 출력한다.
#
# 이 파일에는 ONNX 분석과 binary 직렬화 세부 구현을 직접 넣지 않는다.
