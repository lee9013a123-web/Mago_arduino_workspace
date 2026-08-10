# 생성된 Runtime bundle의 추적 정보를 manifest.json으로 저장하는 모듈이다.
#
# 기록할 정보:
# - canonical ONNX SHA-256
# - graph freeze manifest hash
# - binary format version
# - bucket별 frame 수와 operator 수
# - weights.bin과 plan_*.bin의 크기 및 SHA-256
#
# Runtime은 로드 시 이 정보와 binary header가 일치하는지 확인한다.
