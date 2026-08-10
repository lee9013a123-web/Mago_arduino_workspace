# CAM++ Runtime bundle exporter 패키지의 시작 파일이다.
#
# 이 패키지는 정적 ONNX와 graph IR을 읽어 C Runtime이 사용할
# weights.bin, plan_*.bin, manifest.json을 생성하는 재사용 코드를 제공한다.
# 실제 변환 실행 순서는 scripts/3_runtime/01_export_reference_bundle.py에서 시작한다.
