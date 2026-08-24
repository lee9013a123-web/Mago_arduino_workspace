# Reference Runtime 모델 bundle

Runtime bundle exporter가 생성하는 파일의 위치다.

- `weights.bin`: 네 bucket이 공유하는 가중치, bias, scale, zero point
- `execution_plans/plan_*.bin`: bucket별 Tensor와 Operator 실행 계획
- `manifest.json`: 원본 모델 hash, format version, 산출물 checksum

이 파일들은 생성물이므로 소스 파일과 분리하며, 재생성 명령과 checksum을 결과 보고서에 기록한다.
