# CAM++ Reference Runtime 고정 보고서

- 기준선 ID: `campp-reference-v1-b4092e6bfdcf5443`
- Git SHA: `23f4f504c99aa257dd0d300021ebd112976b863f`
- canonical model SHA-256: `1ea9a4806a3c410046e7e785cfa81fb1393550b2f016b1e94581056b1c092048`
- binary format version: `1`
- weights SHA-256: `595d308c035eccbd338eeba72a7b3b23dd878878cf74418f0730bf60264aecb9`
- validation config SHA-256: `81a39aa99cf34646c6a370bfa623d428646f415b86568b00f7fd2d19d369027f`
- 기준선 고정: `PASS`
- Phase 4 진행 가능: `NO`

## 고정 범위

`manifest.json`에 기록된 canonical ONNX, `weights.bin`, 네 execution plan의 크기와 SHA-256을 다시 계산했습니다. 각 bucket의 1,438개 Operator 출력 비교 파일도 SHA-256으로 고정했습니다.

## 구현한 opcode

`ADD`, `AVERAGE_POOL`, `BATCH_NORMALIZATION`, `CONCAT`, `DEQUANTIZE_LINEAR`, `DIV`, `EXPAND`, `MUL`, `QLINEAR_CONV`, `QUANTIZE_LINEAR`, `REDUCE_MEAN`, `RELU`, `RESHAPE`, `SIGMOID`, `SLICE`, `SQRT`, `SQUEEZE`, `SUB`, `TRANSPOSE`, `UNSQUEEZE`

## 허용 오차

| 항목 | 기준 |
|---|---:|
| FP32 절대 오차(atol) | 0.0001 |
| FP32 상대 오차(rtol) | 0.001 |
| embedding cosine 최솟값 | 0.999999 |
| 정수 Tensor | 원소 단위 완전 일치 |

## Bucket별 결과

| frames | 실행 | 비교 Tensor | 실패 Tensor | embedding cosine | 수치 판정 |
|---:|:---:|---:|---:|---:|:---:|
| 98 | PASS | 1438 | 0 | 0.999999999999998 | PASS |
| 298 | PASS | 1438 | 0 | 0.999999999999998 | PASS |
| 498 | PASS | 1438 | 0 | 0.999999999999998 | PASS |
| 998 | PASS | 1438 | 1059 | 0.998937643103882 | FAIL |

## 알려진 제한 사항

- 이 기준선은 scalar CPU Reference backend의 정확도를 기록하며 성능을 보증하지 않습니다.
- 998-frame은 FP32 1-ULP 차이가 QuantizeLinear 반올림 경계를 넘은 뒤 증폭되어 1059개 Tensor가 실패합니다. 첫 엄격 불일치는 Operator #369 QUANTIZE_LINEAR이며, 현재 기준에서는 PASS로 간주하지 않습니다.
- 전용 End-to-end 상세 보고서가 없는 bucket: [98, 498, 998].
- 현재 results/runtime에는 교차-bucket 거부 summary가 없습니다.

## Phase 4 판정

전체 네 bucket을 대상으로 한 Phase 4 최적화 기준선 승인은 보류합니다.

- 수치 허용 기준을 통과하지 못한 bucket이 있습니다: [998].
- 네 bucket의 전용 end_to_end_*.json이 모두 고정되지 않았습니다. 없는 bucket은 전체 Tensor dump 비교로 실행 완료를 추론했습니다.
- 잘못된 frame 수를 BUCKET_MISMATCH로 거부한 증거가 고정되지 않았습니다.
- 생성 보고서를 제외한 Git working tree가 깨끗하지 않습니다.

최적화 kernel은 같은 입력과 plan으로 실행한 뒤 이 기준선의 dtype, shape, 정수 완전 일치 및 FP32 허용 오차를 그대로 적용해야 합니다. 이 보고서는 성능 기준이 아니라 정확도 기준입니다.
