# Final-98 model binary

`campp_sv_98.camppmodel`은 향후 음성→화자검증 파이프라인이 소비할 고정 bucket
98 화자 embedding 모델이다. 현재 단계에서는 audio frontend나 cosine scorer를
구현하지 않는다. 대신 그 구현이 따라야 할 계약을 model binary 안에 저장한다.

## Build and verify

```bash
python3 scripts/5_model/01_build_model_bin.py --force
python3 scripts/5_model/02_verify_model_bin.py \
  --expect-plan runs/runtime/kernel_optimization/e7/bundle/execution_plans/plan_98.bin \
  --expect-weights runs/runtime/kernel_optimization/e7/bundle/weights.bin
```

기본 산출물:

```text
models/runtime/campp_sv_98/campp_sv_98.camppmodel
models/runtime/campp_sv_98/campp_sv_98.json
```

모델은 execution plan, packed weights, model metadata, frontend contract,
postprocess contract의 다섯 section으로 구성된다. 모든 section과 전체 payload는
SHA-256으로 검증된다.

## Runtime validation on QRB2210

Final-98 V3 runtime을 빌드한 뒤 split 파일과 model package가 같은 embedding을
생성하는지 확인한다.

```bash
bash scripts/4_profill/05_build_final_v3.sh
python3 scripts/5_model/03_validate_model_runtime.py --preflight-only
python3 scripts/5_model/03_validate_model_runtime.py --force
```

검증은 같은 `[1,98,80]` feature에 대해 `plan_98.bin + weights.bin` 실행과
`.camppmodel` 실행의 모든 retained tensor와 192개 float32 embedding을 byte
단위로 비교한다.

## Deliberately not frozen yet

- 긴 음성의 1초 window 선택·aggregation
- 짧은 음성 padding
- enrollment template aggregation
- speaker verification threshold

위 값은 trial 기반 정확도 보정 후 새 contract version 또는 별도 calibrated scoring
section으로 추가한다. 임의의 threshold를 현재 모델에 넣지 않는다.

## ONNX Runtime vs final C Runtime, four buckets

Final V3 배포 바이너리 하나로 원본 ONNX Runtime과 C Runtime을 비교한다.
컴파일된 bucket별 layer-hybrid plan이 있으면 V3를 사용하고, 아직 측정된 plan이
없는 bucket은 검증된 V2(`qconv_v4 + fused_combined_hybrid`)로 fallback한다.
바이너리가 각 실행 결과에 실제 bucket policy를 기록하며 평가기가 이를 자동
검증한다.

계측 필드가 포함된 V3 바이너리를 한 번 빌드한 뒤 preflight와 quick을 실행한다.

```bash
bash scripts/4_profill/05_build_final_v3.sh
python3 scripts/5_model/04_evaluate_onnx_crt_multibucket.py \
  --mode quick --preflight-only
python3 scripts/5_model/04_evaluate_onnx_crt_multibucket.py --mode quick
```

공식 반복 수로 재측정할 때만 `--mode official`을 사용한다. 각 성능 입력을 원본
ONNX Runtime과 C Runtime에 동일하게 넣어 embedding을 직접 비교한다. 정확도
gate는 bitwise가 아니라 finite embedding 및 cosine 0.99 이상이다. cached ORT
retained tensor 차이는 별도 진단 자료로 계속 저장된다. 성능 보고에는
First/Warm p50, p95/p99, RTF,
whole-process Peak RSS와 모델/세션 로드 전 대비 incremental Peak RSS가 포함된다.

최종 표는 다음 위치에 JSON/CSV/Markdown으로 생성된다.

```text
results/model_evaluation/onnx_crt_multibucket/<run-id>/frame_matrix.*
```
