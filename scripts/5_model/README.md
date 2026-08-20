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
