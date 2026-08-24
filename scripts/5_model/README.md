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
평가기의 기본 계약은 98/298/498/998 모두 측정된 layer-hybrid V3 plan을 사용하는
것이다. 하나라도 compiled plan에서 빠져 있으면 V2 fallback으로 측정하지 않고
preflight에서 실패한다. 바이너리가 각 실행 결과에 실제 bucket policy를 기록하며
평가기는 네 bucket 모두 `layer_hybrid_v3`인지 자동 검증한다.

bucket별 승자 측정과 최종 V3 재빌드 후 preflight와 quick을 실행한다.

```bash
python3 scripts/4_profill/optimization/16_select_multibucket_v3.py \
  --buckets 298 498 998 --mode official --force --build-final
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
results/models/campplus/final_v3/onnx_crt_evaluation/<run-id>/frame_matrix.*
```

## Bucket-specific static weight plans

98/298/498/998은 각 execution plan의 Operator input Tensor ID를 따라가며 서로
독립된 정적 weight layout을 만든다. Constant는 최초 사용 Operator 순서로 미리
패킹되고 새 plan에서는 Constant `data_offset`만 바뀐다. Operator table,
attribute section, Tensor metadata와 weight bytes는 유지된다. Runtime은 기존
`weights_base + data_offset`을 그대로 사용하므로 inference 중 weight 복사,
재배치, eviction은 없다.

```bash
python3 scripts/5_model/05_build_weight_plans.py --force
python3 scripts/5_model/06_validate_weight_plans.py

bash scripts/4_profill/05_build_final_v3.sh
python3 scripts/5_model/07_benchmark_weight_plans.py --preflight-only
python3 scripts/5_model/07_benchmark_weight_plans.py --mode quick --force
```

Quick gate는 원본 shared `weights.bin`과 bucket 정적 blob을 같은 V3 binary와
입력으로 실행하여 embedding bitwise, warm p50 1% 이내, p95 3% 이내를 요구한다.
정식 증거는 `--mode official --force`로 만든다.

98 bucket의 Quick/official gate를 통과한 뒤에만 정적 pair를 최종 model package로
승격한다. 기본 `01_build_model_bin.py` 입력은 기존 shared bundle로 유지되므로,
승격할 때 아래 세 입력을 함께 명시해야 한다.

```bash
python3 scripts/5_model/01_build_model_bin.py \
  --plan runs/models/campplus/final_v3/weight_residency/98/plan_98.bin \
  --weights runs/models/campplus/final_v3/weight_residency/98/weights_98.bin \
  --source-manifest runs/models/campplus/final_v3/weight_residency/98/weight_plan_98.json \
  --force
```

```text
runs/models/campplus/final_v3/weight_residency/<bucket>/plan_<bucket>.bin
runs/models/campplus/final_v3/weight_residency/<bucket>/weights_<bucket>.bin
runs/models/campplus/final_v3/weight_residency/<bucket>/weight_plan_<bucket>.json
runs/models/campplus/final_v3/weight_residency/<bucket>/weight_plan_<bucket>.csv
results/models/campplus/final_v3/weight_residency/summary.json
results/models/campplus/final_v3/weight_residency/validation.json
```

## Multibucket mmap weight window experiment

이 단계는 기존 Final V3 바이너리를 덮어쓰지 않는다. bucket 전용 weights를 Operator
최초 사용 순서의 4 KiB block으로 만들고, 별도 candidate binary에서만 read-only
`mmap`, 다음 block `POSIX_FADV_WILLNEED`, 마지막 사용 후 `MADV_DONTNEED`를
적용한다. Tensor마다 `pread`하거나 kernel pointer를 복사하지 않는다.

```bash
python3 scripts/5_model/08_build_weight_streaming.py \
  --bucket-frames 98 298 498 998 --force
bash scripts/5_model/09_build_weight_streaming_98.sh
python3 scripts/5_model/10_benchmark_weight_streaming_multibucket.py \
  --preflight-only
python3 scripts/5_model/10_benchmark_weight_streaming_multibucket.py \
  --mode quick --trace-weight-events --force
```

기존 V3가 아직 빌드되지 않은 보드에서만 먼저 다음을 실행한다.

```bash
bash scripts/4_profill/05_build_final_v3.sh
```

비교 mode는 기존 V3 full-resident, 같은 candidate layout의 malloc, 전체 mmap,
windowed mmap 네 가지다. Quick gate는 V3 대비 embedding bitwise, warm p50 1%,
p95 3% 이내, Peak RSS 5.5 MB 이상 감소, 측정 구간 major fault 0을 모두 요구한다.
성능 측정 전에 각 bucket 고정 입력 한 개의 모든 retained Tensor도 기존 V3와 bitwise로
비교한다. build script는 소스·flag hash가 같으면 두 candidate binary를 재사용한다.
event trace는 별도 진단 process에서만 실행하여 공식 latency와 Peak gate를 오염시키지
않는다. runtime은 오래된 block을 먼저 `DONTNEED`한 뒤 다음 block에 `WILLNEED`를
호출한다. C pipeline은 `campp_runtime_model_load_weight_streaming_bundle(root,
bucket_frames, model)`로 sidecar root에서 bucket별 세 파일을 함께 선택할 수 있다.

```text
runs/models/campplus/final_v3/weight_streaming/<bucket>/plan_<bucket>.bin
runs/models/campplus/final_v3/weight_streaming/<bucket>/weights_<bucket>.bin
runs/models/campplus/final_v3/weight_streaming/<bucket>/weight_schedule_<bucket>.bin
runs/models/campplus/final_v3/weight_streaming/weight_streaming_manifest.json
results/models/campplus/final_v3/weight_streaming/<bucket>/plan.json
results/models/campplus/final_v3/weight_streaming/benchmark_multibucket_quick.json
```

## Deployable fixed-bucket model set

현재 C package loader의 v1 ABI는 package 하나당 bucket 하나를 로드한다. 포맷을
임의로 확장하지 않고 98/298/498/998용 package를 각각 생성하며, 작은 manifest가
feature frame 수와 package를 연결한다.

```bash
python3 scripts/5_model/11_build_bucket_models.py --force

python3 scripts/5_model/12_run_bucket_model.py \
  --input benchmarks/campplus/features/multi__speaker_0000__498.f32 \
  --embedding-output runs/model_packages/example_498_embedding.f32 \
  --preflight-only

python3 scripts/5_model/12_run_bucket_model.py \
  --input benchmarks/campplus/features/multi__speaker_0000__498.f32 \
  --embedding-output runs/model_packages/example_498_embedding.f32

python3 scripts/5_model/13_validate_bucket_models.py --preflight-only
python3 scripts/5_model/13_validate_bucket_models.py --force
```

선택기는 `[1,T,80]` float32 payload 크기에서 `T`를 계산하고 package SHA, 내부
bucket/input shape, runtime의 `camppmodel-v1` 지원과 Final V3 suite가 모두 맞아야
실행한다. `.camppmodel` 방식은 full-resident weights이며, windowed mmap이 필요하면
앞 절의 sidecar 방식을 사용한다.
