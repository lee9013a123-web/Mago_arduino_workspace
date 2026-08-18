# Phase 3: Reference Runtime

이 폴더의 스크립트는 `src/python/runtime_bundle_exporter/`와
`src/c/runtime/`의 기능을 순서대로 호출한다. 모델 변환이나 kernel 계산을
스크립트에 다시 구현하지 않는다.

## 수동 재현: 다섯 명령

기존 `models/compiled/reference`를 보호하기 위해 별도 작업 폴더를 사용한다.

```bash
RUN_ROOT="$PWD/runs/runtime/pipelines/manual-reference"

python3 scripts/3_runtime/01_export_reference_bundle.py --output-dir "$RUN_ROOT/bundle"
python3 scripts/3_runtime/02_dump_ort_references.py --output-dir "$RUN_ROOT/ort_reference" --buckets 98 298 498 998
BUILD_DIR="$RUN_ROOT/build" bash scripts/3_runtime/03_build_reference_runtime.sh
BUILD_DIR="$RUN_ROOT/build" BUNDLE_DIR="$RUN_ROOT/bundle" ORT_DIR="$RUN_ROOT/ort_reference" OUT_DIR="$RUN_ROOT/c_reference" BUCKETS="98 298 498 998" bash scripts/3_runtime/04_run_reference_runtime.sh
python3 scripts/3_runtime/05_compare_runtime_outputs.py --ort-dir "$RUN_ROOT/ort_reference" --c-dir "$RUN_ROOT/c_reference" --results-dir "$RUN_ROOT/results" --buckets 98 298 498 998
```

위 순서는 bundle 생성, ORT 정답, C build, C 실행, 수치 비교를 재현한다.

## 자동 재현: 06

먼저 실행할 명령과 경로만 확인한다. 이 명령은 파일을 만들지 않는다.

```bash
python3 scripts/3_runtime/06_run_reference_pipeline.py \
  --config configs/runtime/reference.json \
  --dry-run
```

전체 파이프라인 실행:

```bash
python3 scripts/3_runtime/06_run_reference_pipeline.py \
  --config configs/runtime/reference.json
```

매 실행은 다음과 같은 고유 UTC timestamp 폴더에 저장된다.

```text
runs/runtime/pipelines/reference/reference-YYYYMMDDTHHMMSSZ/
├── bundle/
├── ort_reference/
├── build/
├── c_reference/
├── results/
│   └── verification_report.json
├── resolved_config.json
└── pipeline_summary.json
```

`06`은 `models/compiled/reference`에 쓰지 않으며 실행 전후 SHA-256 snapshot이
같은지도 확인한다. 같은 `--run-id`를 재사용하려면 `--resume`을 명시해야 한다.
검증된 staging bundle을 canonical 위치로 배포하는 기능은 의도적으로 포함하지
않았다.

## 결과 고정: 07

앞 단계에서 만든 canonical bundle과
`results/runtime/c_runtime_compare/compare_*.json`을 검증한 뒤
Phase 4가 참조할 최종 결과 세 파일을 만든다.

```bash
python3 scripts/3_runtime/07_freeze_reference_results.py
```

`07`은 추론을 다시 실행하지 않는다. 대신 manifest의 canonical model, weights와
네 plan의 크기·SHA-256, kernel registry의 opcode, 네 bucket의 1,438개 Operator
결과를 교차 검증한다. 알려진 수치 실패도 삭제하지 않고 최종 보고서에 고정한다.
출력은 `results/runtime/c_runtime_compare/operator_validation.json`,
`results/runtime/c_runtime_compare/end_to_end_validation.json`,
`results/runtime/c_runtime_compare/reference_runtime_report.md`이다.
CI에서 Phase 4 승인까지 필수로 요구하려면 `--require-phase4-ready`를 추가한다.
이 경우 보고서는 생성하되 `phase4_ready=false`이면 종료 코드 3을 반환한다.

## Tensor Arena plan 준비

기존 Reference bundle을 보존한 채 Arena offset이 들어간 80바이트 Tensor
descriptor를 별도 staging bundle에 생성한다.

```bash
python3 scripts/3_runtime/01_export_reference_bundle.py \
  --tensor-arena \
  --arena-alignment 64 \
  --output-dir runs/runtime/tensor_arena/bundle
```

필요하면 보드 메모리 예산을 byte 단위로 강제한다.

```bash
python3 scripts/3_runtime/01_export_reference_bundle.py \
  --tensor-arena \
  --arena-budget-bytes 25165824 \
  --output-dir runs/runtime/tensor_arena/bundle
```

Arena plan은 `ACTIVATION/OUTPUT`의 `data_offset`과 `DENSE_SLAB` flag만 바꾸며
descriptor 크기, weights와 graph 연산은 바꾸지 않는다.

## Tensor Arena C Runtime 검증: 08

Arena plan을 만든 뒤 C Runtime을 다시 빌드하고, 기존 독립 buffer 실행과 Arena
실행의 각 Operator 출력을 생성 직후 저장해 bit-exact 비교한다.

```bash
bash scripts/3_runtime/03_build_reference_runtime.sh
python3 scripts/3_runtime/08_validate_tensor_arena.py
```

일부 bucket만 확인하려면 다음처럼 지정한다.

```bash
python3 scripts/3_runtime/08_validate_tensor_arena.py --buckets 298
```

결과는
`results/runtime/c_runtime_compare/tensor_arena_validation.json`에 기록된다.
Arena에서는
중간 Tensor 주소가 재사용되므로 `campp_reference_dump`는 graph 종료 후가 아니라
각 Operator 출력 직후 callback으로 값을 기록한다. 보드 디스크를 보호하기 위해
bucket 하나를 비교해 통과하면 큰 binary dump 두 개를 즉시 삭제한다. 실패한
bucket은 원인 분석을 위해 남기며, 통과한 dump도 보존하려면 `--keep-dumps`를 쓴다.

## Dense concat 제거와 slab/view 검증: 10-11

기존 Tensor Arena bundle의 `manifest.json`과 `plan_*.bin`을 직접 분석해
Dense block 3개의 누적 Concat 52개를 slab-backed VIEW로 변환한다.

```bash
python3 scripts/3_runtime/10_export_dense_slab_bundle.py
python3 scripts/3_runtime/11_validate_dense_slab.py
```

생성 bundle은 `runs/runtime/kernel_optimization/dense_slab/bundle/`, 최종 검증 결과는
`results/runtime/dense/dense_slab_validation.json`에 기록된다.

## Cache layout·tiling·weight packing: 12-13

Dense slab bundle을 입력으로 받아 논리 shape는 유지하고 rank 3/4 activation의
물리 stride를 `[N,T,C_padded]` 또는 `[N,H,W,C_padded]`로 바꾼다. Channel은
4-lane 경계로 padding하며 Dense VIEW는 같은 slab의 channel byte offset을 쓴다.
입력의 `[N,T,C] -> [N,C,T]` Transpose는 외부 INPUT을 가리키는 producerless
VIEW로 바꿔 copy를 제거한다. 98-frame graph에서는 singleton 축만 바꾸는
Unsqueeze 54개, Squeeze 1개, Reshape 52개도 direct VIEW로 변환한다. 따라서
descriptor 변경만 필요한 layout copy는 총 108개 제거된다. 실제 data 순서를
바꾸는 `(1,32,10,98) -> (1,320,98)` Reshape는 copy 연산으로 유지한다.

225개 QLinearConv weight는 다음 O4I4 순서로 오프라인 packing한다.

```text
[group][output block][kernel][input block][output lane][input lane]
```

실행 중 packing은 없으며 kernel ID 1이 packed AArch64 backend를 선택한다.
AArch64에서는 NEON 4-lane multiply를 사용하고, 비-AArch64 개발 환경에서는 같은
tile 순서의 scalar fallback으로 정확도를 검증한다.

```bash
python3 scripts/3_runtime/12_export_cache_packed_bundle.py
python3 scripts/3_runtime/13_validate_cache_packed.py
```

생성 bundle은 `runs/runtime/kernel_optimization/cache_packed/bundle/`, 검증 결과는
`results/runtime/cache_layout/cache_packed_validation.json`이다. 검증 입력은
`multi__speaker_0000`, `0005`, `0006`의 98-frame feature 세 개로 고정한다.

## E7 operator fusion과 메모리 계획 재생성: 14-15

Cache-packed bundle에 fusion family를 독립적으로 적용한 다음 죽은 중간
Tensor를 제거하고 Tensor ID, producer/consumer, alias lifetime, Arena offset,
execution table, operator별 kernel ID와 최대 scratch를 전부 다시 생성한다.

```bash
python3 scripts/3_runtime/14_export_e7_fused_bundle.py --force
python3 scripts/3_runtime/15_validate_e7_fusions.py \
  --runtime-binary build/campp_reference_dump
```

각 family는 `--no-fusion-conv-bias-act`, `--no-fusion-bn-relu-quant`,
`--no-fusion-pool-cam`, `--no-fusion-qdq-elementwise`,
`--no-fusion-stats-pooling`으로 독립 비활성화할 수 있다. 생성 bundle은
`runs/runtime/kernel_optimization/e7/bundle/`, bitwise 검증은
`results/runtime/fusion/e7_validation.json`에 기록된다.

```bash
python3 scripts/3_runtime/09_benchmark_runtime.py \
  --config configs/benchmark/runtime_cache_packed_98.json \
  --run-id cache_packed_98

python3 scripts/3_runtime/09_benchmark_runtime.py \
  --config configs/benchmark/runtime_e7_98.json \
  --run-id e7_fused_98
```

## QRB2210 End-to-end 성능 검증: 09

`09_benchmark_runtime.py`는 06-08의 정확도 검증을 반복하지 않는다. 기존 결과의
SHA-256만 기록하고, QRB2210에서 canonical INT8 ONNX와 Tensor Arena C Runtime에
동일한 FBank float32 payload를 공급한다.

측정 범위:

- cold: 새 프로세스의 모델/context 초기화와 첫 inference
- warm: 20회 warm-up 뒤 100회 inference
- p50·p95·p99, RTF, CV, cold/warm Peak RSS
- Arena, weights, plan, model, dataset 및 입력 payload SHA-256

WAV read와 FBank 생성은 두 backend 공통의 측정 제외 구간이다. 실제 데이터에서
미리 만든 `[1,frames,80]` little-endian float32 파일을
`benchmarks/campplus/manifests/runtime_features.json`에 등록한다. 형식은
`benchmarks/campplus/runtime_feature_manifest.schema.json`에 정의되어 있다.

```json
{
  "schema_version": 1,
  "preprocessing": {
    "implementation": "배포 파이프라인의 고정 FBank 구현",
    "crop_padding_policy": "고정한 정책과 버전을 여기에 기록"
  },
  "features": [
    {
      "input_id": "positive__speaker_identify_3sec_positive_jonah_id_0",
      "bucket_frames": 298,
      "audio_seconds": 3.0,
      "path": "benchmarks/campplus/features/positive__speaker_identify_3sec_positive_jonah_id_0__298.f32",
      "sha256": "64자리 소문자 SHA-256"
    }
  ]
}
```

정식 실행 전에는 15개 `latency_selected=1` 입력 각각에 대해 98·298·498·998
frame 항목이 모두 있어야 한다.

이 payload와 두 manifest는 `scripts/dataset_make/build_runtime_features.py`가
만든다. **09보다 먼저 실행해야 한다.** `data/multi_speaker`의 화자별 3개 클립을
이어붙여(원본이 4초 내외라 단일 클립으로는 498/998을 못 채운다) offset 0에서
1·3·5·10초를 자르고, 배포 파이프라인의 FBank를 그대로 적용한다. zero-padding은
쓰지 않는다.

```bash
python3 scripts/dataset_make/build_runtime_features.py

CFLAGS="-std=c11 -O3 -DNDEBUG -Wall -Wextra" \
  bash scripts/3_runtime/03_build_reference_runtime.sh

python3 scripts/3_runtime/09_benchmark_runtime.py \
  --config configs/benchmark/runtime_qrb2210.json \
  --preflight-only

python3 scripts/3_runtime/09_benchmark_runtime.py \
  --config configs/benchmark/runtime_qrb2210.json
```

Dense slab 대 cache-packed A/B 평가는 같은 Release binary로 다음 두 설정을 각각
실행한다. 두 설정 모두 위의 고정 98-frame 입력 세 개만 선택한다.

```bash
python3 scripts/3_runtime/09_benchmark_runtime.py \
  --config configs/benchmark/runtime_dense_slab_98.json \
  --run-id dense_slab_98

python3 scripts/3_runtime/09_benchmark_runtime.py \
  --config configs/benchmark/runtime_cache_packed_98.json \
  --run-id cache_packed_98
```

> **비용 주의.** cpu_reference backend는 1초 음성 추론이 약 35초(RTF 35.5)다.
> 09의 정식 프로토콜은 (input, bucket)마다 130회 추론을 요구하므로 15개 입력 ×
> 4 bucket이면 C 쪽만 약 15일이 걸린다(ORT는 2.5시간). 추론 시간만 빠르게 보려면
> `experiments/rtf/`의 축소 루프를 쓴다.

결과는 `runs/runtime/benchmarks/<run-id>/benchmark_summary.json`과 backend별 raw
JSON에 저장된다. 측정 중 backend 순서는 입력마다 교차해 일방적인 thermal
순서 편향을 줄인다.

현재 `cpu_reference`는 실제 단일 thread이므로 공식 설정도 `threads=1`이다.
4-thread 설정은 worker가 구현되어 C 실행 파일의 `effective_threads`가 4가 된
뒤에만 허용된다. 환경 변수나 affinity만 4로 바꾸면 실행 파일이 오류로 거부한다.
