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

앞 단계에서 만든 canonical bundle과 `results/runtime/compare_*.json`을 검증한 뒤
Phase 4가 참조할 최종 결과 세 파일을 만든다.

```bash
python3 scripts/3_runtime/07_freeze_reference_results.py
```

`07`은 추론을 다시 실행하지 않는다. 대신 manifest의 canonical model, weights와
네 plan의 크기·SHA-256, kernel registry의 opcode, 네 bucket의 1,438개 Operator
결과를 교차 검증한다. 알려진 수치 실패도 삭제하지 않고 최종 보고서에 고정한다.
출력은 `results/runtime/operator_validation.json`,
`results/runtime/end_to_end_validation.json`,
`results/runtime/reference_runtime_report.md`이다.
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

결과는 `results/runtime/tensor_arena_validation.json`에 기록된다. Arena에서는
중간 Tensor 주소가 재사용되므로 `campp_reference_dump`는 graph 종료 후가 아니라
각 Operator 출력 직후 callback으로 값을 기록한다. 보드 디스크를 보호하기 위해
bucket 하나를 비교해 통과하면 큰 binary dump 두 개를 즉시 삭제한다. 실패한
bucket은 원인 분석을 위해 남기며, 통과한 dump도 보존하려면 `--keep-dumps`를 쓴다.
