# E7 Operator Profiling

E7 Bucket 98의 `kernel->run()` exclusive wall time을 Operator ID별로
측정하고, end-to-end latency의 누적 80%를 차지하는 최소 Operator 집합을 만든다.

C 계측기는 시간과 최소 식별자만 출력한다. opcode, dtype, Tensor·weight shape와
fusion 원본 정보는 기존 `read_execution_plan()`과
`fusion_plans/fusion_98.json`을 Python에서 조인한다.

## 측정 범위

```text
Tensor view 준비       제외
bounds check           제외
clock start
kernel->run()          포함
clock end
bounds check/callback  제외
```

end-to-end 시간은 `campp_graph_execute()` 전체를 별도로 감싼다. 따라서
`end-to-end - kernel exclusive 합`은 executor와 bounds check 등을 포함하는
`unattributed_runtime`으로 남는다.

## 빌드

Linux 또는 QRB2210 보드에서 실행한다.

```bash
bash scripts/4_profill/01_build_profiler.sh
build/profill/test_operator_profiler
build/profill/test_profiled_graph_executor
build/profill/test_final_candidate_suite
python3 -m unittest tests.runtime.profill.test_profile_statistics
```

기본 빌드 옵션은 두 실행 파일 모두
`-std=c11 -O3 -DNDEBUG -Wall -Wextra`이다.

## QRB2210 컴파일 옵션 Quick matrix

통합 `final_candidate_suite`의 C 코드를 복제하지 않고 컴파일 옵션만 단계별로
바꿔 빌드·측정한다. O2/O3, CPU target, LTO, inline, loop unroll, frame pointer,
section GC/strip을 순서대로 비교하며 fast-math는 별도 설정으로 격리한다.

```bash
bash scripts/4_profill/compiler/01_run_matrix.sh --preflight-only
bash scripts/4_profill/compiler/01_run_matrix.sh --run-id compile_quick_01
python3 scripts/4_profill/compiler/03_compare_matrix.py --run-id compile_quick_01
```

후보별 build와 raw 측정은 `build/`·`runs/` 아래에 격리되고, 선정 근거만
`results/profiling/e7_98/compiler_matrix/<run-id>/`에 저장된다. 상세한 재개,
최종 후보 profile, fast-math 명령은 `compiler/README.md`를 참고한다. 기본은
6개 조합의 Quick이며, 기존 18개 전수조사는 `--mode full`로 실행한다.

## 실행 모드

기본값은 병목을 빠르게 찾기 위한 Quick 모드다.

| 모드 | Warm-up | Profile 반복/입력 | Baseline 반복/입력 | 용도 |
|---|---:|---:|---:|---|
| `quick` | 5 | 20 | 5 | 약 15분의 1차 Top 80% 탐색 |
| `official` | 20 | 100 | 100 | 최종 성능 증거 |

Quick도 고정 입력 3개를 모두 사용하므로 Operator마다 60개 sample을 수집한다.
Quick의 overhead 판정은 입력당 baseline 5회로 계산한 예비 판정이며 결과에
`preliminary=true`로 표시된다.

## 사전 검사

```bash
python3 scripts/4_profill/02_profile_e7.py --preflight-only
```

다음을 강제한다.

- Bucket 98
- CPU affinity `[0]`
- 실제 thread 1개
- Quick: warm-up 5회, 측정 20회, overhead baseline 5회
- Official: warm-up 20회, 측정 및 baseline 100회
- 입력 `multi__speaker_0000`, `0005`, `0006`
- 기존 E7 bitwise validation 통과

## 정식 실행

```bash
python3 scripts/4_profill/02_profile_e7.py
```

실행을 시작하면 기존 E7 p50을 기준으로 예상 시간을 한 번 표시한다. 측정 중에는
입력 ID나 backend 로그 대신 `진행 중`만 출력하고, 완료 후 latency, overhead,
Top 80%, kernel별 비중과 상위 Operator를 상세히 출력한다. 같은 분석은
`summary.json`의 `analysis`에도 구조화해 저장한다.

최종 공식 측정이 필요할 때만 다음을 실행한다.

```bash
python3 scripts/4_profill/02_profile_e7.py --mode official --force
```

기존 결과를 의도적으로 교체할 때만 `--force`를 사용한다.

Raw 결과:

```text
runs/profiling/e7_98/raw/
```

최종 결과:

```text
results/profiling/e7_98/operator_profile.json
results/profiling/e7_98/operator_profile.csv
results/profiling/e7_98/summary.json
```

`Top bottleneck set`은 Operator exclusive time 내림차순의 최소 prefix이며,
누적 end-to-end 비중이 80%에 도달해야 `complete=true`가 된다. 모든 kernel
exclusive time을 더해도 80% 미만이면 이를 100%로 재정규화하지 않고
`complete=false`와 `unattributed_runtime`을 보고한다.

계측 overhead는 같은 Release 옵션의 비계측 binary와 비교한다. 1% 이상이면
결과 파일은 생성하지만 프로세스는 종료 코드 3을 반환하고
`profile_valid=false`로 기록한다.

## 상위 4개 kernel 내부 진단

전체 profile에서 확인된 일반 QConv, fused Quant-QConv, fused BN-ReLU-Quant,
DequantizeLinear를 최적화하기 전 다음 진단을 실행한다. QConv는 3x3과 1x1을
서로 다른 대표 shape로 측정하므로 실행 case는 총 5개다.

```bash
bash scripts/4_profill/optimization/01_build_optimization.sh
build/profill/optimization/test_optimization_probe

python3 scripts/4_profill/optimization/02_diagnose_top4.py --preflight-only
python3 scripts/4_profill/optimization/02_diagnose_top4.py
```

microbenchmark는 실제 feature로 target 직전 Operator까지 한 번 실행하여 중간
Tensor를 만들고, 입력을 snapshot한 뒤 target `kernel->run()`만 반복한다.
입력 복원, 출력 초기화, output hash 계산은 측정 구간 밖이다. Linux PMU 권한이
있으면 cycles, instructions, cache/branch miss도 target 호출에 한해 기록한다.
PMU 접근이 제한되어도 단계별 wall time 진단은 계속된다.

Raw 결과:

```text
runs/profiling/e7_98/optimization/diagnosis/raw/
```

요약 결과:

```text
results/profiling/e7_98/optimization/diagnosis.json
```

진단용 clock 호출은 공식 E2E profiler에 포함되지 않으므로 기존 profiling
overhead 1% 판정에 영향을 주지 않는다. 후보 구현은 진단 결과가 생성된 이후
`src/c/profill/optimization/candidates/`에서 격리 검증한다.

후보 binary로 같은 진단을 별도 경로에 실행한 뒤 baseline과 비교한다.

```bash
python3 scripts/4_profill/optimization/03_compare_candidate.py \
  --baseline results/profiling/e7_98/optimization/diagnosis.json \
  --candidate results/profiling/e7_98/optimization/candidate_diagnosis.json
```

대상 case가 개선되고 다른 case의 회귀가 1% 이내이며 세 입력의 output hash가
bitwise 동일할 때만 `candidate_microbench_gate_passed=true`가 된다. 이 판정
이후에도 전체 retained tensor 검증과 Quick E2E profiling은 별도로 통과해야 한다.

## QConv MAC 대 주소 비용 분리

`qconv_mac_address` 단계가 지배적이어도 그 결과만으로 실제 MAC이 병목인지,
좌표 계산·분기·입력/weight load가 병목인지 판단할 수 없다. 다음 측정은 기존
profile에서 선택한 `qconv_3x3`(Operator 2)와 `qconv_1x1`(Operator 825)을
그대로 사용하고, target kernel 구간의 cycle sample을 소스 라인별로 분류한다.

```bash
bash scripts/4_profill/optimization/01_build_optimization.sh

python3 scripts/4_profill/optimization/04_profile_qconv_hotspot.py \
  --preflight-only
python3 scripts/4_profill/optimization/04_profile_qconv_hotspot.py
```

`campp_operator_hotspot`은 내부 stage clock을 컴파일하지 않은 sampling 전용
binary다. 외부 `perf record`는 프로세스 전체를 실행하지만 C의
`PR_TASK_PERF_EVENTS_DISABLE/ENABLE` 경계 때문에 반복 중
`run_target_kernel()`에서만 sample을 수집한다. 전처리, target 입력 복원,
warm-up, 출력 hash 계산은 표본에 포함되지 않는다. 보드에 Linux `perf`와
userspace cycle sampling 권한이 필요하며, 빌드의 `-g` line 정보로 표본을
분류한다.

각 shape를 고정 입력 3개로 검사한다. 모든 입력에서 같은 항목이 55% 이상이고
미분류 표본이 20% 이하일 때만 결과를 안정적이라고 판정한다.
기본값은 기존 Quick 진단과 같은 warm-up 5회·측정 20회이며, 최종 확인이
필요하면 `--repeat 100 --force`로 표본을 늘린다.

```text
address_load_control  좌표·offset 계산, bounds/valid 분기, 입력·weight load
mac_reduction         4-lane dot product와 accumulator 갱신
requant_write         scale, rounding, quantized output write
setup_other           QConv 함수 내부의 나머지 setup
unclassified          debug line으로 귀속하지 못한 sample
```

Raw 결과:

```text
runs/profiling/e7_98/optimization/qconv_hotspot/
```

요약 결과:

```text
results/profiling/e7_98/optimization/qconv_hotspot/qconv_hotspot.json
results/profiling/e7_98/optimization/qconv_hotspot/qconv_hotspot.csv
```

hotspot 결과는 address와 MAC 중 구현 순서를 정하는 근거로 사용한다. 두 후보를
모두 구현하는 경우에도 각각을 독립 측정한 뒤 combined 결과를 확인하며,
shape별 효과가 다르면 3x3과 1x1 dispatch를 분리한다.

## QConv 최적화 후보 비교

QConv 후보는 production kernel을 수정하지 않고 target Operator에서만
선택한다. 빌드 후 C equivalence test를 먼저 실행한다.

```bash
bash scripts/4_profill/optimization/01_build_optimization.sh
build/profill/optimization/test_qconv_candidate
```

기존 QConv, address-only, MAC-only, combined를 같은 두 shape와 세 입력으로
측정한다. 각 모드는 별도 raw/result 경로를 사용한다.

```bash
for mode in baseline address mac combined; do
  python3 scripts/4_profill/optimization/02_diagnose_top4.py \
    --qconv-only \
    --qconv-candidate "${mode}" \
    --runs-dir "runs/profiling/e7_98/optimization/qconv_candidates/${mode}" \
    --output "results/profiling/e7_98/optimization/qconv_candidates/${mode}.json" \
    --force
done
```

후보별 bitwise hash와 latency gate는 기존 비교기를 그대로 사용한다.

```bash
for mode in address mac combined; do
  python3 scripts/4_profill/optimization/03_compare_candidate.py \
    --baseline results/profiling/e7_98/optimization/qconv_candidates/baseline.json \
    --candidate "results/profiling/e7_98/optimization/qconv_candidates/${mode}.json" \
    --output "results/profiling/e7_98/optimization/qconv_candidates/${mode}_comparison.json" \
    --force
done
```

`combined`가 세 입력에서 bitwise 동일하고 두 shape 모두 회귀 없이 개선된
경우에만 production QConv의 fast path로 승격한다. 기존 구현은 지원하지 않는
layout을 위한 fallback으로 유지한다.

## QConv fixed microkernel과 compiler spill 비교

현재 `mac` v2를 보존한 상태에서 E7의 완전한 내부 tile만 `mac_fixed`로
교체한다. 고정 경로는 `tile_count=4`인 half tile 두 개, `valid_outputs=8`,
UINT8 input, INT8 O4I4 weight를 사용하며 dtype·tail·NULL padding 분기를 hot
loop 밖으로 이동한다. 경계와 tail은 기존 `mac`으로 fallback한다.

QRB2210에서 GCC와 Clang을 동일한 `-O3 -mcpu=native` 조건으로 한 번에
빌드하고 두 QConv shape와 세 입력을 비교한다.

```bash
python3 scripts/4_profill/optimization/07_compare_qconv_spill.py \
  --preflight-only

python3 scripts/4_profill/optimization/07_compare_qconv_spill.py \
  --force
```

기본 실행은 compiler별 `mac`과 `mac_fixed`를 측정한다. 선택된 fixed
intrinsics의 최대 annotated stack spill이 5%를 넘으면
`decision.assembly_required=true`가 된다. 그때만 assembly까지 추가 측정한다.

```bash
python3 scripts/4_profill/optimization/07_compare_qconv_spill.py \
  --include-assembly \
  --force
```

```text
build/profill/optimization/qconv_spill/{gcc,clang}/
runs/profiling/e7_98/optimization/qconv_spill/{gcc,clang}/{mac,mac_fixed,mac_asm}/
results/profiling/e7_98/optimization/qconv_spill/compiler_comparison.json
results/profiling/e7_98/optimization/qconv_spill/compiler_comparison.csv
results/profiling/e7_98/optimization/qconv_spill/decision.json
```

선택 순서는 output hash bitwise 일치, shape별 1% 초과 회귀 없음, 두 shape의
합산 mean latency, spill 비중이다. spill이 더 낮다는 이유만으로 느린 compiler를
선택하지 않는다. production 승격 전 retained tensor bitwise와 Quick E2E
검증은 별도로 수행한다.

## Fused Quant-QConv 후보 비교

fused 후보는 production의 validation과 scratch 구성을 그대로 사용한다. `mac`,
`combined`, `mac_fixed`는 packed QConv runner만 교체하고, `quant_neon`은 QConv를
baseline으로 유지한 채 FP32→UINT8 pass만 교체한다. `combined_fixed`는
`quant_neon`과 `mac_fixed`를 함께 적용한다. 일반 QConv 코드는 복사하지 않는다.

```bash
bash scripts/4_profill/optimization/01_build_optimization.sh
build/profill/optimization/test_qconv_candidate

for mode in baseline mac_fixed quant_neon combined_fixed; do
  python3 scripts/4_profill/optimization/02_diagnose_top4.py \
    --fused-qconv-only \
    --fused-qconv-candidate "${mode}" \
    --runs-dir "runs/profiling/e7_98/optimization/fused_qconv_candidates/${mode}" \
    --output "results/profiling/e7_98/optimization/fused_qconv_candidates/${mode}.json" \
    --force
done

for mode in mac_fixed quant_neon combined_fixed; do
  python3 scripts/4_profill/optimization/03_compare_candidate.py \
    --baseline results/profiling/e7_98/optimization/fused_qconv_candidates/baseline.json \
    --candidate "results/profiling/e7_98/optimization/fused_qconv_candidates/${mode}.json" \
    --output "results/profiling/e7_98/optimization/fused_qconv_candidates/${mode}_comparison.json" \
    --force
done
```

세 입력의 output hash가 bitwise 동일해야 하며, 후보 적용 뒤
`fused_input_quantize`가 새 병목으로 커지는지는 stage 비중으로 다시 확인한다.
microbench 통과 후에도 retained tensor bitwise와 Quick E2E 검증이 필요하다.

일반 `qlinear_conv_o4i4_neon` 115개를 `mac_fixed/v4/v5`로 전수 비교할 때는
batch family runner를 사용한다. 입력당 모델을 한 번만 읽고 graph를 한 번
순회하며, 각 QConv에서 후보를 연속 측정한 뒤 baseline 출력을 다음 Operator에
전달한다. 기존 Operator별 runner의 1,380회 process/model-load 반복은 없다.

```bash
BUILD_TARGET=qconv_family_batch \
  bash scripts/4_profill/optimization/01_build_optimization.sh

python3 scripts/4_profill/optimization/12_benchmark_qconv_family.py \
  --modes baseline mac_fixed v4 v5 \
  --preflight-only

python3 scripts/4_profill/optimization/12_benchmark_qconv_family.py \
  --modes baseline mac_fixed v4 v5 \
  --force
```

raw batch payload는 `runs/profiling/e7_98/optimization/qconv_family/raw_batch/`,
mode별 결과와 비교표는 `results/profiling/e7_98/optimization/qconv_family/`에
기록된다. 세 입력의 모든 Operator output hash가 baseline과 bitwise 동일해야
후속 layer-hybrid plan의 후보가 된다.

대표 Operator가 아니라 profile의 `fused_quant_qlinear_conv_o4i4` 전체를 비교할
때는 family runner를 사용한다. 현재 E7 profile에서는 110개 Operator가 대상이다.
이 runner는 Operator마다 graph prelude를 다시 실행하지 않는다. 입력당 graph를
한 번 순회하면서 fused Operator를 만날 때 같은 입력 snapshot으로 모든 후보를
측정하고, 마지막 baseline 출력을 다음 Operator에 전달한다.

```bash
python3 scripts/4_profill/optimization/08_benchmark_fused_qconv_family.py \
  --preflight-only

python3 scripts/4_profill/optimization/08_benchmark_fused_qconv_family.py \
  --force
```

기본 Quick은 `baseline + combined_fixed`, warm-up 5회, 측정 20회, 입력 3개다.
원인 분리까지 필요한 경우에만 네 모드를 실행한다.

```bash
python3 scripts/4_profill/optimization/08_benchmark_fused_qconv_family.py \
  --modes baseline mac_fixed quant_neon combined_fixed --force
```

결과는 다음 경로에 생성된다.

```text
runs/profiling/e7_98/optimization/fused_qconv_family/raw/
results/profiling/e7_98/optimization/fused_qconv_family/{mode}.json
results/profiling/e7_98/optimization/fused_qconv_family/{mode}_comparison.json
results/profiling/e7_98/optimization/fused_qconv_family/comparison.csv
results/profiling/e7_98/optimization/fused_qconv_family/summary.json
```

family speedup은 Operator별 speedup 평균이 아니라 110개 baseline mean 합계를
candidate mean 합계로 나눈 값이다. 합산 p50/p95는 동기화된 whole-graph
percentile이 아니므로 E2E percentile 대신 사용하지 않는다.

### Fused Quant-QConv register spill

`combined_fixed`가 실제 S4×O8 microkernel을 실행했는지와 MAC/input-quantize의
stack spill을 확인할 때는 fused spill profiler를 사용한다. 기본값은 profile에
존재하는 6개 weight shape에서 exclusive time이 가장 큰 Operator 하나씩이다.
fixed 심볼 sample이 없으면 IC tail 등의 이유로 v2 fallback을 실행한 것으로
판정하고 `campp_qconv_mac_neon_tile_v2`를 자동으로 annotate한다.

```bash
python3 scripts/4_profill/optimization/09_profile_fused_qconv_spill.py \
  --preflight-only

python3 scripts/4_profill/optimization/09_profile_fused_qconv_spill.py \
  --force
```

Operator 0과 대표 3×3만 먼저 확인하려면 다음처럼 범위를 제한한다.

```bash
python3 scripts/4_profill/optimization/09_profile_fused_qconv_spill.py \
  --operator-ids 0 10 --force \
  --runs-dir runs/profiling/e7_98/optimization/fused_qconv_spill_smoke \
  --results-dir results/profiling/e7_98/optimization/fused_qconv_spill_smoke
```

결과는 `fused_qconv_spill.json`과 `fused_qconv_spill.csv`에 기록한다. 기본
spill gate는 세 입력의 실행 경로가 같고 MAC과 input quantize의 annotated stack
spill이 각각 5% 이하인 경우다. quantize sample이 부족하면 spill 0%가 아니라
inconclusive로 처리하며 `--sample-period`를 낮춰 다시 측정한다.

## BN 병목 분리와 후보 비교

기존 coarse 진단의 `bn_elementwise`를 index/address, parameter load,
sqrt/affine, ReLU/quantize, output write로 다시 분류한다. target symbol이
유지되는 인라인 `tensor_view.h` 표본도 address로 포함하므로 단일 C 파일만
허용하던 분류로 인해 unclassified가 커지는 문제를 피한다.

```bash
bash scripts/4_profill/optimization/01_build_optimization.sh
build/profill/optimization/test_bn_candidate

python3 scripts/4_profill/optimization/05_profile_bn_hotspot.py \
  --preflight-only
python3 scripts/4_profill/optimization/05_profile_bn_hotspot.py
```

세 입력에서 최상위 category가 같고 입력별 unclassified가 20% 이하일 때
`ready=true`다. 분류된 BN core의 누적 80%에 도달하는 최소 category 집합을
`top_bottleneck_set`으로 기록한다.

```text
runs/profiling/e7_98/optimization/bn_hotspot/
results/profiling/e7_98/optimization/bn_hotspot/bn_hotspot.json
results/profiling/e7_98/optimization/bn_hotspot/bn_hotspot.csv
```

후보 다섯 모드를 동일한 BN Operator와 세 입력으로 측정한다.

```bash
for mode in baseline address affine quant combined; do
  python3 scripts/4_profill/optimization/02_diagnose_top4.py \
    --bn-only \
    --bn-candidate "${mode}" \
    --runs-dir "runs/profiling/e7_98/optimization/bn_candidates/${mode}" \
    --output "results/profiling/e7_98/optimization/bn_candidates/${mode}.json" \
    --force
done

for mode in address affine quant combined; do
  python3 scripts/4_profill/optimization/03_compare_candidate.py \
    --baseline results/profiling/e7_98/optimization/bn_candidates/baseline.json \
    --candidate "results/profiling/e7_98/optimization/bn_candidates/${mode}.json" \
    --output "results/profiling/e7_98/optimization/bn_candidates/${mode}_comparison.json" \
    --force
done
```

모든 입력 hash가 bitwise 동일하고 BN case가 1% 넘게 회귀하지 않아야 한다.
`combined`가 가장 빠르다는 가정은 하지 않고 측정 결과로 승격 후보를 정한다.

기존 `combined`를 기준으로 16-channel BN v2 후보를 비교한다. 기본 실행은
채널 수가 작은/중간/큰 대표 Operator 세 개를 선택하고 `--all-operators`는
55개 전체 family를 측정한다. `v2_prescaled`는 실험 결과를 기록하지만 exact
두 모드가 bitwise와 회귀 gate를 통과해야만 승격 준비 상태가 된다.

```bash
bash scripts/4_profill/optimization/01_build_optimization.sh
build/profill/optimization/test_bn_v2_candidate

python3 scripts/4_profill/optimization/11_benchmark_bn_v2.py \
  --preflight-only
python3 scripts/4_profill/optimization/11_benchmark_bn_v2.py --force
python3 scripts/4_profill/optimization/11_benchmark_bn_v2.py \
  --all-operators --force
```

```text
runs/profiling/e7_98/optimization/bn_v2/raw/
results/profiling/e7_98/optimization/bn_v2/summary.json
results/profiling/e7_98/optimization/bn_v2/comparison.csv
```

## DequantizeLinear 병목 분리와 후보 비교

기존 `dequant_elementwise`를 index/address, input load, parameter load,
convert/multiply, output store로 분리한다. 원소마다 clock을 호출하지 않고 기존
target-only `perf record`와 source-line 분류기를 사용한다.

```bash
bash scripts/4_profill/optimization/01_build_optimization.sh
build/profill/optimization/test_dequant_candidate

python3 scripts/4_profill/optimization/06_profile_dequant_hotspot.py \
  --preflight-only
python3 scripts/4_profill/optimization/06_profile_dequant_hotspot.py
```

```text
runs/profiling/e7_98/optimization/dequant_hotspot/
results/profiling/e7_98/optimization/dequant_hotspot/dequant_hotspot.json
results/profiling/e7_98/optimization/dequant_hotspot/dequant_hotspot.csv
```

후보 다섯 모드는 동일한 DequantizeLinear Operator와 세 입력으로 측정한다.

```bash
for mode in baseline address parameter scalar_combined neon_combined; do
  python3 scripts/4_profill/optimization/02_diagnose_top4.py \
    --dequant-only \
    --dequant-candidate "${mode}" \
    --runs-dir "runs/profiling/e7_98/optimization/dequant_candidates/${mode}" \
    --output "results/profiling/e7_98/optimization/dequant_candidates/${mode}.json" \
    --force
done

for mode in address parameter scalar_combined neon_combined; do
  python3 scripts/4_profill/optimization/03_compare_candidate.py \
    --baseline results/profiling/e7_98/optimization/dequant_candidates/baseline.json \
    --candidate "results/profiling/e7_98/optimization/dequant_candidates/${mode}.json" \
    --output "results/profiling/e7_98/optimization/dequant_candidates/${mode}_comparison.json" \
    --force
done
```

세 입력의 output hash가 bitwise 동일하고 1% 초과 회귀가 없어야 한다. 최종
후보는 retained tensor bitwise와 Quick E2E 검증을 통과한 뒤에만 production
AArch64 registry로 승격한다.

## 나머지 10개 연산 공통 후보

Add, ReLU, Expand, Slice, QuantizeLinear, ReduceMean,
`fused_dequant_sigmoid_mul`, AveragePool, Reshape,
`fused_statistics_pooling`에 공통 pointer/NEON/reduction/copy/LUT 후보를 적용한다.

```bash
bash scripts/4_profill/optimization/01_build_optimization.sh
build/profill/optimization/test_remaining_candidates

python3 scripts/4_profill/optimization/07_benchmark_remaining_ops.py \
  --preflight-only
python3 scripts/4_profill/optimization/07_benchmark_remaining_ops.py
```

긴 실행 전에 일부 kernel만 확인할 수 있다.

```bash
python3 scripts/4_profill/optimization/07_benchmark_remaining_ops.py \
  --kernels add_stride relu_stride quantize_linear_stride \
  --force
```

결과는 아래에 저장한다.

```text
runs/profiling/e7_98/optimization/remaining_candidates/
results/profiling/e7_98/optimization/remaining_candidates.json
results/profiling/e7_98/optimization/remaining_candidates.csv
```

세 입력의 output hash가 모두 bitwise 동일하고 어떤 case도 1%를 초과해 회귀하지
않아야 microbench gate를 통과한다. 이 단계는 production registry나 bundle fusion
plan을 수정하지 않는다.

## 최종 후보 통합 E2E 재프로파일

개별 후보 검증이 끝나면 `01_build_profiler.sh`가 stock binary와 별도로 최종
후보를 연결한 비계측/계측 binary 쌍을 만든다. 최종 registry는 production
AArch64 registry를 복사한 뒤 아래 15개 entry만 교체한다.

| 대상 | 최종 mode |
|---|---|
| 일반 QConv (V2) | `v4` |
| fused Quant-QConv (V2) | `combined_hybrid` |
| 일반 QConv (V3) | operator별 `mac_fixed/v4/v5` hybrid |
| fused Quant-QConv (V3) | operator별 `combined_fixed/combined_hybrid/combined_v5` hybrid |
| fused BN-ReLU-Quant | `v2_spatial2` |
| DequantizeLinear | `neon_combined` |
| fused Dequant-ReLU-Quant | `neon` |
| 나머지 10개 연산 | `optimized` |

현재 production runtime source의 requant NEON 구현은 두 binary에 공통으로
컴파일된다. 교체하지 않은 entry는 production registry의 함수와 scratch query를
그대로 유지한다. `RuntimeContext`는 이 최종 registry로 생성되므로 microbench
target 우회가 아니라 실제 E7 graph executor 전체가 실행된다.

```bash
bash scripts/4_profill/04_build_final_v2.sh
build/profill/final_v2_aggressive/test_final_candidate_suite

bash scripts/4_profill/03_profile_final_e7.sh --preflight-only
bash scripts/4_profill/03_profile_final_e7.sh
```

`04_build_final_v2.sh`는 compiler matrix winner인 GCC
`-O3 -mcpu=cortex-a53 -flto`와 aggressive packaged flag, linker GC 및 strip을
재현한다. runtime, profiler, suite test와 retained-Tensor dump를 같은 source와
compile/link flag로 만든다.

최종 공식 측정은 다음과 같다.

```bash
bash scripts/4_profill/03_profile_final_e7.sh --mode official --force
```

### Layer-hybrid final V3와 V2 비교

V3는 `conv_hybrid_plan.json`에서 1% 이상 빠르고 bitwise gate를 통과한
operator만 교체한다. bucket 98에서 일반 QConv는 `v5 112 / mac_fixed 2 /
v4 1`, fused Quant-QConv는 `combined_v5 55 / combined_hybrid 50 /
combined_fixed 5`로 dispatch한다. 측정된 V3 plan이 없는 bucket만 V2 경로로
안전하게 fallback한다.

### 298/498/998 layer-hybrid V3 선택

추가 bucket은 98의 operator ID 표를 복사하지 않는다. 각 bucket의 고정 feature
3개에서 일반 QConv(`mac_fixed/v4/v5`)와 fused Quant-QConv
(`combined_fixed/combined_hybrid/combined_v5`)를 한 graph traversal로 측정하고,
bitwise-valid 후보 중 incumbent보다 1% 이상 빠른 mode만 선택한다. baseline은
output hash 확인용 1회만 실행하므로 후보 측정 시간을 지배하지 않는다.

operator ID는 bucket마다 다르므로 먼저 각 `plan_<bucket>.bin`으로 해당 bucket의
operator profile을 만든다. profile과 plan의 SHA-256 및 전체 operator의
ID/opcode/kernel/Tensor/weight shape가 일치해야 family 측정을 시작한다.

```bash
python3 scripts/4_profill/optimization/16_select_multibucket_v3.py \
  --buckets 298 498 998 --profiles-only

python3 scripts/4_profill/optimization/16_select_multibucket_v3.py \
  --buckets 298 498 998 --mode official --preflight-only

python3 scripts/4_profill/optimization/16_select_multibucket_v3.py \
  --buckets 298 498 998 --mode official --build-final
```

family benchmark binary 2개가 이미 있으면 optimization build는 재사용한다.
없을 때만 `01_build_optimization.sh`를 실행하며, 강제 재빌드는 `--rebuild`로
요청한다. `--preflight-only`는 빌드하지 않고 누락을 보고한다. 측정 재실행은
`--force`, 기존 결과에서 plan/source만 다시 만드는
경우는 `--plan-only --force --build-final`을 사용한다.

산출물은 bucket별로 아래에 생성된다.

```text
runs/models/campplus/final_v3/layer_selection/<bucket>/
results/models/campplus/final_v3/layer_selection/<bucket>/operator_profile/
results/models/campplus/final_v3/layer_selection/<bucket>/qconv_family/
results/models/campplus/final_v3/layer_selection/<bucket>/fused_qconv_family/
results/models/campplus/final_v3/layer_selection/<bucket>/conv_hybrid_plan.json
results/models/campplus/final_v3/layer_selection/<bucket>/conv_hybrid_plan.csv
```

`17_generate_multibucket_v3_source.py`는 98과 추가 bucket plan을 다시 검증한 뒤
`conv_layer_hybrid_plan.c`의 bucket별 정렬 테이블을 생성한다. 측정되지 않은
bucket은 계속 일반 QConv `v4`, fused Quant-QConv `combined_hybrid`로 fallback한다.

V2 빌드는 기존 binary의 capability가 V2 suite와 일치하면 다시 만들 필요가 없다.
V3만 빌드한 뒤 단일 스크립트로 retained tensor gate와 동일 세션 Quick E2E를
진행한다.

```bash
bash scripts/4_profill/05_build_final_v3.sh
build/profill/final_v3_hybrid/test_final_candidate_suite

bash scripts/4_profill/06_profile_final_v2_v3.sh --preflight-only
bash scripts/4_profill/06_profile_final_v2_v3.sh --mode quick --force
```

`06_profile_final_v2_v3.sh`의 순서는 다음과 같다.

1. 같은 plan/weights와 고정 feature 3개로 모든 retained tensor를 byte 단위 비교
2. final V2 Quick E2E/profile
3. final V3 Quick E2E/profile
4. E2E RTF와 operator delta 자동 비교

공식 프로토콜은 아래와 같다.

```bash
bash scripts/4_profill/06_profile_final_v2_v3.sh --mode official --force
```

산출물은 다음 위치에 저장한다.

```text
results/profiling/e7_98/final_v3_vs_v2_retained.json
results/profiling/e7_98/final_v2_aggressive/
results/profiling/e7_98/final_v3_hybrid/
results/profiling/e7_98/final_v3_vs_v2.json
```

### 현재 E7 성능 baseline

이전 `mac_fixed+combined_fixed+BN combined` suite의 2026-08-19 Quick 결과를
새 final v2의 회귀 비교 기준선으로 사용한다. canonical 기준선은
`results/profiling/e7_98/baseline.json`, 원본 profile은
`results/profiling/e7_98/final_combined/summary.json`이다.

| 지표 | 기준값 |
|---|---:|
| RTF p50 | 0.522350 (표기 0.52) |
| E2E mean / p50 / p95 | 522.322 / 522.350 / 523.983 ms |
| CV | 0.167% |
| 표본 | 60 |
| 이전 stock Quick 대비 | 16.22x |

이 값은 `quick-baseline`이며 공식 프로토콜 결과는 아니다. 성능 회귀 비교의 현재
기준으로는 사용하되, 공식 수치가 필요할 때는 `--mode official --force`로 다시
측정한다. final suite 예상 시간 계산도 비계측 mean 522.329 ms를 사용한다.

final binary는 `--capabilities`와 결과 JSON에 `optimization_suite`와 정확한
`optimization_suite_config`를 기록한다. runner는 비계측 binary와 계측 binary의
두 값이 모두 같은지 실행 전에 확인하므로, 과거 final runtime과 새 final v2
profiler를 섞은 잘못된 overhead 비교를 거부한다.

```text
build/profill/final_v2_aggressive/campp_runtime_benchmark_final
build/profill/final_v2_aggressive/campp_e7_profiler_final
runs/profiling/e7_98/final_v2_aggressive/raw/
results/profiling/e7_98/final_v2_aggressive/
```
