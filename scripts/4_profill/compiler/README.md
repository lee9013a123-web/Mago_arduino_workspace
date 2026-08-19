# QRB2210 compiler matrix

The matrix recompiles the integrated `final_candidate_suite` with isolated
compiler options. It does not copy or fork production kernel C sources. Matrix
Quick builds compile only the final runtime and retained-Tensor dump harness.
The profiler is built later only for the baseline and winner.

```text
configs/runtime/
  compiler_qrb2210_quick.json
  compiler_qrb2210_strict.json
  compiler_qrb2210_fast_math.json
configs/benchmark/
  runtime_e7_98_compile_quick.json
scripts/4_profill/compiler/
  01_run_matrix.sh          board entry point
  02_run_matrix.py          staged build, correctness gates, Quick benchmark
  03_compare_matrix.py      decision reader
  04_profile_winner.sh      baseline/winner operator profile
build/profill/compiler_matrix/<run-id>/<variant>/
runs/profiling/e7_98/compiler_matrix/<run-id>/<variant>/
results/profiling/e7_98/compiler_matrix/<run-id>/
```

## Quick matrix

```bash
bash scripts/4_profill/compiler/01_run_matrix.sh --preflight-only
bash scripts/4_profill/compiler/01_run_matrix.sh --run-id compile_quick_01
python3 scripts/4_profill/compiler/03_compare_matrix.py --run-id compile_quick_01
```

Quick is the default. It compares four performance states (`O2`, `O3`, A53,
A53+LTO), then compares the performance winner with one aggressive packaged
state. All three fixed inputs run once per candidate with no warm-up or cold
run. Both embedding and retained-Tensor payload hashes must stay identical.
Dump payloads are deleted after hashing. At the measured ~7-second E7 latency,
inference itself is about 4-5 minutes; board compilation normally brings the
whole Quick matrix to roughly 15-30 minutes, excluding thermal cooling.

The exhaustive 18-build experiment remains available:

```bash
bash scripts/4_profill/compiler/01_run_matrix.sh \
  --mode full --run-id compile_full_01
```

Use `--resume` after an interrupted board run. Use `--force` only when the
existing build, raw run and compact result directories for the same run ID may
be replaced.

## Operator profile for finalists

```bash
bash scripts/4_profill/compiler/04_profile_winner.sh \
  --run-id compile_quick_01
```

Only the compile baseline and final winner get the expensive operator-level
profile. Missing profiler binaries are rebuilt automatically with the selected
flags. Add `--mode official` after Quick validation passes.

## Isolated fast-math experiment

```bash
bash scripts/4_profill/compiler/01_run_matrix.sh \
  --config configs/runtime/compiler_qrb2210_fast_math.json \
  --base-decision results/profiling/e7_98/compiler_matrix/compile_quick_01/decision.json \
  --run-id fast_math_01
```

Fast-math is never part of the strict matrix and cannot win unless all retained
tensors are bitwise-identical.

Generated builds and raw logs remain under `build/` and `runs/`. Commit only
the compact `matrix_summary.csv`, `matrix_summary.json`, and `decision.json`
when they are intentional experiment evidence.
