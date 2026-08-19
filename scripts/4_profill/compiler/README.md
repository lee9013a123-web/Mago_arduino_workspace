# QRB2210 compiler matrix

The matrix recompiles the integrated `final_candidate_suite` with isolated
compiler options. It does not copy or fork production kernel C sources. Matrix
builds use `BUILD_SCOPE=compiler_matrix`, so only the final runtime, matching
profiler, final-suite registry test, and retained-Tensor dump harness are
compiled for each candidate.

```text
configs/runtime/
  compiler_qrb2210_strict.json
  compiler_qrb2210_fast_math.json
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
bash scripts/4_profill/compiler/01_run_matrix.sh --run-id strict_quick_01
python3 scripts/4_profill/compiler/03_compare_matrix.py --run-id strict_quick_01
```

The strict matrix advances one stage at a time: optimization level, CPU,
LTO, inline, loop unroll, frame pointer, then packaging. A candidate must keep
both the baseline embedding hashes and all retained-Tensor payload hashes, plus
the CV and p95/RSS gates, before it can win. Dump payloads are deleted after
hashing; compact counts and hashes remain in the result.

Use `--resume` after an interrupted board run. Use `--force` only when the
existing build, raw run and compact result directories for the same run ID may
be replaced.

## Operator profile for finalists

```bash
bash scripts/4_profill/compiler/04_profile_winner.sh \
  --run-id strict_quick_01
```

Only the compile baseline and final winner get the expensive operator-level
profile. Add `--mode official` after Quick and retained-tensor validation pass.

## Isolated fast-math experiment

```bash
bash scripts/4_profill/compiler/01_run_matrix.sh \
  --config configs/runtime/compiler_qrb2210_fast_math.json \
  --base-decision results/profiling/e7_98/compiler_matrix/strict_quick_01/decision.json \
  --run-id fast_math_01
```

Fast-math is never part of the strict matrix and cannot win unless all retained
tensors are bitwise-identical.

Generated builds and raw logs remain under `build/` and `runs/`. Commit only
the compact `matrix_summary.csv`, `matrix_summary.json`, and `decision.json`
when they are intentional experiment evidence.
