# CAM++ ORT activation-plan baseline

This experiment reuses `experiments/baseline_ort_c/ort_benchmark.c`. The RSS
therefore belongs to the native ONNX Runtime C API process; Python and NumPy are
not part of the measured process.

It answers two separate questions for each fixed `[1, 98, 80]` input:

1. Does ORT actually reuse a memory pattern on the second call in the same
   session?
2. How much of the native process RSS is the estimated activation-pattern
   block, and how much is everything else (ORT code, weights, prepacked data,
   thread pools, allocator reserve, input/output, and process overhead)?

## Inputs

- `benchmarks/campplus/features/multi__speaker_0000__98.f32`
- `benchmarks/campplus/features/multi__speaker_0005__98.f32`
- `benchmarks/campplus/features/multi__speaker_0006__98.f32`

The model is `results/static/campp_static_98.onnx`. The default is four intra-op
threads, sequential execution, graph optimization `all`, and CPU arena enabled.

## Run on the target device

First rebuild the existing native benchmark after pulling this change:

```bash
bash experiments/baseline_ort_c/build.sh
python3 experiments/baseline_activation_plan/measure_activation_plan.py
```

Use `--cpu -1` if CPU affinity is unavailable. Results are written to
`experiments/baseline_activation_plan/result/<UTC>__activation_plan.json`.

## Decision rule

Each input is run in two fresh native processes: memory pattern ON and OFF.
Each process calls the same session exactly twice. `activation_plan_running` is
true only when all output hashes match and the ON second call uses fewer
allocator events than both the ON first call and the OFF second call.

`VmHWM` is cumulative, and the CPU arena may retain pages after call 1. For that
reason a flat call-1/call-2 RSS is not evidence that the plan failed. The plan
decision uses the session CPU allocator's counter deltas. RSS is used only for
the process-level memory decomposition.

The activation estimate is the ON call-2 allocator byte delta minus the model
output bytes. It is an empirical estimate of ORT's internal memory-pattern
block, not an exact tensor-by-tensor export. The JSON retains every raw counter
and RSS snapshot so this assumption can be audited. The result also includes
the existing 98-frame offline tensor-arena size (2,007,040 bytes) for comparison.
