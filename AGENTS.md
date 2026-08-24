# Repository Guidelines

## Project Structure & Module Organization

- `src/python/runtime_bundle_exporter/` builds compiled runtime bundles, memory plans, packed weights, and fusion plans.
- `src/c/runtime/` contains the portable reference backend, AArch64/NEON kernels, model loading, execution, memory management, and command-line tools.
- `scripts/1_benchmark/`, `scripts/2_graph/`, and `scripts/3_runtime/` implement the benchmark, graph-export, and C-runtime pipelines. Keep reusable logic in `src/`, not in scripts.
- `tests/runtime/` separates kernel, operator-replay, end-to-end, memory-safety, arena, dense-slab, and fusion tests.
- `configs/` stores reproducible runtime and benchmark profiles. `benchmarks/` and `models/` hold canonical inputs and models; `runs/` holds generated runs, while `results/` holds compact comparison summaries.

## Build, Test, and Development Commands

Use Linux or the QRB2210 board for the production C build:

```bash
bash scripts/3_runtime/03_build_reference_runtime.sh
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/3_runtime/06_run_reference_pipeline.py --config configs/runtime/reference.json --dry-run
python3 scripts/3_runtime/14_export_e7_fused_bundle.py --force
python3 scripts/3_runtime/15_validate_e7_fusions.py --runtime-binary build/campp_reference_dump
```

The build script compiles runtime tools directly with GCC. The dry run checks pipeline paths without writing artifacts. Consult `scripts/3_runtime/README.md` before running long board benchmarks.

## Coding Style & Naming Conventions

Use four spaces in Python, type hints for public interfaces, `snake_case` for modules/functions, and `PascalCase` for classes. Follow existing C11 style: four-space indentation, `campp_` function prefixes, `Campp` type prefixes, and uppercase constants. Keep kernel IDs and binary-format changes synchronized between Python writers, C headers, loaders, and validators. Prefer small, single-purpose planner and kernel files.

## Testing Guidelines

Name Python tests `test_*.py` and use `unittest`; name C tests `test_*.c`. Every numerical optimization must compare retained tensors against the cache-packed or ORT reference, not only the final embedding. Report bitwise results, tolerances, backend name, input bucket, and fixed feature inputs. Add memory-safety coverage when changing aliases, lifetimes, arena offsets, or scratch buffers.

## Generated Artifacts & Configuration

Do not commit build products, tensor dumps, virtual environments, or validation scratch files. Keep large outputs under ignored `runs/` paths and commit only intentional manifests, plans, or compact JSON summaries. Never embed machine-specific absolute paths or Windows backslashes in portable bundle manifests.

## Commit & Pull Request Guidelines

History contains both descriptive commits (`Implement E7 fused runtime kernels and validation`) and terse placeholders; use the descriptive form: imperative, scoped, and outcome-focused. Pull requests should state the affected backend/buckets, commands run, accuracy result, latency/RTF/RSS deltas, and any binary-format compatibility impact. Link related issues and include tables rather than screenshots for benchmark results.
