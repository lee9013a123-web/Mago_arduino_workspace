# QConv v5 validation and profiling

V5 is built as a separate candidate mode from V4. Runtime dispatch reuses V4
planning/requant/store and tries the v5 raw MAC kernel for full tiles first.
The V5 address provider is also built and tested, but address dispatch is a
separate gate.

```bash
bash scripts/4_profill/optimization/qconv_v5/01_build_qconv_v5.sh
python3 scripts/4_profill/optimization/qconv_v5/03_benchmark_qconv_v5.py \
    --preflight-only
python3 scripts/4_profill/optimization/qconv_v5/03_benchmark_qconv_v5.py \
    --force
```

Outputs:

```text
runs/profiling/e7_98/optimization/qconv_v5/<mode>/
results/profiling/e7_98/optimization/qconv_v5/<mode>.json
```
