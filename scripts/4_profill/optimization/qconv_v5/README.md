# QConv v5 profiling

This folder runs the v5 staging candidate with the same E7 protocol used for
v4.  The initial v5 candidate delegates to v4, so latency should match v4 until
MAC-side kernels replace the dispatch path.

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
