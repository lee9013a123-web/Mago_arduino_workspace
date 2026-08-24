#!/usr/bin/env python3
"""bucket별 Final V3 검증: V2 bitwise / ORT tolerance / E2E latency.

ORT와의 bitwise 일치는 목표가 아니다.  `cam_layer/ReduceMean`의 float32 누적
순서가 ORT와 달라(~1e-07) 구조적으로 bit가 어긋나며, 그 차이가 양자화 경계에서
1 LSB 뒤집힘으로 증폭된다.  V2와 V3가 bitwise 동일하므로 이는 최적화 회귀가
아니라 C runtime 계열의 고유 특성이고, EER 영향은 없는 것으로 측정됐다.
따라서 gate를 두 축으로 나눈다.

    V2  <-> V3 : bitwise      (회귀 gate)
    ORT <-> V3 : cosine/max_abs (정확도 gate)
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
FEATURES = ROOT / "benchmarks/campplus/features"
SPEAKERS = ("0000", "0005", "0006")
V2 = ROOT / "build/profill/final_v2_aggressive"
V3 = ROOT / "build/profill/final_v3_hybrid"
COSINE_GATE = 0.99
# /tmp는 RAM 기반 tmpfs다.  bucket 998 dump가 194MB라 tmpfs에 쓰면 그만큼
# RAM을 잠식해 실패할 수 있다.  기본값을 디스크 경로로 둔다.
TMP = pathlib.Path(
    os.environ.get("CAMPP_VERIFY_TMPDIR", ROOT / "build/.verify_tmp"))


class VerifyError(RuntimeError):
    """검증을 수행할 수 없었다."""


def embedding(build_dir: pathlib.Path, bucket: int,
              feature: pathlib.Path) -> np.ndarray:
    TMP.mkdir(parents=True, exist_ok=True)
    prefix = TMP / f"dump_{bucket}"
    subprocess.run(
        [str(build_dir / "campp_reference_dump_final"),
         str(BUNDLE / f"execution_plans/plan_{bucket}.bin"),
         str(BUNDLE / "weights.bin"), str(feature), str(prefix)],
        check=True, capture_output=True)
    meta = json.loads(pathlib.Path(f"{prefix}.json").read_text())
    tensor = meta["tensors"][-1]
    if tensor["shape"] != [1, 192]:
        raise VerifyError(f"마지막 tensor가 embedding이 아니다: {tensor['shape']}")
    blob = np.fromfile(f"{prefix}.bin", dtype=np.uint8)
    out = blob[tensor["offset"]:tensor["offset"] + tensor["byte_size"]] \
        .view(np.float32).copy()
    os.remove(f"{prefix}.bin")          # dump는 bucket당 수십 MB라 즉시 지운다
    return out


def latency_ms(build_dir: pathlib.Path, bucket: int,
               feature: pathlib.Path, repeat: int = 30) -> float:
    proc = subprocess.run(
        ["taskset", "-c", "0", str(build_dir / "campp_runtime_benchmark_final"),
         "--plan", str(BUNDLE / f"execution_plans/plan_{bucket}.bin"),
         "--weights", str(BUNDLE / "weights.bin"), "--input", str(feature),
         "--audio-seconds", "1", "--warmup", "5",
         "--repeat", str(repeat), "--threads", "1"],
        check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return statistics.mean(payload["warm"]["timings_ms"])


def ort_session():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(ROOT / "models/source/campplus_int8_static_qop.onnx"), options,
        providers=["CPUExecutionProvider"])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.astype(np.float64), b.astype(np.float64)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def main(argv: list[str]) -> int:
    buckets = [int(v) for v in argv[1:]] or [98, 298, 498, 998]
    sess = ort_session()
    rows, failures = [], []

    for bucket in buckets:
        feats = [FEATURES / f"multi__speaker_{s}__{bucket}.f32"
                 for s in SPEAKERS]
        missing = [f for f in feats if not f.exists()]
        if missing:
            failures.append(f"bucket {bucket}: feature 없음 {missing[0].name}")
            continue

        bitwise, cosines, max_abs = True, [], 0.0
        for feature in feats:
            e2 = embedding(V2, bucket, feature)
            e3 = embedding(V3, bucket, feature)
            if not np.array_equal(e2.view(np.uint32), e3.view(np.uint32)):
                bitwise = False
            raw = np.fromfile(feature, dtype=np.float32).reshape(1, bucket, 80)
            ort_emb = sess.run(["embedding"],
                               {"feature": raw})[0].reshape(-1)
            cosines.append(cosine(ort_emb, e3))
            max_abs = max(max_abs, float(np.abs(
                ort_emb.astype(np.float64) - e3.astype(np.float64)).max()))
            if not np.isfinite(e3).all():
                failures.append(f"bucket {bucket}: V3 embedding에 NaN/Inf")

        v2_ms = latency_ms(V2, bucket, feats[0])
        v3_ms = latency_ms(V3, bucket, feats[0])
        rows.append({
            "bucket": bucket, "v2_v3_bitwise": bitwise,
            "ort_cosine_min": min(cosines), "ort_max_abs": max_abs,
            "v2_mean_ms": v2_ms, "v3_mean_ms": v3_ms,
            "improvement_pct": (v2_ms - v3_ms) / v2_ms * 100.0,
        })
        if not bitwise:
            failures.append(f"bucket {bucket}: V2/V3 bitwise 불일치")
        if min(cosines) < COSINE_GATE:
            failures.append(
                f"bucket {bucket}: ORT cosine {min(cosines):.6f} "
                f"< {COSINE_GATE}")

    print(f"\n{'bucket':>7}{'V2=V3':>8}{'ORT cos min':>14}{'ORT max_abs':>13}"
          f"{'V2 ms':>10}{'V3 ms':>10}{'개선':>9}")
    for r in rows:
        print(f"{r['bucket']:>7}{str(r['v2_v3_bitwise']):>8}"
              f"{r['ort_cosine_min']:>14.10f}{r['ort_max_abs']:>13.3e}"
              f"{r['v2_mean_ms']:>10.2f}{r['v3_mean_ms']:>10.2f}"
              f"{r['improvement_pct']:>8.2f}%")

    out = ROOT / "results/model_evaluation/final_v3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "v3_vs_ort_by_bucket.json").write_text(
        json.dumps({"schema_version": 1, "cosine_gate": COSINE_GATE,
                    "note": "ORT bitwise는 ReduceMean 누적 순서 차이로 달성 불가",
                    "buckets": rows, "failures": failures},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n결과: {(out / 'v3_vs_ort_by_bucket.json').relative_to(ROOT)}")
    if failures:
        print("\n실패:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n모든 gate 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
