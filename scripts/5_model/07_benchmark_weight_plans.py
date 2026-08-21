#!/usr/bin/env python3
"""Compare source and static bucket weights for bitwise output and RTF."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
DEFAULT_RUNS = ROOT / "runs/models/campplus/final_v3/weight_residency"
DEFAULT_RESULTS = ROOT / "results/models/campplus/final_v3/weight_residency"
DEFAULT_BINARY = (
    ROOT / "build/profill/final_v3_hybrid/campp_runtime_benchmark_final"
)
DEFAULT_FEATURE_DIR = ROOT / "benchmarks/campplus/features"
PROTOCOLS = {
    "quick": {"warmup": 2, "repeat": 10},
    "official": {"warmup": 20, "repeat": 100},
}


class WeightBenchmarkError(RuntimeError):
    """The static weight benchmark did not produce comparable evidence."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise WeightBenchmarkError("empty timing sample")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: Sequence[float]) -> dict:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
    }


def _run_json(command: list[str]) -> dict:
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "ORT_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, text=True,
        capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise WeightBenchmarkError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise WeightBenchmarkError("runtime did not return JSON") from exc
    if not isinstance(payload, dict):
        raise WeightBenchmarkError("runtime JSON is not an object")
    return payload


def _command(
    binary: Path, plan: Path, weights: Path, feature: Path,
    embedding: Path, bucket: int, warmup: int, repeat: int,
) -> list[str]:
    return [
        str(binary),
        "--plan", str(plan),
        "--weights", str(weights),
        "--input", str(feature),
        "--audio-seconds", f"{(bucket + 2) / 100.0:g}",
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--threads", "1",
        "--embedding-output", str(embedding),
    ]


def _validate_payload(payload: dict, bucket: int, repeat: int) -> list[float]:
    if payload.get("runtime") != "campp-c-runtime":
        raise WeightBenchmarkError("unexpected runtime payload")
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("bucket_frames") != bucket:
        raise WeightBenchmarkError(f"runtime bucket is not {bucket}")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict) or (
        configuration.get("effective_threads") != 1
        or configuration.get("repeat") != repeat
    ):
        raise WeightBenchmarkError("runtime protocol mismatch")
    warm = payload.get("warm")
    values = warm.get("timings_ms") if isinstance(warm, dict) else None
    if not isinstance(values, list) or len(values) != repeat:
        raise WeightBenchmarkError("runtime timing count mismatch")
    return [float(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=_path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--buckets", type=int, nargs="+", default=[98, 298, 498, 998]
    )
    parser.add_argument("--binary", type=_path, default=DEFAULT_BINARY)
    parser.add_argument("--feature-dir", type=_path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--input-id", default="multi__speaker_0000")
    parser.add_argument("--runs-root", type=_path, default=DEFAULT_RUNS)
    parser.add_argument("--results-root", type=_path, default=DEFAULT_RESULTS)
    parser.add_argument("--mode", choices=PROTOCOLS, default="quick")
    parser.add_argument("--max-p50-regression-pct", type=float, default=1.0)
    parser.add_argument("--max-p95-regression-pct", type=float, default=3.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        buckets = list(dict.fromkeys(args.buckets))
        protocol = PROTOCOLS[args.mode]
        required = [args.binary, args.bundle / "weights.bin"]
        per_bucket = {}
        for bucket in buckets:
            paths = {
                "feature": args.feature_dir / f"{args.input_id}__{bucket}.f32",
                "source_plan": (
                    args.bundle / "execution_plans" / f"plan_{bucket}.bin"
                ),
                "source_weights": args.bundle / "weights.bin",
                "static_plan": args.runs_root / str(bucket) / f"plan_{bucket}.bin",
                "static_weights": (
                    args.runs_root / str(bucket) / f"weights_{bucket}.bin"
                ),
            }
            per_bucket[bucket] = paths
            required.extend(paths.values())
        missing = [path for path in dict.fromkeys(required) if not path.is_file()]
        if missing:
            raise WeightBenchmarkError(
                "required artifacts are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        capabilities = _run_json([str(args.binary), "--capabilities"])
        if capabilities.get("effective_threads") != 1:
            raise WeightBenchmarkError("runtime is not single-threaded")
        if args.preflight_only:
            print(json.dumps({
                "ready": True,
                "buckets": buckets,
                "mode": args.mode,
                "warmup": protocol["warmup"],
                "repeat": protocol["repeat"],
                "comparison": "source_shared_blob_vs_bucket_static_blob",
            }, ensure_ascii=False, indent=2))
            return 0

        target_json = args.results_root / f"benchmark_{args.mode}.json"
        target_csv = args.results_root / f"benchmark_{args.mode}.csv"
        if (target_json.exists() or target_csv.exists()) and not args.force:
            raise WeightBenchmarkError("benchmark output exists; use --force")

        rows = []
        for index, bucket in enumerate(buckets):
            paths = per_bucket[bucket]
            raw_dir = args.runs_root / str(bucket) / "benchmark" / args.mode
            raw_dir.mkdir(parents=True, exist_ok=True)
            order = ("source", "static") if index % 2 == 0 else ("static", "source")
            payloads = {}
            embeddings = {}
            for variant in order:
                plan = paths[f"{variant}_plan"]
                weights = paths[f"{variant}_weights"]
                embedding = raw_dir / f"{variant}.f32"
                payload = _run_json(_command(
                    args.binary, plan, weights, paths["feature"], embedding,
                    bucket, protocol["warmup"], protocol["repeat"],
                ))
                timings = _validate_payload(payload, bucket, protocol["repeat"])
                payloads[variant] = (payload, _summary(timings))
                embeddings[variant] = embedding.read_bytes()
                (raw_dir / f"{variant}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n",
                )
            source_payload, source = payloads["source"]
            static_payload, static = payloads["static"]
            p50_delta_pct = (
                (static["p50_ms"] - source["p50_ms"]) / source["p50_ms"] * 100.0
            )
            p95_delta_pct = (
                (static["p95_ms"] - source["p95_ms"]) / source["p95_ms"] * 100.0
            )
            bitwise = embeddings["source"] == embeddings["static"]
            gate = (
                bitwise
                and p50_delta_pct <= args.max_p50_regression_pct
                and p95_delta_pct <= args.max_p95_regression_pct
            )
            audio_seconds = (bucket + 2) / 100.0
            rows.append({
                "bucket_frames": bucket,
                "bitwise_embedding": bitwise,
                "source_first_ms": source_payload["lifecycle"]["first_inference_ms"],
                "static_first_ms": static_payload["lifecycle"]["first_inference_ms"],
                "source_p50_ms": source["p50_ms"],
                "static_p50_ms": static["p50_ms"],
                "p50_delta_pct": p50_delta_pct,
                "source_p95_ms": source["p95_ms"],
                "static_p95_ms": static["p95_ms"],
                "p95_delta_pct": p95_delta_pct,
                "source_rtf_p50": source["p50_ms"] / 1000.0 / audio_seconds,
                "static_rtf_p50": static["p50_ms"] / 1000.0 / audio_seconds,
                "source_peak_rss_bytes": source_payload["memory"][
                    "after_measurement"
                ]["peak_rss_bytes"],
                "static_peak_rss_bytes": static_payload["memory"][
                    "after_measurement"
                ]["peak_rss_bytes"],
                "gate_passed": gate,
            })
            print(
                f"bucket {bucket}: bitwise={bitwise} "
                f"p50_delta={p50_delta_pct:+.3f}% gate={gate}",
                flush=True,
            )

        result = {
            "schema_version": 1,
            "mode": args.mode,
            "comparison": "source_shared_blob_vs_bucket_static_blob",
            "inference_weight_moves": 0,
            "all_bitwise": all(row["bitwise_embedding"] for row in rows),
            "no_rtf_regression": all(row["gate_passed"] for row in rows),
            "ready": all(row["gate_passed"] for row in rows),
            "thresholds": {
                "max_p50_regression_pct": args.max_p50_regression_pct,
                "max_p95_regression_pct": args.max_p95_regression_pct,
            },
            "buckets": rows,
        }
        args.results_root.mkdir(parents=True, exist_ok=True)
        target_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        with target_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return 0 if result["ready"] else 3
    except (WeightBenchmarkError, OSError, ValueError, KeyError) as exc:
        print(f"weight plan benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
