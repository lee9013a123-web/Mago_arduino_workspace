#!/usr/bin/env python3
"""Gate bucket-98 mmap weight windows against the incumbent Final V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
DEFAULT_CANDIDATE = (
    ROOT / "runs/models/campplus/final_v3/weight_streaming/98"
)
DEFAULT_RESULTS = (
    ROOT / "results/models/campplus/final_v3/weight_streaming/98"
)
DEFAULT_CANDIDATE_BINARY = (
    ROOT / "build/profill/weight_streaming_98/campp_runtime_benchmark_final"
)
DEFAULT_BASELINE_BINARY = (
    ROOT / "build/profill/final_v3_hybrid/campp_runtime_benchmark_final"
)
DEFAULT_CANDIDATE_DUMP_BINARY = (
    ROOT / "build/profill/weight_streaming_98/campp_reference_dump_final"
)
DEFAULT_BASELINE_DUMP_BINARY = (
    ROOT / "build/profill/final_v3_hybrid/campp_reference_dump_final"
)
DEFAULT_FEATURE = (
    ROOT / "benchmarks/campplus/features/multi__speaker_0000__98.f32"
)
PROTOCOLS = {
    "quick": {"warmup": 2, "repeat": 10},
    "official": {"warmup": 20, "repeat": 100},
}


class WeightStreamingBenchmarkError(RuntimeError):
    pass


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(payload: dict, repeat: int) -> dict:
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict) or (
        configuration.get("effective_threads") != 1
        or configuration.get("repeat") != repeat
    ):
        raise WeightStreamingBenchmarkError("runtime protocol mismatch")
    warm = payload.get("warm")
    values = warm.get("timings_ms") if isinstance(warm, dict) else None
    if not isinstance(values, list) or len(values) != repeat:
        raise WeightStreamingBenchmarkError("runtime timing count mismatch")
    samples = [float(value) for value in values]
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
    }


def _run(command: list[str]) -> dict:
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
        raise WeightStreamingBenchmarkError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise WeightStreamingBenchmarkError(
            "runtime output is not JSON"
        ) from exc
    if not isinstance(value, dict):
        raise WeightStreamingBenchmarkError("runtime output is not an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_dump(command: list[str], prefix: Path) -> dict:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise WeightStreamingBenchmarkError(
            f"retained tensor dump failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr.strip()}"
        )
    payload_path = prefix.with_suffix(".bin")
    index_path = prefix.with_suffix(".json")
    if not payload_path.is_file() or not index_path.is_file():
        raise WeightStreamingBenchmarkError(
            f"retained tensor dump is incomplete: {prefix}"
        )
    document = json.loads(index_path.read_text(encoding="utf-8"))
    tensors = document.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise WeightStreamingBenchmarkError(
            f"retained tensor index is empty: {index_path}"
        )
    return {
        "tensor_count": len(tensors),
        "tensors": tensors,
        "payload_bytes": payload_path.stat().st_size,
        "payload_sha256": _sha256(payload_path),
    }


def _command(
    *, binary: Path, plan: Path, weights: Path, feature: Path,
    embedding: Path, mode: str | None, schedule: Path | None,
    warmup: int, repeat: int, cpu: int | None,
) -> list[str]:
    command = [str(binary)]
    if cpu is not None and shutil.which("taskset") is not None:
        command = ["taskset", "-c", str(cpu), *command]
    command.extend([
        "--plan", str(plan),
        "--weights", str(weights),
        "--input", str(feature),
        "--audio-seconds", "1",
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--threads", "1",
        "--embedding-output", str(embedding),
    ])
    if mode is not None:
        command.extend(["--weight-mode", mode])
    if schedule is not None:
        command.extend(["--weight-schedule", str(schedule)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=_path, default=DEFAULT_BUNDLE)
    parser.add_argument("--candidate", type=_path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--results", type=_path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--baseline-binary", type=_path, default=DEFAULT_BASELINE_BINARY
    )
    parser.add_argument(
        "--candidate-binary", type=_path, default=DEFAULT_CANDIDATE_BINARY
    )
    parser.add_argument(
        "--baseline-dump-binary", type=_path,
        default=DEFAULT_BASELINE_DUMP_BINARY,
    )
    parser.add_argument(
        "--candidate-dump-binary", type=_path,
        default=DEFAULT_CANDIDATE_DUMP_BINARY,
    )
    parser.add_argument("--feature", type=_path, default=DEFAULT_FEATURE)
    parser.add_argument("--mode", choices=PROTOCOLS, default="quick")
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--max-p50-regression-pct", type=float, default=1.0)
    parser.add_argument("--max-p95-regression-pct", type=float, default=3.0)
    parser.add_argument("--min-peak-rss-reduction-bytes", type=int,
                        default=5_500_000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        protocol = PROTOCOLS[args.mode]
        source_plan = args.bundle / "execution_plans/plan_98.bin"
        source_weights = args.bundle / "weights.bin"
        candidate_plan = args.candidate / "plan_98.bin"
        candidate_weights = args.candidate / "weights_98.bin"
        schedule = args.candidate / "weight_schedule_98.bin"
        required = (
            args.baseline_binary, args.candidate_binary, args.feature,
            args.baseline_dump_binary, args.candidate_dump_binary,
            source_plan, source_weights,
            candidate_plan, candidate_weights, schedule,
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise WeightStreamingBenchmarkError(
                "required artifact is missing: " + str(missing[0])
            )
        capabilities = _run([str(args.candidate_binary), "--capabilities"])
        modes = capabilities.get("weight_residency_modes")
        if modes != ["malloc", "mmap", "windowed"]:
            raise WeightStreamingBenchmarkError(
                "runtime does not expose mmap weight modes"
            )
        if args.preflight_only:
            print(json.dumps({
                "ready": True,
                "bucket_frames": 98,
                "protocol": protocol,
                "variants": [
                    "v3_full", "stream_full", "stream_mmap", "stream_windowed"
                ],
            }, ensure_ascii=False, indent=2))
            return 0

        output_json = args.results / f"benchmark_{args.mode}.json"
        raw_dir = args.candidate / "benchmark" / args.mode
        if output_json.exists() and not args.force:
            raise WeightStreamingBenchmarkError("output exists; use --force")
        raw_dir.mkdir(parents=True, exist_ok=True)
        baseline_retained_prefix = raw_dir / "retained_v3_full"
        windowed_retained_prefix = raw_dir / "retained_stream_windowed"
        baseline_retained = _run_dump([
            str(args.baseline_dump_binary),
            str(source_plan),
            str(source_weights),
            str(args.feature),
            str(baseline_retained_prefix),
        ], baseline_retained_prefix)
        windowed_retained = _run_dump([
            str(args.candidate_dump_binary),
            str(candidate_plan),
            str(candidate_weights),
            str(args.feature),
            str(windowed_retained_prefix),
            "--weight-mode", "windowed",
            "--weight-schedule", str(schedule),
        ], windowed_retained_prefix)
        retained_index_identical = (
            baseline_retained["tensors"] == windowed_retained["tensors"]
        )
        retained_payload_bitwise = (
            baseline_retained["payload_sha256"]
            == windowed_retained["payload_sha256"]
            and baseline_retained["payload_bytes"]
            == windowed_retained["payload_bytes"]
        )
        retained_bitwise = (
            retained_index_identical and retained_payload_bitwise
        )
        variants = (
            ("v3_full", args.baseline_binary, source_plan, source_weights,
             None, None),
            ("stream_full", args.candidate_binary, candidate_plan,
             candidate_weights, "malloc", None),
            ("stream_mmap", args.candidate_binary, candidate_plan,
             candidate_weights, "mmap", None),
            ("stream_windowed", args.candidate_binary, candidate_plan,
             candidate_weights, "windowed", schedule),
        )
        rows: dict[str, dict] = {}
        embeddings: dict[str, bytes] = {}
        for (name, binary, plan, weights,
             weight_mode, schedule_path) in variants:
            embedding = raw_dir / f"{name}.f32"
            payload = _run(_command(
                binary=binary,
                plan=plan,
                weights=weights,
                feature=args.feature,
                embedding=embedding,
                mode=weight_mode,
                schedule=schedule_path,
                warmup=protocol["warmup"],
                repeat=protocol["repeat"],
                cpu=args.cpu,
            ))
            if payload.get("model", {}).get("bucket_frames") != 98:
                raise WeightStreamingBenchmarkError("runtime bucket mismatch")
            summary = _timing_summary(payload, protocol["repeat"])
            memory = payload["memory"]["after_measurement"]
            faults = payload.get("faults", {})
            rows[name] = {
                **summary,
                "first_ms": payload["lifecycle"]["first_inference_ms"],
                "current_rss_bytes": memory["current_rss_bytes"],
                "peak_rss_bytes": memory["peak_rss_bytes"],
                "rtf_p50": summary["p50_ms"] / 1000.0,
                "runtime_weight_mode": payload["configuration"].get(
                    "weight_mode", "malloc"
                ),
                "measurement_minor_faults": faults.get("measurement_minor"),
                "measurement_major_faults": faults.get("measurement_major"),
            }
            embeddings[name] = embedding.read_bytes()
            (raw_dir / f"{name}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
            print(
                f"{name}: p50={summary['p50_ms']:.3f} ms "
                f"peak_rss={memory['peak_rss_bytes']} B",
                flush=True,
            )

        baseline = rows["v3_full"]
        candidate = rows["stream_windowed"]
        p50_delta = (
            (candidate["p50_ms"] - baseline["p50_ms"])
            / baseline["p50_ms"] * 100.0
        )
        p95_delta = (
            (candidate["p95_ms"] - baseline["p95_ms"])
            / baseline["p95_ms"] * 100.0
        )
        peak_rss_reduction = (
            baseline["peak_rss_bytes"] - candidate["peak_rss_bytes"]
        )
        bitwise = all(
            payload == embeddings["v3_full"]
            for payload in embeddings.values()
        )
        gate = (
            retained_bitwise
            and bitwise
            and p50_delta <= args.max_p50_regression_pct
            and p95_delta <= args.max_p95_regression_pct
            and peak_rss_reduction >= args.min_peak_rss_reduction_bytes
            and candidate["measurement_major_faults"] == 0
        )
        result = {
            "schema_version": 1,
            "ready": gate,
            "bucket_frames": 98,
            "mode": args.mode,
            "comparison": "final_v3_full_vs_page_windowed_mmap",
            "retained_tensor_validation": {
                "backend": "cpu_aarch64_o4i4_final_v3",
                "input": args.feature.name,
                "tolerance": {"kind": "bitwise", "atol": 0.0, "rtol": 0.0},
                "tensor_count": baseline_retained["tensor_count"],
                "index_identical": retained_index_identical,
                "all_tensor_bytes_bitwise_identical": (
                    retained_payload_bitwise
                ),
                "reference_sha256": baseline_retained["payload_sha256"],
                "candidate_sha256": windowed_retained["payload_sha256"],
                "passed": retained_bitwise,
            },
            "all_embeddings_bitwise_identical": bitwise,
            "p50_delta_pct": p50_delta,
            "p95_delta_pct": p95_delta,
            "peak_rss_reduction_bytes": peak_rss_reduction,
            "windowed_measurement_major_faults": candidate[
                "measurement_major_faults"
            ],
            "thresholds": {
                "max_p50_regression_pct": args.max_p50_regression_pct,
                "max_p95_regression_pct": args.max_p95_regression_pct,
                "min_peak_rss_reduction_bytes": (
                    args.min_peak_rss_reduction_bytes
                ),
                "max_measurement_major_faults": 0,
            },
            "variants": rows,
        }
        args.results.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        return 0 if gate else 3
    except (WeightStreamingBenchmarkError, OSError, ValueError, KeyError) as exc:
        print(f"weight streaming benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
