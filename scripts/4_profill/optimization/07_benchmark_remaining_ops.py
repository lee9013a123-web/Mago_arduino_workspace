#!/usr/bin/env python3
"""E7 나머지 10개 kernel의 baseline/optimized 후보를 bitwise 비교한다."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DIAGNOSIS_SCRIPT = SCRIPT_DIR / "02_diagnose_top4.py"
TARGET_KERNELS = (
    "add_stride",
    "relu_stride",
    "expand_stride",
    "slice_stride",
    "quantize_linear_stride",
    "reduce_mean_stride",
    "fused_dequant_sigmoid_mul",
    "average_pool_stride",
    "reshape_stride",
    "fused_statistics_pooling",
)


class RemainingBenchmarkError(RuntimeError):
    """나머지 연산 후보의 입력, 실행 또는 결과가 유효하지 않다."""


def _load_diagnosis() -> Any:
    spec = importlib.util.spec_from_file_location(
        "e7_remaining_diagnosis", DIAGNOSIS_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RemainingBenchmarkError(f"cannot import {DIAGNOSIS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSIS = _load_diagnosis()


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def select_representatives(
    operators: Sequence[dict[str, Any]],
    kernels: Sequence[str] = TARGET_KERNELS,
) -> list[dict[str, Any]]:
    selected = []
    for kernel in kernels:
        matches = [item for item in operators if item.get("kernel_name") == kernel]
        if not matches:
            raise RemainingBenchmarkError(f"profile does not contain {kernel}")
        operator = max(matches, key=lambda item: float(item["mean_ms"]))
        selected.append(
            {
                "case_name": kernel,
                "operator_id": int(operator["operator_id"]),
                "opcode": int(operator.get("opcode", 0)),
                "kernel_id": int(operator["kernel_id"]),
                "kernel_name": kernel,
                "operator_type": str(operator["operator_type"]),
                "profile_mean_ms": float(operator["mean_ms"]),
                "profile_share_pct": float(operator["end_to_end_share_pct"]),
                "input_shapes": [
                    list(tensor.get("shape", []))
                    for tensor in operator.get("input_tensors", [])
                ],
                "output_shapes": [
                    list(tensor.get("shape", []))
                    for tensor in operator.get("output_tensors", [])
                ],
            }
        )
    return selected


def _percentile(samples: Sequence[int], percentile: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(samples_ns: Sequence[int]) -> dict[str, float | int]:
    if not samples_ns:
        raise RemainingBenchmarkError("candidate returned no samples")
    return {
        "sample_count": len(samples_ns),
        "mean_ms": statistics.fmean(samples_ns) / 1_000_000.0,
        "p50_ms": _percentile(samples_ns, 0.50) / 1_000_000.0,
        "p95_ms": _percentile(samples_ns, 0.95) / 1_000_000.0,
        "min_ms": min(samples_ns) / 1_000_000.0,
        "max_ms": max(samples_ns) / 1_000_000.0,
    }


def _command(
    binary: Path,
    *,
    plan: Path,
    weights: Path,
    feature: Path,
    operator_id: int,
    warmup: int,
    repeat: int,
    mode: str,
    expected_hash: str | None = None,
) -> list[str]:
    command = DIAGNOSIS._command(
        binary,
        plan=plan,
        weights=weights,
        feature=feature,
        operator_id=operator_id,
        warmup=warmup,
        repeat=repeat,
        expected_hash=expected_hash,
    )
    command.extend(("--remaining-candidate", mode))
    return command


def _validate_payload(
    payload: dict[str, Any], case: dict[str, Any], mode: str, repeat: int
) -> None:
    operator = payload.get("operator", {})
    samples = payload.get("samples_ns")
    if (
        payload.get("mode") != "operator_microbench"
        or payload.get("remaining_candidate") != mode
        or operator.get("operator_id") != case["operator_id"]
        or operator.get("kernel_id") != case["kernel_id"]
        or operator.get("kernel_name") != case["kernel_name"]
        or payload.get("output_hash_matches") is not True
        or not isinstance(samples, list)
        or len(samples) != repeat
        or any(not isinstance(value, int) or value <= 0 for value in samples)
    ):
        raise RemainingBenchmarkError(
            f"invalid {mode} payload for {case['case_name']}"
        )


def build_result(
    cases: Sequence[dict[str, Any]],
    *,
    warmup: int,
    repeat: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    all_bitwise = all(case["bitwise_output_hashes"] for case in cases)
    no_regression = all(
        case["optimized"]["mean_ms"] <= case["baseline"]["mean_ms"] * 1.01
        for case in cases
    )
    any_faster = any(
        case["optimized"]["mean_ms"] < case["baseline"]["mean_ms"]
        for case in cases
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "benchmark common fastpaths on remaining E7 operators",
        "configuration": {
            "bucket_frames": 98,
            "threads": 1,
            "cpu_affinity": [0],
            "warmup": warmup,
            "repeat_per_input": repeat,
            "input_count": 3,
            "candidate_modes": ["baseline", "optimized"],
            "elapsed_seconds": elapsed_seconds,
        },
        "all_output_hashes_bitwise_identical": all_bitwise,
        "no_case_regressed_over_1pct": no_regression,
        "any_case_faster": any_faster,
        "candidate_gate_passed": all_bitwise and no_regression and any_faster,
        "cases": list(cases),
        "next_gate": "retained-tensor bitwise validation and Quick E2E profiling",
    }


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    fields = [
        "case_name",
        "operator_id",
        "operator_type",
        "kernel_name",
        "baseline_mean_ms",
        "optimized_mean_ms",
        "speedup_ratio",
        "latency_reduction_pct",
        "baseline_p50_ms",
        "optimized_p50_ms",
        "baseline_p95_ms",
        "optimized_p95_ms",
        "bitwise_output_hashes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            writer.writerow(
                {
                    "case_name": case["case_name"],
                    "operator_id": case["operator_id"],
                    "operator_type": case["operator_type"],
                    "kernel_name": case["kernel_name"],
                    "baseline_mean_ms": case["baseline"]["mean_ms"],
                    "optimized_mean_ms": case["optimized"]["mean_ms"],
                    "speedup_ratio": case["speedup_ratio"],
                    "latency_reduction_pct": case["latency_reduction_pct"],
                    "baseline_p50_ms": case["baseline"]["p50_ms"],
                    "optimized_p50_ms": case["optimized"]["p50_ms"],
                    "baseline_p95_ms": case["baseline"]["p95_ms"],
                    "optimized_p95_ms": case["optimized"]["p95_ms"],
                    "bitwise_output_hashes": case["bitwise_output_hashes"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=_path,
        default=ROOT / "results/profiling/e7_98/operator_profile.json",
    )
    parser.add_argument(
        "--binary",
        type=_path,
        default=ROOT / "build/profill/optimization/campp_operator_microbench",
    )
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument(
        "--features", type=_path, nargs="+", default=list(DIAGNOSIS.EXPECTED_INPUTS)
    )
    parser.add_argument("--kernels", nargs="+", choices=TARGET_KERNELS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--runs-dir",
        type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/remaining_candidates",
    )
    parser.add_argument(
        "--output",
        type=_path,
        default=ROOT
        / "results/profiling/e7_98/optimization/remaining_candidates.json",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.warmup < 0 or args.repeat <= 0:
            raise RemainingBenchmarkError("invalid warmup or repeat")
        if os.name != "posix" and not args.preflight_only:
            raise RemainingBenchmarkError("actual benchmark requires Linux/QRB2210")
        if not args.profile.is_file():
            raise RemainingBenchmarkError(f"profile not found: {args.profile}")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        kernels = tuple(args.kernels or TARGET_KERNELS)
        cases = select_representatives(profile.get("operators", []), kernels)
        plan = args.plan or DIAGNOSIS._resolve_profile_artifact(profile, "plan")
        weights = args.weights or DIAGNOSIS._resolve_profile_artifact(
            profile, "weights"
        )
        required = [args.binary, plan, weights, *args.features]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise RemainingBenchmarkError(
                "required files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3 or any(
            path.stat().st_size != 98 * 80 * 4 for path in args.features
        ):
            raise RemainingBenchmarkError(
                "exactly three float32 [1,98,80] inputs are required"
            )
        capabilities = DIAGNOSIS._run_payload(
            [str(args.binary), "--capabilities"]
        )
        if capabilities.get("remaining_candidates") != [
            "baseline",
            "optimized",
        ]:
            raise RemainingBenchmarkError(
                "binary does not expose remaining candidates"
            )
        if args.output.exists() and not args.force and not args.preflight_only:
            raise RemainingBenchmarkError(
                f"output exists: {args.output} (use --force)"
            )
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "binary": _display_path(args.binary),
                        "plan": _display_path(plan),
                        "weights": _display_path(weights),
                        "features": [_display_path(path) for path in args.features],
                        "cases": cases,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        estimated_minutes = max(1, math.ceil(len(cases) * len(args.features) / 4))
        print(f"예상 시간: 약 {estimated_minutes}분")
        print("진행 중", flush=True)
        started = time.monotonic()
        results = []
        for case in cases:
            mode_payloads: dict[str, list[dict[str, Any]]] = {
                "baseline": [],
                "optimized": [],
            }
            for feature in args.features:
                baseline = DIAGNOSIS._run_payload(
                    _command(
                        args.binary,
                        plan=plan,
                        weights=weights,
                        feature=feature,
                        operator_id=case["operator_id"],
                        warmup=args.warmup,
                        repeat=args.repeat,
                        mode="baseline",
                    )
                )
                _validate_payload(baseline, case, "baseline", args.repeat)
                optimized = DIAGNOSIS._run_payload(
                    _command(
                        args.binary,
                        plan=plan,
                        weights=weights,
                        feature=feature,
                        operator_id=case["operator_id"],
                        warmup=args.warmup,
                        repeat=args.repeat,
                        mode="optimized",
                        expected_hash=str(baseline["output_hash"]),
                    )
                )
                _validate_payload(optimized, case, "optimized", args.repeat)
                for mode, payload in (
                    ("baseline", baseline),
                    ("optimized", optimized),
                ):
                    payload["input"] = _display_path(feature)
                    raw = args.runs_dir / mode / case["case_name"]
                    raw.mkdir(parents=True, exist_ok=True)
                    raw_path = raw / f"{feature.stem}.json"
                    raw_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    mode_payloads[mode].append(payload)
            baseline_samples = [
                int(sample)
                for payload in mode_payloads["baseline"]
                for sample in payload["samples_ns"]
            ]
            optimized_samples = [
                int(sample)
                for payload in mode_payloads["optimized"]
                for sample in payload["samples_ns"]
            ]
            baseline_stats = summarize_samples(baseline_samples)
            optimized_stats = summarize_samples(optimized_samples)
            bitwise = all(
                before["output_hash"] == after["output_hash"]
                for before, after in zip(
                    mode_payloads["baseline"],
                    mode_payloads["optimized"],
                    strict=True,
                )
            )
            speedup = baseline_stats["mean_ms"] / optimized_stats["mean_ms"]
            results.append(
                {
                    **case,
                    "baseline": baseline_stats,
                    "optimized": optimized_stats,
                    "speedup_ratio": speedup,
                    "latency_reduction_pct": (1.0 - 1.0 / speedup) * 100.0,
                    "bitwise_output_hashes": bitwise,
                    "output_hashes": [
                        payload["output_hash"]
                        for payload in mode_payloads["baseline"]
                    ],
                }
            )
        result = build_result(
            results,
            warmup=args.warmup,
            repeat=args.repeat,
            elapsed_seconds=time.monotonic() - started,
        )
        result["artifacts"] = {
            "profile": _display_path(args.profile),
            "plan": _display_path(plan),
            "weights": _display_path(weights),
            "raw_dir": _display_path(args.runs_dir),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_csv(args.output.with_suffix(".csv"), result)
        print(f"완료: {_display_path(args.output)}")
        return 0 if result["candidate_gate_passed"] else 3
    except (
        RemainingBenchmarkError,
        DIAGNOSIS.DiagnosisError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"remaining benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
