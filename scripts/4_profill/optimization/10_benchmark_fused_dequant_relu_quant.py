#!/usr/bin/env python3
"""fused_dequant_relu_quant의 scalar/NEON 후보를 shape별로 Quick 비교한다."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_KERNEL = "fused_dequant_relu_quant"
MODES = ("baseline", "scalar", "neon")


class FusedDqRqBenchmarkError(RuntimeError):
    """후보 입력, 실행 결과 또는 검증 게이트가 유효하지 않다."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FusedDqRqBenchmarkError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSIS = _load_module(
    "e7_fused_dqrq_diagnosis", SCRIPT_DIR / "02_diagnose_top4.py"
)
COMMON = _load_module(
    "e7_fused_dqrq_statistics", SCRIPT_DIR / "07_benchmark_remaining_ops.py"
)


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _input_shape(operator: dict[str, Any]) -> tuple[int, ...]:
    tensors = operator.get("input_tensors", [])
    if not tensors:
        raise FusedDqRqBenchmarkError("operator input shape is missing")
    return tuple(int(value) for value in tensors[0].get("shape", []))


def select_cases(
    operators: Sequence[dict[str, Any]], *, all_operators: bool = False
) -> list[dict[str, Any]]:
    family = [
        item for item in operators if item.get("kernel_name") == TARGET_KERNEL
    ]
    if not family:
        raise FusedDqRqBenchmarkError(
            f"profile does not contain {TARGET_KERNEL}"
        )
    selected = family
    if not all_operators:
        representatives: dict[tuple[int, ...], dict[str, Any]] = {}
        for operator in family:
            shape = _input_shape(operator)
            current = representatives.get(shape)
            if current is None or float(operator["mean_ms"]) > float(
                current["mean_ms"]
            ):
                representatives[shape] = operator
        selected = list(representatives.values())
    return [
        {
            "case_name": f"op_{int(operator['operator_id'])}",
            "operator_id": int(operator["operator_id"]),
            "kernel_id": int(operator["kernel_id"]),
            "kernel_name": TARGET_KERNEL,
            "operator_type": str(operator["operator_type"]),
            "input_shape": list(_input_shape(operator)),
            "profile_mean_ms": float(operator["mean_ms"]),
            "profile_share_pct": float(operator["end_to_end_share_pct"]),
        }
        for operator in sorted(selected, key=lambda item: int(item["operator_id"]))
    ]


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
    command.extend(("--fused-dqrq-candidate", mode))
    return command


def _validate_payload(
    payload: dict[str, Any], case: dict[str, Any], mode: str, repeat: int
) -> None:
    operator = payload.get("operator", {})
    samples = payload.get("samples_ns")
    if payload.get("output_hash_matches") is not True:
        raise FusedDqRqBenchmarkError(
            f"bitwise mismatch for {case['case_name']} ({mode}): "
            f"{payload.get('output_hash', '<missing>')}"
        )
    if (
        payload.get("mode") != "operator_microbench"
        or payload.get("fused_dqrq_candidate") != mode
        or operator.get("operator_id") != case["operator_id"]
        or operator.get("kernel_id") != case["kernel_id"]
        or operator.get("kernel_name") != TARGET_KERNEL
        or not isinstance(samples, list)
        or len(samples) != repeat
        or any(not isinstance(value, int) or value <= 0 for value in samples)
    ):
        raise FusedDqRqBenchmarkError(
            f"invalid {mode} payload for {case['case_name']}"
        )


def build_result(
    cases: Sequence[dict[str, Any]],
    *,
    warmup: int,
    repeat: int,
    all_operators: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    winners = []
    all_bitwise = True
    no_regression = True
    any_faster = False
    for case in cases:
        baseline = case["modes"]["baseline"]
        candidates = {
            name: stats
            for name, stats in case["modes"].items()
            if name != "baseline"
        }
        winner_name, winner = min(
            candidates.items(), key=lambda item: item[1]["mean_ms"]
        )
        speedup = baseline["mean_ms"] / winner["mean_ms"]
        case["winner"] = winner_name
        case["winner_speedup_ratio"] = speedup
        case["winner_latency_reduction_pct"] = (1.0 - 1.0 / speedup) * 100.0
        winners.append(winner_name)
        all_bitwise = all_bitwise and all(case["bitwise_by_mode"].values())
        no_regression = no_regression and (
            winner["mean_ms"] <= baseline["mean_ms"] * 1.01
        )
        any_faster = any_faster or winner["mean_ms"] < baseline["mean_ms"]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "vectorize fused Dequantize/ReLU/Quantize with NEON",
        "configuration": {
            "bucket_frames": 98,
            "threads": 1,
            "cpu_affinity": [0],
            "warmup": warmup,
            "repeat_per_input": repeat,
            "input_count": 3,
            "modes": list(MODES),
            "scope": "all_operators" if all_operators else "shape_representatives",
            "elapsed_seconds": elapsed_seconds,
        },
        "all_output_hashes_bitwise_identical": all_bitwise,
        "no_winner_regressed_over_1pct": no_regression,
        "any_case_faster": any_faster,
        "candidate_microbench_gate_passed": (
            all_bitwise and no_regression and any_faster
        ),
        "winner_modes": sorted(set(winners)),
        "cases": list(cases),
        "next_gate": (
            "all-operator family benchmark, retained-tensor bitwise validation, "
            "and Quick E2E profiling"
        ),
    }


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    fields = [
        "operator_id", "input_shape", "baseline_mean_ms", "scalar_mean_ms",
        "neon_mean_ms", "neon_speedup_ratio", "baseline_p50_ms",
        "neon_p50_ms", "baseline_p95_ms", "neon_p95_ms",
        "scalar_bitwise", "neon_bitwise", "winner",
    ]
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            baseline = case["modes"]["baseline"]
            scalar = case["modes"]["scalar"]
            neon = case["modes"]["neon"]
            writer.writerow(
                {
                    "operator_id": case["operator_id"],
                    "input_shape": json.dumps(case["input_shape"]),
                    "baseline_mean_ms": baseline["mean_ms"],
                    "scalar_mean_ms": scalar["mean_ms"],
                    "neon_mean_ms": neon["mean_ms"],
                    "neon_speedup_ratio": (
                        baseline["mean_ms"] / neon["mean_ms"]
                    ),
                    "baseline_p50_ms": baseline["p50_ms"],
                    "neon_p50_ms": neon["p50_ms"],
                    "baseline_p95_ms": baseline["p95_ms"],
                    "neon_p95_ms": neon["p95_ms"],
                    "scalar_bitwise": case["bitwise_by_mode"]["scalar"],
                    "neon_bitwise": case["bitwise_by_mode"]["neon"],
                    "winner": case["winner"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=_path,
        default=ROOT / "results/profiling/e7_98/operator_profile.json",
    )
    parser.add_argument(
        "--binary", type=_path,
        default=ROOT / "build/profill/optimization/campp_operator_microbench",
    )
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument(
        "--features", type=_path, nargs="+", default=list(DIAGNOSIS.EXPECTED_INPUTS)
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--all-operators", action="store_true")
    parser.add_argument(
        "--runs-dir", type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/fused_dequant_relu_quant/raw",
    )
    parser.add_argument(
        "--output-dir", type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/fused_dequant_relu_quant",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.warmup < 0 or args.repeat <= 0:
            raise FusedDqRqBenchmarkError("invalid warmup or repeat")
        if os.name != "posix" and not args.preflight_only:
            raise FusedDqRqBenchmarkError("actual benchmark requires Linux/QRB2210")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        cases = select_cases(
            profile.get("operators", []), all_operators=args.all_operators
        )
        plan = args.plan or DIAGNOSIS._resolve_profile_artifact(profile, "plan")
        weights = args.weights or DIAGNOSIS._resolve_profile_artifact(
            profile, "weights"
        )
        missing = [
            path for path in (args.binary, plan, weights, *args.features)
            if not path.is_file()
        ]
        if missing:
            raise FusedDqRqBenchmarkError(
                "required files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3:
            raise FusedDqRqBenchmarkError("exactly three inputs are required")
        capabilities = DIAGNOSIS._run_payload([str(args.binary), "--capabilities"])
        if capabilities.get("fused_dqrq_candidates") != list(MODES):
            raise FusedDqRqBenchmarkError(
                "binary does not expose fused DQ/ReLU/Q candidates"
            )
        summary_path = args.output_dir / "summary.json"
        if summary_path.exists() and not args.force and not args.preflight_only:
            raise FusedDqRqBenchmarkError(
                f"output exists: {summary_path} (use --force)"
            )
        if args.preflight_only:
            print(json.dumps({
                "ready": True,
                "scope": "all_operators" if args.all_operators else "shape_representatives",
                "case_count": len(cases),
                "modes": list(MODES),
                "cases": cases,
            }, ensure_ascii=False, indent=2))
            return 0

        estimated_minutes = max(
            1, math.ceil(len(cases) * len(args.features) * len(MODES) / 6)
        )
        print(f"예상 시간: 약 {estimated_minutes}분")
        print("진행 중", flush=True)
        started = time.monotonic()
        measured_cases = []
        for case in cases:
            payloads: dict[str, list[dict[str, Any]]] = {
                mode: [] for mode in MODES
            }
            for feature in args.features:
                expected_hash: str | None = None
                for mode in MODES:
                    payload = DIAGNOSIS._run_payload(_command(
                        args.binary, plan=plan, weights=weights,
                        feature=feature, operator_id=case["operator_id"],
                        warmup=args.warmup, repeat=args.repeat, mode=mode,
                        expected_hash=expected_hash,
                    ))
                    raw_dir = args.runs_dir / mode / case["case_name"]
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    (raw_dir / f"{feature.stem}.json").write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n",
                    )
                    _validate_payload(payload, case, mode, args.repeat)
                    payloads[mode].append(payload)
                    if mode == "baseline":
                        expected_hash = str(payload["output_hash"])
            stats = {
                mode: COMMON.summarize_samples([
                    int(sample)
                    for payload in payloads[mode]
                    for sample in payload["samples_ns"]
                ])
                for mode in MODES
            }
            baseline_hashes = [item["output_hash"] for item in payloads["baseline"]]
            measured_cases.append({
                **case,
                "modes": stats,
                "bitwise_by_mode": {
                    mode: [item["output_hash"] for item in payloads[mode]]
                    == baseline_hashes
                    for mode in MODES if mode != "baseline"
                },
                "output_hashes": baseline_hashes,
            })
        result = build_result(
            measured_cases, warmup=args.warmup, repeat=args.repeat,
            all_operators=args.all_operators,
            elapsed_seconds=time.monotonic() - started,
        )
        result["artifacts"] = {
            "profile": _display_path(args.profile),
            "plan": _display_path(plan),
            "weights": _display_path(weights),
            "raw_dir": _display_path(args.runs_dir),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        _write_csv(args.output_dir / "comparison.csv", result)
        print(f"완료: {_display_path(summary_path)}")
        return 0 if result["candidate_microbench_gate_passed"] else 3
    except (
        FusedDqRqBenchmarkError, DIAGNOSIS.DiagnosisError,
        OSError, ValueError, KeyError,
    ) as exc:
        print(f"fused DQ/ReLU/Q benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
