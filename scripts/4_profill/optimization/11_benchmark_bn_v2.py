#!/usr/bin/env python3
"""Compare fused BN/ReLU/Quant combined with v2 candidates on QRB2210."""

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
TARGET_KERNELS = {
    "fused_bn_relu_quant",
    "fused_bn_relu_quant_combined",
}
MODES = ("combined", "v2_exact16", "v2_spatial2", "v2_prescaled")
REQUIRED_BITWISE_MODES = ("v2_exact16", "v2_spatial2")


class BnV2BenchmarkError(RuntimeError):
    """The benchmark input, payload, or candidate gate is invalid."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BnV2BenchmarkError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSIS = _load_module(
    "e7_bn_v2_diagnosis", SCRIPT_DIR / "02_diagnose_top4.py"
)
COMMON = _load_module(
    "e7_bn_v2_statistics", SCRIPT_DIR / "07_benchmark_remaining_ops.py"
)


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _shape(operator: dict[str, Any]) -> tuple[int, ...]:
    tensors = operator.get("output_tensors", [])
    if not tensors:
        raise BnV2BenchmarkError("BN output shape is missing")
    return tuple(int(value) for value in tensors[0].get("shape", []))


def select_cases(
    operators: Sequence[dict[str, Any]], *, all_operators: bool = False
) -> list[dict[str, Any]]:
    family = [
        item
        for item in operators
        if item.get("kernel_name") in TARGET_KERNELS
        and item.get("operator_type") == "BATCH_NORMALIZATION"
    ]
    if not family:
        raise BnV2BenchmarkError("profile does not contain fused BN/ReLU/Quant")
    ordered = sorted(family, key=lambda item: (_shape(item)[1], int(item["operator_id"])))
    selected = ordered
    if not all_operators and len(ordered) > 3:
        selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    return [
        {
            "case_name": f"op_{int(operator['operator_id'])}",
            "operator_id": int(operator["operator_id"]),
            "kernel_id": int(operator["kernel_id"]),
            "kernel_name": str(operator["kernel_name"]),
            "operator_type": str(operator["operator_type"]),
            "output_shape": list(_shape(operator)),
            "profile_mean_ms": float(operator["mean_ms"]),
            "profile_share_pct": float(operator["end_to_end_share_pct"]),
        }
        for operator in selected
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
    index = command.index("--bn-candidate")
    command[index + 1] = mode
    return command


def _validate_payload(
    payload: dict[str, Any], case: dict[str, Any], mode: str, repeat: int
) -> None:
    operator = payload.get("operator", {})
    samples = payload.get("samples_ns")
    if mode != "combined" and payload.get("output_hash_matches") is not True:
        raise BnV2BenchmarkError(
            f"bitwise mismatch for {case['case_name']} ({mode})"
        )
    if (
        payload.get("mode") != "operator_microbench"
        or payload.get("bn_candidate") != mode
        or operator.get("operator_id") != case["operator_id"]
        or operator.get("kernel_id") != case["kernel_id"]
        or not isinstance(samples, list)
        or len(samples) != repeat
        or any(not isinstance(value, int) or value <= 0 for value in samples)
    ):
        raise BnV2BenchmarkError(
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
    all_required_bitwise = True
    no_required_regression = True
    any_required_faster = False
    winner_modes: list[str] = []
    for case in cases:
        baseline = case["modes"]["combined"]
        valid_candidates = {
            mode: case["modes"][mode]
            for mode in MODES[1:]
            if case["bitwise_by_mode"].get(mode, False)
        }
        if valid_candidates:
            winner_name, winner = min(
                valid_candidates.items(), key=lambda item: item[1]["mean_ms"]
            )
            case["winner"] = winner_name
            case["winner_speedup_ratio"] = (
                baseline["mean_ms"] / winner["mean_ms"]
            )
            winner_modes.append(winner_name)
        else:
            case["winner"] = "combined"
            case["winner_speedup_ratio"] = 1.0
        for mode in REQUIRED_BITWISE_MODES:
            bitwise = bool(case["bitwise_by_mode"].get(mode, False))
            all_required_bitwise = all_required_bitwise and bitwise
            if bitwise:
                candidate = case["modes"][mode]
                no_required_regression = no_required_regression and (
                    candidate["mean_ms"] <= baseline["mean_ms"] * 1.01
                )
                any_required_faster = any_required_faster or (
                    candidate["mean_ms"] < baseline["mean_ms"]
                )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "select the exact 16-lane fused BN/ReLU/Quant v2 path",
        "configuration": {
            "bucket_frames": 98,
            "threads": 1,
            "cpu_affinity": [0],
            "warmup": warmup,
            "repeat_per_input": repeat,
            "input_count": 3,
            "modes": list(MODES),
            "required_bitwise_modes": list(REQUIRED_BITWISE_MODES),
            "scope": "all_operators" if all_operators else "channel_representatives",
            "elapsed_seconds": elapsed_seconds,
        },
        "all_required_output_hashes_bitwise_identical": all_required_bitwise,
        "no_required_case_regressed_over_1pct": no_required_regression,
        "any_required_case_faster": any_required_faster,
        "candidate_microbench_gate_passed": (
            all_required_bitwise
            and no_required_regression
            and any_required_faster
        ),
        "winner_modes": sorted(set(winner_modes)),
        "cases": list(cases),
        "next_gate": (
            "all-operator family benchmark, retained-tensor bitwise validation, "
            "all E7 buckets, and Quick E2E profiling"
        ),
    }


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    fields = [
        "operator_id",
        "output_shape",
        "combined_mean_ms",
        "exact16_mean_ms",
        "spatial2_mean_ms",
        "prescaled_mean_ms",
        "exact16_bitwise",
        "spatial2_bitwise",
        "prescaled_bitwise",
        "winner",
        "winner_speedup_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            writer.writerow(
                {
                    "operator_id": case["operator_id"],
                    "output_shape": json.dumps(case["output_shape"]),
                    "combined_mean_ms": case["modes"]["combined"]["mean_ms"],
                    "exact16_mean_ms": case["modes"]["v2_exact16"]["mean_ms"],
                    "spatial2_mean_ms": case["modes"]["v2_spatial2"]["mean_ms"],
                    "prescaled_mean_ms": case["modes"]["v2_prescaled"]["mean_ms"],
                    "exact16_bitwise": case["bitwise_by_mode"]["v2_exact16"],
                    "spatial2_bitwise": case["bitwise_by_mode"]["v2_spatial2"],
                    "prescaled_bitwise": case["bitwise_by_mode"]["v2_prescaled"],
                    "winner": case["winner"],
                    "winner_speedup_ratio": case["winner_speedup_ratio"],
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=_path,
        default=ROOT / "results/profiling/e7_98/final_combined/operator_profile.json",
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--all-operators", action="store_true")
    parser.add_argument(
        "--runs-dir",
        type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/bn_v2/raw",
    )
    parser.add_argument(
        "--output-dir",
        type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/bn_v2",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.warmup < 0 or args.repeat <= 0:
            raise BnV2BenchmarkError("invalid warmup or repeat")
        if os.name != "posix" and not args.preflight_only:
            raise BnV2BenchmarkError("actual benchmark requires Linux/QRB2210")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        cases = select_cases(
            profile.get("operators", []), all_operators=args.all_operators
        )
        plan = args.plan or DIAGNOSIS._resolve_profile_artifact(profile, "plan")
        weights = args.weights or DIAGNOSIS._resolve_profile_artifact(
            profile, "weights"
        )
        missing = [
            path
            for path in (args.binary, plan, weights, *args.features)
            if not path.is_file()
        ]
        if missing:
            raise BnV2BenchmarkError(
                "required files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3:
            raise BnV2BenchmarkError("exactly three inputs are required")
        capabilities = DIAGNOSIS._run_payload(
            [str(args.binary), "--capabilities"]
        )
        exposed = capabilities.get("bn_candidates", [])
        if any(mode not in exposed for mode in MODES):
            raise BnV2BenchmarkError("binary does not expose BN v2 candidates")
        summary_path = args.output_dir / "summary.json"
        if summary_path.exists() and not args.force and not args.preflight_only:
            raise BnV2BenchmarkError(
                f"output exists: {summary_path} (use --force)"
            )
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "scope": (
                            "all_operators"
                            if args.all_operators
                            else "channel_representatives"
                        ),
                        "case_count": len(cases),
                        "modes": list(MODES),
                        "cases": cases,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        estimated_minutes = max(
            1, math.ceil(len(cases) * len(args.features) * len(MODES) / 6)
        )
        print(f"예상 시간: 약 {estimated_minutes}분", flush=True)
        started = time.monotonic()
        measured_cases: list[dict[str, Any]] = []
        for case in cases:
            payloads: dict[str, list[dict[str, Any]]] = {
                mode: [] for mode in MODES
            }
            for feature in args.features:
                expected_hash: str | None = None
                for mode in MODES:
                    payload = DIAGNOSIS._run_payload(
                        _command(
                            args.binary,
                            plan=plan,
                            weights=weights,
                            feature=feature,
                            operator_id=case["operator_id"],
                            warmup=args.warmup,
                            repeat=args.repeat,
                            mode=mode,
                            expected_hash=expected_hash,
                        )
                    )
                    raw_dir = args.runs_dir / mode / case["case_name"]
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    (raw_dir / f"{feature.stem}.json").write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    _validate_payload(payload, case, mode, args.repeat)
                    payloads[mode].append(payload)
                    if mode == "combined":
                        expected_hash = str(payload["output_hash"])
            stats = {
                mode: COMMON.summarize_samples(
                    [
                        int(sample)
                        for payload in payloads[mode]
                        for sample in payload["samples_ns"]
                    ]
                )
                for mode in MODES
            }
            baseline_hashes = [
                item["output_hash"] for item in payloads["combined"]
            ]
            measured_cases.append(
                {
                    **case,
                    "modes": stats,
                    "bitwise_by_mode": {
                        mode: [
                            item["output_hash"] for item in payloads[mode]
                        ]
                        == baseline_hashes
                        for mode in MODES[1:]
                    },
                    "output_hashes": baseline_hashes,
                }
            )
        result = build_result(
            measured_cases,
            warmup=args.warmup,
            repeat=args.repeat,
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
            encoding="utf-8",
            newline="\n",
        )
        _write_csv(args.output_dir / "comparison.csv", result)
        print(
            f"결과: gate={'PASS' if result['candidate_microbench_gate_passed'] else 'FAIL'} "
            f"winner={','.join(result['winner_modes']) or 'none'}"
        )
        return 0 if result["candidate_microbench_gate_passed"] else 3
    except (BnV2BenchmarkError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
