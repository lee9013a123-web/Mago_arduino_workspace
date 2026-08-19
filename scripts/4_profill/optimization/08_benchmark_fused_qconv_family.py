#!/usr/bin/env python3
"""Measure all E7 fused Quant-QConv operators with one graph pass per input."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
PROFILL_SCRIPT_DIR = ROOT / "scripts" / "4_profill"
if str(PROFILL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILL_SCRIPT_DIR))

from profile_statistics import summarize_ns  # noqa: E402


COMPARE = ROOT / "scripts/4_profill/optimization/03_compare_candidate.py"
EXPECTED_INPUTS = (
    ROOT / "benchmarks/campplus/features/multi__speaker_0000__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0005__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0006__98.f32",
)
SUPPORTED_MODES = (
    "baseline", "mac", "combined", "mac_fixed", "quant_neon",
    "combined_fixed", "combined_v4",
)
DEFAULT_MODES = ("baseline", "combined_fixed", "combined_v4")
FULL_DIAGNOSTIC_MODES = (
    "baseline", "mac_fixed", "quant_neon", "combined_fixed", "combined_v4",
)


class FamilyBenchmarkError(RuntimeError):
    """The family matrix could not be executed or summarized."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _pin_cpu_zero() -> None:
    os.sched_setaffinity(0, {0})


def _run(
    command: list[str], allowed_returncodes: tuple[int, ...] = (0,),
) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in allowed_returncodes:
        raise FamilyBenchmarkError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


def _run_payload(
    command: Sequence[str], allowed_returncodes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        list(command), cwd=ROOT, env=environment, text=True,
        capture_output=True, check=False,
        preexec_fn=_pin_cpu_zero if os.name == "posix" else None,
    )
    if completed.returncode not in allowed_returncodes:
        raise FamilyBenchmarkError(
            f"batch benchmark failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FamilyBenchmarkError(
            "batch benchmark did not return one JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise FamilyBenchmarkError("batch benchmark payload is not an object")
    return payload


def _meaningful_weight_shape(operator: dict[str, Any]) -> tuple[int, ...]:
    shapes = operator.get("weight_shapes")
    if not isinstance(shapes, list):
        return ()
    candidates = [
        tuple(int(value) for value in shape)
        for shape in shapes
        if isinstance(shape, list) and len(shape) >= 3
    ]
    return max(candidates, key=lambda shape: math.prod(shape), default=())


def select_family_cases(profile: dict[str, Any]) -> list[dict[str, Any]]:
    operators = profile.get("operators")
    if not isinstance(operators, list):
        raise FamilyBenchmarkError("profile operators are missing")
    selected = [
        operator for operator in operators
        if operator.get("kernel_name") == "fused_quant_qlinear_conv_o4i4"
    ]
    selected.sort(key=lambda item: int(item["operator_id"]))
    if not selected:
        raise FamilyBenchmarkError("profile has no fused Quant-QConv operators")
    return [
        {
            "case_name": f"fused_quant_qconv_op_{int(operator['operator_id'])}",
            "operator_id": int(operator["operator_id"]),
            "kernel_id": int(operator["kernel_id"]),
            "kernel_name": str(operator["kernel_name"]),
            "operator_type": str(operator["operator_type"]),
            "weight_shape": list(_meaningful_weight_shape(operator)),
            "profile_mean_ms": float(operator["mean_ms"]),
            "profile_share_pct": float(operator["end_to_end_share_pct"]),
        }
        for operator in selected
    ]


def _resolve_profile_artifact(profile: dict[str, Any], name: str) -> Path:
    artifacts = profile.get("artifacts")
    entry = artifacts.get(name) if isinstance(artifacts, dict) else None
    value = entry.get("path") if isinstance(entry, dict) else None
    if not isinstance(value, str) or not value:
        raise FamilyBenchmarkError(f"profile artifact is missing: {name}")
    return _path(value)


def _mode_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modes = case.get("modes")
    if not isinstance(modes, list):
        raise FamilyBenchmarkError("batch case modes are missing")
    result = {
        str(mode["name"]): mode for mode in modes if isinstance(mode, dict)
    }
    if len(result) != len(modes):
        raise FamilyBenchmarkError("batch case has duplicate or invalid modes")
    return result


def validate_batch_payload(
    payload: dict[str, Any], cases: Sequence[dict[str, Any]],
    modes: Sequence[str], repeat: int,
) -> None:
    if payload.get("mode") != "fused_qconv_family_batch":
        raise FamilyBenchmarkError("unexpected batch benchmark mode")
    if payload.get("clock") != "CLOCK_MONOTONIC_RAW":
        raise FamilyBenchmarkError("batch benchmark clock is not monotonic raw")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict) or (
        configuration.get("effective_threads") != 1
        or configuration.get("repeat") != repeat
        or configuration.get("graph_traversals") != 1
    ):
        raise FamilyBenchmarkError("batch benchmark configuration mismatch")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise FamilyBenchmarkError("batch benchmark cases are missing")
    by_id = {
        int(case["operator_id"]): case for case in raw_cases
        if isinstance(case, dict) and "operator_id" in case
    }
    expected_ids = {int(case["operator_id"]) for case in cases}
    if set(by_id) != expected_ids or len(by_id) != len(raw_cases):
        raise FamilyBenchmarkError("batch benchmark operator set mismatch")
    expected_modes = set(modes)
    for expected in cases:
        raw = by_id[int(expected["operator_id"])]
        if (
            int(raw.get("kernel_id", -1)) != expected["kernel_id"]
            or raw.get("kernel_name") != expected["kernel_name"]
        ):
            raise FamilyBenchmarkError(
                f"operator identity mismatch: {expected['operator_id']}"
            )
        mode_map = _mode_map(raw)
        if set(mode_map) != expected_modes:
            raise FamilyBenchmarkError(
                f"candidate mode mismatch: {expected['operator_id']}"
            )
        for mode in modes:
            result = mode_map[mode]
            samples = result.get("samples_ns")
            if not isinstance(samples, list) or len(samples) != repeat or any(
                isinstance(value, bool) or not isinstance(value, int)
                or value <= 0 for value in samples
            ):
                raise FamilyBenchmarkError(
                    f"invalid samples: op {expected['operator_id']} {mode}"
                )
            if not isinstance(result.get("output_hash"), str) or (
                result.get("matches_baseline") not in (True, False)
            ):
                raise FamilyBenchmarkError(
                    f"invalid output hash: op {expected['operator_id']} {mode}"
                )


def _batch_command(
    binary: Path, plan: Path, weights: Path, feature: Path,
    modes: Sequence[str], warmup: int, repeat: int,
) -> list[str]:
    command = [
        str(binary), "--plan", str(plan), "--weights", str(weights),
        "--input", str(feature), "--warmup", str(warmup),
        "--repeat", str(repeat), "--threads", "1",
    ]
    for mode in modes:
        command.extend(("--mode", mode))
    return command


def _payload_cases_by_id(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(case["operator_id"]): case for case in payload["cases"]}


def build_mode_documents(
    cases: Sequence[dict[str, Any]],
    input_payloads: Sequence[tuple[Path, dict[str, Any]]],
    modes: Sequence[str], *, warmup: int, repeat: int,
    elapsed_seconds: float, artifacts: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Convert batch payloads to the existing comparison-compatible schema."""

    payload_maps = [
        (feature, _payload_cases_by_id(payload))
        for feature, payload in input_payloads
    ]
    documents: dict[str, dict[str, Any]] = {}
    for mode in modes:
        result_cases = []
        all_bitwise = True
        for case in cases:
            operator_id = int(case["operator_id"])
            samples: list[int] = []
            hashes: dict[str, str] = {}
            for feature, raw_cases in payload_maps:
                mode_result = _mode_map(raw_cases[operator_id])[mode]
                samples.extend(int(value) for value in mode_result["samples_ns"])
                hashes[_display_path(feature)] = str(mode_result["output_hash"])
                all_bitwise = (
                    all_bitwise and bool(mode_result["matches_baseline"])
                )
            result_cases.append(
                {
                    **case,
                    "input_count": len(input_payloads),
                    "output_hashes": hashes,
                    "wall": summarize_ns(samples),
                    "stages": [],
                    "dominant_stage": None,
                    "dominant_stage_wall_share_pct": 0.0,
                    "pmu": {
                        "available": False,
                        "unavailable_errno": [],
                        "totals": {},
                        "instructions_per_cycle": None,
                        "cache_miss_pct": None,
                        "branch_miss_pct": None,
                    },
                    "diagnosis_scope": (
                        "target-only latency and output hash from a single "
                        "graph traversal per input; PMU/stage probes disabled"
                    ),
                }
            )
        documents[mode] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "objective": "benchmark all fused_quant_qlinear_conv_o4i4 operators",
            "configuration": {
                "bucket_frames": 98,
                "threads": 1,
                "cpu_affinity": [0],
                "warmup": warmup,
                "repeat_per_input": repeat,
                "input_count": len(input_payloads),
                "elapsed_seconds": elapsed_seconds,
                "fused_qconv_candidate": mode,
                "graph_traversals_per_input": 1,
            },
            "target_kernel_families": ["fused_quant_qlinear_conv_o4i4"],
            "case_count": len(result_cases),
            "cases": result_cases,
            "optimization_gate": {
                "ready": all_bitwise,
                "all_output_hashes_match_batch_baseline": all_bitwise,
            },
            "artifacts": artifacts,
        }
    return documents


def _shape_groups(document: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, ...], dict[str, Any]] = {}
    for case in document.get("cases", []):
        shape = tuple(int(value) for value in case["weight_shape"])
        group = grouped.setdefault(
            shape,
            {
                "weight_shape": list(shape),
                "operator_count": 0,
                "baseline_mean_ms": 0.0,
                "candidate_mean_ms": 0.0,
                "all_output_hashes_bitwise_identical": True,
                "regressed_operator_ids": [],
            },
        )
        baseline = float(case["baseline_mean_ms"])
        candidate = float(case["candidate_mean_ms"])
        group["operator_count"] += 1
        group["baseline_mean_ms"] += baseline
        group["candidate_mean_ms"] += candidate
        group["all_output_hashes_bitwise_identical"] = (
            group["all_output_hashes_bitwise_identical"]
            and bool(case["bitwise_output_hashes"])
        )
        if candidate > baseline * 1.01:
            group["regressed_operator_ids"].append(case["operator_id"])
    result = []
    for shape in sorted(grouped):
        group = grouped[shape]
        candidate = float(group["candidate_mean_ms"])
        group["mean_speedup_ratio"] = (
            float(group["baseline_mean_ms"]) / candidate
            if candidate else None
        )
        result.append(group)
    return result


def build_summary(
    modes: Sequence[str], result_dir: Path,
) -> dict[str, Any]:
    comparisons = []
    for mode in modes:
        if mode == "baseline":
            continue
        path = result_dir / f"{mode}_comparison.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        aggregate = document["family_operator_sum"]
        comparisons.append(
            {
                "mode": mode,
                "gate_passed": document["candidate_microbench_gate_passed"],
                "all_output_hashes_bitwise_identical": document[
                    "all_output_hashes_bitwise_identical"
                ],
                "no_case_regressed_over_1pct": document[
                    "no_case_regressed_over_1pct"
                ],
                **aggregate,
                "shape_groups": _shape_groups(document),
                "comparison": _display_path(path),
            }
        )
    eligible = [item for item in comparisons if item["gate_passed"]]
    winner = min(
        eligible, key=lambda item: float(item["candidate_mean_ms"]),
        default=None,
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "all fused_quant_qlinear_conv_o4i4 operators",
        "measurement_strategy": "one graph traversal per input",
        "modes": list(modes),
        "comparisons": comparisons,
        "winner": winner["mode"] if winner is not None else None,
        "production_gate_ready": winner is not None,
        "next_gate": (
            "retained-tensor bitwise validation and Quick E2E profiling"
            if winner is not None else "inspect failing family cases"
        ),
    }


def write_comparison_csv(
    modes: Sequence[str], result_dir: Path, output: Path,
) -> None:
    fields = (
        "mode", "operator_id", "weight_shape", "baseline_mean_ms",
        "candidate_mean_ms", "speedup_ratio", "baseline_p50_ms",
        "candidate_p50_ms", "p50_speedup_ratio", "baseline_p95_ms",
        "candidate_p95_ms", "p95_speedup_ratio",
        "bitwise_output_hashes",
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode in modes:
            if mode == "baseline":
                continue
            document = json.loads(
                (result_dir / f"{mode}_comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            for case in document["cases"]:
                writer.writerow(
                    {
                        field: (
                            "x".join(str(value) for value in case[field])
                            if field == "weight_shape" else case.get(field)
                        )
                        for field in fields
                    }
                    | {"mode": mode}
                )


def estimate_seconds(
    profile: dict[str, Any], cases: Sequence[dict[str, Any]],
    modes: Sequence[str], warmup: int, repeat: int, input_count: int,
) -> float:
    family_ms = sum(float(case["profile_mean_ms"]) for case in cases)
    factors = {
        "baseline": 1.0,
        "mac": 0.12,
        "combined": 0.11,
        "mac_fixed": 0.08,
        "quant_neon": 0.98,
        "combined_fixed": 0.08,
        "combined_v4": 0.08,
    }
    measured = (
        family_ms / 1000.0 * (warmup + repeat + 1) * input_count
        * sum(factors[mode] for mode in modes)
    )
    timing = profile.get("timing")
    end_to_end = timing.get("end_to_end") if isinstance(timing, dict) else None
    graph_ms = (
        float(end_to_end.get("mean_ms", 0.0))
        if isinstance(end_to_end, dict) else 0.0
    )
    non_target = max(0.0, graph_ms - family_ms) / 1000.0 * input_count
    return (measured + non_target + 5.0) * 1.10


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes", nargs="+", choices=SUPPORTED_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--binary", type=_path,
        default=(
            ROOT / "build/profill/optimization/"
            "campp_fused_qconv_family_bench"
        ),
    )
    parser.add_argument(
        "--profile", type=_path,
        default=ROOT / "results/profiling/e7_98/operator_profile.json",
    )
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument(
        "--features", type=_path, nargs="+", default=list(EXPECTED_INPUTS),
    )
    parser.add_argument(
        "--runs-dir", type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/fused_qconv_family",
    )
    parser.add_argument(
        "--results-dir", type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/fused_qconv_family",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.warmup < 0 or args.repeat <= 0:
            raise FamilyBenchmarkError("warmup must be >= 0 and repeat > 0")
        modes = list(dict.fromkeys(args.modes))
        if "baseline" not in modes:
            modes.insert(0, "baseline")
        if not args.profile.is_file():
            raise FamilyBenchmarkError(f"profile not found: {args.profile}")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        cases = select_family_cases(profile)
        plan = args.plan or _resolve_profile_artifact(profile, "plan")
        weights = args.weights or _resolve_profile_artifact(profile, "weights")
        required = [args.binary, plan, weights, *args.features]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FamilyBenchmarkError(
                "required files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3 or any(
            path.stat().st_size != 98 * 80 * 4 for path in args.features
        ):
            raise FamilyBenchmarkError(
                "exactly three float32 [1,98,80] inputs are required"
            )
        capabilities = _run_payload([str(args.binary), "--capabilities"])
        if capabilities.get("batch_graph_traversal") is not True:
            raise FamilyBenchmarkError("binary does not support batch traversal")
        available = capabilities.get("fused_qconv_candidates", [])
        if any(mode not in available for mode in modes):
            raise FamilyBenchmarkError("binary does not expose requested modes")
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "binary": _display_path(args.binary),
                        "plan": _display_path(plan),
                        "weights": _display_path(weights),
                        "features": [_display_path(path) for path in args.features],
                        "operator_count": len(cases),
                        "modes": modes,
                        "graph_traversals": len(args.features),
                    },
                    ensure_ascii=False, indent=2,
                )
            )
            return 0
        if os.name != "posix":
            raise FamilyBenchmarkError("actual benchmark requires Linux/QRB2210")
        summary_path = args.results_dir / "summary.json"
        planned_outputs = [
            summary_path,
            args.results_dir / "comparison.csv",
            *(args.results_dir / f"{mode}.json" for mode in modes),
            *(
                args.results_dir / f"{mode}_comparison.json"
                for mode in modes if mode != "baseline"
            ),
        ]
        existing_outputs = [path for path in planned_outputs if path.exists()]
        if existing_outputs and not args.force:
            raise FamilyBenchmarkError(
                f"output exists: {existing_outputs[0]} (use --force)"
            )

        expected = estimate_seconds(
            profile, cases, modes, args.warmup, args.repeat,
            len(args.features),
        )
        print(f"예상 시간: 약 {max(1, math.ceil(expected / 60.0))}분")
        raw_dir = args.runs_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        input_payloads: list[tuple[Path, dict[str, Any]]] = []
        started = time.monotonic()
        for feature in args.features:
            print("진행 중", flush=True)
            payload = _run_payload(
                _batch_command(
                    args.binary, plan, weights, feature, modes,
                    args.warmup, args.repeat,
                ),
                (0, 3),
            )
            validate_batch_payload(payload, cases, modes, args.repeat)
            input_payloads.append((feature, payload))
            (raw_dir / f"{feature.stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
        elapsed = time.monotonic() - started
        artifacts = {
            "profile": _display_path(args.profile),
            "plan": _display_path(plan),
            "weights": _display_path(weights),
            "raw_dir": _display_path(raw_dir),
        }
        documents = build_mode_documents(
            cases, input_payloads, modes,
            warmup=args.warmup, repeat=args.repeat,
            elapsed_seconds=elapsed, artifacts=artifacts,
        )
        args.results_dir.mkdir(parents=True, exist_ok=True)
        for mode, document in documents.items():
            (args.results_dir / f"{mode}.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )

        baseline = args.results_dir / "baseline.json"
        for mode in modes:
            if mode == "baseline":
                continue
            command = [
                sys.executable, str(COMPARE),
                "--baseline", str(baseline),
                "--candidate", str(args.results_dir / f"{mode}.json"),
                "--output", str(args.results_dir / f"{mode}_comparison.json"),
            ]
            if args.force:
                command.append("--force")
            _run(command, (0, 3))

        summary = build_summary(modes, args.results_dir)
        summary["elapsed_seconds"] = elapsed
        summary["estimated_seconds"] = expected
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        write_comparison_csv(
            modes, args.results_dir, args.results_dir / "comparison.csv"
        )
        print(f"fused family summary: {_display_path(summary_path)}")
        return 0 if summary["production_gate_ready"] else 3
    except (FamilyBenchmarkError, OSError, ValueError, KeyError) as exc:
        print(f"fused family benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
