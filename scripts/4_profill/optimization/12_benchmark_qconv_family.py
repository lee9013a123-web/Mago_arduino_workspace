#!/usr/bin/env python3
"""Measure all ordinary E7 QLinearConv operators in one graph pass per input."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
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
from bucket_profile_identity import (  # noqa: E402
    BucketProfileIdentityError,
    validate_profile_against_plan,
)


DIAGNOSIS_PATH = Path(__file__).with_name("02_diagnose_top4.py")
COMPARE = Path(__file__).with_name("03_compare_candidate.py")
SUPPORTED_MODES = ("baseline", "mac_fixed", "v4", "v5")
DEFAULT_MODES = ("baseline", "mac_fixed", "v4", "v5")
ORDINARY_PROFILE_KERNELS = {
    "qlinear_conv_o4i4_neon",
    "qlinear_conv_o4i4_layer_hybrid_v3",
}

SPEC = importlib.util.spec_from_file_location("diagnose_top4", DIAGNOSIS_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {DIAGNOSIS_PATH}")
DIAGNOSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DIAGNOSIS
SPEC.loader.exec_module(DIAGNOSIS)


class QconvFamilyError(RuntimeError):
    """QConv family benchmark input or execution failed."""


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


def select_qconv_family_cases(profile: dict[str, Any]) -> list[dict[str, Any]]:
    operators = profile.get("operators")
    if not isinstance(operators, list):
        raise QconvFamilyError("profile operator list is missing")
    selected = [
        operator for operator in operators
        if operator.get("operator_type") == "QLINEAR_CONV"
        and operator.get("kernel_name") in ORDINARY_PROFILE_KERNELS
    ]
    if not selected:
        raise QconvFamilyError("profile is missing ordinary QLinearConv ops")
    selected.sort(key=lambda item: int(item["operator_id"]))
    return [
        {
            "case_name": f"qconv_op_{int(operator['operator_id'])}",
            "operator_id": int(operator["operator_id"]),
            "kernel_id": int(operator["kernel_id"]),
            "kernel_name": "qlinear_conv_o4i4_neon",
            "profile_kernel_name": str(operator["kernel_name"]),
            "operator_type": str(operator["operator_type"]),
            "weight_shape": list(DIAGNOSIS._meaningful_weight_shape(operator)),
            "profile_mean_ms": float(operator["mean_ms"]),
            "profile_share_pct": float(operator["end_to_end_share_pct"]),
        }
        for operator in selected
    ]


def _run(command: list[str], allowed_returncodes: tuple[int, ...] = (0,)) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in allowed_returncodes:
        raise QconvFamilyError(
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
        raise QconvFamilyError(
            f"batch benchmark failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise QconvFamilyError(
            "batch benchmark did not return one JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise QconvFamilyError("batch benchmark payload is not an object")
    return payload


def _mode_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modes = case.get("modes")
    if not isinstance(modes, list):
        raise QconvFamilyError("batch case modes are missing")
    result = {
        str(mode["name"]): mode for mode in modes if isinstance(mode, dict)
    }
    if len(result) != len(modes):
        raise QconvFamilyError("batch case has duplicate or invalid modes")
    return result


def validate_batch_payload(
    payload: dict[str, Any], cases: Sequence[dict[str, Any]],
    modes: Sequence[str], repeat: int, baseline_check_only: bool = False,
    bucket_frames: int = 98,
    expected_operator_count: int | None = None,
) -> None:
    if payload.get("mode") != "qconv_family_batch":
        raise QconvFamilyError("unexpected batch benchmark mode")
    if payload.get("clock") != "CLOCK_MONOTONIC_RAW":
        raise QconvFamilyError("batch benchmark clock is not monotonic raw")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict) or (
        configuration.get("effective_threads") != 1
        or configuration.get("repeat") != repeat
        or configuration.get("graph_traversals") != 1
        or bool(configuration.get("baseline_check_only")) != baseline_check_only
    ):
        raise QconvFamilyError("batch benchmark configuration mismatch")
    model = payload.get("model")
    if not isinstance(model, dict) or (
        model.get("bucket_frames") != bucket_frames
    ):
        raise QconvFamilyError("batch benchmark model bucket mismatch")
    if expected_operator_count is not None and (
        model.get("operator_count") != expected_operator_count
    ):
        raise QconvFamilyError("batch benchmark model operator count mismatch")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise QconvFamilyError("batch benchmark cases are missing")
    by_id = {
        int(case["operator_id"]): case for case in raw_cases
        if isinstance(case, dict) and "operator_id" in case
    }
    expected_ids = {int(case["operator_id"]) for case in cases}
    if set(by_id) != expected_ids or len(by_id) != len(raw_cases):
        raise QconvFamilyError("batch benchmark operator set mismatch")
    expected_modes = set(modes)
    for expected in cases:
        raw = by_id[int(expected["operator_id"])]
        if (
            int(raw.get("kernel_id", -1)) != expected["kernel_id"]
            or raw.get("kernel_name") != expected["kernel_name"]
        ):
            raise QconvFamilyError(
                f"operator identity mismatch: {expected['operator_id']}"
            )
        mode_map = _mode_map(raw)
        if set(mode_map) != expected_modes:
            raise QconvFamilyError(
                f"candidate mode mismatch: {expected['operator_id']}"
            )
        for mode in modes:
            result = mode_map[mode]
            samples = result.get("samples_ns")
            # baseline_check_only에서 baseline은 hash 확인 겸 1회만 실행한다.
            expected_samples = (
                1 if baseline_check_only and mode == "baseline" else repeat
            )
            if not isinstance(samples, list) or (
                len(samples) != expected_samples
            ) or any(
                isinstance(value, bool) or not isinstance(value, int)
                or value <= 0 for value in samples
            ):
                raise QconvFamilyError(
                    f"invalid samples: op {expected['operator_id']} {mode}"
                )
            if not isinstance(result.get("output_hash"), str) or (
                result.get("matches_baseline") not in (True, False)
            ):
                raise QconvFamilyError(
                    f"invalid output hash: op {expected['operator_id']} {mode}"
                )


def _batch_command(
    binary: Path, plan: Path, weights: Path, feature: Path,
    modes: Sequence[str], warmup: int, repeat: int,
    baseline_check_only: bool = False,
) -> list[str]:
    command = [
        str(binary), "--plan", str(plan), "--weights", str(weights),
        "--input", str(feature), "--warmup", str(warmup),
        "--repeat", str(repeat), "--threads", "1",
    ]
    if baseline_check_only:
        command.append("--baseline-check-only")
    for mode in modes:
        command.extend(("--mode", mode))
    return command


def _payload_cases_by_id(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(case["operator_id"]): case for case in payload["cases"]}


def build_mode_documents(
    cases: Sequence[dict[str, Any]],
    input_payloads: Sequence[tuple[Path, dict[str, Any]]],
    modes: Sequence[str], *, warmup: int, repeat: int,
    elapsed_seconds: float, artifacts: dict[str, Any],
    bucket_frames: int = 98,
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
            "schema_version": 2,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "objective": "benchmark all ordinary qlinear_conv_o4i4 operators",
            "configuration": {
                "bucket_frames": bucket_frames,
                "threads": 1,
                "cpu_affinity": [0],
                "warmup": warmup,
                "repeat_per_input": repeat,
                "input_count": len(input_payloads),
                "elapsed_seconds": elapsed_seconds,
                "qconv_candidate": mode,
                "graph_traversals_per_input": 1,
            },
            "target_kernel_families": ["qlinear_conv_o4i4_neon"],
            "case_count": len(result_cases),
            "cases": result_cases,
            "optimization_gate": {
                "ready": all_bitwise,
                "all_output_hashes_match_batch_baseline": all_bitwise,
            },
            "artifacts": artifacts,
        }
    return documents


def build_summary(modes: Sequence[str], results_dir: Path) -> dict[str, Any]:
    comparisons = []
    for mode in modes:
        if mode == "baseline":
            continue
        path = results_dir / f"{mode}_comparison.json"
        document = json.loads(path.read_text(encoding="utf-8"))
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
                **document["family_operator_sum"],
                "comparison": _display_path(path),
            }
        )
    eligible = [item for item in comparisons if item["gate_passed"]]
    winner = min(
        eligible, key=lambda item: float(item["candidate_mean_ms"]),
        default=None,
    )
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "all ordinary qlinear_conv_o4i4 operators",
        "measurement_strategy": "one graph traversal per input",
        "modes": list(modes),
        "comparisons": comparisons,
        "winner": winner["mode"] if winner is not None else None,
        "production_gate_ready": winner is not None,
        "next_gate": "build layer-hybrid plan, then retained-tensor and Quick E2E",
    }


def compare_modes(modes: Sequence[str], results_dir: Path, force: bool) -> None:
    for mode in modes:
        if mode == "baseline":
            continue
        output = results_dir / f"{mode}_comparison.json"
        command = [
            sys.executable,
            str(COMPARE),
            "--baseline",
            str(results_dir / "baseline.json"),
            "--candidate",
            str(results_dir / f"{mode}.json"),
            "--output",
            str(output),
        ]
        if force:
            command.append("--force")
        _run(command, allowed_returncodes=(0, 3))


def write_comparison_csv(
    modes: Sequence[str], results_dir: Path, output: Path,
) -> None:
    fields = (
        "mode", "operator_id", "weight_shape", "baseline_mean_ms",
        "candidate_mean_ms", "speedup_ratio", "baseline_p50_ms",
        "candidate_p50_ms", "p50_speedup_ratio", "baseline_p95_ms",
        "candidate_p95_ms", "p95_speedup_ratio", "bitwise_output_hashes",
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode in modes:
            if mode == "baseline":
                continue
            document = json.loads(
                (results_dir / f"{mode}_comparison.json").read_text(
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
    baseline_check_only: bool = False,
    bucket_frames: int = 98,
) -> float:
    """Estimate batch time from the profiled reference family sum.

    The canonical profile contains the 5.63 s reference QConv family total.
    Board measurements put mac_fixed/v4/v5 near 3--5% of that total. The
    estimator includes every timed kernel call but excludes the per-operator
    process/model-load overhead removed by the batch runner.
    """

    frame_scale = bucket_frames / 98.0
    family_ms = (
        sum(float(case["profile_mean_ms"]) for case in cases) * frame_scale
    )
    factors = {
        "baseline": 1.0,
        "mac_fixed": 0.05,
        "v4": 0.04,
        "v5": 0.04,
    }
    # baseline_check_only에서 baseline은 hash 확인 1회 + 실행 1회뿐이다.
    runs = {
        mode: (
            2 if baseline_check_only and mode == "baseline"
            else warmup + repeat + 1
        )
        for mode in modes
    }
    measured = (
        family_ms / 1000.0 * input_count
        * sum(factors[mode] * runs[mode] for mode in modes)
    )
    timing = profile.get("timing")
    end_to_end = timing.get("end_to_end") if isinstance(timing, dict) else None
    graph_ms = (
        float(end_to_end.get("mean_ms", 0.0)) * frame_scale
        if isinstance(end_to_end, dict) else 0.0
    )
    non_target = max(0.0, graph_ms - family_ms) / 1000.0 * input_count
    return (measured + non_target + 5.0) * 1.10


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=_path,
        default=ROOT / "results/profiling/e7_98/operator_profile.json",
    )
    parser.add_argument(
        "--bucket-frames", type=int, default=98,
        help="input frame bucket recorded in results and used for size checks",
    )
    parser.add_argument(
        "--binary", type=_path,
        default=ROOT / "build/profill/optimization/campp_qconv_family_bench",
    )
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument(
        "--features", type=_path, nargs="+",
        default=list(DIAGNOSIS.EXPECTED_INPUTS),
    )
    parser.add_argument(
        "--modes", nargs="+", choices=SUPPORTED_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--runs-dir", type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/qconv_family",
    )
    parser.add_argument(
        "--results-dir", type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/qconv_family",
    )
    parser.add_argument(
        "--baseline-check-only", action="store_true",
        help=(
            "baseline은 op당 1회만 실행해 hash 기준과 graph 진행에만 쓰고 "
            "반복 성능 측정은 후보 3개에만 적용한다"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.warmup < 0 or args.repeat <= 0:
            raise QconvFamilyError("warmup must be >= 0 and repeat > 0")
        if args.bucket_frames <= 0:
            raise QconvFamilyError("bucket-frames must be positive")
        modes = list(dict.fromkeys(args.modes))
        if "baseline" not in modes:
            modes.insert(0, "baseline")
        if not args.profile.is_file():
            raise QconvFamilyError(f"profile not found: {args.profile}")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        plan = args.plan or DIAGNOSIS._resolve_profile_artifact(profile, "plan")
        weights = args.weights or DIAGNOSIS._resolve_profile_artifact(profile, "weights")
        required = [args.binary, plan, weights, *args.features]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise QconvFamilyError(
                "required files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        identity = validate_profile_against_plan(
            args.profile, plan, args.bucket_frames
        )
        cases = select_qconv_family_cases(profile)
        expected_feature_bytes = args.bucket_frames * 80 * 4
        if not args.features or any(
            path.stat().st_size != expected_feature_bytes
            for path in args.features
        ):
            raise QconvFamilyError(
                "one or more float32 [1,"
                f"{args.bucket_frames},80] inputs are required"
            )
        capabilities = _run_payload([str(args.binary), "--capabilities"])
        if capabilities.get("batch_graph_traversal") is not True:
            raise QconvFamilyError("binary does not support batch traversal")
        if capabilities.get("bucket_support") != "plan_header":
            raise QconvFamilyError(
                "family binary is stale: bucket_support is not plan_header"
            )
        available = capabilities.get("qconv_candidates", [])
        if any(mode not in available for mode in modes):
            raise QconvFamilyError("binary does not expose requested QConv modes")
        if args.preflight_only:
            print(json.dumps({
                "ready": True,
                "bucket_frames": args.bucket_frames,
                "operator_count": len(cases),
                "modes": modes,
                "plan": _display_path(plan),
                "weights": _display_path(weights),
                "features": [_display_path(path) for path in args.features],
                "graph_traversals": len(args.features),
                "profile_plan_identity": identity,
            }, ensure_ascii=False, indent=2))
            return 0
        if os.name != "posix":
            raise QconvFamilyError("actual benchmark requires Linux/QRB2210")
        planned = [
            args.results_dir / "summary.json",
            args.results_dir / "comparison.csv",
            *(args.results_dir / f"{mode}.json" for mode in modes),
            *(
                args.results_dir / f"{mode}_comparison.json"
                for mode in modes if mode != "baseline"
            ),
        ]
        existing = [path for path in planned if path.exists()]
        if existing and not args.force:
            raise QconvFamilyError(f"output exists: {existing[0]} (use --force)")
        estimated = estimate_seconds(
            profile, cases, modes, args.warmup, args.repeat,
            len(args.features),
            baseline_check_only=args.baseline_check_only,
            bucket_frames=args.bucket_frames,
        )
        print(f"예상 시간: 약 {max(1, math.ceil(estimated / 60.0))}분")
        raw_dir = args.runs_dir / "raw_batch"
        raw_dir.mkdir(parents=True, exist_ok=True)
        input_payloads: list[tuple[Path, dict[str, Any]]] = []
        started = time.monotonic()
        for feature_index, feature in enumerate(args.features, start=1):
            print(
                f"진행 중 input {feature_index}/{len(args.features)}",
                flush=True,
            )
            payload = _run_payload(
                _batch_command(
                    args.binary, plan, weights, feature, modes,
                    args.warmup, args.repeat,
                    baseline_check_only=args.baseline_check_only,
                ),
                (0, 3),
            )
            validate_batch_payload(
                payload, cases, modes, args.repeat,
                baseline_check_only=args.baseline_check_only,
                bucket_frames=args.bucket_frames,
                expected_operator_count=int(identity["operator_count"]),
            )
            input_payloads.append((feature, payload))
            (raw_dir / f"{feature.stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
            print(
                f"완료 input {feature_index}/{len(args.features)} "
                f"(operators={len(payload['cases'])})",
                flush=True,
            )
        elapsed = time.monotonic() - started
        artifacts = {
            "profile": _display_path(args.profile),
            "plan": _display_path(plan),
            "weights": _display_path(weights),
            "raw_dir": _display_path(raw_dir),
            "profile_plan_identity": identity,
        }
        documents = build_mode_documents(
            cases, input_payloads, modes,
            warmup=args.warmup, repeat=args.repeat,
            elapsed_seconds=elapsed, artifacts=artifacts,
            bucket_frames=args.bucket_frames,
        )
        args.results_dir.mkdir(parents=True, exist_ok=True)
        for mode, document in documents.items():
            (args.results_dir / f"{mode}.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
        compare_modes(modes, args.results_dir, args.force)
        summary = build_summary(modes, args.results_dir)
        summary["elapsed_seconds"] = elapsed
        summary["estimated_seconds"] = estimated
        (args.results_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        write_comparison_csv(
            modes, args.results_dir, args.results_dir / "comparison.csv"
        )
        print(
            "qconv family summary: "
            f"{_display_path(args.results_dir / 'summary.json')}"
        )
        return 0 if summary["production_gate_ready"] else 3
    except (
        QconvFamilyError, BucketProfileIdentityError,
        OSError, ValueError, KeyError,
    ) as exc:
        print(f"qconv family benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
