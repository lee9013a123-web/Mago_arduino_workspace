#!/usr/bin/env python3
"""Build GCC/Clang QConv variants and compare latency, bitwise hashes and spill."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = Path(__file__).with_name("01_build_optimization.sh")
HOTSPOT_SCRIPT = Path(__file__).with_name("04_profile_qconv_hotspot.py")
DEFAULT_CFLAGS = "-std=c11 -O3 -g -DNDEBUG -Wall -Wextra -mcpu=native"


class SpillComparisonError(RuntimeError):
    """Compiler build, profiling artifact or comparison gate is invalid."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise SpillComparisonError("latency samples are missing")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_profile(
    summary_path: Path, *, compiler: str, mode: str
) -> dict[str, Any]:
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    configured_mode = document.get("configuration", {}).get("qconv_candidate")
    if configured_mode != mode:
        raise SpillComparisonError(
            f"candidate mismatch in {summary_path}: {configured_mode} != {mode}"
        )
    cases = []
    for case in document.get("cases", []):
        if mode in ("mac_fixed", "mac_asm") and int(
            case.get("aggregate", {}).get("fixed_microkernel_sample_count", 0)
        ) <= 0:
            raise SpillComparisonError(
                f"fixed microkernel was not sampled for {case.get('case_name')}"
            )
        samples_ms: list[float] = []
        hashes = []
        for input_result in case.get("inputs", []):
            artifact = Path(input_result["artifacts"]["microbench"])
            if not artifact.is_absolute():
                artifact = ROOT / artifact
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            if payload.get("qconv_candidate") != mode:
                raise SpillComparisonError(f"microbench mode mismatch: {artifact}")
            if payload.get("output_hash_matches") is not True:
                raise SpillComparisonError(f"output hash gate failed: {artifact}")
            samples_ms.extend(float(value) / 1_000_000.0 for value in payload["samples_ns"])
            hashes.append(str(payload["output_hash"]))
        spill = float(
            case.get("aggregate", {}).get(
                "stack_spill_share_of_annotated_pct", 0.0
            )
        )
        cases.append(
            {
                "case_name": str(case["case_name"]),
                "operator_id": int(case["operator_id"]),
                "weight_shape": case.get("weight_shape"),
                "sample_count": len(samples_ms),
                "mean_ms": statistics.fmean(samples_ms),
                "p50_ms": _percentile(samples_ms, 0.50),
                "p95_ms": _percentile(samples_ms, 0.95),
                "output_hashes": hashes,
                "stack_spill_share_of_annotated_pct": spill,
            }
        )
    if not cases:
        raise SpillComparisonError(f"profile contains no cases: {summary_path}")
    return {
        "compiler": compiler,
        "mode": mode,
        "profile": _display_path(summary_path),
        "cases": cases,
    }


def _case_map(measurement: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_name"]): case for case in measurement["cases"]}


def build_decision(
    measurements: Sequence[dict[str, Any]], spill_limit_pct: float = 5.0
) -> dict[str, Any]:
    indexed = {
        (str(item["compiler"]), str(item["mode"])): item
        for item in measurements
    }
    compilers = sorted({str(item["compiler"]) for item in measurements})
    evaluations = []
    for compiler in compilers:
        baseline = indexed.get((compiler, "mac"))
        fixed = indexed.get((compiler, "mac_fixed"))
        if baseline is None or fixed is None:
            continue
        baseline_cases = _case_map(baseline)
        fixed_cases = _case_map(fixed)
        if set(baseline_cases) != set(fixed_cases):
            raise SpillComparisonError(f"case set differs for {compiler}")
        case_results = []
        for name in baseline_cases:
            before = baseline_cases[name]
            after = fixed_cases[name]
            bitwise = before["output_hashes"] == after["output_hashes"]
            before_mean = float(before["mean_ms"])
            after_mean = float(after["mean_ms"])
            case_results.append(
                {
                    "case_name": name,
                    "bitwise": bitwise,
                    "mac_mean_ms": before_mean,
                    "mac_fixed_mean_ms": after_mean,
                    "speedup_ratio": before_mean / after_mean,
                    "fixed_spill_pct": float(
                        after["stack_spill_share_of_annotated_pct"]
                    ),
                }
            )
        all_bitwise = all(case["bitwise"] for case in case_results)
        no_regression = all(
            case["mac_fixed_mean_ms"] <= case["mac_mean_ms"] * 1.01
            for case in case_results
        )
        maximum_spill = max(case["fixed_spill_pct"] for case in case_results)
        evaluations.append(
            {
                "compiler": compiler,
                "all_output_hashes_bitwise_identical": all_bitwise,
                "no_case_regressed_over_1pct": no_regression,
                "maximum_fixed_spill_pct": maximum_spill,
                "fixed_total_mean_ms": sum(
                    case["mac_fixed_mean_ms"] for case in case_results
                ),
                "cases": case_results,
            }
        )
    eligible = [
        item
        for item in evaluations
        if item["all_output_hashes_bitwise_identical"]
        and item["no_case_regressed_over_1pct"]
    ]
    selected = min(
        eligible, key=lambda item: float(item["fixed_total_mean_ms"]), default=None
    )
    assembly_required = (
        selected is not None
        and float(selected["maximum_fixed_spill_pct"]) > spill_limit_pct
    )
    return {
        "ready": selected is not None,
        "selected_compiler": selected["compiler"] if selected else None,
        "spill_limit_pct": spill_limit_pct,
        "assembly_required": assembly_required,
        "next_mode": "mac_asm" if assembly_required else "mac_fixed",
        "rule": (
            "bitwise hashes must match, no case may regress over 1%; "
            "choose the lowest total mean latency, then require assembly "
            "when maximum fixed spill exceeds the configured limit"
        ),
        "compiler_evaluations": evaluations,
    }


def _run(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        list(command), cwd=ROOT, env=env, text=True, check=False
    )
    if completed.returncode != 0:
        raise SpillComparisonError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


def _compiler_version(compiler: str) -> str:
    completed = subprocess.run(
        [compiler, "--version"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise SpillComparisonError(f"cannot query compiler: {compiler}")
    return completed.stdout.splitlines()[0]


def _write_outputs(
    measurements: Sequence[dict[str, Any]], metadata: dict[str, Any],
    decision: dict[str, Any], results_dir: Path
) -> None:
    document = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "select the lowest-spill, fastest fixed QConv compiler",
        "configuration": metadata,
        "measurements": list(measurements),
        "decision": decision,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = results_dir / "compiler_comparison.json"
    comparison_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    rows = []
    for measurement in measurements:
        for case in measurement["cases"]:
            rows.append(
                {
                    "compiler": measurement["compiler"],
                    "mode": measurement["mode"],
                    "case_name": case["case_name"],
                    "operator_id": case["operator_id"],
                    "sample_count": case["sample_count"],
                    "mean_ms": case["mean_ms"],
                    "p50_ms": case["p50_ms"],
                    "p95_ms": case["p95_ms"],
                    "stack_spill_pct": case[
                        "stack_spill_share_of_annotated_pct"
                    ],
                    "output_hashes": ";".join(case["output_hashes"]),
                }
            )
    with (results_dir / "compiler_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compilers", nargs="+", default=["gcc", "clang"])
    parser.add_argument("--cflags", default=DEFAULT_CFLAGS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sample-period", type=int, default=100_000)
    parser.add_argument("--spill-limit-pct", type=float, default=5.0)
    parser.add_argument("--include-assembly", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--build-root", type=_path,
        default=ROOT / "build/profill/optimization/qconv_spill",
    )
    parser.add_argument(
        "--runs-dir", type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/qconv_spill",
    )
    parser.add_argument(
        "--results-dir", type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/qconv_spill",
    )
    args = parser.parse_args(argv)
    try:
        if args.warmup < 0 or args.repeat <= 0 or args.spill_limit_pct < 0.0:
            raise SpillComparisonError("invalid warmup, repeat or spill limit")
        missing = [
            name for name in ("bash", *args.compilers)
            if shutil.which(name) is None
        ]
        required = [BUILD_SCRIPT, HOTSPOT_SCRIPT]
        missing.extend(str(path) for path in required if not path.is_file())
        preflight = {
            "ready": not missing,
            "missing": missing,
            "compilers": args.compilers,
            "cflags": args.cflags,
            "profile_count": len(args.compilers) * (3 if args.include_assembly else 2),
            "note": "run on the fixed QRB2210 CPU; perf time dominates build time",
        }
        if args.preflight_only:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0 if preflight["ready"] else 2
        if missing:
            raise SpillComparisonError(f"preflight missing: {', '.join(missing)}")
        if os.name != "posix":
            raise SpillComparisonError("actual comparison requires Linux/QRB2210")
        if (args.results_dir / "compiler_comparison.json").exists() and not args.force:
            raise SpillComparisonError("result exists; use --force")

        modes = ["mac", "mac_fixed"]
        if args.include_assembly:
            modes.append("mac_asm")
        measurements = []
        versions = {}
        for compiler in args.compilers:
            versions[compiler] = _compiler_version(compiler)
            build_dir = args.build_root / compiler
            if not args.skip_build:
                environment = os.environ.copy()
                environment.update(
                    {"CC": compiler, "CFLAGS": args.cflags,
                     "BUILD_DIR": str(build_dir)}
                )
                _run(["bash", str(BUILD_SCRIPT)], env=environment)
            for test_name in (
                "test_qconv_candidate", "test_qconv_microkernel_4x8"
            ):
                _run([str(build_dir / test_name)])
            for mode in modes:
                profile_runs = args.runs_dir / compiler / mode
                profile_results = args.results_dir / "profiles" / compiler / mode
                command = [
                    sys.executable, str(HOTSPOT_SCRIPT),
                    "--binary", str(build_dir / "campp_operator_hotspot"),
                    "--qconv-candidate", mode,
                    "--warmup", str(args.warmup),
                    "--repeat", str(args.repeat),
                    "--sample-period", str(args.sample_period),
                    "--runs-dir", str(profile_runs),
                    "--results-dir", str(profile_results),
                    "--force",
                ]
                _run(command)
                measurements.append(
                    summarize_profile(
                        profile_results / "qconv_hotspot.json",
                        compiler=compiler, mode=mode,
                    )
                )
        decision = build_decision(measurements, args.spill_limit_pct)
        metadata = {
            "bucket_frames": 98,
            "threads": 1,
            "cpu_affinity": [0],
            "warmup": args.warmup,
            "repeat_per_input": args.repeat,
            "input_count": 3,
            "cflags": args.cflags,
            "compiler_versions": versions,
            "modes": modes,
        }
        _write_outputs(measurements, metadata, decision, args.results_dir)
        print(f"qconv spill comparison: {_display_path(args.results_dir)}")
        return 0 if decision["ready"] else 3
    except (SpillComparisonError, OSError, ValueError, KeyError) as exc:
        print(f"qconv spill comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
