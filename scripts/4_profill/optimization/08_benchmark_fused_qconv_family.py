#!/usr/bin/env python3
"""Run the full E7 fused Quant-QConv family candidate matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
DIAGNOSE = ROOT / "scripts/4_profill/optimization/02_diagnose_top4.py"
COMPARE = ROOT / "scripts/4_profill/optimization/03_compare_candidate.py"
DEFAULT_MODES = ("baseline", "mac_fixed", "quant_neon", "combined_fixed")


class FamilyBenchmarkError(RuntimeError):
    """The family matrix could not be executed or summarized."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _run(command: list[str], allowed_returncodes: tuple[int, ...] = (0,)) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in allowed_returncodes:
        raise FamilyBenchmarkError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


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
                "comparison": str(path),
            }
        )
    eligible = [item for item in comparisons if item["gate_passed"]]
    winner = min(
        eligible,
        key=lambda item: float(item["candidate_mean_ms"]),
        default=None,
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "all fused_quant_qlinear_conv_o4i4 operators",
        "modes": list(modes),
        "comparisons": comparisons,
        "winner": winner["mode"] if winner is not None else None,
        "production_gate_ready": winner is not None,
        "next_gate": (
            "retained-tensor bitwise validation and Quick E2E profiling"
            if winner is not None
            else "inspect failing family cases"
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
                            if field == "weight_shape"
                            else case.get(field)
                        )
                        for field in fields
                    }
                    | {"mode": mode}
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes", nargs="+", choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
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
        modes = list(dict.fromkeys(args.modes))
        if "baseline" not in modes:
            modes.insert(0, "baseline")
        if args.preflight_only:
            _run(
                [
                    sys.executable, str(DIAGNOSE),
                    "--fused-qconv-family",
                    "--fused-qconv-candidate", modes[-1],
                    "--warmup", str(args.warmup),
                    "--repeat", str(args.repeat),
                    "--preflight-only",
                ]
            )
            return 0

        args.results_dir.mkdir(parents=True, exist_ok=True)
        for mode in modes:
            command = [
                sys.executable, str(DIAGNOSE),
                "--fused-qconv-family",
                "--fused-qconv-candidate", mode,
                "--warmup", str(args.warmup),
                "--repeat", str(args.repeat),
                "--runs-dir", str(args.runs_dir / mode),
                "--output", str(args.results_dir / f"{mode}.json"),
            ]
            if args.force:
                command.append("--force")
            _run(command)

        baseline = args.results_dir / "baseline.json"
        for mode in modes:
            if mode == "baseline":
                continue
            command = [
                sys.executable, str(COMPARE),
                "--baseline", str(baseline),
                "--candidate", str(args.results_dir / f"{mode}.json"),
                "--output", str(
                    args.results_dir / f"{mode}_comparison.json"
                ),
            ]
            if args.force:
                command.append("--force")
            _run(command, (0, 3))

        summary = build_summary(modes, args.results_dir)
        summary_path = args.results_dir / "summary.json"
        if summary_path.exists() and not args.force:
            raise FamilyBenchmarkError(
                f"output exists: {summary_path} (use --force)"
            )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        write_comparison_csv(
            modes, args.results_dir, args.results_dir / "comparison.csv"
        )
        print(f"fused family summary: {summary_path}")
        return 0 if summary["production_gate_ready"] else 3
    except (FamilyBenchmarkError, OSError, ValueError, KeyError) as exc:
        print(f"fused family benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
