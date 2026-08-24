#!/usr/bin/env python3
"""두 E7 내부 진단 결과의 bitwise hash와 latency를 비교한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]


class ComparisonError(RuntimeError):
    """비교할 진단 결과가 누락되거나 서로 호환되지 않는다."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _case_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ComparisonError("diagnosis cases are missing")
    result = {str(case["case_name"]): case for case in cases}
    if len(result) != len(cases):
        raise ComparisonError("diagnosis contains duplicate case names")
    return result


def compare_diagnoses(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_cases = _case_map(baseline)
    candidate_cases = _case_map(candidate)
    if set(baseline_cases) != set(candidate_cases):
        raise ComparisonError("baseline and candidate case sets differ")

    comparisons = []
    all_bitwise = True
    for name in baseline_cases:
        before = baseline_cases[name]
        after = candidate_cases[name]
        identity_fields = ("operator_id", "kernel_id", "kernel_name", "weight_shape")
        if any(before.get(field) != after.get(field) for field in identity_fields):
            raise ComparisonError(f"operator identity differs for {name}")
        bitwise = before.get("output_hashes") == after.get("output_hashes")
        all_bitwise = all_bitwise and bitwise
        before_mean = float(before["wall"]["mean_ms"])
        after_mean = float(after["wall"]["mean_ms"])
        before_p50 = float(before["wall"]["p50_ms"])
        after_p50 = float(after["wall"]["p50_ms"])
        before_p95 = float(before["wall"]["p95_ms"])
        after_p95 = float(after["wall"]["p95_ms"])
        comparisons.append(
            {
                "case_name": name,
                "operator_id": before["operator_id"],
                "kernel_name": before["kernel_name"],
                "weight_shape": before["weight_shape"],
                "bitwise_output_hashes": bitwise,
                "baseline_mean_ms": before_mean,
                "candidate_mean_ms": after_mean,
                "mean_delta_ms": after_mean - before_mean,
                "speedup_ratio": before_mean / after_mean if after_mean else None,
                "latency_reduction_pct": (
                    (before_mean - after_mean) / before_mean * 100.0
                    if before_mean
                    else None
                ),
                "baseline_p50_ms": before_p50,
                "candidate_p50_ms": after_p50,
                "p50_speedup_ratio": (
                    before_p50 / after_p50 if after_p50 else None
                ),
                "baseline_p95_ms": before_p95,
                "candidate_p95_ms": after_p95,
                "p95_speedup_ratio": (
                    before_p95 / after_p95 if after_p95 else None
                ),
            }
        )
    any_faster = any(
        item["candidate_mean_ms"] < item["baseline_mean_ms"]
        for item in comparisons
    )
    no_case_regressed = all(
        item["candidate_mean_ms"] <= item["baseline_mean_ms"] * 1.01
        for item in comparisons
    )
    baseline_sum_mean = sum(item["baseline_mean_ms"] for item in comparisons)
    candidate_sum_mean = sum(item["candidate_mean_ms"] for item in comparisons)
    baseline_sum_p50 = sum(item["baseline_p50_ms"] for item in comparisons)
    candidate_sum_p50 = sum(item["candidate_p50_ms"] for item in comparisons)
    baseline_sum_p95 = sum(item["baseline_p95_ms"] for item in comparisons)
    candidate_sum_p95 = sum(item["candidate_p95_ms"] for item in comparisons)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "e7_operator_candidate_vs_baseline",
        "all_output_hashes_bitwise_identical": all_bitwise,
        "any_case_faster": any_faster,
        "no_case_regressed_over_1pct": no_case_regressed,
        "candidate_microbench_gate_passed": (
            all_bitwise and any_faster and no_case_regressed
        ),
        "family_operator_sum": {
            "operator_count": len(comparisons),
            "baseline_mean_ms": baseline_sum_mean,
            "candidate_mean_ms": candidate_sum_mean,
            "mean_speedup_ratio": (
                baseline_sum_mean / candidate_sum_mean
                if candidate_sum_mean else None
            ),
            "baseline_sum_p50_ms": baseline_sum_p50,
            "candidate_sum_p50_ms": candidate_sum_p50,
            "sum_p50_speedup_ratio": (
                baseline_sum_p50 / candidate_sum_p50
                if candidate_sum_p50 else None
            ),
            "baseline_sum_p95_ms": baseline_sum_p95,
            "candidate_sum_p95_ms": candidate_sum_p95,
            "sum_p95_speedup_ratio": (
                baseline_sum_p95 / candidate_sum_p95
                if candidate_sum_p95 else None
            ),
            "note": (
                "p50/p95 sums are operator-level aggregate indicators, not "
                "a synchronized whole-graph percentile"
            ),
        },
        "cases": comparisons,
        "next_gate": "full retained-tensor bitwise validation and Quick E2E profiling",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=_path, required=True)
    parser.add_argument("--candidate", type=_path, required=True)
    parser.add_argument(
        "--output",
        type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/candidate_comparison.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() and not args.force:
            raise ComparisonError(f"output exists: {args.output} (use --force)")
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        result = compare_diagnoses(baseline, candidate)
        result["artifacts"] = {
            "baseline": _display_path(args.baseline),
            "candidate": _display_path(args.candidate),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"candidate comparison: {args.output}")
        return 0 if result["candidate_microbench_gate_passed"] else 3
    except (ComparisonError, OSError, ValueError, KeyError) as exc:
        print(f"candidate comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
