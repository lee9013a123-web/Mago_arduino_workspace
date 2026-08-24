#!/usr/bin/env python3
"""Compare final V2 and layer-hybrid V3 E2E/operator profiles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]


class FinalComparisonError(RuntimeError):
    """Final profile inputs are missing or inconsistent."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalComparisonError(f"cannot read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise FinalComparisonError(f"JSON root is not an object: {path}")
    return document


def _e2e(summary: dict[str, Any]) -> dict[str, Any]:
    timing = summary.get("timing")
    value = timing.get("end_to_end") if isinstance(timing, dict) else None
    if not isinstance(value, dict):
        raise FinalComparisonError("summary has no end_to_end timing")
    return value


def _validate_compatible(
    v2_summary: dict[str, Any], v3_summary: dict[str, Any],
) -> None:
    v2_config = v2_summary.get("configuration", {})
    v3_config = v3_summary.get("configuration", {})
    for key in (
        "protocol_mode", "bucket_frames", "threads", "cpu_affinity",
        "warmup", "repeat", "overhead_baseline_repeat", "input_ids",
    ):
        if v2_config.get(key) != v3_config.get(key):
            raise FinalComparisonError(f"V2/V3 configuration differs: {key}")
    for artifact in ("plan", "weights", "fusion_plan"):
        v2_value = v2_summary.get("artifacts", {}).get(artifact, {})
        v3_value = v3_summary.get("artifacts", {}).get(artifact, {})
        if not v2_value.get("sha256") or (
            v2_value.get("sha256") != v3_value.get("sha256")
        ):
            raise FinalComparisonError(f"V2/V3 artifact differs: {artifact}")
    for label, summary in (("V2", v2_summary), ("V3", v3_summary)):
        validity = summary.get("validity", {})
        if validity.get("profile_valid") is not True:
            raise FinalComparisonError(f"{label} profile validity gate failed")


def _operator_deltas(
    v2_profile: dict[str, Any], v3_profile: dict[str, Any],
    selections: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    v2 = {
        int(item["operator_id"]): item
        for item in v2_profile.get("operators", [])
    }
    v3 = {
        int(item["operator_id"]): item
        for item in v3_profile.get("operators", [])
    }
    if not v2 or set(v2) != set(v3):
        raise FinalComparisonError("V2/V3 operator sets differ")
    rows = []
    for operator_id in sorted(v2):
        before = float(v2[operator_id]["mean_ms"])
        after = float(v3[operator_id]["mean_ms"])
        row = {
            "operator_id": operator_id,
            "operator_type": v3[operator_id]["operator_type"],
            "v2_kernel_name": v2[operator_id]["kernel_name"],
            "v3_kernel_name": v3[operator_id]["kernel_name"],
            "v2_mean_ms": before,
            "v3_mean_ms": after,
            "delta_ms": after - before,
            "latency_reduction_pct": (
                (1.0 - after / before) * 100.0 if before else None
            ),
        }
        if selections is not None and operator_id in selections:
            row["v3_selected_mode"] = selections[operator_id]
        rows.append(row)
    rows.sort(key=lambda item: abs(float(item["delta_ms"])), reverse=True)
    return rows


def build_comparison(
    v2_summary: dict[str, Any], v3_summary: dict[str, Any],
    v2_profile: dict[str, Any], v3_profile: dict[str, Any],
    selections: dict[int, str] | None = None,
) -> dict[str, Any]:
    _validate_compatible(v2_summary, v3_summary)
    before = _e2e(v2_summary)
    after = _e2e(v3_summary)
    before_mean = float(before["mean_ms"])
    after_mean = float(after["mean_ms"])
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "final_v2_vs_layer_hybrid_v3",
        "v2": before,
        "v3": after,
        "e2e": {
            "delta_mean_ms": after_mean - before_mean,
            "speedup_ratio": before_mean / after_mean if after_mean else None,
            "latency_reduction_pct": (
                (1.0 - after_mean / before_mean) * 100.0
                if before_mean else None
            ),
            "rtf_v2": before_mean / 1000.0,
            "rtf_v3": after_mean / 1000.0,
            "p50_delta_ms": float(after["p50_ms"]) - float(before["p50_ms"]),
            "p95_delta_ms": float(after["p95_ms"]) - float(before["p95_ms"]),
        },
        "validity": {
            "v2": v2_summary.get("validity"),
            "v3": v3_summary.get("validity"),
        },
        "largest_operator_deltas": _operator_deltas(
            v2_profile, v3_profile, selections
        )[:30],
    }


def _selection_map(plan: dict[str, Any]) -> dict[int, str]:
    selections: dict[int, str] = {}
    for family in plan.get("families", []):
        for item in family.get("operators", []):
            operator_id = int(item["operator_id"])
            if operator_id in selections:
                raise FinalComparisonError(
                    f"duplicate operator in hybrid plan: {operator_id}"
                )
            selections[operator_id] = str(item["selected"])
    if not selections:
        raise FinalComparisonError("hybrid plan has no selections")
    return selections


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-dir", type=_path,
        default=ROOT / "results/profiling/e7_98/final_v2_aggressive",
    )
    parser.add_argument(
        "--v3-dir", type=_path,
        default=ROOT / "results/profiling/e7_98/final_v3_hybrid",
    )
    parser.add_argument(
        "--output", type=_path,
        default=(
            ROOT / "results/profiling/e7_98/final_v3_vs_v2.json"
        ),
    )
    parser.add_argument(
        "--hybrid-plan", type=_path,
        default=(
            ROOT / "results/profiling/e7_98/optimization"
            / "conv_hybrid_plan.json"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() and not args.force:
            raise FinalComparisonError("output exists (use --force)")
        comparison = build_comparison(
            _load(args.v2_dir / "summary.json"),
            _load(args.v3_dir / "summary.json"),
            _load(args.v2_dir / "operator_profile.json"),
            _load(args.v3_dir / "operator_profile.json"),
            _selection_map(_load(args.hybrid_plan)),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(f"V2/V3 comparison: {args.output}")
        return 0
    except (FinalComparisonError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"V2/V3 comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
