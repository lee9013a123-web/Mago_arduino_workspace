#!/usr/bin/env python3
"""Build an operator-level QConv/fused-QConv hybrid plan from benchmark results."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
ORDINARY_MODES = ("mac_fixed", "v4", "v5")
FUSED_MODES = ("combined_fixed", "combined_hybrid", "combined_v5")


class HybridPlanError(RuntimeError):
    """Hybrid plan inputs are missing or inconsistent."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HybridPlanError(f"cannot read {path}: {exc}") from exc


def _comparison_cases(result_dir: Path, modes: Sequence[str]) -> dict[int, dict[str, Any]]:
    by_operator: dict[int, dict[str, Any]] = {}
    for mode in modes:
        path = result_dir / f"{mode}_comparison.json"
        document = _load_json(path)
        if not document.get("all_output_hashes_bitwise_identical", False):
            raise HybridPlanError(f"{path} is not fully bitwise-identical")
        for case in document.get("cases", []):
            operator_id = int(case["operator_id"])
            entry = by_operator.setdefault(
                operator_id,
                {
                    "operator_id": operator_id,
                    "case_name": str(case["case_name"]),
                    "kernel_name": str(case["kernel_name"]),
                    "weight_shape": list(case["weight_shape"]),
                    "modes": {},
                },
            )
            identity = (
                entry["kernel_name"] == str(case["kernel_name"]) and
                entry["weight_shape"] == list(case["weight_shape"])
            )
            if not identity:
                raise HybridPlanError(f"operator identity mismatch: {operator_id}")
            entry["modes"][mode] = {
                "mean_ms": float(case["candidate_mean_ms"]),
                "p50_ms": float(case["candidate_p50_ms"]),
                "p95_ms": float(case["candidate_p95_ms"]),
                "bitwise": bool(case["bitwise_output_hashes"]),
            }
    for operator_id, entry in by_operator.items():
        missing = [mode for mode in modes if mode not in entry["modes"]]
        if missing:
            raise HybridPlanError(f"operator {operator_id} missing modes: {missing}")
    return by_operator


def _choose_mode(
    modes: dict[str, dict[str, Any]], incumbent: str, min_margin_pct: float
) -> tuple[str, str, float]:
    bitwise_modes = {
        mode: data for mode, data in modes.items() if bool(data["bitwise"])
    }
    if incumbent not in bitwise_modes:
        raise HybridPlanError(f"incumbent mode is not bitwise-valid: {incumbent}")
    winner = min(bitwise_modes, key=lambda mode: float(bitwise_modes[mode]["mean_ms"]))
    winner_ms = float(bitwise_modes[winner]["mean_ms"])
    incumbent_ms = float(bitwise_modes[incumbent]["mean_ms"])
    margin = (
        (incumbent_ms - winner_ms) / incumbent_ms * 100.0
        if incumbent_ms else 0.0
    )
    if winner != incumbent and margin < min_margin_pct:
        return incumbent, "incumbent_within_margin", margin
    return winner, "fastest_bitwise", margin


def _family_plan(
    *,
    family: str,
    result_dir: Path,
    modes: Sequence[str],
    incumbent: str,
    min_margin_pct: float,
) -> dict[str, Any]:
    operators = _comparison_cases(result_dir, modes)
    rows = []
    totals = {mode: 0.0 for mode in modes}
    selected_total = 0.0
    incumbent_total = 0.0
    for operator_id in sorted(operators):
        entry = operators[operator_id]
        selected, reason, margin = _choose_mode(
            entry["modes"], incumbent, min_margin_pct
        )
        for mode in modes:
            totals[mode] += float(entry["modes"][mode]["mean_ms"])
        selected_ms = float(entry["modes"][selected]["mean_ms"])
        incumbent_ms = float(entry["modes"][incumbent]["mean_ms"])
        selected_total += selected_ms
        incumbent_total += incumbent_ms
        rows.append({
            "family": family,
            "operator_id": operator_id,
            "case_name": entry["case_name"],
            "kernel_name": entry["kernel_name"],
            "weight_shape": entry["weight_shape"],
            "incumbent": incumbent,
            "selected": selected,
            "selection_reason": reason,
            "margin_vs_incumbent_pct": margin,
            "selected_mean_ms": selected_ms,
            "incumbent_mean_ms": incumbent_ms,
            "modes": entry["modes"],
        })
    return {
        "family": family,
        "operator_count": len(rows),
        "incumbent": incumbent,
        "candidate_modes": list(modes),
        "total_mean_ms_by_mode": totals,
        "selected_total_mean_ms": selected_total,
        "incumbent_total_mean_ms": incumbent_total,
        "selected_speedup_vs_incumbent": (
            incumbent_total / selected_total if selected_total else None
        ),
        "selected_latency_reduction_pct": (
            (1.0 - selected_total / incumbent_total) * 100.0
            if incumbent_total else None
        ),
        "selection_counts": {
            mode: sum(1 for row in rows if row["selected"] == mode)
            for mode in modes
        },
        "operators": rows,
    }


def _write_csv(plan: dict[str, Any], output: Path) -> None:
    fields = (
        "family", "operator_id", "weight_shape", "incumbent", "selected",
        "selection_reason", "margin_vs_incumbent_pct", "incumbent_mean_ms",
        "selected_mean_ms",
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for family in plan["families"]:
            for row in family["operators"]:
                writer.writerow({
                    "family": row["family"],
                    "operator_id": row["operator_id"],
                    "weight_shape": "x".join(str(v) for v in row["weight_shape"]),
                    "incumbent": row["incumbent"],
                    "selected": row["selected"],
                    "selection_reason": row["selection_reason"],
                    "margin_vs_incumbent_pct": row["margin_vs_incumbent_pct"],
                    "incumbent_mean_ms": row["incumbent_mean_ms"],
                    "selected_mean_ms": row["selected_mean_ms"],
                })


def build_plan(
    *,
    qconv_dir: Path,
    fused_dir: Path,
    min_margin_pct: float,
) -> dict[str, Any]:
    qconv = _family_plan(
        family="qlinear_conv",
        result_dir=qconv_dir,
        modes=ORDINARY_MODES,
        incumbent="v4",
        min_margin_pct=min_margin_pct,
    )
    fused = _family_plan(
        family="fused_quant_qconv",
        result_dir=fused_dir,
        modes=FUSED_MODES,
        incumbent="combined_hybrid",
        min_margin_pct=min_margin_pct,
    )
    incumbent_total = (
        float(qconv["incumbent_total_mean_ms"]) +
        float(fused["incumbent_total_mean_ms"])
    )
    selected_total = (
        float(qconv["selected_total_mean_ms"]) +
        float(fused["selected_total_mean_ms"])
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "operator-level hybrid selection for ordinary and fused QConv",
        "selection_rule": {
            "metric": "mean_ms",
            "correctness": "candidate comparison must be bitwise-identical",
            "min_margin_pct": min_margin_pct,
            "incumbents": {
                "qlinear_conv": "v4",
                "fused_quant_qconv": "combined_hybrid",
            },
        },
        "artifacts": {
            "qconv_dir": _display_path(qconv_dir),
            "fused_dir": _display_path(fused_dir),
        },
        "families": [qconv, fused],
        "combined": {
            "incumbent_total_mean_ms": incumbent_total,
            "selected_total_mean_ms": selected_total,
            "selected_speedup_vs_incumbent": (
                incumbent_total / selected_total if selected_total else None
            ),
            "selected_latency_reduction_pct": (
                (1.0 - selected_total / incumbent_total) * 100.0
                if incumbent_total else None
            ),
        },
        "next_gate": (
            "translate selected operator IDs into C dispatch tables, then run "
            "retained-tensor bitwise validation and Quick E2E profiling"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qconv-dir", type=_path, default=ROOT / "results/profiling/e7_98/optimization/qconv_family")
    parser.add_argument("--fused-dir", type=_path, default=ROOT / "results/profiling/e7_98/optimization/fused_qconv_family")
    parser.add_argument("--min-margin-pct", type=float, default=1.0)
    parser.add_argument("--output", type=_path, default=ROOT / "results/profiling/e7_98/optimization/conv_hybrid_plan.json")
    parser.add_argument("--csv", type=_path, default=ROOT / "results/profiling/e7_98/optimization/conv_hybrid_plan.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.min_margin_pct < 0.0:
            raise HybridPlanError("min-margin-pct must be non-negative")
        if (args.output.exists() or args.csv.exists()) and not args.force:
            raise HybridPlanError("output exists (use --force)")
        plan = build_plan(
            qconv_dir=args.qconv_dir,
            fused_dir=args.fused_dir,
            min_margin_pct=args.min_margin_pct,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        _write_csv(plan, args.csv)
        print(f"conv hybrid plan: {_display_path(args.output)}")
        return 0
    except (HybridPlanError, OSError, ValueError, KeyError) as exc:
        print(f"conv hybrid plan failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
