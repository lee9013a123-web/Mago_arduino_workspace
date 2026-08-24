#!/usr/bin/env python3
"""Build four offline static weight layouts with no inference-time movement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_bundle_exporter.planner.weight_residency_planner import (  # noqa: E402
    WeightResidencyPlanError,
    build_static_weight_plan,
    load_weight_index,
)
from runtime_bundle_exporter.reporting.weight_residency_report import (  # noqa: E402
    build_weight_residency_summary,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
    read_execution_plan,
)
from runtime_bundle_exporter.writer.weight_residency_manifest_writer import (  # noqa: E402
    write_static_weight_artifacts,
)


DEFAULT_BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
DEFAULT_RUNS = ROOT / "runs/models/campplus/final_v3/weight_residency"
DEFAULT_RESULTS = ROOT / "results/models/campplus/final_v3/weight_residency"


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _ensure_outputs_available(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise WeightResidencyPlanError(
            f"output exists: {existing[0]} (use --force)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=_path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--buckets", type=int, nargs="+", default=[98, 298, 498, 998]
    )
    parser.add_argument("--runs-root", type=_path, default=DEFAULT_RUNS)
    parser.add_argument("--results-root", type=_path, default=DEFAULT_RESULTS)
    parser.add_argument("--cache-alignment", type=int, default=64)
    parser.add_argument("--scalar-alignment", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        buckets = list(dict.fromkeys(args.buckets))
        if not buckets or any(bucket <= 0 for bucket in buckets):
            raise WeightResidencyPlanError("buckets must be positive")
        source_weights_path = args.bundle / "weights.bin"
        source_manifest_path = args.bundle / "manifest.json"
        required = [source_weights_path, source_manifest_path]
        required.extend(
            args.bundle / "execution_plans" / f"plan_{bucket}.bin"
            for bucket in buckets
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise WeightResidencyPlanError(
                "required artifacts are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )

        targets: list[Path] = []
        for bucket in buckets:
            targets.extend((
                args.runs_root / str(bucket) / f"plan_{bucket}.bin",
                args.runs_root / str(bucket) / f"weights_{bucket}.bin",
                args.runs_root / str(bucket) / f"weight_plan_{bucket}.json",
                args.runs_root / str(bucket) / f"weight_plan_{bucket}.csv",
            ))
        targets.extend((
            args.results_root / "summary.json",
            args.results_root / "summary.csv",
        ))
        _ensure_outputs_available(targets, args.force)

        source_weights = source_weights_path.read_bytes()
        weight_index = load_weight_index(source_manifest_path)
        plans = []
        for bucket in buckets:
            source_plan_path = (
                args.bundle / "execution_plans" / f"plan_{bucket}.bin"
            )
            source_plan_bytes = source_plan_path.read_bytes()
            loaded = read_execution_plan(source_plan_path)
            if loaded.header.bucket_frames != bucket:
                raise WeightResidencyPlanError(
                    f"plan bucket mismatch: {source_plan_path}"
                )
            static_plan = build_static_weight_plan(
                loaded,
                source_plan_bytes,
                source_weights,
                weight_index=weight_index,
                cache_alignment=args.cache_alignment,
                scalar_alignment=args.scalar_alignment,
            )
            run_dir = args.runs_root / str(bucket)
            plan_path = run_dir / f"plan_{bucket}.bin"
            weights_path = run_dir / f"weights_{bucket}.bin"
            write_static_weight_artifacts(
                static_plan,
                execution_plan_path=plan_path,
                weights_path=weights_path,
                manifest_path=run_dir / f"weight_plan_{bucket}.json",
                csv_path=run_dir / f"weight_plan_{bucket}.csv",
                artifact_root=ROOT,
            )
            plans.append(static_plan)
            print(
                f"bucket {bucket}: {len(source_weights):,} -> "
                f"{len(static_plan.weight_bytes):,} bytes "
                f"({static_plan.saved_pct:.2f}% saved)"
            )

        summary = build_weight_residency_summary(plans)
        summary["artifacts"] = {
            "runs_root": _display(args.runs_root),
            "results_root": _display(args.results_root),
        }
        args.results_root.mkdir(parents=True, exist_ok=True)
        (args.results_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with (args.results_root / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            rows = summary["buckets"]
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"summary: {_display(args.results_root / 'summary.json')}")
        return 0
    except (WeightResidencyPlanError, OSError, ValueError, KeyError) as exc:
        print(f"weight plan build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
