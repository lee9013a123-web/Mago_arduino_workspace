#!/usr/bin/env python3
"""Build bucket-specific mmap weight windows and evict-first schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_bundle_exporter.planner.weight_residency_planner import (  # noqa: E402
    WeightResidencyPlanError,
    load_weight_index,
)
from runtime_bundle_exporter.planner.weight_streaming_planner import (  # noqa: E402
    build_windowed_weight_plan,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
    read_execution_plan,
)


SUPPORTED_BUCKETS = (98, 298, 498, 998)
DEFAULT_BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
DEFAULT_STATIC_ROOT = ROOT / "runs/models/campplus/final_v3/weight_residency"
DEFAULT_RUN_ROOT = ROOT / "runs/models/campplus/final_v3/weight_streaming"
DEFAULT_RESULT_ROOT = ROOT / "results/models/campplus/final_v3/weight_streaming"


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _static_weight_index(manifest: Mapping[str, Any]) -> tuple[dict, ...]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise WeightResidencyPlanError("source manifest has no weight index")
    return tuple({
        "offset": entry["destination_offset"],
        "byte_size": entry["byte_size"],
        "name": entry.get("name"),
        "dtype": entry.get("dtype"),
        "shape": entry.get("shape", []),
    } for entry in entries)


def _resolve_source(
    *, bucket: int, bundle: Path, static_root: Path,
    explicit_plan: Path | None, explicit_weights: Path | None,
    explicit_manifest: Path | None,
) -> tuple[Path, Path, Path]:
    if explicit_plan is not None:
        if explicit_weights is None or explicit_manifest is None:
            raise WeightResidencyPlanError(
                "explicit source requires plan, weights, and manifest"
            )
        return explicit_plan, explicit_weights, explicit_manifest
    bundle_plan = bundle / f"execution_plans/plan_{bucket}.bin"
    bundle_weights = bundle / "weights.bin"
    bundle_manifest = bundle / "manifest.json"
    if all(path.is_file() for path in (
        bundle_plan, bundle_weights, bundle_manifest
    )):
        return bundle_plan, bundle_weights, bundle_manifest
    static = static_root / str(bucket)
    return (
        static / f"plan_{bucket}.bin",
        static / f"weights_{bucket}.bin",
        static / f"weight_plan_{bucket}.json",
    )


def _build_bucket(args: argparse.Namespace, bucket: int) -> dict[str, Any]:
    explicit = len(args.bucket_frames) == 1
    source_plan, source_weights, source_manifest = _resolve_source(
        bucket=bucket,
        bundle=args.bundle,
        static_root=args.static_root,
        explicit_plan=args.source_plan if explicit else None,
        explicit_weights=args.source_weights if explicit else None,
        explicit_manifest=args.source_manifest if explicit else None,
    )
    missing = [
        path for path in (source_plan, source_weights, source_manifest)
        if not path.is_file()
    ]
    if missing:
        raise WeightResidencyPlanError(
            f"bucket {bucket} artifact is missing: {missing[0]}"
        )
    run_dir = args.run_root / str(bucket)
    result_dir = args.result_root / str(bucket)
    outputs = (
        run_dir / f"plan_{bucket}.bin",
        run_dir / f"weights_{bucket}.bin",
        run_dir / f"weight_schedule_{bucket}.bin",
        run_dir / f"weight_schedule_{bucket}.json",
        result_dir / "plan.json",
    )
    if not args.force and any(path.exists() for path in outputs):
        raise WeightResidencyPlanError(
            f"bucket {bucket} output exists; use --force"
        )

    plan_bytes = source_plan.read_bytes()
    weights = source_weights.read_bytes()
    loaded = read_execution_plan(source_plan)
    if loaded.header.bucket_frames != bucket:
        raise WeightResidencyPlanError(
            f"source plan bucket mismatch: {loaded.header.bucket_frames} != {bucket}"
        )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if isinstance(manifest.get("weight_index"), list):
        weight_index = load_weight_index(source_manifest)
    else:
        weight_index = _static_weight_index(manifest)
    result = build_windowed_weight_plan(
        loaded,
        plan_bytes,
        weights,
        weight_index=weight_index,
        page_size=args.page_size,
        target_block_bytes=args.target_block_kib * 1024,
    )
    for entry in result.entries:
        source = weights[
            entry.source_offset:entry.source_offset + entry.byte_size
        ]
        output = result.weight_bytes[
            entry.destination_offset:entry.destination_offset + entry.byte_size
        ]
        if source != output:
            raise WeightResidencyPlanError(
                f"Tensor {entry.tensor_id} payload changed"
            )

    document = result.to_dict()
    document["offline_validation"] = {
        "constant_payloads_bitwise_identical": True,
        "validated_constant_count": len(result.entries),
        "operator_and_attribute_rewrite": False,
        "only_constant_data_offsets_rewritten": True,
    }
    document["artifacts"] = {
        "plan": _relative(outputs[0]),
        "weights": _relative(outputs[1]),
        "schedule": _relative(outputs[2]),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_bytes(result.plan_bytes)
    outputs[1].write_bytes(result.weight_bytes)
    outputs[2].write_bytes(result.schedule_bytes)
    outputs[3].write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    compact = {key: value for key, value in document.items()
               if key not in ("entries", "blocks")}
    compact["blocks"] = [
        {key: value for key, value in block.items() if key != "tensor_ids"}
        for block in document["blocks"]
    ]
    result_dir.mkdir(parents=True, exist_ok=True)
    outputs[4].write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return {
        "bucket_frames": bucket,
        "ready": True,
        "block_count": len(result.blocks),
        "scheduled_peak_bytes": result.scheduled_window_peak["byte_size"],
        "scheduled_peak_block_count": result.scheduled_window_peak["block_count"],
        "logical_weight_bytes": result.logical_weight_bytes,
        "theoretical_weight_rss_reduction_bytes": (
            result.theoretical_weight_rss_reduction_bytes
        ),
        "plan_result": _relative(outputs[4]),
        "artifacts": document["artifacts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket-frames", type=int, nargs="+", default=[98],
        choices=SUPPORTED_BUCKETS,
    )
    parser.add_argument("--bundle", type=_path, default=DEFAULT_BUNDLE)
    parser.add_argument("--static-root", type=_path, default=DEFAULT_STATIC_ROOT)
    parser.add_argument("--source-plan", type=_path)
    parser.add_argument("--source-weights", type=_path)
    parser.add_argument("--source-manifest", type=_path)
    parser.add_argument("--run-root", type=_path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--result-root", type=_path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--target-block-kib", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        buckets = tuple(dict.fromkeys(args.bucket_frames))
        explicit_values = (
            args.source_plan, args.source_weights, args.source_manifest
        )
        if any(value is not None for value in explicit_values) and not all(
            value is not None for value in explicit_values
        ):
            raise WeightResidencyPlanError(
                "source plan, weights, and manifest must be provided together"
            )
        if len(buckets) != 1 and any(value is not None for value in explicit_values):
            raise WeightResidencyPlanError(
                "explicit source paths are valid for one bucket only"
            )
        rows = [_build_bucket(args, bucket) for bucket in buckets]
        summary = {
            "schema_version": 1,
            "ready": all(row["ready"] for row in rows),
            "strategy": "evict_then_prefetch_page_window_v2",
            "buckets": rows,
        }
        sidecar_manifest = {
            "schema_version": 1,
            "format": "campp-weight-streaming-sidecar-v1",
            "schedule_format_version": 2,
            "selection_key": "bucket_frames",
            "loader_api": "campp_runtime_model_load_weight_streaming_bundle",
            "buckets": {
                str(row["bucket_frames"]): row["artifacts"] for row in rows
            },
        }
        args.run_root.mkdir(parents=True, exist_ok=True)
        (args.run_root / "weight_streaming_manifest.json").write_text(
            json.dumps(sidecar_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        args.result_root.mkdir(parents=True, exist_ok=True)
        (args.result_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (
        WeightResidencyPlanError, OSError, ValueError, KeyError, TypeError
    ) as exc:
        print(f"weight streaming build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
