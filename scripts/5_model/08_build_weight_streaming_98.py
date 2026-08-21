#!/usr/bin/env python3
"""Build the experimental bucket-98 mmap weight window and schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


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


DEFAULT_BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
DEFAULT_STATIC = (
    ROOT / "runs/models/campplus/final_v3/weight_residency/98"
)
DEFAULT_RUN = (
    ROOT / "runs/models/campplus/final_v3/weight_streaming/98"
)
DEFAULT_RESULT = (
    ROOT / "results/models/campplus/final_v3/weight_streaming/98"
)


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=_path, default=DEFAULT_BUNDLE)
    parser.add_argument("--source-plan", type=_path)
    parser.add_argument("--source-weights", type=_path)
    parser.add_argument("--source-manifest", type=_path)
    parser.add_argument("--run-dir", type=_path, default=DEFAULT_RUN)
    parser.add_argument("--result-dir", type=_path, default=DEFAULT_RESULT)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--target-block-kib", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        bundle_available = (
            args.bundle / "execution_plans/plan_98.bin"
        ).is_file()
        source_plan = args.source_plan or (
            args.bundle / "execution_plans/plan_98.bin"
            if bundle_available else DEFAULT_STATIC / "plan_98.bin"
        )
        source_weights = args.source_weights or (
            args.bundle / "weights.bin"
            if bundle_available else DEFAULT_STATIC / "weights_98.bin"
        )
        source_manifest = args.source_manifest or (
            args.bundle / "manifest.json"
            if bundle_available else DEFAULT_STATIC / "weight_plan_98.json"
        )
        required = (source_plan, source_weights, source_manifest)
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise WeightResidencyPlanError(
                "required artifact is missing: " + str(missing[0])
            )
        outputs = (
            args.run_dir / "plan_98.bin",
            args.run_dir / "weights_98.bin",
            args.run_dir / "weight_schedule_98.bin",
            args.run_dir / "weight_schedule_98.json",
            args.result_dir / "plan.json",
        )
        if not args.force and any(path.exists() for path in outputs):
            raise WeightResidencyPlanError("output exists; use --force")

        plan_bytes = source_plan.read_bytes()
        weights = source_weights.read_bytes()
        loaded = read_execution_plan(source_plan)
        if loaded.header.bucket_frames != 98:
            raise WeightResidencyPlanError("source plan is not bucket 98")
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        if isinstance(manifest.get("weight_index"), list):
            weight_index = load_weight_index(source_manifest)
        elif isinstance(manifest.get("entries"), list):
            weight_index = tuple({
                "offset": entry["destination_offset"],
                "byte_size": entry["byte_size"],
                "name": entry.get("name"),
                "dtype": entry.get("dtype"),
                "shape": entry.get("shape", []),
            } for entry in manifest["entries"])
        else:
            raise WeightResidencyPlanError("source manifest has no weight index")
        result = build_windowed_weight_plan(
            loaded,
            plan_bytes,
            weights,
            weight_index=weight_index,
            page_size=args.page_size,
            target_block_bytes=args.target_block_kib * 1024,
        )
        document = result.to_dict()
        document["artifacts"] = {
            "plan": "runs/models/campplus/final_v3/weight_streaming/98/plan_98.bin",
            "weights": "runs/models/campplus/final_v3/weight_streaming/98/weights_98.bin",
            "schedule": "runs/models/campplus/final_v3/weight_streaming/98/weight_schedule_98.bin",
        }
        args.run_dir.mkdir(parents=True, exist_ok=True)
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
            {key: value for key, value in block.items()
             if key != "tensor_ids"}
            for block in document["blocks"]
        ]
        args.result_dir.mkdir(parents=True, exist_ok=True)
        outputs[4].write_text(
            json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(json.dumps({
            "ready": True,
            "bucket_frames": 98,
            "block_count": len(result.blocks),
            "max_block_bytes": result.max_block_bytes,
            "double_buffer_bound_bytes": result.double_buffer_bound_bytes,
            "output_weight_bytes": len(result.weight_bytes),
        }, ensure_ascii=False, indent=2))
        return 0
    except (WeightResidencyPlanError, OSError, ValueError) as exc:
        print(f"weight streaming build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
