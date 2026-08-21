#!/usr/bin/env python3
"""Validate static bucket weight plans without executing inference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_bundle_exporter.format.binary_format_schema import (  # noqa: E402
    TensorStorageType,
)
from runtime_bundle_exporter.planner.weight_residency_planner import (  # noqa: E402
    WeightResidencyPlanError,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
    read_execution_plan,
)


DEFAULT_BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
DEFAULT_RUNS = ROOT / "runs/models/campplus/final_v3/weight_residency"
DEFAULT_RESULTS = ROOT / "results/models/campplus/final_v3/weight_residency"


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_bucket(
    bucket: int, bundle: Path, runs_root: Path, results_root: Path,
) -> dict:
    source_plan_path = bundle / "execution_plans" / f"plan_{bucket}.bin"
    source_weights_path = bundle / "weights.bin"
    output_plan_path = runs_root / str(bucket) / f"plan_{bucket}.bin"
    output_weights_path = runs_root / str(bucket) / f"weights_{bucket}.bin"
    manifest_path = runs_root / str(bucket) / f"weight_plan_{bucket}.json"
    required = (
        source_plan_path, source_weights_path, output_plan_path,
        output_weights_path, manifest_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise WeightResidencyPlanError(
            f"bucket {bucket} artifact is missing: {missing[0]}"
        )

    source_plan_bytes = source_plan_path.read_bytes()
    source_weights = source_weights_path.read_bytes()
    output_plan_bytes = output_plan_path.read_bytes()
    output_weights = output_weights_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bucket_frames") != bucket:
        raise WeightResidencyPlanError(f"bucket {bucket} manifest mismatch")
    if manifest.get("runtime_contract", {}).get("inference_weight_moves") != 0:
        raise WeightResidencyPlanError("manifest permits inference weight movement")
    if manifest.get("source", {}).get("plan_sha256") != _sha256(source_plan_bytes):
        raise WeightResidencyPlanError(f"bucket {bucket} source plan changed")
    if manifest.get("source", {}).get("weights_sha256") != _sha256(source_weights):
        raise WeightResidencyPlanError(f"bucket {bucket} source weights changed")
    if manifest.get("output", {}).get("plan_sha256") != _sha256(output_plan_bytes):
        raise WeightResidencyPlanError(f"bucket {bucket} output plan hash mismatch")
    if manifest.get("output", {}).get("weights_sha256") != _sha256(output_weights):
        raise WeightResidencyPlanError(f"bucket {bucket} output weights hash mismatch")

    source_plan = read_execution_plan(source_plan_path)
    output_plan = read_execution_plan(output_plan_path)
    if source_plan.header.bucket_frames != bucket or (
        output_plan.header.bucket_frames != bucket
    ):
        raise WeightResidencyPlanError(f"bucket {bucket} plan header mismatch")
    if source_plan.operators != output_plan.operators:
        raise WeightResidencyPlanError(
            f"bucket {bucket} operator table changed during weight packing"
        )
    if source_plan.attribute_section != output_plan.attribute_section:
        raise WeightResidencyPlanError(
            f"bucket {bucket} attribute section changed"
        )
    if len(source_plan.tensors) != len(output_plan.tensors):
        raise WeightResidencyPlanError(f"bucket {bucket} Tensor count changed")

    rows = manifest.get("entries")
    if not isinstance(rows, list):
        raise WeightResidencyPlanError(f"bucket {bucket} entries are missing")
    by_tensor = {int(row["tensor_id"]): row for row in rows}
    constant_ids = {
        tensor.tensor_id for tensor in source_plan.tensors
        if tensor.storage_type == TensorStorageType.CONSTANT
    }
    if set(by_tensor) != constant_ids:
        raise WeightResidencyPlanError(
            f"bucket {bucket} manifest does not cover every constant"
        )

    intervals = []
    for source, output in zip(source_plan.tensors, output_plan.tensors):
        if source.storage_type != TensorStorageType.CONSTANT:
            if source != output:
                raise WeightResidencyPlanError(
                    f"bucket {bucket} non-constant Tensor {source.tensor_id} changed"
                )
            continue
        if replace(output, data_offset=source.data_offset) != source:
            raise WeightResidencyPlanError(
                f"bucket {bucket} Tensor {source.tensor_id} changed beyond offset"
            )
        row = by_tensor[source.tensor_id]
        if int(row["source_offset"]) != source.data_offset or (
            int(row["destination_offset"]) != output.data_offset
        ) or int(row["byte_size"]) != source.storage_span_bytes:
            raise WeightResidencyPlanError(
                f"bucket {bucket} Tensor {source.tensor_id} manifest mismatch"
            )
        source_payload = source_weights[
            source.data_offset : source.data_offset + source.storage_span_bytes
        ]
        output_payload = output_weights[
            output.data_offset : output.data_offset + output.storage_span_bytes
        ]
        if source_payload != output_payload or _sha256(output_payload) != row["sha256"]:
            raise WeightResidencyPlanError(
                f"bucket {bucket} Tensor {source.tensor_id} bytes changed"
            )
        alignment = int(row["alignment"])
        if output.data_offset % alignment:
            raise WeightResidencyPlanError(
                f"bucket {bucket} Tensor {source.tensor_id} is misaligned"
            )
        intervals.append((
            output.data_offset,
            output.data_offset + output.storage_span_bytes,
            source.tensor_id,
        ))

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise WeightResidencyPlanError(
                f"bucket {bucket} Tensor {previous[2]}/{current[2]} overlap"
            )
    if intervals and intervals[-1][1] != len(output_weights):
        raise WeightResidencyPlanError(
            f"bucket {bucket} output blob has untracked trailing bytes"
        )
    return {
        "bucket_frames": bucket,
        "ready": True,
        "operator_table_identical": True,
        "attribute_section_identical": True,
        "constant_bytes_identical": True,
        "constant_count": len(constant_ids),
        "source_weights_bytes": len(source_weights),
        "output_weights_bytes": len(output_weights),
        "saved_bytes": len(source_weights) - len(output_weights),
        "inference_weight_moves": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=_path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--buckets", type=int, nargs="+", default=[98, 298, 498, 998]
    )
    parser.add_argument("--runs-root", type=_path, default=DEFAULT_RUNS)
    parser.add_argument("--results-root", type=_path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    try:
        rows = [
            _validate_bucket(
                bucket, args.bundle, args.runs_root, args.results_root
            )
            for bucket in dict.fromkeys(args.buckets)
        ]
        result = {
            "schema_version": 1,
            "ready": all(row["ready"] for row in rows),
            "production_invariant": "no weight relocation during inference",
            "buckets": rows,
        }
        target = args.results_root / "validation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (WeightResidencyPlanError, OSError, ValueError, KeyError) as exc:
        print(f"weight plan validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
