#!/usr/bin/env python3
"""Build a Dense slab bundle directly from an existing Tensor Arena bundle."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "src" / "python"


class DenseSlabExportError(RuntimeError):
    """The source bundle is invalid or the output would be unsafe."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(target: Path, base: Path) -> str:
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def _runtime_graph_from_plan(loaded: object, weights: bytes) -> object:
    from runtime_bundle_exporter.format.binary_format_schema import (
        OperatorCode,
        TensorDType,
        TensorStorageType,
    )
    from runtime_bundle_exporter.runtime_ir import (
        RuntimeGraph,
        RuntimeInitializer,
        RuntimeOperator,
        RuntimeTensor,
    )

    producers: dict[int, int] = {}
    consumers: dict[int, list[int]] = {
        tensor_id: [] for tensor_id in range(len(loaded.tensors))
    }
    for operator in loaded.operators:
        for tensor_id in operator.input_tensor_ids[: operator.input_count]:
            bucket = consumers[tensor_id]
            if not bucket or bucket[-1] != operator.operator_id:
                bucket.append(operator.operator_id)
        for tensor_id in operator.output_tensor_ids:
            producers[tensor_id] = operator.operator_id

    tensors = []
    initializers = []
    for descriptor in loaded.tensors:
        name = f"tensor_{descriptor.tensor_id}"
        storage_type = TensorStorageType(descriptor.storage_type)
        shape = tuple(descriptor.dimensions[: descriptor.rank])
        tensors.append(
            RuntimeTensor(
                name=name,
                tensor_id=descriptor.tensor_id,
                dtype=TensorDType(descriptor.dtype),
                shape=shape,
                strides=tuple(descriptor.byte_strides[: descriptor.rank]),
                byte_size=descriptor.logical_byte_size,
                storage_type=storage_type,
                producer=producers.get(descriptor.tensor_id),
                consumers=tuple(consumers[descriptor.tensor_id]),
                storage_span_bytes=descriptor.storage_span_bytes,
            )
        )
        if storage_type is TensorStorageType.CONSTANT:
            start = descriptor.data_offset
            end = start + descriptor.logical_byte_size
            if end > len(weights):
                raise DenseSlabExportError(
                    f"Tensor {descriptor.tensor_id} constant range exceeds weights.bin"
                )
            initializers.append(
                RuntimeInitializer(
                    name=name,
                    tensor_id=descriptor.tensor_id,
                    dtype=TensorDType(descriptor.dtype),
                    shape=shape,
                    raw_data=weights[start:end],
                )
            )

    operators = tuple(
        RuntimeOperator(
            name=f"operator_{descriptor.operator_id}",
            operator_id=descriptor.operator_id,
            opcode=OperatorCode(descriptor.opcode),
            input_tensor_ids=tuple(
                descriptor.input_tensor_ids[: descriptor.input_count]
            ),
            output_tensor_ids=tuple(descriptor.output_tensor_ids),
            attributes=loaded.attributes_of(descriptor.operator_id),
        )
        for descriptor in loaded.operators
    )
    return RuntimeGraph(
        tensors=tuple(tensors),
        operators=operators,
        initializers=tuple(initializers),
        name=f"compiled_{loaded.header.bucket_frames}f",
        bucket_frames=loaded.header.bucket_frames,
    )


def _rebase_source_paths(document: dict, source: Path, output: Path) -> None:
    for key in ("canonical_model", "graph_manifest"):
        entry = document.get(key)
        if not isinstance(entry, dict) or not entry.get("path"):
            continue
        target = (source / entry["path"]).resolve()
        entry["path"] = _relative_path(target, output)


def export_dense_slab_bundle(args: argparse.Namespace) -> None:
    if str(PYTHON_SOURCE) not in sys.path:
        sys.path.insert(0, str(PYTHON_SOURCE))
    from runtime_bundle_exporter.planner.dense_slab_planner import (
        inspect_compiled_dense_concats,
        rewrite_dense_concats_as_slabs,
    )
    from runtime_bundle_exporter.planner.tensor_arena_planner import (
        plan_tensor_arena,
    )
    from runtime_bundle_exporter.builder.tensor_table_builder import (
        WeightBlobLayout,
    )
    from runtime_bundle_exporter.format.binary_format_schema import (
        TensorStorageType,
    )
    from runtime_bundle_exporter.writer.bundle_manifest_writer import (
        verify_bundle_manifest,
    )
    from runtime_bundle_exporter.writer.execution_plan_writer import (
        read_execution_plan,
        write_execution_plan,
    )

    source = args.source_bundle.resolve()
    output = args.output_dir.resolve()
    if source == output:
        raise DenseSlabExportError("source bundle을 같은 위치에 덮어쓸 수 없다")
    source_manifest_path = source / "manifest.json"
    source_weights_path = source / "weights.bin"
    if not source_manifest_path.is_file() or not source_weights_path.is_file():
        raise DenseSlabExportError(f"source bundle 파일이 부족하다: {source}")
    source_manifest = verify_bundle_manifest(source_manifest_path)
    if source_manifest.get("memory_layout") != "tensor_arena":
        raise DenseSlabExportError("source bundle이 Tensor Arena bundle이 아니다")
    weights = source_weights_path.read_bytes()

    target_files = [output / "manifest.json", output / "weights.bin"]
    target_files.extend(
        output / "execution_plans" / f"plan_{entry['bucket_frames']}.bin"
        for entry in source_manifest["plans"]
    )
    existing = [path for path in target_files if path.exists()]
    if existing and not args.force:
        raise DenseSlabExportError(
            f"output 파일이 이미 있다: {existing[0]} (--force로 명시적 덮어쓰기)"
        )

    loaded_by_bucket = {}
    tensor_offsets: dict[tuple[int | None, int], int] = {}
    for entry in source_manifest["plans"]:
        frames = int(entry["bucket_frames"])
        plan_path = source / entry["path"]
        loaded = read_execution_plan(plan_path)
        loaded_by_bucket[frames] = (loaded, entry)
        for descriptor in loaded.tensors:
            if descriptor.storage_type == int(TensorStorageType.CONSTANT):
                tensor_offsets[(frames, descriptor.tensor_id)] = descriptor.data_offset
    weight_layout = WeightBlobLayout(
        entries=(),
        total_bytes=len(weights),
        alignment=int(source_manifest["weights"].get("alignment", 8)),
        tensor_offsets=tensor_offsets,
    )

    output.mkdir(parents=True, exist_ok=True)
    plan_dir = output / "execution_plans"
    report_dir = output / "dense_slab_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_weights_path, output / "weights.bin")

    plan_results = []
    reports = []
    for frames in sorted(loaded_by_bucket):
        loaded, source_entry = loaded_by_bucket[frames]
        compiled_blocks = inspect_compiled_dense_concats(loaded)
        if [len(block.concat_operator_ids) for block in compiled_blocks] != [
            12,
            24,
            16,
        ]:
            raise DenseSlabExportError(
                f"bucket {frames} Dense chain이 12/24/16 구조가 아니다"
            )
        graph = _runtime_graph_from_plan(loaded, weights)
        rewrite = rewrite_dense_concats_as_slabs(
            graph, expected_block_count=3, expected_concat_count=52
        )
        if tuple(block.concat_operator_ids for block in compiled_blocks) != tuple(
            block.concat_operator_ids for block in rewrite.blocks
        ):
            raise DenseSlabExportError(
                f"bucket {frames} binary 분석과 graph rewrite 대상이 다르다"
            )
        arena = plan_tensor_arena(rewrite.graph, alignment=args.arena_alignment)
        result = write_execution_plan(
            rewrite.graph,
            weight_layout,
            plan_dir / f"plan_{frames}.bin",
            arena_layout=arena,
        )
        plan_results.append(result)
        report = rewrite.to_dict()
        report.update(
            {
                "source_plan_sha256": source_entry["sha256"],
                "source_arena_size_bytes": source_entry["arena_size_bytes"],
                "arena_size_bytes": arena.arena_size,
                "view_tensor_count": sum(
                    tensor.storage_type is TensorStorageType.VIEW
                    for tensor in rewrite.graph.tensors
                ),
            }
        )
        (report_dir / f"dense_slab_{frames}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
        print(
            f"{frames:>4} frames: operators "
            f"{rewrite.original_operator_count}->{len(rewrite.graph.operators)}, "
            f"views={report['view_tensor_count']}, arena={arena.arena_size:,} B"
        )

    manifest = deepcopy(source_manifest)
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _rebase_source_paths(manifest, source, output)
    manifest["weights"]["path"] = "weights.bin"
    manifest["plans"] = [
        {
            "bucket_frames": result.bucket_frames,
            "path": f"execution_plans/plan_{result.bucket_frames}.bin",
            "size_bytes": result.byte_size,
            "sha256": result.sha256,
            "tensor_count": result.tensor_count,
            "operator_count": result.operator_count,
            "attribute_section_bytes": result.attribute_section_bytes,
            "attribute_blocks": result.attribute_blocks,
            "arena_size_bytes": result.arena_size,
            "arena_alignment": result.arena_alignment,
        }
        for result in plan_results
    ]
    manifest["optimization"] = {
        "name": "dense_slab_view_alias",
        "source_bundle": _relative_path(source, output),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "removed_dense_concat_count_per_bucket": 52,
        "operator_count_before": 1438,
        "operator_count_after": 1386,
        "view_tensor_count_per_bucket": 104,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_bundle": str(source),
        "output_bundle": str(output),
        "bucket_results": reports,
    }
    (report_dir / "dense_slab_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verify_bundle_manifest(output / "manifest.json")
    if _sha256(output / "weights.bin") != source_manifest["weights"]["sha256"]:
        raise DenseSlabExportError("copied weights.bin hash가 source manifest와 다르다")
    print(f"Dense slab bundle: {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-bundle",
        type=_path,
        default=ROOT / "runs" / "runtime" / "tensor_arena" / "bundle",
    )
    parser.add_argument(
        "--output-dir",
        type=_path,
        default=(
            ROOT
            / "runs"
            / "runtime"
            / "kernel_optimization"
            / "dense_slab"
            / "bundle"
        ),
    )
    parser.add_argument("--arena-alignment", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        export_dense_slab_bundle(args)
        return 0
    except (DenseSlabExportError, OSError, ValueError, KeyError) as exc:
        print(f"Dense slab export 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
