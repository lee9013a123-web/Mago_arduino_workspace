#!/usr/bin/env python3
"""Export NTC/NHWC-padded activations and O4I4-packed QConv weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "src" / "python"


class CachePackedExportError(RuntimeError):
    """The source bundle or requested rewrite is invalid."""


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


def _source_path(manifest: dict, key: str, bundle_dir: Path) -> Path | None:
    entry = manifest.get(key)
    if not isinstance(entry, dict) or not entry.get("path"):
        return None
    return (bundle_dir / entry["path"]).resolve()


def _weight_records_by_offset(manifest: dict) -> dict[int, dict]:
    records = manifest.get("weight_index")
    if not isinstance(records, list):
        raise CachePackedExportError("source manifest has no weight_index")
    result: dict[int, dict] = {}
    for record in records:
        if not isinstance(record, dict) or "offset" not in record:
            raise CachePackedExportError("invalid source weight_index record")
        offset = int(record["offset"])
        if offset in result:
            raise CachePackedExportError(f"duplicate weight offset: {offset}")
        result[offset] = record
    return result


def _runtime_graph_from_plan(
    loaded: object, weights: bytes, weight_records: dict[int, dict]
) -> object:
    from runtime_bundle_exporter.format.binary_format_schema import (
        OperatorCode,
        TensorDType,
        TensorFlags,
        TensorStorageType,
    )
    from runtime_bundle_exporter.runtime_ir import (
        InitializerScope,
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
            if not consumers[tensor_id] or consumers[tensor_id][-1] != operator.operator_id:
                consumers[tensor_id].append(operator.operator_id)
        for tensor_id in operator.output_tensor_ids:
            producers[tensor_id] = operator.operator_id

    tensors = []
    initializers = []
    for descriptor in loaded.tensors:
        storage_type = TensorStorageType(descriptor.storage_type)
        shape = tuple(descriptor.dimensions[: descriptor.rank])
        record = (
            weight_records.get(descriptor.data_offset)
            if storage_type is TensorStorageType.CONSTANT
            else None
        )
        name = record["name"] if record is not None else f"tensor_{descriptor.tensor_id}"
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
                alias_of_tensor_id=(
                    descriptor.alias_of_tensor_id
                    if storage_type is TensorStorageType.VIEW
                    else None
                ),
                view_byte_offset=(
                    descriptor.data_offset
                    if storage_type is TensorStorageType.VIEW
                    else 0
                ),
                packed_qconv_o4i4=bool(
                    descriptor.flags & int(TensorFlags.PACKED_QCONV_O4I4)
                ),
            )
        )
        if storage_type is TensorStorageType.CONSTANT:
            if record is None:
                raise CachePackedExportError(
                    f"Tensor {descriptor.tensor_id} has no weight_index record"
                )
            start = descriptor.data_offset
            end = start + descriptor.storage_span_bytes
            if end > len(weights) or int(record["byte_size"]) != descriptor.storage_span_bytes:
                raise CachePackedExportError(
                    f"Tensor {descriptor.tensor_id} constant metadata is inconsistent"
                )
            initializers.append(
                RuntimeInitializer(
                    name=name,
                    tensor_id=descriptor.tensor_id,
                    dtype=TensorDType(descriptor.dtype),
                    shape=shape,
                    raw_data=weights[start:end],
                    scope=InitializerScope(record["scope"]),
                    storage_span_bytes=descriptor.storage_span_bytes,
                    packed_qconv_o4i4=bool(
                        descriptor.flags & int(TensorFlags.PACKED_QCONV_O4I4)
                    ),
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
        name=f"cache_source_{loaded.header.bucket_frames}f",
        bucket_frames=loaded.header.bucket_frames,
    )


def export_cache_packed_bundle(args: argparse.Namespace) -> None:
    if str(PYTHON_SOURCE) not in sys.path:
        sys.path.insert(0, str(PYTHON_SOURCE))
    from runtime_bundle_exporter.planner.cache_layout_planner import (
        rewrite_cache_friendly_activations,
    )
    from runtime_bundle_exporter.planner.tensor_arena_planner import (
        plan_tensor_arena,
    )
    from runtime_bundle_exporter.planner.weight_packing_planner import (
        pack_qlinearconv_weights,
    )
    from runtime_bundle_exporter.builder.tensor_table_builder import plan_weight_blob
    from runtime_bundle_exporter.runtime_ir import RuntimeBundle
    from runtime_bundle_exporter.writer.bundle_manifest_writer import (
        verify_bundle_manifest,
        write_bundle_manifest,
    )
    from runtime_bundle_exporter.writer.execution_plan_writer import (
        read_execution_plan,
        write_execution_plan,
    )
    from runtime_bundle_exporter.writer.weight_blob_writer import write_weight_blob

    source = args.source_bundle.resolve()
    output = args.output_dir.resolve()
    if source == output:
        raise CachePackedExportError("source bundle cannot be overwritten in place")
    source_manifest_path = source / "manifest.json"
    source_manifest = verify_bundle_manifest(source_manifest_path)
    source_weights_path = source / source_manifest["weights"]["path"]
    weights = source_weights_path.read_bytes()
    records_by_offset = _weight_records_by_offset(source_manifest)

    expected = [output / "manifest.json", output / "weights.bin"]
    expected.extend(
        output / "execution_plans" / f"plan_{entry['bucket_frames']}.bin"
        for entry in source_manifest["plans"]
    )
    existing = [path for path in expected if path.exists()]
    if existing and not args.force:
        raise CachePackedExportError(
            f"output already exists: {existing[0]} (use --force to overwrite)"
        )

    graphs = []
    reports = []
    for entry in sorted(source_manifest["plans"], key=lambda item: item["bucket_frames"]):
        loaded = read_execution_plan(source / entry["path"])
        graph = _runtime_graph_from_plan(loaded, weights, records_by_offset)
        layout = rewrite_cache_friendly_activations(
            graph, channel_block=args.channel_block
        )
        packing = pack_qlinearconv_weights(layout.graph)
        graphs.append(packing.graph)
        report = {
            "source_plan_sha256": entry["sha256"],
            "activation_layout": layout.to_dict(),
            "weight_packing": packing.to_dict(),
        }
        reports.append(report)

    bundle = RuntimeBundle(tuple(graphs))
    weight_layout = plan_weight_blob(bundle, alignment=args.weight_alignment)
    output.mkdir(parents=True, exist_ok=True)
    plan_dir = output / "execution_plans"
    report_dir = output / "cache_layout_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    weights_result = write_weight_blob(bundle, weight_layout, output / "weights.bin")

    plan_results = []
    for graph, report in zip(bundle.graphs, reports):
        arena = plan_tensor_arena(graph, alignment=args.arena_alignment)
        result = write_execution_plan(
            graph,
            weight_layout,
            plan_dir / f"plan_{graph.bucket_frames}.bin",
            kernel_id=1,
            arena_layout=arena,
        )
        plan_results.append(result)
        report["arena_size_bytes"] = arena.arena_size
        report["operator_count"] = len(graph.operators)
        (report_dir / f"cache_layout_{graph.bucket_frames}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"{graph.bucket_frames:>4} frames: operators={len(graph.operators)}, "
            f"arena={arena.arena_size:,} B, "
            f"layout={report['activation_layout']['channels_last_tensor_count']} tensors"
        )

    manifest_result = write_bundle_manifest(
        output / "manifest.json",
        bundle=bundle,
        layout=weight_layout,
        weights=weights_result,
        plans=plan_results,
        canonical_model=_source_path(source_manifest, "canonical_model", source),
        graph_manifest=_source_path(source_manifest, "graph_manifest", source),
    )
    manifest = manifest_result.document
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["optimization"] = {
        "name": "cache_layout_qconv_o4i4",
        "source_bundle": _relative_path(source, output),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_optimization": source_manifest.get("optimization"),
        "physical_activation_layout": "NTC_padded/NHWC_padded",
        "channel_block": args.channel_block,
        "weight_layout": "G_O4_K_I4_OL_IL",
        "weight_alignment": args.weight_alignment,
        "kernel_id": 1,
        "spatial_tile": 8,
        "runtime_weight_packing_count": 0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "source_bundle": str(source),
        "output_bundle": str(output),
        "bucket_results": reports,
    }
    (report_dir / "cache_layout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_bundle_manifest(output / "manifest.json")
    print(f"Cache-packed bundle: {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-bundle",
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
    parser.add_argument(
        "--output-dir",
        type=_path,
        default=(
            ROOT
            / "runs"
            / "runtime"
            / "kernel_optimization"
            / "cache_packed"
            / "bundle"
        ),
    )
    parser.add_argument("--channel-block", type=int, default=4)
    parser.add_argument("--arena-alignment", type=int, default=64)
    parser.add_argument("--weight-alignment", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        export_cache_packed_bundle(args)
        return 0
    except (CachePackedExportError, OSError, ValueError, KeyError) as exc:
        print(f"Cache-packed export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
