#!/usr/bin/env python3
"""Export the E7 fused graph and regenerate every memory/execution plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "src" / "python"


class E7ExportError(RuntimeError):
    """The source cache-packed bundle cannot be rewritten as E7."""


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


def _cache_export_module() -> object:
    source = ROOT / "scripts" / "3_runtime" / "12_export_cache_packed_bundle.py"
    spec = importlib.util.spec_from_file_location("cache_packed_export", source)
    if spec is None or spec.loader is None:
        raise E7ExportError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_path(manifest: dict, key: str, bundle_dir: Path) -> Path | None:
    entry = manifest.get(key)
    if not isinstance(entry, dict) or not entry.get("path"):
        return None
    return (bundle_dir / entry["path"]).resolve()


def export_e7(args: argparse.Namespace) -> None:
    if str(PYTHON_SOURCE) not in sys.path:
        sys.path.insert(0, str(PYTHON_SOURCE))
    cache_export = _cache_export_module()
    from runtime_bundle_exporter.builder.tensor_table_builder import plan_weight_blob
    from runtime_bundle_exporter.format.binary_format_schema import OperatorCode
    from runtime_bundle_exporter.planner.fusion_planner import (
        FusionConfig,
        rewrite_operator_fusions,
    )
    from runtime_bundle_exporter.planner.tensor_arena_planner import plan_tensor_arena
    from runtime_bundle_exporter.runtime_ir import RuntimeBundle, TensorStorageType
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
        raise E7ExportError("source bundle cannot be overwritten in place")
    source_manifest_path = source / "manifest.json"
    source_manifest = verify_bundle_manifest(source_manifest_path)
    source_weights = source / source_manifest["weights"]["path"]
    weights = source_weights.read_bytes()
    records_by_offset = cache_export._weight_records_by_offset(source_manifest)
    config = FusionConfig(
        conv_bias_act=args.fusion_conv_bias_act,
        bn_relu_quant=args.fusion_bn_relu_quant,
        pool_cam=args.fusion_pool_cam,
        qdq_elementwise=args.fusion_qdq_elementwise,
        stats_pooling=args.fusion_stats_pooling,
    )

    expected = [output / "manifest.json", output / "weights.bin"]
    expected.extend(
        output / "execution_plans" / f"plan_{entry['bucket_frames']}.bin"
        for entry in source_manifest["plans"]
    )
    existing = [path for path in expected if path.exists()]
    if existing and not args.force:
        raise E7ExportError(
            f"output already exists: {existing[0]} (use --force to overwrite)"
        )

    graphs = []
    rewrites = []
    for entry in sorted(source_manifest["plans"], key=lambda item: item["bucket_frames"]):
        loaded = read_execution_plan(source / entry["path"])
        graph = cache_export._runtime_graph_from_plan(
            loaded, weights, records_by_offset
        )
        rewrite = rewrite_operator_fusions(graph, config=config, default_kernel_id=1)
        graphs.append(rewrite.graph)
        rewrites.append(rewrite)

    bundle = RuntimeBundle(tuple(graphs))
    weight_layout = plan_weight_blob(bundle, alignment=args.weight_alignment)
    output.mkdir(parents=True, exist_ok=True)
    plan_dir = output / "execution_plans"
    report_dir = output / "fusion_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    weights_result = write_weight_blob(bundle, weight_layout, output / "weights.bin")

    plan_results = []
    reports = []
    for graph, rewrite in zip(bundle.graphs, rewrites):
        arena = plan_tensor_arena(graph, alignment=args.arena_alignment)
        result = write_execution_plan(
            graph,
            weight_layout,
            plan_dir / f"plan_{graph.bucket_frames}.bin",
            kernel_id=1,
            kernel_ids=rewrite.kernel_ids,
            arena_layout=arena,
        )
        plan_results.append(result)
        alias_bases = {
            tensor.alias_of_tensor_id
            for tensor in graph.tensors
            if tensor.storage_type is TensorStorageType.VIEW
        }
        report = rewrite.to_dict()
        report.update(
            {
                "arena_size_bytes": arena.arena_size,
                "arena_alignment": arena.alignment,
                "arena_allocation_count": len(arena.allocations),
                "alias_backing_bytes": sum(
                    tensor.storage_span_bytes or tensor.byte_size
                    for tensor in graph.tensors
                    if tensor.tensor_id in alias_bases
                ),
                "runtime_dynamic_allocation_count": 0,
                "runtime_weight_packing_count": 0,
                "profiling_instrumentation_enabled": False,
                "execution_table": [
                    {
                        "operator_id": operator.operator_id,
                        "opcode": OperatorCode(operator.opcode).name,
                        "kernel_id": rewrite.kernel_ids[operator.operator_id],
                        "input_tensor_ids": list(operator.input_tensor_ids),
                        "output_tensor_ids": list(operator.output_tensor_ids),
                    }
                    for operator in graph.operators
                ],
                "arena_allocations": [
                    {
                        "tensor_id": item.tensor_id,
                        "offset": item.offset,
                        "byte_size": item.byte_size,
                        "reserved_bytes": item.reserved_bytes,
                        "first_use": item.first_use,
                        "last_use": item.last_use,
                    }
                    for item in arena.allocations
                ],
            }
        )
        reports.append(report)
        (report_dir / f"fusion_{graph.bucket_frames}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"{graph.bucket_frames:>4} frames: "
            f"operators={rewrite.original_operator_count}->{len(graph.operators)}, "
            f"tensors={rewrite.original_tensor_count}->{len(graph.tensors)}, "
            f"arena={arena.arena_size:,} B, scratch={rewrite.maximum_scratch_bytes:,} B"
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
        "name": "E7_operator_fusion",
        "source_bundle": _relative_path(source, output),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_optimization": source_manifest.get("optimization"),
        "fusion_flags": config.to_dict(),
        "ordinary_kernel_id": 1,
        "fused_kernel_ids": [2, 3, 4, 5, 6],
        "runtime_dynamic_allocation_count": 0,
        "runtime_weight_packing_count": 0,
        "profiling_instrumentation_enabled": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "source_bundle": str(source),
        "output_bundle": str(output),
        "fusion_flags": config.to_dict(),
        "bucket_results": reports,
    }
    (report_dir / "fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_bundle_manifest(output / "manifest.json")
    print(f"E7 fused bundle: {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-bundle", type=_path,
        default=ROOT / "runs/runtime/kernel_optimization/cache_packed/bundle",
    )
    parser.add_argument(
        "--output-dir", type=_path,
        default=ROOT / "runs/runtime/kernel_optimization/e7/bundle",
    )
    parser.add_argument("--arena-alignment", type=int, default=64)
    parser.add_argument("--weight-alignment", type=int, default=64)
    parser.add_argument(
        "--fusion-conv-bias-act", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fusion-bn-relu-quant", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fusion-pool-cam", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fusion-qdq-elementwise", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fusion-stats-pooling", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        export_e7(args)
    except (E7ExportError, OSError, ValueError, KeyError) as exc:
        print(f"E7 export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
