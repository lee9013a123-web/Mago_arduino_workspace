#!/usr/bin/env python3
"""정적 ONNX 4개를 CAM++ Reference Runtime bundle로 변환한다.

이 파일은 경로와 실행 순서만 담당한다. ONNX 해석, RuntimeGraph 변환, binary
직렬화와 검증은 ``src/python/runtime_bundle_exporter`` 모듈에 맡긴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "src" / "python"

# (고정 frame 수, graph IR 파일 접미사)
BUCKETS: tuple[tuple[int, str], ...] = (
    (98, "1s"),
    (298, "3s"),
    (498, "5s"),
    (998, "10s"),
)


class BundleExportCliError(RuntimeError):
    """입력 경로나 CLI 실행 조건이 맞지 않을 때 발생한다."""


def _repository_path(value: str) -> Path:
    """상대 경로를 저장소 root 기준 절대 경로로 바꾼다."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "정적 CAM++ ONNX와 graph IR을 weights.bin, bucket별 plan_*.bin, "
            "manifest.json으로 변환하고 즉시 되읽어 검증한다."
        )
    )
    parser.add_argument(
        "--static-dir",
        type=_repository_path,
        default=ROOT / "results" / "static",
        help="campp_static_{frames}.onnx가 있는 폴더 (기본: results/static)",
    )
    parser.add_argument(
        "--graph-dir",
        type=_repository_path,
        default=ROOT / "results" / "graph",
        help="ir_{1s,3s,5s,10s}.json이 있는 폴더 (기본: results/graph)",
    )
    parser.add_argument(
        "--canonical-model",
        type=_repository_path,
        default=ROOT / "models" / "source" / "campplus_int8_static_qop.onnx",
        help="manifest에 출처와 SHA-256을 기록할 canonical ONNX",
    )
    parser.add_argument(
        "--graph-manifest",
        type=_repository_path,
        default=None,
        help="있다면 manifest에 출처와 SHA-256을 기록할 graph manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=_repository_path,
        default=ROOT / "models" / "compiled" / "reference",
        help="bundle 출력 폴더 (기본: models/compiled/reference)",
    )
    weight_index = parser.add_mutually_exclusive_group()
    weight_index.add_argument(
        "--include-weight-index",
        dest="include_weight_index",
        action="store_true",
        default=True,
        help="manifest에 initializer별 이름·offset·checksum을 기록한다 (기본)",
    )
    weight_index.add_argument(
        "--no-weight-index",
        dest="include_weight_index",
        action="store_false",
        help="보드 배포용으로 큰 weight index를 manifest에서 제외한다",
    )
    parser.add_argument(
        "--tensor-arena",
        action="store_true",
        help="ACTIVATION/OUTPUT data_offset을 계산한 Arena plan을 별도 output에 생성",
    )
    parser.add_argument(
        "--dense-slab",
        action="store_true",
        help="누적 Dense Concat을 slab-backed VIEW로 바꾸고 새 plan을 생성",
    )
    parser.add_argument(
        "--dense-slab-reference-bundle",
        type=_repository_path,
        default=ROOT / "runs" / "runtime" / "tensor_arena" / "bundle",
        help=(
            "변환 전 graph와 대조할 기존 Tensor Arena bundle "
            "(기본: runs/runtime/tensor_arena/bundle)"
        ),
    )
    parser.add_argument(
        "--arena-alignment",
        type=int,
        default=64,
        help="Tensor Arena byte 정렬. 현재 C Runtime ABI는 64만 지원 (기본: 64)",
    )
    parser.add_argument(
        "--arena-budget-bytes",
        type=int,
        default=None,
        help="계산된 Arena가 이 byte 예산을 넘으면 export 실패",
    )
    return parser


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise BundleExportCliError(f"{label} 파일이 없다: {path}")


def _load_exporter_api() -> dict[str, object]:
    """--help는 onnx 없이도 동작하도록 실제 import를 실행 시점까지 늦춘다."""

    source = str(PYTHON_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)

    try:
        from runtime_bundle_exporter.writer.bundle_manifest_writer import (
            MANIFEST_FILE_NAME,
            verify_bundle_manifest,
            write_bundle_manifest,
        )
        from runtime_bundle_exporter.writer.execution_plan_writer import (
            EXECUTION_PLAN_DIR_NAME,
            plan_file_name,
            read_execution_plan,
            verify_plan,
            write_execution_plan,
        )
        from runtime_bundle_exporter.reader.graph_ir_reader import read_graph_ir
        from runtime_bundle_exporter.runtime_ir import RuntimeBundle
        from runtime_bundle_exporter.reader.static_model_reader import (
            build_runtime_graph,
            read_static_model,
        )
        from runtime_bundle_exporter.builder.tensor_table_builder import plan_weight_blob
        from runtime_bundle_exporter.builder.planner.tensor_arena_planner import (
            plan_tensor_arena,
        )
        from runtime_bundle_exporter.builder.planner.dense_slab_planner import (
            inspect_compiled_dense_concats,
            rewrite_dense_concats_as_slabs,
        )
        from runtime_bundle_exporter.writer.weight_blob_writer import (
            WEIGHT_BLOB_FILE_NAME,
            read_weight_blob,
            verify_weight_blob,
            write_weight_blob,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "onnx":
            raise BundleExportCliError(
                "onnx가 설치되어 있지 않다. 다음 명령으로 현재 가상환경에 "
                "설치해야 한다:\n"
                '  & ".\\.venv\\Scripts\\python.exe" -m pip install onnx'
            ) from exc
        raise

    return locals()


def _print_summary(weights: object, plans: Sequence[object], manifest: object) -> None:
    print()
    print("생성 결과")
    print(
        f"weights.bin  {weights.byte_size:>12,} B  sha={weights.sha256[:12]}… "
        f"(shared {weights.shared_bytes:,} / bucket {weights.bucket_bytes:,} / "
        f"padding {weights.padding_bytes:,})"
    )
    has_arena = any(plan.arena_size is not None for plan in plans)
    arena_heading = f" {'arena bytes':>12}" if has_arena else ""
    print(
        f"{'bucket':>8} {'tensors':>9} {'operators':>10} "
        f"{'attributes':>12} {'plan bytes':>12}{arena_heading}  sha"
    )
    for plan in sorted(plans, key=lambda item: item.bucket_frames):
        arena_value = (
            f" {plan.arena_size:>12,}" if plan.arena_size is not None else ""
        )
        print(
            f"{plan.bucket_frames:>8} {plan.tensor_count:>9} "
            f"{plan.operator_count:>10} {plan.attribute_section_bytes:>12,} "
            f"{plan.byte_size:>12,}{arena_value}  {plan.sha256[:12]}…"
        )
    print(
        f"manifest.json {manifest.byte_size:>10,} B  "
        f"weight_index={'포함' if 'weight_index' in manifest.document else '제외'}"
    )


def export_reference_bundle(args: argparse.Namespace) -> None:
    canonical_reference = ROOT / "models" / "compiled" / "reference"
    if (
        (args.tensor_arena or args.dense_slab)
        and args.output_dir.resolve() == canonical_reference.resolve()
    ):
        raise BundleExportCliError(
            "Arena plan은 기존 Reference bundle을 덮어쓸 수 없다. "
            "--output-dir로 runs/runtime/tensor_arena 아래의 별도 경로를 지정해야 한다"
        )
    if not args.tensor_arena and args.arena_budget_bytes is not None:
        raise BundleExportCliError(
            "--arena-budget-bytes는 --tensor-arena와 함께 사용해야 한다"
        )
    if args.dense_slab and not args.tensor_arena:
        raise BundleExportCliError(
            "--dense-slab은 slab backing을 배치할 --tensor-arena와 함께 사용해야 한다"
        )
    if args.tensor_arena and args.arena_alignment != 64:
        raise BundleExportCliError(
            "현재 C Runtime의 Tensor Arena alignment는 64바이트로 고정되어 있다"
        )

    api = _load_exporter_api()

    read_static_model = api["read_static_model"]
    read_graph_ir = api["read_graph_ir"]
    build_runtime_graph = api["build_runtime_graph"]
    RuntimeBundle = api["RuntimeBundle"]
    plan_weight_blob = api["plan_weight_blob"]
    plan_tensor_arena = api["plan_tensor_arena"]
    rewrite_dense_concats_as_slabs = api["rewrite_dense_concats_as_slabs"]
    inspect_compiled_dense_concats = api["inspect_compiled_dense_concats"]
    write_weight_blob = api["write_weight_blob"]
    read_weight_blob = api["read_weight_blob"]
    verify_weight_blob = api["verify_weight_blob"]
    write_execution_plan = api["write_execution_plan"]
    read_execution_plan = api["read_execution_plan"]
    verify_plan = api["verify_plan"]
    write_bundle_manifest = api["write_bundle_manifest"]
    verify_bundle_manifest = api["verify_bundle_manifest"]
    weight_file_name = api["WEIGHT_BLOB_FILE_NAME"]
    plan_directory_name = api["EXECUTION_PLAN_DIR_NAME"]
    plan_file_name = api["plan_file_name"]
    manifest_file_name = api["MANIFEST_FILE_NAME"]

    _require_file(args.canonical_model, "canonical ONNX")
    if args.graph_manifest is not None:
        _require_file(args.graph_manifest, "graph manifest")

    inputs: list[tuple[int, Path, Path]] = []
    for frames, tag in BUCKETS:
        static_path = args.static_dir / f"campp_static_{frames}.onnx"
        graph_path = args.graph_dir / f"ir_{tag}.json"
        _require_file(static_path, f"bucket {frames} 정적 ONNX")
        _require_file(graph_path, f"bucket {frames} graph IR")
        inputs.append((frames, static_path, graph_path))

    started = time.perf_counter()
    graphs = []
    print("입력 검증 및 RuntimeGraph 생성")
    for frames, static_path, graph_path in inputs:
        static = read_static_model(static_path)
        graph_ir = read_graph_ir(graph_path)
        if graph_ir.frames != frames:
            raise BundleExportCliError(
                f"{graph_path}의 frames가 파일 규칙과 다르다: "
                f"기대 {frames}, 실제 {graph_ir.frames}"
            )
        graph = build_runtime_graph(
            static,
            graph_ir,
            validate=True,
            name=f"campp_{frames}f",
        )
        graphs.append(graph)
        print(
            f"  {frames:>4} frames  tensors={len(graph.tensors):,}  "
            f"operators={len(graph.operators):,}  "
            f"initializers={len(graph.initializers):,}"
        )

    bundle = RuntimeBundle(graphs=tuple(graphs))
    layout = plan_weight_blob(bundle)
    dense_slab_results = {}
    if args.dense_slab:
        reference_manifest_path = (
            args.dense_slab_reference_bundle / manifest_file_name
        )
        _require_file(reference_manifest_path, "Dense slab reference manifest")
        reference_manifest = verify_bundle_manifest(reference_manifest_path)
        if reference_manifest.get("memory_layout") != "tensor_arena":
            raise BundleExportCliError(
                "Dense slab reference bundle이 Tensor Arena bundle이 아니다"
            )

        optimized_graphs = []
        print("기존 plan_*.bin 대조 및 Dense slab graph rewrite")
        for graph in bundle.graphs:
            reference_plan_path = (
                args.dense_slab_reference_bundle
                / plan_directory_name
                / plan_file_name(graph.bucket_frames)
            )
            _require_file(
                reference_plan_path,
                f"bucket {graph.bucket_frames} Dense slab reference plan",
            )
            reference_plan = read_execution_plan(reference_plan_path)
            reference_arena = plan_tensor_arena(
                graph, alignment=args.arena_alignment
            )
            verify_plan(
                reference_plan,
                graph,
                layout,
                arena_layout=reference_arena,
            )
            compiled_blocks = inspect_compiled_dense_concats(reference_plan)
            if len(compiled_blocks) != 3 or sum(
                len(block.concat_operator_ids) for block in compiled_blocks
            ) != 52:
                raise BundleExportCliError(
                    f"bucket {graph.bucket_frames} 기존 plan의 Dense chain이 "
                    "3 blocks / 52 Concats가 아니다"
                )
            result = rewrite_dense_concats_as_slabs(
                graph,
                expected_block_count=3,
                expected_concat_count=52,
            )
            if tuple(
                block.concat_operator_ids for block in compiled_blocks
            ) != tuple(block.concat_operator_ids for block in result.blocks):
                raise BundleExportCliError(
                    f"bucket {graph.bucket_frames}의 기존 plan과 RuntimeGraph가 "
                    "서로 다른 Dense Concat chain을 가리킨다"
                )
            dense_slab_results[graph.bucket_frames] = result
            optimized_graphs.append(result.graph)
            print(
                f"  {graph.bucket_frames:>4} frames  blocks={len(result.blocks)}  "
                f"concat={result.removed_concat_count} removed  "
                f"operators={result.original_operator_count}→"
                f"{len(result.graph.operators)}"
            )

        bundle = RuntimeBundle(graphs=tuple(optimized_graphs))
        layout = plan_weight_blob(bundle)

    arena_layouts = {}
    if args.tensor_arena:
        print("Tensor Arena offset 계산")
        for graph in bundle.graphs:
            arena = plan_tensor_arena(
                graph,
                alignment=args.arena_alignment,
                max_arena_bytes=args.arena_budget_bytes,
            )
            arena_layouts[graph.bucket_frames] = arena
            print(
                f"  {graph.bucket_frames:>4} frames  arena={arena.arena_size:,} B  "
                f"independent={arena.naive_aligned_bytes:,} B  "
                f"peak={arena.theoretical_peak_bytes:,} B"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_dir = args.output_dir / plan_directory_name
    plan_dir.mkdir(parents=True, exist_ok=True)
    if dense_slab_results:
        report_dir = args.output_dir / "dense_slab_plans"
        report_dir.mkdir(parents=True, exist_ok=True)
        for frames, result in sorted(dense_slab_results.items()):
            report = result.to_dict()
            report["reference_bundle"] = str(
                args.dense_slab_reference_bundle.resolve()
            )
            report["arena_size_bytes"] = arena_layouts[frames].arena_size
            (report_dir / f"dense_slab_{frames}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    weights = write_weight_blob(
        bundle,
        layout,
        args.output_dir / weight_file_name,
    )
    plans = tuple(
        write_execution_plan(
            graph,
            layout,
            plan_dir / plan_file_name(graph.bucket_frames),
            arena_layout=arena_layouts.get(graph.bucket_frames),
        )
        for graph in bundle.graphs
    )
    manifest = write_bundle_manifest(
        args.output_dir / manifest_file_name,
        bundle=bundle,
        layout=layout,
        weights=weights,
        plans=plans,
        canonical_model=args.canonical_model,
        graph_manifest=args.graph_manifest,
        include_weight_index=args.include_weight_index,
    )

    # 완료 조건: 쓴 파일을 다시 읽어 RuntimeGraph와 byte 단위로 대조한다.
    print("생성 파일 되읽기 검증")
    blob = read_weight_blob(weights.path)
    verify_weight_blob(
        blob,
        bundle,
        layout,
        records=weights.records,
    )
    graph_by_bucket = {graph.bucket_frames: graph for graph in bundle.graphs}
    for plan in plans:
        loaded = read_execution_plan(plan.path)
        verify_plan(
            loaded,
            graph_by_bucket[plan.bucket_frames],
            layout,
            arena_layout=arena_layouts.get(plan.bucket_frames),
        )
        print(f"  plan_{plan.bucket_frames}.bin  OK")
    verify_bundle_manifest(manifest.path)
    print("  weights.bin / manifest.json  OK")

    _print_summary(weights, plans, manifest)
    print(f"검증 완료 ({time.perf_counter() - started:.2f}s)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        export_reference_bundle(args)
    except KeyboardInterrupt:
        print("export cancelled", file=sys.stderr)
        return 130
    except Exception as exc:  # CLI 경계에서 오류 종류와 원인을 한 줄로 보고한다.
        print(f"export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
