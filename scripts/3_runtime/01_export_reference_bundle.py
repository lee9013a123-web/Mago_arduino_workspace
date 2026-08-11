#!/usr/bin/env python3
"""정적 ONNX 4개를 CAM++ Reference Runtime bundle로 변환한다.

이 파일은 경로와 실행 순서만 담당한다. ONNX 해석, RuntimeGraph 변환, binary
직렬화와 검증은 ``src/python/runtime_bundle_exporter`` 모듈에 맡긴다.
"""

from __future__ import annotations

import argparse
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
        from runtime_bundle_exporter.bundle_manifest_writer import (
            MANIFEST_FILE_NAME,
            verify_bundle_manifest,
            write_bundle_manifest,
        )
        from runtime_bundle_exporter.execution_plan_writer import (
            EXECUTION_PLAN_DIR_NAME,
            plan_file_name,
            read_execution_plan,
            verify_plan,
            write_execution_plan,
        )
        from runtime_bundle_exporter.graph_ir_reader import read_graph_ir
        from runtime_bundle_exporter.runtime_ir import RuntimeBundle
        from runtime_bundle_exporter.static_model_reader import (
            build_runtime_graph,
            read_static_model,
        )
        from runtime_bundle_exporter.tensor_table_builder import plan_weight_blob
        from runtime_bundle_exporter.weight_blob_writer import (
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
    print(
        f"{'bucket':>8} {'tensors':>9} {'operators':>10} "
        f"{'attributes':>12} {'plan bytes':>12}  sha"
    )
    for plan in sorted(plans, key=lambda item: item.bucket_frames):
        print(
            f"{plan.bucket_frames:>8} {plan.tensor_count:>9} "
            f"{plan.operator_count:>10} {plan.attribute_section_bytes:>12,} "
            f"{plan.byte_size:>12,}  {plan.sha256[:12]}…"
        )
    print(
        f"manifest.json {manifest.byte_size:>10,} B  "
        f"weight_index={'포함' if 'weight_index' in manifest.document else '제외'}"
    )


def export_reference_bundle(args: argparse.Namespace) -> None:
    api = _load_exporter_api()

    read_static_model = api["read_static_model"]
    read_graph_ir = api["read_graph_ir"]
    build_runtime_graph = api["build_runtime_graph"]
    RuntimeBundle = api["RuntimeBundle"]
    plan_weight_blob = api["plan_weight_blob"]
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_dir = args.output_dir / plan_directory_name
    plan_dir.mkdir(parents=True, exist_ok=True)

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
        verify_plan(loaded, graph_by_bucket[plan.bucket_frames], layout)
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
