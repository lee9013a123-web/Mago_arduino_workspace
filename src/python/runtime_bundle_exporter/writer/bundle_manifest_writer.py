# 생성된 Runtime bundle의 추적 정보를 manifest.json으로 저장하는 모듈이다.
#
# 기록할 정보:
# - canonical ONNX SHA-256
# - graph freeze manifest hash
# - binary format version
# - bucket별 frame 수와 operator 수
# - weights.bin과 plan_*.bin의 크기 및 SHA-256
#
# Runtime은 로드 시 이 정보와 binary header가 일치하는지 확인한다.

"""weights.bin과 plan 파일들을 하나의 bundle로 묶어 기록한다.

manifest는 산출물의 출처를 되짚기 위한 파일이다. 어떤 canonical ONNX와 어떤
graph freeze manifest에서 나왔는지, 각 파일의 크기와 SHA-256이 무엇인지,
bucket마다 Tensor와 Operator가 몇 개인지를 적는다. C Runtime은 로드할 때
plan header의 값이 여기 적힌 값과 같은지 확인할 수 있다.

경로는 manifest 파일이 있는 디렉터리 기준 상대 경로로 적는다. bundle 폴더를
통째로 보드에 복사해도 그대로 유효하다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from ..format.binary_format_schema import PLAN_FORMAT_VERSION
from .execution_plan_writer import ExecutionPlanResult
from ..runtime_ir import RuntimeBundle
from ..builder.tensor_table_builder import WeightBlobLayout
from .weight_blob_writer import WeightBlobResult


MANIFEST_FILE_NAME: str = "manifest.json"
MANIFEST_SCHEMA_VERSION: str = "1.0"

_HASH_CHUNK_SIZE: int = 1 << 20


class BundleManifestError(ValueError):
    """manifest를 쓰거나 되읽을 수 없을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class BundleManifestResult:
    """manifest.json 한 개를 쓴 결과."""

    path: Path
    byte_size: int
    document: dict


def file_sha256(path: Path | str) -> str:
    """큰 파일도 메모리에 다 올리지 않고 해시한다."""

    source = Path(path)
    if not source.exists():
        raise BundleManifestError(f"해시할 파일이 없다: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(path: Path, root: Path) -> str:
    """bundle 안의 파일은 manifest 위치 기준 상대 경로로 적는다."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _source_entry(path: Path | str | None, root: Path) -> dict | None:
    """입력 산출물은 bundle 밖에 있으므로 ``..``를 포함한 상대 경로로 적는다.

    절대 경로를 적으면 만든 사람의 디렉터리 구조가 산출물에 남고, 저장소를
    옮기는 순간 무의미해진다. 출처를 실제로 보증하는 것은 경로가 아니라
    SHA-256이다.
    """

    if path is None:
        return None
    source = Path(path)
    try:
        relative = os.path.relpath(source.resolve(), root.resolve())
    except ValueError:  # 다른 드라이브 등 상대화가 불가능한 경우
        relative = str(source)
    return {
        "path": relative,
        "size_bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def build_bundle_manifest(
    *,
    bundle: RuntimeBundle,
    layout: WeightBlobLayout,
    weights: WeightBlobResult,
    plans: Sequence[ExecutionPlanResult],
    root: Path,
    canonical_model: Path | str | None = None,
    graph_manifest: Path | str | None = None,
    include_weight_index: bool = True,
) -> dict:
    """manifest 문서를 만든다. 파일로 쓰지는 않는다."""

    if not plans:
        raise BundleManifestError("manifest에 실을 plan이 없다")

    graphs = {graph.bucket_frames: graph for graph in bundle.graphs}
    plan_entries = []
    for plan in sorted(plans, key=lambda item: item.bucket_frames):
        graph = graphs.get(plan.bucket_frames)
        if graph is None:
            raise BundleManifestError(
                f"bundle에 없는 bucket의 plan이다: {plan.bucket_frames}"
            )
        plan_entries.append(
            {
                "bucket_frames": plan.bucket_frames,
                "path": _relative_to(plan.path, root),
                "size_bytes": plan.byte_size,
                "sha256": plan.sha256,
                "tensor_count": plan.tensor_count,
                "operator_count": plan.operator_count,
                "attribute_section_bytes": plan.attribute_section_bytes,
                "attribute_blocks": plan.attribute_blocks,
            }
        )

    document: dict = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": PLAN_FORMAT_VERSION,
        "canonical_model": _source_entry(canonical_model, root),
        "graph_manifest": _source_entry(graph_manifest, root),
        "weights": {
            "path": _relative_to(weights.path, root),
            "size_bytes": weights.byte_size,
            "sha256": weights.sha256,
            "entry_count": len(weights.records),
            "shared_bytes": weights.shared_bytes,
            "bucket_bytes": weights.bucket_bytes,
            "padding_bytes": weights.padding_bytes,
            "alignment": layout.alignment,
        },
        "plans": plan_entries,
    }
    if include_weight_index:
        document["weight_index"] = [
            record.to_dict() for record in weights.records
        ]
    return document


def write_bundle_manifest(
    path: Path | str,
    *,
    bundle: RuntimeBundle,
    layout: WeightBlobLayout,
    weights: WeightBlobResult,
    plans: Sequence[ExecutionPlanResult],
    canonical_model: Path | str | None = None,
    graph_manifest: Path | str | None = None,
    include_weight_index: bool = True,
) -> BundleManifestResult:
    """manifest.json을 쓴다."""

    target = Path(path)
    document = build_bundle_manifest(
        bundle=bundle,
        layout=layout,
        weights=weights,
        plans=plans,
        root=target.parent,
        canonical_model=canonical_model,
        graph_manifest=graph_manifest,
        include_weight_index=include_weight_index,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    # Windows의 기본 newline 변환에 맡기면 같은 bundle도 CRLF/LF 차이로 파일
    # 크기와 hash가 달라진다. 산출물은 운영체제와 무관하게 LF로 고정한다.
    target.write_text(text, encoding="utf-8", newline="\n")

    return BundleManifestResult(
        path=target, byte_size=len(text.encode("utf-8")), document=document
    )


def read_bundle_manifest(path: Path | str) -> dict:
    """manifest.json을 읽는다."""

    source = Path(path)
    if not source.exists():
        raise BundleManifestError(f"manifest가 없다: {source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleManifestError(f"manifest JSON을 읽을 수 없다: {exc}") from exc


def verify_bundle_manifest(path: Path | str) -> dict:
    """manifest가 가리키는 파일들이 적힌 크기와 해시 그대로인지 확인한다."""

    source = Path(path)
    document = read_bundle_manifest(source)
    root = source.parent

    entries = [document["weights"], *document["plans"]]
    for entry in entries:
        target = root / entry["path"]
        if not target.exists():
            raise BundleManifestError(f"manifest가 가리키는 파일이 없다: {target}")
        size = target.stat().st_size
        if size != entry["size_bytes"]:
            raise BundleManifestError(
                f"{entry['path']}의 크기가 다르다: 파일 {size}, "
                f"manifest {entry['size_bytes']}"
            )
        digest = file_sha256(target)
        if digest != entry["sha256"]:
            raise BundleManifestError(f"{entry['path']}의 SHA-256이 다르다")

    return document


__all__ = [
    "BundleManifestError",
    "BundleManifestResult",
    "MANIFEST_FILE_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "build_bundle_manifest",
    "file_sha256",
    "read_bundle_manifest",
    "verify_bundle_manifest",
    "write_bundle_manifest",
]
