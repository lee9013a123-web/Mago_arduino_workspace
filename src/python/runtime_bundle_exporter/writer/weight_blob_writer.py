# ONNX initializer를 weights.bin에 직렬화하는 모듈이다.
#
# 저장 대상:
# - convolution weight와 bias
# - BatchNormalization parameter
# - quantization scale과 zero point
# - runtime에서 필요한 작은 상수 Tensor
#
# Phase 3에서는 ONNX 원본 layout을 유지한다.
# NEON용 channel padding과 block packing은 이후 단계에서 별도 적용한다.

"""RuntimeBundle의 initializer를 하나의 weights.bin으로 굳힌다.

배치는 :func:`~.tensor_table_builder.plan_weight_blob`이 이미 정해 두었다.
이 모듈은 그 배치대로 바이트를 채우고, 항목마다 위치와 checksum을 기록한다.
값을 다시 배열하거나 채널을 패딩하지 않는다. ONNX가 준 바이트 그대로 쓴다.

같은 bundle을 두 번 써도 같은 파일이 나온다. 정렬 때문에 생기는 빈틈은 항상
0으로 채우므로 checksum이 흔들리지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from ..format.binary_format_schema import TensorDType
from ..runtime_ir import InitializerScope, RuntimeBundle, RuntimeInitializer
from ..builder.tensor_table_builder import WeightBlobEntry, WeightBlobLayout


WEIGHT_BLOB_FILE_NAME: str = "weights.bin"


class WeightBlobError(ValueError):
    """weights.bin을 쓰거나 되읽을 수 없을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class WeightRecord:
    """weights.bin 안의 항목 하나가 어디에 무엇으로 들어갔는지."""

    name: str
    dtype: TensorDType
    shape: tuple[int, ...]
    offset: int
    byte_size: int
    sha256: str
    scope: InitializerScope
    bucket_frames: int | None

    def to_dict(self) -> dict:
        """manifest에 실을 수 있는 평범한 값으로 바꾼다."""

        return {
            "name": self.name,
            "dtype": self.dtype.name,
            "shape": list(self.shape),
            "offset": self.offset,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "scope": self.scope.value,
            "bucket_frames": self.bucket_frames,
        }


@dataclass(frozen=True, slots=True)
class WeightBlobResult:
    """weights.bin 한 개를 쓴 결과."""

    path: Path
    byte_size: int
    sha256: str
    records: tuple[WeightRecord, ...]

    @property
    def shared_bytes(self) -> int:
        return sum(
            record.byte_size
            for record in self.records
            if record.scope is InitializerScope.SHARED
        )

    @property
    def bucket_bytes(self) -> int:
        return sum(
            record.byte_size
            for record in self.records
            if record.scope is InitializerScope.BUCKET_LOCAL
        )

    @property
    def padding_bytes(self) -> int:
        return self.byte_size - self.shared_bytes - self.bucket_bytes


def _initializers_by_key(
    bundle: RuntimeBundle,
) -> Mapping[tuple[object, ...], RuntimeInitializer]:
    """layout이 쓰는 키로 initializer를 찾을 수 있게 모은다."""

    sources: dict[tuple[object, ...], RuntimeInitializer] = {}
    for item in bundle.graphs[0].initializers:
        if item.scope is InitializerScope.SHARED:
            sources[("shared", item.name)] = item
    for graph in bundle.graphs:
        for item in graph.initializers:
            if item.scope is InitializerScope.BUCKET_LOCAL:
                sources[("bucket", graph.bucket_frames, item.name)] = item
    return sources


def build_weight_blob(
    bundle: RuntimeBundle, layout: WeightBlobLayout
) -> tuple[bytes, tuple[WeightRecord, ...]]:
    """weights.bin의 내용을 메모리에서 만든다."""

    sources = _initializers_by_key(bundle)
    missing = [entry.key for entry in layout.entries if entry.key not in sources]
    if missing:
        raise WeightBlobError(f"배치에는 있으나 bundle에 없는 항목: {missing[:3]}")

    buffer = bytearray(layout.total_bytes)
    records: list[WeightRecord] = []
    for entry in layout.entries:
        item = sources[entry.key]
        if item.byte_size != entry.byte_size:
            raise WeightBlobError(
                f"{entry.name!r}의 크기가 배치({entry.byte_size})와 "
                f"데이터({item.byte_size})에서 다르다"
            )
        end = entry.offset + entry.byte_size
        if end > layout.total_bytes:
            raise WeightBlobError(
                f"{entry.name!r}가 blob 끝({layout.total_bytes})을 넘는다: {end}"
            )
        buffer[entry.offset : end] = item.raw_data
        records.append(
            WeightRecord(
                name=entry.name,
                dtype=item.dtype,
                shape=item.shape,
                offset=entry.offset,
                byte_size=entry.byte_size,
                sha256=hashlib.sha256(item.raw_data).hexdigest(),
                scope=entry.scope,
                bucket_frames=entry.bucket_frames,
            )
        )

    return bytes(buffer), tuple(records)


def write_weight_blob(
    bundle: RuntimeBundle, layout: WeightBlobLayout, path: Path | str
) -> WeightBlobResult:
    """weights.bin을 파일로 쓴다."""

    target = Path(path)
    blob, records = build_weight_blob(bundle, layout)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)

    return WeightBlobResult(
        path=target,
        byte_size=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(),
        records=records,
    )


def read_weight_blob(path: Path | str) -> bytes:
    """weights.bin 전체를 읽는다."""

    source = Path(path)
    if not source.exists():
        raise WeightBlobError(f"weights.bin이 없다: {source}")
    return source.read_bytes()


def extract(blob: bytes, entry: WeightBlobEntry) -> bytes:
    """배치 항목 하나에 해당하는 바이트를 잘라낸다."""

    end = entry.offset + entry.byte_size
    if end > len(blob):
        raise WeightBlobError(
            f"{entry.name!r}가 파일 끝({len(blob)})을 넘는다: {end}"
        )
    return blob[entry.offset : end]


def verify_weight_blob(
    blob: bytes,
    bundle: RuntimeBundle,
    layout: WeightBlobLayout,
    *,
    records: Sequence[WeightRecord] | None = None,
) -> None:
    """되읽은 blob이 bundle의 initializer와 바이트 단위로 같은지 확인한다."""

    if len(blob) != layout.total_bytes:
        raise WeightBlobError(
            f"weights.bin 크기가 배치와 다르다: 파일 {len(blob)}, "
            f"배치 {layout.total_bytes}"
        )

    for graph in bundle.graphs:
        for item in graph.initializers:
            offset = layout.offset_of(graph.bucket_frames, item.tensor_id)
            stored = blob[offset : offset + item.byte_size]
            if stored != item.raw_data:
                raise WeightBlobError(
                    f"bucket {graph.bucket_frames}의 {item.name!r} 바이트가 "
                    f"offset {offset}에서 다르다"
                )

    for record in records or ():
        stored = blob[record.offset : record.offset + record.byte_size]
        if hashlib.sha256(stored).hexdigest() != record.sha256:
            raise WeightBlobError(f"{record.name!r}의 checksum이 맞지 않는다")


__all__ = [
    "WEIGHT_BLOB_FILE_NAME",
    "WeightBlobError",
    "WeightBlobResult",
    "WeightRecord",
    "build_weight_blob",
    "extract",
    "read_weight_blob",
    "verify_weight_blob",
    "write_weight_blob",
]
