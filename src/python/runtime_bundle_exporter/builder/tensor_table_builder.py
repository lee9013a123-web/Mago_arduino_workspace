"""RuntimeGraph의 Tensor를 plan의 Tensor table로 굳히는 모듈.

Tensor ID 자체는 :func:`~.static_model_reader.build_runtime_graph`가 이미
0부터 조밀하게 매겨 두었다. 여기서는 그 ID에 붙는 디스크 표현을 만든다.
storage 종류별로 ``data_offset``의 기준이 달라지므로, 이 모듈이 weight blob
배치도 함께 계산한다.

::

    INPUT       외부에서 pointer를 연결한다        -> INVALID_DATA_OFFSET
    CONSTANT    weights.bin 시작부터의 offset      -> WeightBlobLayout이 결정
    ACTIVATION  Tensor Arena 시작부터의 offset     -> Arena layout을 줄 때 배정
    OUTPUT      Tensor Arena 시작부터의 offset     -> Arena layout을 줄 때 배정
    VIEW        alias Tensor 시작부터의 offset     -> 아직 생성하지 않는다

기본 Reference 경로는 Arena를 계획하지 않으므로 ACTIVATION과 OUTPUT을
``INVALID_DATA_OFFSET``으로 남긴다. ``TensorArenaLayout``을 명시한 경우에만
같은 80바이트 descriptor의 ``data_offset``을 채우고 ``DENSE_SLAB`` flag를 켠다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..format.binary_format_schema import (
    INVALID_DATA_OFFSET,
    INVALID_OPERATOR_INDEX,
    INVALID_QUANTIZATION_INDEX,
    INVALID_TENSOR_ID,
    TENSOR_DESCRIPTOR_SIZE,
    TENSOR_MAX_RANK,
    TensorDescriptor,
    TensorFlags,
    TensorStorageType,
)
from ..runtime_ir import InitializerScope, RuntimeBundle, RuntimeGraph, RuntimeTensor
from .tensor_arena_planner import TensorArenaLayout


# weights.bin의 모든 항목을 8바이트 경계에 두어 C가 int64까지 그대로 읽게 한다.
WEIGHT_BLOB_ALIGNMENT: int = 8


class TensorTableError(ValueError):
    """Tensor table이나 weight blob 배치를 만들 수 없을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class WeightBlobEntry:
    """weights.bin 안의 항목 하나."""

    key: tuple[object, ...]
    name: str
    scope: InitializerScope
    bucket_frames: int | None
    offset: int
    byte_size: int


@dataclass(frozen=True, slots=True)
class WeightBlobLayout:
    """bundle 전체가 공유하는 weights.bin의 배치.

    ``SHARED``는 한 번만 기록하고, ``BUCKET_LOCAL``은 bucket마다 따로 기록한다.
    따라서 같은 Tensor ID라도 plan마다 다른 offset을 가리킬 수 있다.
    """

    entries: tuple[WeightBlobEntry, ...]
    total_bytes: int
    alignment: int
    tensor_offsets: Mapping[tuple[int | None, int], int]
    _by_key: Mapping[tuple[object, ...], WeightBlobEntry] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(
            self, "tensor_offsets", MappingProxyType(dict(self.tensor_offsets))
        )
        object.__setattr__(
            self,
            "_by_key",
            MappingProxyType({entry.key: entry for entry in self.entries}),
        )

    @property
    def shared_bytes(self) -> int:
        return sum(
            entry.byte_size
            for entry in self.entries
            if entry.scope is InitializerScope.SHARED
        )

    @property
    def bucket_bytes(self) -> int:
        return sum(
            entry.byte_size
            for entry in self.entries
            if entry.scope is InitializerScope.BUCKET_LOCAL
        )

    def entry(self, key: tuple[object, ...]) -> WeightBlobEntry:
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise TensorTableError(f"weight blob에 없는 항목: {key!r}") from exc

    def offset_of(self, bucket_frames: int | None, tensor_id: int) -> int:
        """한 bucket의 Tensor ID가 가리킬 weights.bin offset을 돌려준다."""

        try:
            return self.tensor_offsets[(bucket_frames, tensor_id)]
        except KeyError as exc:
            raise TensorTableError(
                f"bucket {bucket_frames}의 Tensor {tensor_id}는 상수가 아니다"
            ) from exc


def _aligned(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + (alignment - remainder)


def plan_weight_blob(
    bundle: RuntimeBundle, *, alignment: int = WEIGHT_BLOB_ALIGNMENT
) -> WeightBlobLayout:
    """bundle의 initializer를 weights.bin 위에 배치한다.

    배치 순서는 ``SHARED`` 전부, 그 뒤에 bucket 순서대로 ``BUCKET_LOCAL``이다.
    이렇게 하면 학습 weight가 파일 앞쪽에 연속으로 모여 mmap 지역성이 좋아지고,
    bucket을 추가해도 앞쪽 offset이 흔들리지 않는다.
    """

    if alignment <= 0:
        raise TensorTableError(f"weight blob 정렬은 양수여야 한다: {alignment}")

    entries: list[WeightBlobEntry] = []
    shared_offsets: dict[str, int] = {}
    bucket_offsets: dict[tuple[int | None, str], int] = {}
    cursor = 0

    for item in bundle.graphs[0].initializers:
        if item.scope is not InitializerScope.SHARED:
            continue
        cursor = _aligned(cursor, alignment)
        entries.append(
            WeightBlobEntry(
                key=("shared", item.name),
                name=item.name,
                scope=item.scope,
                bucket_frames=None,
                offset=cursor,
                byte_size=item.byte_size,
            )
        )
        shared_offsets[item.name] = cursor
        cursor += item.byte_size

    for graph in bundle.graphs:
        for item in graph.initializers:
            if item.scope is not InitializerScope.BUCKET_LOCAL:
                continue
            cursor = _aligned(cursor, alignment)
            entries.append(
                WeightBlobEntry(
                    key=("bucket", graph.bucket_frames, item.name),
                    name=item.name,
                    scope=item.scope,
                    bucket_frames=graph.bucket_frames,
                    offset=cursor,
                    byte_size=item.byte_size,
                )
            )
            bucket_offsets[(graph.bucket_frames, item.name)] = cursor
            cursor += item.byte_size

    tensor_offsets: dict[tuple[int | None, int], int] = {}
    for graph in bundle.graphs:
        for item in graph.initializers:
            if item.scope is InitializerScope.SHARED:
                offset = shared_offsets[item.name]
            else:
                offset = bucket_offsets[(graph.bucket_frames, item.name)]
            tensor_offsets[(graph.bucket_frames, item.tensor_id)] = offset

    return WeightBlobLayout(
        entries=tuple(entries),
        total_bytes=cursor,
        alignment=alignment,
        tensor_offsets=tensor_offsets,
    )


@dataclass(frozen=True, slots=True)
class TensorTable:
    """한 bucket의 plan에 실릴 Tensor descriptor 표."""

    bucket_frames: int | None
    descriptors: tuple[TensorDescriptor, ...]
    names: tuple[str, ...]
    ids: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        object.__setattr__(self, "names", tuple(self.names))
        if len(self.descriptors) != len(self.names):
            raise TensorTableError("descriptor 수와 이름 수가 다르다")
        ids = {name: index for index, name in enumerate(self.names)}
        if len(ids) != len(self.names):
            raise TensorTableError("Tensor 이름이 중복되었다")
        object.__setattr__(self, "ids", MappingProxyType(ids))

    def __len__(self) -> int:
        return len(self.descriptors)

    @property
    def byte_size(self) -> int:
        return len(self.descriptors) * TENSOR_DESCRIPTOR_SIZE

    def id_of(self, name: str) -> int:
        try:
            return self.ids[name]
        except KeyError as exc:
            raise TensorTableError(f"Tensor table에 없는 이름: {name!r}") from exc

    def name_of(self, tensor_id: int) -> str:
        """디버깅과 ORT 대조에만 쓴다. plan 자체에는 이름이 실리지 않는다."""

        if not 0 <= tensor_id < len(self.names):
            raise TensorTableError(f"Tensor ID 범위를 벗어남: {tensor_id}")
        return self.names[tensor_id]

    def pack(self) -> bytes:
        """표 전체를 plan에 실을 바이트열로 직렬화한다."""

        return b"".join(descriptor.pack() for descriptor in self.descriptors)


def _lifetime(tensor: RuntimeTensor) -> tuple[int, int]:
    """Tensor가 살아 있어야 하는 operator 구간을 구한다."""

    if tensor.producer is not None:
        first_use = tensor.producer
    elif tensor.consumers:
        first_use = tensor.consumers[0]
    else:
        first_use = INVALID_OPERATOR_INDEX

    if tensor.consumers:
        last_use = tensor.consumers[-1]
    elif tensor.producer is not None:
        last_use = tensor.producer
    else:
        last_use = INVALID_OPERATOR_INDEX

    return first_use, last_use


def _flags_for(
    storage_type: TensorStorageType, *, arena_managed: bool = False
) -> int:
    flags = TensorFlags.CONTIGUOUS
    if storage_type is TensorStorageType.CONSTANT:
        # weights.bin은 plan 밖에 있고 실행 중에 바뀌지 않는다.
        flags |= TensorFlags.READ_ONLY | TensorFlags.EXTERNAL
    if storage_type is TensorStorageType.VIEW:
        flags |= TensorFlags.ALIASED
    if arena_managed:
        flags |= TensorFlags.DENSE_SLAB
    return int(flags)


def _data_offset_for(
    tensor: RuntimeTensor,
    graph: RuntimeGraph,
    layout: WeightBlobLayout,
    arena_layout: TensorArenaLayout | None,
) -> int:
    if tensor.storage_type is TensorStorageType.CONSTANT:
        return layout.offset_of(graph.bucket_frames, tensor.tensor_id)
    if tensor.storage_type is TensorStorageType.VIEW:
        raise TensorTableError(
            f"Tensor {tensor.name!r}가 VIEW인데 alias 대상이 없다. "
            "RuntimeTensor가 alias_of를 들고 다니게 된 뒤에 켜야 한다"
        )
    if tensor.storage_type in (
        TensorStorageType.ACTIVATION,
        TensorStorageType.OUTPUT,
    ):
        return (
            INVALID_DATA_OFFSET
            if arena_layout is None
            else arena_layout.offset_of(tensor.tensor_id)
        )
    # INPUT은 외부 pointer를 연결하므로 offset이 없다.
    return INVALID_DATA_OFFSET


def build_tensor_table(
    graph: RuntimeGraph,
    layout: WeightBlobLayout,
    *,
    arena_layout: TensorArenaLayout | None = None,
) -> TensorTable:
    """RuntimeGraph 하나를 기존 80바이트 Tensor descriptor 표로 굳힌다.

    ``arena_layout``을 생략하면 Phase 3 Reference plan과 byte-compatible한 표를
    만든다. 명시하면 ACTIVATION/OUTPUT descriptor만 Arena offset과 실제 Arena
    수명으로 바뀐다. INPUT과 CONSTANT의 주소 규칙은 변하지 않는다.
    """

    if arena_layout is not None:
        arena_layout.validate_graph(graph)

    descriptors: list[TensorDescriptor] = []
    names: list[str] = []

    for tensor in graph.tensors:
        rank = len(tensor.shape)
        if rank > TENSOR_MAX_RANK:
            raise TensorTableError(
                f"Tensor {tensor.name!r}의 rank {rank}가 형식 한계를 넘는다"
            )
        padding = TENSOR_MAX_RANK - rank
        arena_managed = (
            arena_layout is not None
            and tensor.storage_type
            in (TensorStorageType.ACTIVATION, TensorStorageType.OUTPUT)
        )
        if arena_managed:
            allocation = arena_layout.allocation(tensor.tensor_id)
            first_use, last_use = allocation.first_use, allocation.last_use
        else:
            first_use, last_use = _lifetime(tensor)

        descriptors.append(
            TensorDescriptor(
                tensor_id=tensor.tensor_id,
                dtype=int(tensor.dtype),
                rank=rank,
                storage_type=int(tensor.storage_type),
                flags=_flags_for(
                    tensor.storage_type, arena_managed=arena_managed
                ),
                dimensions=tuple(tensor.shape) + (1,) * padding,
                byte_strides=tuple(tensor.strides) + (0,) * padding,
                data_offset=_data_offset_for(
                    tensor, graph, layout, arena_layout
                ),
                logical_byte_size=tensor.byte_size,
                storage_span_bytes=tensor.byte_size,
                alias_of_tensor_id=INVALID_TENSOR_ID,
                quantization_index=INVALID_QUANTIZATION_INDEX,
                first_use=first_use,
                last_use=last_use,
            )
        )
        names.append(tensor.name)

    return TensorTable(
        bucket_frames=graph.bucket_frames,
        descriptors=tuple(descriptors),
        names=tuple(names),
    )


__all__ = [
    "TensorTable",
    "TensorTableError",
    "WEIGHT_BLOB_ALIGNMENT",
    "WeightBlobEntry",
    "WeightBlobLayout",
    "build_tensor_table",
    "plan_weight_blob",
]
