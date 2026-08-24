"""RuntimeGraph의 Activation을 하나의 Tensor Arena에 정적으로 배치한다.

이 모듈은 메모리를 실제로 할당하지 않는다. 각 Tensor의 포괄 수명 구간
``[first_use, last_use]``와 크기를 이용해 ``tensor_id -> byte offset``을
결정한다. C Runtime은 plan에 저장된 offset을 ``arena_base``에 더하기만 한다.

수명 끝과 다음 수명 시작이 같은 Operator면 입력과 출력이 동시에 필요하므로
메모리를 공유하지 않는다. 오직 ``old.last_use < new.first_use``일 때만 이전
영역을 재사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from ..format.binary_format_schema import (
    INVALID_OPERATOR_INDEX,
    TensorDescriptor,
    TensorStorageType,
    UINT64_MAX,
)
from ..runtime_ir import RuntimeGraph, RuntimeTensor


DEFAULT_ARENA_ALIGNMENT = 64
_ARENA_STORAGE_TYPES = frozenset(
    (TensorStorageType.ACTIVATION, TensorStorageType.OUTPUT)
)


class TensorArenaPlanningError(ValueError):
    """Tensor 수명이나 Arena 배치가 안전하지 않을 때 발생한다."""


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TensorArenaPlanningError(f"{name}은 양의 정수여야 한다: {value!r}")


def _aligned(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def _ranges_overlap(
    left_offset: int, left_size: int, right_offset: int, right_size: int
) -> bool:
    return left_offset < right_offset + right_size and right_offset < left_offset + left_size


def _lifetimes_overlap(
    left_first: int, left_last: int, right_first: int, right_last: int
) -> bool:
    return left_first <= right_last and right_first <= left_last


def _arena_lifetime(
    tensor: RuntimeTensor, operator_count: int
) -> tuple[int, int]:
    if tensor.producer is None:
        raise TensorArenaPlanningError(
            f"Arena Tensor {tensor.tensor_id} {tensor.name!r}에 producer가 없다"
        )
    first_use = tensor.producer
    last_use = tensor.consumers[-1] if tensor.consumers else tensor.producer
    if tensor.storage_type is TensorStorageType.OUTPUT:
        # 호출자가 graph 실행 후 output을 읽을 때까지 덮어쓰지 않는다.
        last_use = operator_count - 1
    if first_use > last_use:
        raise TensorArenaPlanningError(
            f"Tensor {tensor.tensor_id} 수명이 뒤집혔다: {first_use}>{last_use}"
        )
    return first_use, last_use


def _view_lifetime(tensor: RuntimeTensor) -> tuple[int, int]:
    uses: list[int] = []
    if tensor.producer is not None:
        uses.append(tensor.producer)
    uses.extend(tensor.consumers)
    if not uses:
        raise TensorArenaPlanningError(
            f"VIEW Tensor {tensor.tensor_id} {tensor.name!r} has no execution use"
        )
    return min(uses), max(uses)


def _arena_lifetimes(graph: RuntimeGraph) -> dict[int, tuple[int, int]]:
    lifetimes = {
        tensor.tensor_id: _arena_lifetime(tensor, len(graph.operators))
        for tensor in graph.tensors
        if tensor.storage_type in _ARENA_STORAGE_TYPES
    }
    for tensor in graph.tensors:
        if tensor.storage_type is not TensorStorageType.VIEW:
            continue
        assert tensor.alias_of_tensor_id is not None
        base_tensor = graph.tensor(tensor.alias_of_tensor_id)
        if base_tensor.storage_type is TensorStorageType.INPUT:
            # External input remains live for the whole inference.
            continue
        try:
            base_first, base_last = lifetimes[tensor.alias_of_tensor_id]
        except KeyError as exc:
            raise TensorArenaPlanningError(
                f"VIEW Tensor {tensor.tensor_id} aliases non-arena Tensor "
                f"{tensor.alias_of_tensor_id}"
            ) from exc
        view_first, view_last = _view_lifetime(tensor)
        lifetimes[tensor.alias_of_tensor_id] = (
            min(base_first, view_first),
            max(base_last, view_last),
        )
    return lifetimes


@dataclass(frozen=True, slots=True)
class TensorArenaAllocation:
    """Arena 안에서 Tensor 하나가 차지하는 고정 구간과 실행 수명."""

    tensor_id: int
    offset: int
    byte_size: int
    reserved_bytes: int
    first_use: int
    last_use: int

    @property
    def end_offset(self) -> int:
        return self.offset + self.reserved_bytes


@dataclass(frozen=True, slots=True)
class TensorArenaLayout:
    """한 bucket의 결정적인 Tensor Arena 배치 결과."""

    bucket_frames: int | None
    alignment: int
    arena_size: int
    naive_activation_bytes: int
    naive_aligned_bytes: int
    theoretical_peak_bytes: int
    allocations: tuple[TensorArenaAllocation, ...]
    _by_tensor_id: Mapping[int, TensorArenaAllocation] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "allocations", tuple(self.allocations))
        by_tensor_id = {item.tensor_id: item for item in self.allocations}
        if len(by_tensor_id) != len(self.allocations):
            raise TensorArenaPlanningError("Arena layout에 Tensor ID가 중복되었다")
        object.__setattr__(
            self, "_by_tensor_id", MappingProxyType(by_tensor_id)
        )
        self.validate()

    @property
    def tensor_count(self) -> int:
        return len(self.allocations)

    @property
    def saved_bytes(self) -> int:
        return self.naive_aligned_bytes - self.arena_size

    @property
    def packing_efficiency(self) -> float:
        if self.arena_size == 0:
            return 1.0
        return self.theoretical_peak_bytes / self.arena_size

    def allocation(self, tensor_id: int) -> TensorArenaAllocation:
        try:
            return self._by_tensor_id[tensor_id]
        except KeyError as exc:
            raise TensorArenaPlanningError(
                f"Tensor {tensor_id}는 Arena 할당 대상이 아니다"
            ) from exc

    def offset_of(self, tensor_id: int) -> int:
        return self.allocation(tensor_id).offset

    def validate(self) -> None:
        _require_positive_integer("alignment", self.alignment)
        if self.alignment & (self.alignment - 1):
            raise TensorArenaPlanningError("alignment는 2의 거듭제곱이어야 한다")
        for name, value in (
            ("arena_size", self.arena_size),
            ("naive_activation_bytes", self.naive_activation_bytes),
            ("naive_aligned_bytes", self.naive_aligned_bytes),
            ("theoretical_peak_bytes", self.theoretical_peak_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TensorArenaPlanningError(f"{name}이 유효하지 않다: {value!r}")
            if value > UINT64_MAX:
                raise TensorArenaPlanningError(f"{name}이 uint64 범위를 넘는다")

        maximum_end = 0
        ordered = sorted(
            self.allocations, key=lambda item: (item.first_use, item.tensor_id)
        )
        active: list[TensorArenaAllocation] = []
        for item in ordered:
            for name, value in (
                ("tensor_id", item.tensor_id),
                ("offset", item.offset),
                ("byte_size", item.byte_size),
                ("reserved_bytes", item.reserved_bytes),
                ("first_use", item.first_use),
                ("last_use", item.last_use),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise TensorArenaPlanningError(
                        f"Tensor {item.tensor_id}의 {name}이 유효하지 않다: {value!r}"
                    )
            if item.byte_size == 0 or item.reserved_bytes < item.byte_size:
                raise TensorArenaPlanningError(
                    f"Tensor {item.tensor_id}의 Arena 크기가 유효하지 않다"
                )
            if item.offset % self.alignment != 0:
                raise TensorArenaPlanningError(
                    f"Tensor {item.tensor_id} offset이 {self.alignment}바이트 정렬이 아니다"
                )
            if item.reserved_bytes != _aligned(item.byte_size, self.alignment):
                raise TensorArenaPlanningError(
                    f"Tensor {item.tensor_id} reserved_bytes가 정렬 크기와 다르다"
                )
            if item.first_use > item.last_use:
                raise TensorArenaPlanningError(
                    f"Tensor {item.tensor_id}의 수명이 뒤집혔다"
                )
            if item.end_offset > self.arena_size:
                raise TensorArenaPlanningError(
                    f"Tensor {item.tensor_id}가 Arena 범위를 벗어난다"
                )

            active = [prior for prior in active if prior.last_use >= item.first_use]
            for prior in active:
                if _ranges_overlap(
                    prior.offset,
                    prior.reserved_bytes,
                    item.offset,
                    item.reserved_bytes,
                ):
                    raise TensorArenaPlanningError(
                        f"수명이 겹치는 Tensor {prior.tensor_id}와 "
                        f"{item.tensor_id}의 Arena 구간이 겹친다"
                    )
            active.append(item)
            maximum_end = max(maximum_end, item.end_offset)

        if maximum_end != self.arena_size:
            raise TensorArenaPlanningError(
                f"arena_size가 실제 최대 offset과 다르다: "
                f"{self.arena_size}!={maximum_end}"
            )
        if self.theoretical_peak_bytes > self.arena_size:
            raise TensorArenaPlanningError("이론적 peak가 실제 Arena보다 크다")
        if self.arena_size > self.naive_aligned_bytes:
            raise TensorArenaPlanningError("Arena가 독립 정렬 buffer 합계보다 크다")

    def validate_graph(self, graph: RuntimeGraph) -> None:
        if self.bucket_frames != graph.bucket_frames:
            raise TensorArenaPlanningError(
                f"Arena bucket {self.bucket_frames}가 graph bucket "
                f"{graph.bucket_frames}와 다르다"
            )
        expected = {
            tensor.tensor_id: tensor
            for tensor in graph.tensors
            if tensor.storage_type in _ARENA_STORAGE_TYPES
        }
        if set(expected) != set(self._by_tensor_id):
            missing = sorted(set(expected) - set(self._by_tensor_id))
            extra = sorted(set(self._by_tensor_id) - set(expected))
            raise TensorArenaPlanningError(
                f"Arena 대상 Tensor가 graph와 다르다: missing={missing}, extra={extra}"
            )
        lifetimes = _arena_lifetimes(graph)
        for tensor_id, tensor in expected.items():
            allocation = self._by_tensor_id[tensor_id]
            first_use, last_use = lifetimes[tensor_id]
            if (
                allocation.byte_size != tensor.storage_span_bytes
                or allocation.first_use != first_use
                or allocation.last_use != last_use
            ):
                raise TensorArenaPlanningError(
                    f"Tensor {tensor_id}의 Arena 수명 또는 크기가 graph와 다르다"
                )


@dataclass(frozen=True, slots=True)
class _ArenaRequest:
    tensor_id: int
    byte_size: int
    reserved_bytes: int
    first_use: int
    last_use: int


def _theoretical_peak(requests: tuple[_ArenaRequest, ...]) -> int:
    events: dict[int, int] = {}
    for item in requests:
        events[item.first_use] = events.get(item.first_use, 0) + item.reserved_bytes
        events[item.last_use + 1] = (
            events.get(item.last_use + 1, 0) - item.reserved_bytes
        )
    live = 0
    peak = 0
    for operator_id in sorted(events):
        live += events[operator_id]
        peak = max(peak, live)
    return peak


def _build_layout(
    requests: tuple[_ArenaRequest, ...],
    *,
    bucket_frames: int | None,
    alignment: int,
    max_arena_bytes: int | None = None,
) -> TensorArenaLayout:
    if not requests:
        raise TensorArenaPlanningError("graph에 Arena 할당 대상 Tensor가 없다")

    # 그래프 전체를 미리 아는 offline 배치다. 큰 Tensor부터 배치하면 작은
    # Tensor가 큰 연속 구간을 잘게 쪼개는 현상을 줄일 수 있다. 각 Tensor는
    # 수명이 겹치는 이미 배치된 Tensor들의 메모리 구간만 피해서 가장 낮은
    # 정렬 offset에 둔다. 동일 입력에서는 순서와 결과가 항상 같다.
    allocations: list[TensorArenaAllocation] = []
    placement_order = sorted(
        requests,
        key=lambda item: (
            -item.reserved_bytes,
            item.first_use,
            item.last_use,
            item.tensor_id,
        ),
    )
    for request in placement_order:
        occupied = sorted(
            (item.offset, item.end_offset)
            for item in allocations
            if _lifetimes_overlap(
                request.first_use,
                request.last_use,
                item.first_use,
                item.last_use,
            )
        )
        offset = 0
        for occupied_start, occupied_end in occupied:
            if offset + request.reserved_bytes <= occupied_start:
                break
            offset = max(offset, occupied_end)
        allocation = TensorArenaAllocation(
            tensor_id=request.tensor_id,
            offset=offset,
            byte_size=request.byte_size,
            reserved_bytes=request.reserved_bytes,
            first_use=request.first_use,
            last_use=request.last_use,
        )
        allocations.append(allocation)

    arena_size = max(item.end_offset for item in allocations)
    if max_arena_bytes is not None and arena_size > max_arena_bytes:
        raise TensorArenaPlanningError(
            f"계산된 Arena {arena_size}바이트가 예산 "
            f"{max_arena_bytes}바이트를 넘는다"
        )
    layout = TensorArenaLayout(
        bucket_frames=bucket_frames,
        alignment=alignment,
        arena_size=arena_size,
        naive_activation_bytes=sum(item.byte_size for item in requests),
        naive_aligned_bytes=sum(item.reserved_bytes for item in requests),
        theoretical_peak_bytes=_theoretical_peak(requests),
        allocations=tuple(sorted(allocations, key=lambda item: item.tensor_id)),
    )
    return layout


def plan_tensor_arena(
    graph: RuntimeGraph,
    *,
    alignment: int = DEFAULT_ARENA_ALIGNMENT,
    max_arena_bytes: int | None = None,
) -> TensorArenaLayout:
    """한 graph의 ACTIVATION/OUTPUT offset을 deterministic offline 배치한다."""

    _require_positive_integer("alignment", alignment)
    if alignment & (alignment - 1):
        raise TensorArenaPlanningError("alignment는 2의 거듭제곱이어야 한다")
    if max_arena_bytes is not None:
        _require_positive_integer("max_arena_bytes", max_arena_bytes)
    lifetimes = _arena_lifetimes(graph)
    requests_list: list[_ArenaRequest] = []
    for tensor in graph.tensors:
        if tensor.storage_type not in _ARENA_STORAGE_TYPES:
            continue
        first_use, last_use = lifetimes[tensor.tensor_id]
        assert tensor.storage_span_bytes is not None
        requests_list.append(
            _ArenaRequest(
                tensor_id=tensor.tensor_id,
                byte_size=tensor.storage_span_bytes,
                reserved_bytes=_aligned(tensor.storage_span_bytes, alignment),
                first_use=first_use,
                last_use=last_use,
            )
        )
    layout = _build_layout(
        tuple(requests_list),
        bucket_frames=graph.bucket_frames,
        alignment=alignment,
        max_arena_bytes=max_arena_bytes,
    )
    layout.validate_graph(graph)
    return layout


def plan_tensor_arena_from_descriptors(
    descriptors: Sequence[TensorDescriptor],
    *,
    bucket_frames: int | None,
    operator_count: int,
    alignment: int = DEFAULT_ARENA_ALIGNMENT,
    max_arena_bytes: int | None = None,
) -> TensorArenaLayout:
    """기존 Reference plan의 descriptor만으로 같은 Arena 배치를 계산한다.

    ONNX가 없는 C loader 검증 환경에서도 plan에 이미 기록된 수명과 크기를
    이용해 Arena 요구량을 검사할 수 있다. Exporter의 정식 생성 경로는
    :func:`plan_tensor_arena`이며, 이 함수는 분석과 교차 검증용이다.
    """

    _require_positive_integer("operator_count", operator_count)
    _require_positive_integer("alignment", alignment)
    if alignment & (alignment - 1):
        raise TensorArenaPlanningError("alignment는 2의 거듭제곱이어야 한다")
    if max_arena_bytes is not None:
        _require_positive_integer("max_arena_bytes", max_arena_bytes)

    descriptor_tuple = tuple(descriptors)
    descriptors_by_id = {
        descriptor.tensor_id: descriptor for descriptor in descriptor_tuple
    }
    requests_by_id: dict[int, _ArenaRequest] = {}
    for descriptor in descriptor_tuple:
        storage_type = TensorStorageType(descriptor.storage_type)
        if storage_type not in _ARENA_STORAGE_TYPES:
            continue
        if (
            descriptor.first_use == INVALID_OPERATOR_INDEX
            or descriptor.last_use == INVALID_OPERATOR_INDEX
            or descriptor.first_use >= operator_count
            or descriptor.last_use >= operator_count
        ):
            raise TensorArenaPlanningError(
                f"Tensor {descriptor.tensor_id}의 descriptor 수명이 유효하지 않다"
            )
        last_use = descriptor.last_use
        if storage_type is TensorStorageType.OUTPUT:
            last_use = operator_count - 1
        byte_size = descriptor.storage_span_bytes
        requests_by_id[descriptor.tensor_id] = _ArenaRequest(
            tensor_id=descriptor.tensor_id,
            byte_size=byte_size,
            reserved_bytes=_aligned(byte_size, alignment),
            first_use=descriptor.first_use,
            last_use=last_use,
        )
    for descriptor in descriptor_tuple:
        storage_type = TensorStorageType(descriptor.storage_type)
        if storage_type is not TensorStorageType.VIEW:
            continue
        base_descriptor = descriptors_by_id.get(descriptor.alias_of_tensor_id)
        if (
            base_descriptor is not None
            and TensorStorageType(base_descriptor.storage_type)
            is TensorStorageType.INPUT
        ):
            continue
        if (
            descriptor.first_use == INVALID_OPERATOR_INDEX
            or descriptor.last_use == INVALID_OPERATOR_INDEX
            or descriptor.first_use >= operator_count
            or descriptor.last_use >= operator_count
        ):
            raise TensorArenaPlanningError(
                f"VIEW Tensor {descriptor.tensor_id} has invalid descriptor lifetime"
            )
        try:
            base = requests_by_id[descriptor.alias_of_tensor_id]
        except KeyError as exc:
            raise TensorArenaPlanningError(
                f"VIEW Tensor {descriptor.tensor_id} aliases non-arena Tensor "
                f"{descriptor.alias_of_tensor_id}"
            ) from exc
        requests_by_id[base.tensor_id] = _ArenaRequest(
            tensor_id=base.tensor_id,
            byte_size=base.byte_size,
            reserved_bytes=base.reserved_bytes,
            first_use=min(base.first_use, descriptor.first_use),
            last_use=max(base.last_use, descriptor.last_use),
        )
    return _build_layout(
        tuple(requests_by_id.values()),
        bucket_frames=bucket_frames,
        alignment=alignment,
        max_arena_bytes=max_arena_bytes,
    )


__all__ = [
    "DEFAULT_ARENA_ALIGNMENT",
    "TensorArenaAllocation",
    "TensorArenaLayout",
    "TensorArenaPlanningError",
    "plan_tensor_arena",
    "plan_tensor_arena_from_descriptors",
]
