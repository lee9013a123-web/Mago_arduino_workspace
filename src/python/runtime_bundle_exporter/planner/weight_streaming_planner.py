"""Build a page-granular, zero-copy weight schedule for one bucket.

The output weight file is ordered by first operator use.  Constants needed by
the same operator stay in the same page-aligned block, so the runtime can map
the file read-only, prefetch the next block, and discard completed blocks
without copying or rebasing kernel pointers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import struct
from typing import Any, Mapping, Sequence

from ..format.binary_format_schema import TensorStorageType
from ..writer.execution_plan_writer import LoadedPlan, build_loaded_plan_bytes
from .weight_residency_planner import (
    DEFAULT_CACHE_ALIGNMENT,
    DEFAULT_SCALAR_ALIGNMENT,
    StaticWeightEntry,
    WeightResidencyPlanError,
    build_static_weight_plan,
)


WEIGHT_SCHEDULE_MAGIC = b"CAMPPWS1"
WEIGHT_SCHEDULE_VERSION = 2
WEIGHT_SCHEDULE_HEADER_SIZE = 64
WEIGHT_SCHEDULE_RECORD_SIZE = 40
WEIGHT_SCHEDULE_USED = 1
WEIGHT_SCHEDULE_PREFETCH_AFTER_OPERATOR = 2


def _align(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise WeightResidencyPlanError(
            f"alignment must be a positive power of two: {alignment}"
        )
    return (value + alignment - 1) & ~(alignment - 1)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class WeightStreamingBlock:
    block_id: int
    file_offset: int
    byte_size: int
    logical_bytes: int
    first_operator: int
    last_operator: int
    prefetch_operator: int
    prefetch_after_operator: bool
    tensor_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "file_offset": self.file_offset,
            "byte_size": self.byte_size,
            "logical_bytes": self.logical_bytes,
            "first_operator": self.first_operator,
            "last_operator": self.last_operator,
            "prefetch_operator": self.prefetch_operator,
            "prefetch_phase": (
                "after_operator" if self.prefetch_after_operator
                else "before_operator"
            ),
            "tensor_ids": list(self.tensor_ids),
        }


def _scheduled_window_peak(
    blocks: Sequence[WeightStreamingBlock], operator_count: int,
) -> dict[str, Any]:
    """Simulate runtime event order and return the advised-window peak."""

    by_id = {block.block_id: block for block in blocks}
    resident: set[int] = {
        block.block_id for block in blocks if block.first_operator == 0
    }
    peak_bytes = 0
    peak_ids: tuple[int, ...] = ()
    peak_operator = 0
    peak_phase = "initial"

    def observe(operator_id: int, phase: str) -> None:
        nonlocal peak_bytes, peak_ids, peak_operator, peak_phase
        current = sum(by_id[block_id].byte_size for block_id in resident)
        if current > peak_bytes:
            peak_bytes = current
            peak_ids = tuple(sorted(resident))
            peak_operator = operator_id
            peak_phase = phase

    observe(0, "initial")
    for operator_id in range(operator_count):
        for block in blocks:
            if (
                not block.prefetch_after_operator
                and block.prefetch_operator == operator_id
            ):
                resident.add(block.block_id)
        observe(operator_id, "before_operator")
        missing = [
            block.block_id for block in blocks
            if block.first_operator == operator_id
            and block.block_id not in resident
        ]
        if missing:
            raise WeightResidencyPlanError(
                f"blocks are not prefetched before first use: {missing}"
            )
        for block in blocks:
            if block.last_operator == operator_id:
                resident.discard(block.block_id)
        for block in blocks:
            if (
                block.prefetch_after_operator
                and block.prefetch_operator == operator_id
            ):
                resident.add(block.block_id)
        observe(operator_id, "after_operator")

    return {
        "byte_size": peak_bytes,
        "block_count": len(peak_ids),
        "block_ids": list(peak_ids),
        "operator_id": peak_operator,
        "phase": peak_phase,
    }


@dataclass(frozen=True, slots=True)
class WindowedWeightPlan:
    bucket_frames: int
    operator_count: int
    page_size: int
    target_block_bytes: int
    source_plan_sha256: str
    source_weights_sha256: str
    plan_bytes: bytes
    weight_bytes: bytes
    schedule_bytes: bytes
    entries: tuple[StaticWeightEntry, ...]
    blocks: tuple[WeightStreamingBlock, ...]

    @property
    def plan_sha256(self) -> str:
        return _sha256(self.plan_bytes)

    @property
    def weights_sha256(self) -> str:
        return _sha256(self.weight_bytes)

    @property
    def schedule_sha256(self) -> str:
        return _sha256(self.schedule_bytes)

    @property
    def logical_weight_bytes(self) -> int:
        return sum(entry.byte_size for entry in self.entries)

    @property
    def max_block_bytes(self) -> int:
        return max((block.byte_size for block in self.blocks), default=0)

    @property
    def largest_two_blocks_bytes(self) -> int:
        sizes = sorted(
            (block.byte_size for block in self.blocks), reverse=True
        )
        return sum(sizes[:2])

    @property
    def scheduled_window_peak(self) -> dict[str, Any]:
        return _scheduled_window_peak(self.blocks, self.operator_count)

    @property
    def double_buffer_bound_bytes(self) -> int:
        """Compatibility field; now reflects the actual event simulation."""

        return int(self.scheduled_window_peak["byte_size"])

    @property
    def theoretical_weight_rss_reduction_bytes(self) -> int:
        return max(
            0, self.logical_weight_bytes - self.double_buffer_bound_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "bucket_frames": self.bucket_frames,
            "strategy": "page_aligned_mmap_window",
            "runtime_contract": {
                "read_only_mmap": True,
                "userspace_weight_copies_per_inference": 0,
                "prefetch": "POSIX_FADV_WILLNEED",
                "release": "MADV_DONTNEED",
                "fallback": "full_resident_v3",
            },
            "page_size": self.page_size,
            "target_block_bytes": self.target_block_bytes,
            "operator_count": self.operator_count,
            "constant_count": len(self.entries),
            "block_count": len(self.blocks),
            "logical_weight_bytes": self.logical_weight_bytes,
            "output_weight_bytes": len(self.weight_bytes),
            "max_block_bytes": self.max_block_bytes,
            "largest_two_blocks_bytes": self.largest_two_blocks_bytes,
            "double_buffer_bound_bytes": self.double_buffer_bound_bytes,
            "scheduled_window_peak": self.scheduled_window_peak,
            "theoretical_weight_rss_reduction_bytes": (
                self.theoretical_weight_rss_reduction_bytes
            ),
            "source": {
                "plan_sha256": self.source_plan_sha256,
                "weights_sha256": self.source_weights_sha256,
            },
            "output": {
                "plan_sha256": self.plan_sha256,
                "weights_sha256": self.weights_sha256,
                "schedule_sha256": self.schedule_sha256,
            },
            "blocks": [block.to_dict() for block in self.blocks],
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _build_schedule_bytes(
    *,
    bucket_frames: int,
    page_size: int,
    operator_count: int,
    weights_size: int,
    blocks: Sequence[WeightStreamingBlock],
) -> bytes:
    records = bytearray()
    for block in blocks:
        records.extend(struct.pack(
            "<6I2Q",
            block.block_id,
            block.first_operator,
            block.last_operator,
            block.prefetch_operator,
            WEIGHT_SCHEDULE_USED | (
                WEIGHT_SCHEDULE_PREFETCH_AFTER_OPERATOR
                if block.prefetch_after_operator else 0
            ),
            0,
            block.file_offset,
            block.byte_size,
        ))
    if len(records) != len(blocks) * WEIGHT_SCHEDULE_RECORD_SIZE:
        raise AssertionError("weight schedule record size drift")
    header = struct.pack(
        "<8s6I4Q",
        WEIGHT_SCHEDULE_MAGIC,
        WEIGHT_SCHEDULE_VERSION,
        WEIGHT_SCHEDULE_HEADER_SIZE,
        bucket_frames,
        page_size,
        len(blocks),
        operator_count,
        weights_size,
        WEIGHT_SCHEDULE_HEADER_SIZE,
        len(records),
        0,
    )
    if len(header) != WEIGHT_SCHEDULE_HEADER_SIZE:
        raise AssertionError("weight schedule header size drift")
    return header + bytes(records)


def build_windowed_weight_plan(
    loaded: LoadedPlan,
    source_plan_bytes: bytes,
    source_weights: bytes,
    *,
    weight_index: Sequence[Mapping[str, Any]] | None = None,
    page_size: int = 4096,
    target_block_bytes: int = 512 * 1024,
    cache_alignment: int = DEFAULT_CACHE_ALIGNMENT,
    scalar_alignment: int = DEFAULT_SCALAR_ALIGNMENT,
) -> WindowedWeightPlan:
    """Build one bucket's direct-mapped weights and operator schedule."""

    _align(0, page_size)
    if target_block_bytes < page_size:
        raise WeightResidencyPlanError(
            "target block size must be at least one page"
        )
    if target_block_bytes % page_size != 0:
        raise WeightResidencyPlanError(
            "target block size must be page aligned"
        )

    ordered = build_static_weight_plan(
        loaded,
        source_plan_bytes,
        source_weights,
        weight_index=weight_index,
        cache_alignment=cache_alignment,
        scalar_alignment=scalar_alignment,
    )
    used = [entry for entry in ordered.entries if entry.used]
    unused = [entry for entry in ordered.entries if not entry.used]
    if not used:
        raise WeightResidencyPlanError("plan has no used constant tensors")

    groups: list[list[StaticWeightEntry]] = []
    for entry in used:
        if (
            not groups
            or groups[-1][0].first_use_operator != entry.first_use_operator
        ):
            groups.append([])
        groups[-1].append(entry)

    block_groups: list[list[StaticWeightEntry]] = []
    current: list[StaticWeightEntry] = []
    current_estimate = 0
    for group in groups:
        group_estimate = sum(
            _align(entry.byte_size, entry.alignment) for entry in group
        )
        if current and current_estimate + group_estimate > target_block_bytes:
            block_groups.append(current)
            current = []
            current_estimate = 0
        current.extend(group)
        current_estimate += group_estimate
    if current:
        block_groups.append(current)

    output = bytearray()
    descriptors = list(loaded.tensors)
    remapped_entries: list[StaticWeightEntry] = []
    blocks: list[WeightStreamingBlock] = []
    for block_id, entries in enumerate(block_groups):
        block_offset = _align(len(output), page_size)
        output.extend(b"\0" * (block_offset - len(output)))
        tensor_ids: list[int] = []
        logical_bytes = 0
        for entry in entries:
            destination = _align(len(output), entry.alignment)
            output.extend(b"\0" * (destination - len(output)))
            payload = source_weights[
                entry.source_offset:entry.source_offset + entry.byte_size
            ]
            output.extend(payload)
            descriptor = descriptors[entry.tensor_id]
            if descriptor.storage_type != TensorStorageType.CONSTANT:
                raise WeightResidencyPlanError(
                    f"Tensor {entry.tensor_id} is not constant"
                )
            descriptors[entry.tensor_id] = replace(
                descriptor, data_offset=destination
            )
            remapped_entries.append(replace(
                entry, destination_offset=destination
            ))
            tensor_ids.append(entry.tensor_id)
            logical_bytes += entry.byte_size
        block_end = _align(len(output), page_size)
        output.extend(b"\0" * (block_end - len(output)))
        first_operator = min(
            int(entry.first_use_operator) for entry in entries
            if entry.first_use_operator is not None
        )
        last_operator = max(
            int(entry.last_use_operator) for entry in entries
            if entry.last_use_operator is not None
        )
        if block_id == 0:
            prefetch_operator = 0
            prefetch_after_operator = False
        elif block_id == 1:
            prefetch_operator = 0
            prefetch_after_operator = first_operator > 0
        else:
            oldest_release = blocks[block_id - 2].last_operator
            if oldest_release < first_operator:
                prefetch_operator = oldest_release
                prefetch_after_operator = True
            else:
                # Three truly overlapping lifetimes cannot be reduced by
                # scheduling.  Prefetch immediately before first use.
                prefetch_operator = first_operator
                prefetch_after_operator = False
        blocks.append(WeightStreamingBlock(
            block_id=block_id,
            file_offset=block_offset,
            byte_size=block_end - block_offset,
            logical_bytes=logical_bytes,
            first_operator=first_operator,
            last_operator=last_operator,
            prefetch_operator=prefetch_operator,
            prefetch_after_operator=prefetch_after_operator,
            tensor_ids=tuple(tensor_ids),
        ))

    # Unused constants remain valid for loader validation but never fault into
    # the process during normal inference and therefore need no schedule block.
    for entry in unused:
        destination = _align(len(output), entry.alignment)
        output.extend(b"\0" * (destination - len(output)))
        payload = source_weights[
            entry.source_offset:entry.source_offset + entry.byte_size
        ]
        output.extend(payload)
        descriptors[entry.tensor_id] = replace(
            descriptors[entry.tensor_id], data_offset=destination
        )
        remapped_entries.append(replace(
            entry, destination_offset=destination
        ))

    remapped = replace(loaded, tensors=tuple(descriptors))
    plan_bytes = build_loaded_plan_bytes(remapped)
    schedule_bytes = _build_schedule_bytes(
        bucket_frames=loaded.header.bucket_frames,
        page_size=page_size,
        operator_count=len(loaded.operators),
        weights_size=len(output),
        blocks=blocks,
    )
    return WindowedWeightPlan(
        bucket_frames=loaded.header.bucket_frames,
        operator_count=len(loaded.operators),
        page_size=page_size,
        target_block_bytes=target_block_bytes,
        source_plan_sha256=_sha256(source_plan_bytes),
        source_weights_sha256=_sha256(source_weights),
        plan_bytes=plan_bytes,
        weight_bytes=bytes(output),
        schedule_bytes=schedule_bytes,
        entries=tuple(remapped_entries),
        blocks=tuple(blocks),
    )


__all__ = [
    "WEIGHT_SCHEDULE_HEADER_SIZE",
    "WEIGHT_SCHEDULE_MAGIC",
    "WEIGHT_SCHEDULE_RECORD_SIZE",
    "WEIGHT_SCHEDULE_VERSION",
    "WeightStreamingBlock",
    "WindowedWeightPlan",
    "build_windowed_weight_plan",
]
