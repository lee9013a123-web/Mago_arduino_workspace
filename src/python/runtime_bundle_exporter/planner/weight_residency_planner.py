"""Build bucket-local static weight layouts from compiled execution plans.

The planner runs offline.  It follows each operator's input Tensor IDs, places
constant tensors in first-use order, and rewrites only the decoded constant
``data_offset`` values.  Runtime kernels still receive direct pointers into one
immutable blob; no weight copy, eviction, or relocation occurs during
inference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..format.binary_format_schema import TensorFlags, TensorStorageType
from ..writer.execution_plan_writer import LoadedPlan, build_loaded_plan_bytes


DEFAULT_CACHE_ALIGNMENT = 64
DEFAULT_SCALAR_ALIGNMENT = 8


class WeightResidencyPlanError(ValueError):
    """A static weight layout cannot be produced without changing semantics."""


def _aligned(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise WeightResidencyPlanError(
            f"alignment must be a positive power of two: {alignment}"
        )
    return (value + alignment - 1) & ~(alignment - 1)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class StaticWeightEntry:
    tensor_id: int
    name: str | None
    dtype: str | None
    shape: tuple[int, ...]
    source_offset: int
    destination_offset: int
    byte_size: int
    alignment: int
    first_use_operator: int | None
    last_use_operator: int | None
    operator_ids: tuple[int, ...]
    sha256: str

    @property
    def used(self) -> bool:
        return bool(self.operator_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "source_offset": self.source_offset,
            "destination_offset": self.destination_offset,
            "byte_size": self.byte_size,
            "alignment": self.alignment,
            "used": self.used,
            "first_use_operator": self.first_use_operator,
            "last_use_operator": self.last_use_operator,
            "operator_ids": list(self.operator_ids),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class StaticWeightPlan:
    bucket_frames: int
    source_plan_sha256: str
    source_weights_sha256: str
    source_weights_bytes: int
    plan_bytes: bytes
    weight_bytes: bytes
    entries: tuple[StaticWeightEntry, ...]
    output_padding_bytes: int
    cache_alignment: int
    scalar_alignment: int

    @property
    def output_plan_sha256(self) -> str:
        return _sha256(self.plan_bytes)

    @property
    def output_weights_sha256(self) -> str:
        return _sha256(self.weight_bytes)

    @property
    def used_constant_count(self) -> int:
        return sum(entry.used for entry in self.entries)

    @property
    def logical_weight_bytes(self) -> int:
        return sum(entry.byte_size for entry in self.entries)

    @property
    def saved_bytes(self) -> int:
        return self.source_weights_bytes - len(self.weight_bytes)

    @property
    def saved_pct(self) -> float:
        return (
            self.saved_bytes / self.source_weights_bytes * 100.0
            if self.source_weights_bytes else 0.0
        )

    def to_dict(self, *, include_entries: bool = True) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": 1,
            "bucket_frames": self.bucket_frames,
            "strategy": "offline_first_use_static_blob",
            "runtime_contract": {
                "direct_base_plus_offset": True,
                "inference_weight_moves": 0,
                "inference_weight_allocations": 0,
                "descriptor_fields_changed": ["constant.data_offset"],
            },
            "alignment": {
                "cache_bytes": self.cache_alignment,
                "scalar_bytes": self.scalar_alignment,
                "rule": "packed_or_at_least_cache_line_else_scalar",
            },
            "source": {
                "plan_sha256": self.source_plan_sha256,
                "weights_sha256": self.source_weights_sha256,
                "weights_bytes": self.source_weights_bytes,
            },
            "output": {
                "plan_sha256": self.output_plan_sha256,
                "weights_sha256": self.output_weights_sha256,
                "weights_bytes": len(self.weight_bytes),
                "logical_weight_bytes": self.logical_weight_bytes,
                "padding_bytes": self.output_padding_bytes,
                "saved_bytes": self.saved_bytes,
                "saved_pct": self.saved_pct,
            },
            "constant_count": len(self.entries),
            "used_constant_count": self.used_constant_count,
            "unused_constant_count": len(self.entries) - self.used_constant_count,
        }
        if include_entries:
            document["entries"] = [entry.to_dict() for entry in self.entries]
        return document


def _weight_index_by_range(
    records: Sequence[Mapping[str, Any]] | None,
) -> Mapping[tuple[int, int], Mapping[str, Any]]:
    by_range: dict[tuple[int, int], Mapping[str, Any]] = {}
    for record in records or ():
        try:
            key = (int(record["offset"]), int(record["byte_size"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise WeightResidencyPlanError("invalid source weight index") from exc
        if key in by_range:
            raise WeightResidencyPlanError(
                f"duplicate source weight range: {key}"
            )
        by_range[key] = record
    return MappingProxyType(by_range)


def _operator_references(loaded: LoadedPlan) -> dict[int, tuple[int, ...]]:
    references: dict[int, list[int]] = {}
    for operator in loaded.operators:
        for tensor_id in operator.input_tensor_ids[: operator.input_count]:
            if tensor_id >= len(loaded.tensors):
                raise WeightResidencyPlanError(
                    f"operator {operator.operator_id} input Tensor ID is invalid: "
                    f"{tensor_id}"
                )
            tensor = loaded.tensors[tensor_id]
            if tensor.storage_type == TensorStorageType.CONSTANT:
                references.setdefault(tensor_id, []).append(operator.operator_id)
    return {tensor_id: tuple(ids) for tensor_id, ids in references.items()}


def _alignment_for(descriptor: Any, cache: int, scalar: int) -> int:
    packed = bool(int(descriptor.flags) & int(TensorFlags.PACKED_QCONV_O4I4))
    return cache if packed or descriptor.storage_span_bytes >= cache else scalar


def build_static_weight_plan(
    loaded: LoadedPlan,
    source_plan_bytes: bytes,
    source_weights: bytes,
    *,
    weight_index: Sequence[Mapping[str, Any]] | None = None,
    cache_alignment: int = DEFAULT_CACHE_ALIGNMENT,
    scalar_alignment: int = DEFAULT_SCALAR_ALIGNMENT,
) -> StaticWeightPlan:
    """Pack all constants required by one plan into a static bucket blob."""

    if loaded.header.bucket_frames <= 0:
        raise WeightResidencyPlanError("plan bucket must be positive")
    _aligned(0, cache_alignment)
    _aligned(0, scalar_alignment)
    if scalar_alignment > cache_alignment:
        raise WeightResidencyPlanError(
            "scalar alignment must not exceed cache alignment"
        )

    references = _operator_references(loaded)
    source_index = _weight_index_by_range(weight_index)
    constants = [
        descriptor for descriptor in loaded.tensors
        if descriptor.storage_type == TensorStorageType.CONSTANT
    ]
    if not constants:
        raise WeightResidencyPlanError("execution plan has no constant tensors")

    def order_key(descriptor: Any) -> tuple[int, int, int]:
        operator_ids = references.get(descriptor.tensor_id, ())
        return (
            0 if operator_ids else 1,
            operator_ids[0] if operator_ids else len(loaded.operators),
            descriptor.tensor_id,
        )

    destination = bytearray()
    output_descriptors = list(loaded.tensors)
    entries: list[StaticWeightEntry] = []
    padding_bytes = 0
    for descriptor in sorted(constants, key=order_key):
        start = int(descriptor.data_offset)
        byte_size = int(descriptor.storage_span_bytes)
        end = start + byte_size
        if start < 0 or byte_size <= 0 or end > len(source_weights):
            raise WeightResidencyPlanError(
                f"Tensor {descriptor.tensor_id} exceeds source weights: "
                f"[{start}, {end}) / {len(source_weights)}"
            )
        alignment = _alignment_for(
            descriptor, cache_alignment, scalar_alignment
        )
        destination_offset = _aligned(len(destination), alignment)
        if destination_offset > len(destination):
            padding_bytes += destination_offset - len(destination)
            destination.extend(b"\0" * (destination_offset - len(destination)))
        payload = source_weights[start:end]
        destination.extend(payload)
        output_descriptors[descriptor.tensor_id] = replace(
            descriptor, data_offset=destination_offset
        )

        metadata = source_index.get((start, byte_size), {})
        shape_value = metadata.get("shape", ())
        shape = tuple(int(value) for value in shape_value)
        operator_ids = references.get(descriptor.tensor_id, ())
        entries.append(
            StaticWeightEntry(
                tensor_id=descriptor.tensor_id,
                name=(
                    str(metadata["name"])
                    if metadata.get("name") is not None else None
                ),
                dtype=(
                    str(metadata["dtype"])
                    if metadata.get("dtype") is not None else None
                ),
                shape=shape,
                source_offset=start,
                destination_offset=destination_offset,
                byte_size=byte_size,
                alignment=alignment,
                first_use_operator=operator_ids[0] if operator_ids else None,
                last_use_operator=operator_ids[-1] if operator_ids else None,
                operator_ids=operator_ids,
                sha256=_sha256(payload),
            )
        )

    remapped = replace(loaded, tensors=tuple(output_descriptors))
    output_plan = build_loaded_plan_bytes(remapped)
    return StaticWeightPlan(
        bucket_frames=loaded.header.bucket_frames,
        source_plan_sha256=_sha256(source_plan_bytes),
        source_weights_sha256=_sha256(source_weights),
        source_weights_bytes=len(source_weights),
        plan_bytes=output_plan,
        weight_bytes=bytes(destination),
        entries=tuple(entries),
        output_padding_bytes=padding_bytes,
        cache_alignment=cache_alignment,
        scalar_alignment=scalar_alignment,
    )


def load_weight_index(manifest_path: Path) -> tuple[Mapping[str, Any], ...]:
    """Read the optional descriptive weight index from a bundle manifest."""

    import json

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = document.get("weight_index")
    if not isinstance(records, list):
        raise WeightResidencyPlanError("bundle manifest has no weight_index")
    if not all(isinstance(record, dict) for record in records):
        raise WeightResidencyPlanError("bundle weight_index is invalid")
    return tuple(records)


__all__ = [
    "DEFAULT_CACHE_ALIGNMENT",
    "DEFAULT_SCALAR_ALIGNMENT",
    "StaticWeightEntry",
    "StaticWeightPlan",
    "WeightResidencyPlanError",
    "build_static_weight_plan",
    "load_weight_index",
]
