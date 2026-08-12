"""Runtime-only intermediate representation for the bundle exporter.

Readers must convert source model objects to these plain Python values before
the binary writers are called.  This keeps serialization independent from the
source model format and gives both writers one validated graph contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping, TypeAlias

from .format.binary_format_schema import (
    OPERATOR_INPUT_CAPACITY,
    OPERATOR_OUTPUT_CAPACITY,
    PLAN_FORMAT_VERSION,
    TENSOR_MAX_RANK,
    OperatorCode,
    TensorDType,
    TensorStorageType,
)


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

_DTYPE_BYTE_SIZE: Mapping[TensorDType, int] = MappingProxyType(
    {
        TensorDType.FLOAT32: 4,
        TensorDType.UINT8: 1,
        TensorDType.INT8: 1,
        TensorDType.INT32: 4,
        TensorDType.INT64: 8,
        TensorDType.BOOL: 1,
        TensorDType.FLOAT16: 2,
    }
)

AttributeScalar: TypeAlias = int | float | bool | str | bytes | None
AttributeValue: TypeAlias = AttributeScalar | tuple["AttributeValue", ...]


class InitializerScope(Enum):
    """How far one initializer's bytes reach across a bundle's buckets.

    ``SHARED`` is a trained parameter: every bucket must see the same bytes, so
    an accidental divergence is an error.  ``BUCKET_LOCAL`` is a constant the
    static export folded out of the shape domain (a Reshape target, a Slice
    bound, a pooling length); it is allowed - not required - to differ per
    bucket, and the weight blob stores one copy per bucket.
    """

    SHARED = "shared"
    BUCKET_LOCAL = "bucket_local"


class RuntimeIRError(ValueError):
    """A Runtime IR value or graph relationship is invalid."""


def _require_name(kind: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise RuntimeIRError(f"{kind} name must be a non-empty string")


def _require_uint32(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeIRError(f"{name} must be an integer")
    if not 0 <= value <= UINT32_MAX:
        raise RuntimeIRError(f"{name} is outside 0..{UINT32_MAX}: {value}")


def _require_uint64(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeIRError(f"{name} must be an integer")
    if not 0 <= value <= UINT64_MAX:
        raise RuntimeIRError(f"{name} is outside 0..{UINT64_MAX}: {value}")


def _coerce_enum(name: str, value: int, enum_type: type) -> object:
    try:
        result = enum_type(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeIRError(f"unsupported {name}: {value!r}") from exc
    if int(result) == 0:
        raise RuntimeIRError(f"{name} must not be INVALID")
    return result


def _coerce_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    try:
        result = tuple(shape)
    except TypeError as exc:
        raise RuntimeIRError("shape must be an iterable of dimensions") from exc
    if len(result) > TENSOR_MAX_RANK:
        raise RuntimeIRError(
            f"rank exceeds Runtime maximum {TENSOR_MAX_RANK}: {len(result)}"
        )
    for axis, dimension in enumerate(result):
        _require_uint32(f"shape[{axis}]", dimension)
        if dimension == 0:
            raise RuntimeIRError(f"shape[{axis}] must be greater than zero")
    return result


def _logical_byte_size(dtype: TensorDType, shape: tuple[int, ...]) -> int:
    return math.prod(shape) * _DTYPE_BYTE_SIZE[dtype]


def _freeze_attribute(value: object, path: str) -> AttributeValue:
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeIRError(f"{path} must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_attribute(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise RuntimeIRError(
        f"{path} has unsupported Runtime attribute type "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class RuntimeTensor:
    """A Tensor known to the Runtime, including graph relationships."""

    name: str
    tensor_id: int
    dtype: TensorDType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    byte_size: int
    storage_type: TensorStorageType
    producer: int | None
    consumers: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dtype", _coerce_enum("dtype", self.dtype, TensorDType)
        )
        object.__setattr__(
            self,
            "storage_type",
            _coerce_enum("storage_type", self.storage_type, TensorStorageType),
        )
        object.__setattr__(self, "shape", _coerce_shape(self.shape))
        try:
            object.__setattr__(self, "strides", tuple(self.strides))
            object.__setattr__(self, "consumers", tuple(self.consumers))
        except TypeError as exc:
            raise RuntimeIRError("strides and consumers must be iterable") from exc
        self.validate()

    def validate(self) -> None:
        _require_name("Tensor", self.name)
        _require_uint32("tensor_id", self.tensor_id)
        if len(self.strides) != len(self.shape):
            raise RuntimeIRError(
                f"Tensor {self.name!r} has rank {len(self.shape)} but "
                f"{len(self.strides)} strides"
            )
        for axis, stride in enumerate(self.strides):
            _require_uint32(f"strides[{axis}]", stride)
            if stride == 0 and self.storage_type != TensorStorageType.VIEW:
                raise RuntimeIRError(
                    f"only VIEW Tensors may use a zero stride (axis {axis})"
                )

        _require_uint64("byte_size", self.byte_size)
        expected_size = _logical_byte_size(self.dtype, self.shape)
        if self.byte_size != expected_size:
            raise RuntimeIRError(
                f"Tensor {self.name!r} byte_size is {self.byte_size}, "
                f"expected {expected_size} from dtype and shape"
            )

        if self.producer is not None:
            _require_uint32("producer", self.producer)
        if self.storage_type in (
            TensorStorageType.INPUT,
            TensorStorageType.CONSTANT,
        ) and self.producer is not None:
            raise RuntimeIRError(
                f"{self.storage_type.name} Tensor {self.name!r} cannot have a producer"
            )

        seen: set[int] = set()
        previous = -1
        for index, consumer in enumerate(self.consumers):
            _require_uint32(f"consumers[{index}]", consumer)
            if consumer in seen:
                raise RuntimeIRError(
                    f"Tensor {self.name!r} contains duplicate consumer {consumer}"
                )
            if consumer <= previous:
                raise RuntimeIRError(
                    f"Tensor {self.name!r} consumers must be in execution order"
                )
            seen.add(consumer)
            previous = consumer


@dataclass(frozen=True, slots=True)
class RuntimeOperator:
    """One executable Runtime instruction in topological order."""

    name: str
    operator_id: int
    opcode: OperatorCode
    input_tensor_ids: tuple[int, ...]
    output_tensor_ids: tuple[int, ...]
    attributes: Mapping[str, AttributeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "opcode", _coerce_enum("opcode", self.opcode, OperatorCode)
        )
        try:
            object.__setattr__(self, "input_tensor_ids", tuple(self.input_tensor_ids))
            object.__setattr__(
                self, "output_tensor_ids", tuple(self.output_tensor_ids)
            )
        except TypeError as exc:
            raise RuntimeIRError("operator Tensor IDs must be iterable") from exc

        if not isinstance(self.attributes, Mapping):
            raise RuntimeIRError("attributes must be a mapping")
        frozen_attributes: dict[str, AttributeValue] = {}
        for name, value in self.attributes.items():
            if not isinstance(name, str) or not name:
                raise RuntimeIRError("attribute names must be non-empty strings")
            frozen_attributes[name] = _freeze_attribute(value, f"attribute {name!r}")
        object.__setattr__(self, "attributes", MappingProxyType(frozen_attributes))
        self.validate()

    def validate(self) -> None:
        _require_name("Operator", self.name)
        _require_uint32("operator_id", self.operator_id)
        if not self.input_tensor_ids:
            raise RuntimeIRError(f"Operator {self.name!r} must have an input")
        if len(self.input_tensor_ids) > OPERATOR_INPUT_CAPACITY:
            raise RuntimeIRError(
                f"Operator {self.name!r} exceeds the format's "
                f"{OPERATOR_INPUT_CAPACITY}-input capacity"
            )
        if len(self.output_tensor_ids) != OPERATOR_OUTPUT_CAPACITY:
            raise RuntimeIRError(
                f"Operator {self.name!r} must have exactly "
                f"{OPERATOR_OUTPUT_CAPACITY} output"
            )
        for index, tensor_id in enumerate(self.input_tensor_ids):
            _require_uint32(f"input_tensor_ids[{index}]", tensor_id)
        seen_outputs: set[int] = set()
        for index, tensor_id in enumerate(self.output_tensor_ids):
            _require_uint32(f"output_tensor_ids[{index}]", tensor_id)
            if tensor_id in seen_outputs:
                raise RuntimeIRError(
                    f"Operator {self.name!r} contains duplicate output Tensor "
                    f"{tensor_id}"
                )
            seen_outputs.add(tensor_id)


@dataclass(frozen=True, slots=True)
class RuntimeInitializer:
    """Raw little-endian bytes for a CONSTANT Runtime Tensor."""

    name: str
    tensor_id: int
    dtype: TensorDType
    shape: tuple[int, ...]
    raw_data: bytes
    scope: InitializerScope = InitializerScope.SHARED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dtype", _coerce_enum("dtype", self.dtype, TensorDType)
        )
        object.__setattr__(self, "shape", _coerce_shape(self.shape))
        if isinstance(self.raw_data, (bytearray, memoryview)):
            object.__setattr__(self, "raw_data", bytes(self.raw_data))
        try:
            object.__setattr__(self, "scope", InitializerScope(self.scope))
        except ValueError as exc:
            raise RuntimeIRError(
                f"unsupported initializer scope: {self.scope!r}"
            ) from exc
        self.validate()

    @property
    def byte_size(self) -> int:
        return len(self.raw_data)

    def validate(self) -> None:
        _require_name("Initializer", self.name)
        _require_uint32("tensor_id", self.tensor_id)
        if not isinstance(self.raw_data, bytes):
            raise RuntimeIRError("initializer raw_data must be bytes")
        expected_size = _logical_byte_size(self.dtype, self.shape)
        if len(self.raw_data) != expected_size:
            raise RuntimeIRError(
                f"Initializer {self.name!r} contains {len(self.raw_data)} bytes, "
                f"expected {expected_size} from dtype and shape"
            )


@dataclass(frozen=True, slots=True)
class RuntimeGraph:
    """A validated, serialization-ready graph for one input bucket."""

    tensors: tuple[RuntimeTensor, ...]
    operators: tuple[RuntimeOperator, ...]
    initializers: tuple[RuntimeInitializer, ...]
    input_tensor_ids: tuple[int, ...] | None = None
    output_tensor_ids: tuple[int, ...] | None = None
    name: str = ""
    bucket_frames: int | None = None
    _tensors_by_id: Mapping[int, RuntimeTensor] = field(
        init=False, repr=False, compare=False
    )
    _operators_by_id: Mapping[int, RuntimeOperator] = field(
        init=False, repr=False, compare=False
    )
    _initializers_by_tensor_id: Mapping[int, RuntimeInitializer] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "tensors", tuple(self.tensors))
            object.__setattr__(self, "operators", tuple(self.operators))
            object.__setattr__(self, "initializers", tuple(self.initializers))
        except TypeError as exc:
            raise RuntimeIRError(
                "tensors, operators, and initializers must be iterable"
            ) from exc

        inputs = (
            tuple(
                tensor.tensor_id
                for tensor in self.tensors
                if tensor.storage_type == TensorStorageType.INPUT
            )
            if self.input_tensor_ids is None
            else tuple(self.input_tensor_ids)
        )
        outputs = (
            tuple(
                tensor.tensor_id
                for tensor in self.tensors
                if tensor.storage_type == TensorStorageType.OUTPUT
            )
            if self.output_tensor_ids is None
            else tuple(self.output_tensor_ids)
        )
        object.__setattr__(self, "input_tensor_ids", inputs)
        object.__setattr__(self, "output_tensor_ids", outputs)

        object.__setattr__(
            self,
            "_tensors_by_id",
            MappingProxyType({tensor.tensor_id: tensor for tensor in self.tensors}),
        )
        object.__setattr__(
            self,
            "_operators_by_id",
            MappingProxyType(
                {operator.operator_id: operator for operator in self.operators}
            ),
        )
        object.__setattr__(
            self,
            "_initializers_by_tensor_id",
            MappingProxyType(
                {
                    initializer.tensor_id: initializer
                    for initializer in self.initializers
                }
            ),
        )
        self.validate()

    def tensor(self, tensor_id: int) -> RuntimeTensor:
        return self._tensors_by_id[tensor_id]

    def operator(self, operator_id: int) -> RuntimeOperator:
        return self._operators_by_id[operator_id]

    def initializer(self, tensor_id: int) -> RuntimeInitializer:
        return self._initializers_by_tensor_id[tensor_id]

    def validate(self) -> None:
        if not isinstance(self.name, str):
            raise RuntimeIRError("graph name must be a string")
        if self.bucket_frames is not None:
            _require_uint32("bucket_frames", self.bucket_frames)
            if self.bucket_frames == 0:
                raise RuntimeIRError("bucket_frames must be greater than zero")
        if not self.tensors:
            raise RuntimeIRError("RuntimeGraph must contain at least one Tensor")
        if not self.operators:
            raise RuntimeIRError("RuntimeGraph must contain at least one Operator")

        self._validate_unique_and_dense_ids()
        self._validate_graph_io()
        self._validate_operator_links()
        self._validate_initializers()

    def _validate_unique_and_dense_ids(self) -> None:
        tensor_ids = [tensor.tensor_id for tensor in self.tensors]
        if len(self._tensors_by_id) != len(self.tensors):
            raise RuntimeIRError("RuntimeGraph contains duplicate Tensor IDs")
        if tensor_ids != list(range(len(self.tensors))):
            raise RuntimeIRError("Tensor IDs must be dense and ordered from zero")
        if len({tensor.name for tensor in self.tensors}) != len(self.tensors):
            raise RuntimeIRError("RuntimeGraph contains duplicate Tensor names")

        operator_ids = [operator.operator_id for operator in self.operators]
        if len(self._operators_by_id) != len(self.operators):
            raise RuntimeIRError("RuntimeGraph contains duplicate Operator IDs")
        if operator_ids != list(range(len(self.operators))):
            raise RuntimeIRError("Operator IDs must be dense execution order from zero")
        if len({operator.name for operator in self.operators}) != len(self.operators):
            raise RuntimeIRError("RuntimeGraph contains duplicate Operator names")

    def _validate_graph_io(self) -> None:
        assert self.input_tensor_ids is not None
        assert self.output_tensor_ids is not None
        if not self.input_tensor_ids:
            raise RuntimeIRError("RuntimeGraph must contain at least one input Tensor")
        if not self.output_tensor_ids:
            raise RuntimeIRError("RuntimeGraph must contain at least one output Tensor")
        actual_inputs = tuple(
            tensor.tensor_id
            for tensor in self.tensors
            if tensor.storage_type == TensorStorageType.INPUT
        )
        actual_outputs = tuple(
            tensor.tensor_id
            for tensor in self.tensors
            if tensor.storage_type == TensorStorageType.OUTPUT
        )
        if self.input_tensor_ids != actual_inputs:
            raise RuntimeIRError(
                "input_tensor_ids must exactly match Tensors with INPUT storage"
            )
        if self.output_tensor_ids != actual_outputs:
            raise RuntimeIRError(
                "output_tensor_ids must exactly match Tensors with OUTPUT storage"
            )

    def _validate_operator_links(self) -> None:
        expected_producers: dict[int, int] = {}
        expected_consumers: dict[int, list[int]] = {
            tensor.tensor_id: [] for tensor in self.tensors
        }
        for operator in self.operators:
            for tensor_id in operator.input_tensor_ids:
                if tensor_id not in self._tensors_by_id:
                    raise RuntimeIRError(
                        f"Operator {operator.name!r} references unknown input "
                        f"Tensor {tensor_id}"
                    )
                consumers = expected_consumers[tensor_id]
                if not consumers or consumers[-1] != operator.operator_id:
                    consumers.append(operator.operator_id)
                producer = self._tensors_by_id[tensor_id].producer
                if producer is not None and producer >= operator.operator_id:
                    raise RuntimeIRError(
                        f"Operator {operator.name!r} consumes Tensor {tensor_id} "
                        "before it is produced"
                    )
            for tensor_id in operator.output_tensor_ids:
                if tensor_id not in self._tensors_by_id:
                    raise RuntimeIRError(
                        f"Operator {operator.name!r} references unknown output "
                        f"Tensor {tensor_id}"
                    )
                if tensor_id in expected_producers:
                    raise RuntimeIRError(
                        f"Tensor {tensor_id} is produced by more than one Operator"
                    )
                expected_producers[tensor_id] = operator.operator_id

        for tensor in self.tensors:
            expected_producer = expected_producers.get(tensor.tensor_id)
            if tensor.producer != expected_producer:
                raise RuntimeIRError(
                    f"Tensor {tensor.name!r} producer is {tensor.producer!r}, "
                    f"expected {expected_producer!r} from Operator outputs"
                )
            if tensor.storage_type not in (
                TensorStorageType.INPUT,
                TensorStorageType.CONSTANT,
            ) and expected_producer is None:
                raise RuntimeIRError(f"Tensor {tensor.name!r} has no producer")
            expected = tuple(expected_consumers[tensor.tensor_id])
            if tensor.consumers != expected:
                raise RuntimeIRError(
                    f"Tensor {tensor.name!r} consumers are {tensor.consumers}, "
                    f"expected {expected} from Operator inputs"
                )

    def _validate_initializers(self) -> None:
        if len(self._initializers_by_tensor_id) != len(self.initializers):
            raise RuntimeIRError("RuntimeGraph contains duplicate initializer Tensor IDs")
        if len({item.name for item in self.initializers}) != len(self.initializers):
            raise RuntimeIRError("RuntimeGraph contains duplicate initializer names")
        constant_ids = {
            tensor.tensor_id
            for tensor in self.tensors
            if tensor.storage_type == TensorStorageType.CONSTANT
        }
        if set(self._initializers_by_tensor_id) != constant_ids:
            raise RuntimeIRError(
                "initializers must correspond exactly to CONSTANT Tensors"
            )
        for initializer in self.initializers:
            tensor = self._tensors_by_id[initializer.tensor_id]
            if initializer.name != tensor.name:
                raise RuntimeIRError(
                    f"Initializer {initializer.name!r} does not match Tensor "
                    f"name {tensor.name!r}"
                )
            if (
                initializer.dtype != tensor.dtype
                or initializer.shape != tensor.shape
                or initializer.byte_size != tensor.byte_size
            ):
                raise RuntimeIRError(
                    f"Initializer {initializer.name!r} metadata does not match its Tensor"
                )


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    """Graphs whose execution plans share one immutable weight blob.

    Every graph carries the same initializer names in the same order.  The
    ``SHARED`` ones are byte-identical across buckets and the blob stores one
    copy; the ``BUCKET_LOCAL`` ones are stored once per bucket, so a plan's
    Tensor descriptor points at its own bucket's offset.  The Tensor ID and the
    ``CONSTANT -> weights_base + data_offset`` rule are unchanged either way.
    """

    graphs: tuple[RuntimeGraph, ...]
    format_version: int = PLAN_FORMAT_VERSION

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "graphs", tuple(self.graphs))
        except TypeError as exc:
            raise RuntimeIRError("graphs must be iterable") from exc
        self.validate()

    @property
    def shared_initializers(self) -> tuple[RuntimeInitializer, ...]:
        """Initializers written to the weight blob exactly once."""

        return tuple(
            item
            for item in self.graphs[0].initializers
            if item.scope is InitializerScope.SHARED
        )

    def bucket_initializers(
        self, bucket_frames: int
    ) -> tuple[RuntimeInitializer, ...]:
        """Initializers the weight blob stores separately for this bucket."""

        return tuple(
            item
            for item in self.graph_for_bucket(bucket_frames).initializers
            if item.scope is InitializerScope.BUCKET_LOCAL
        )

    def graph_for_bucket(self, bucket_frames: int) -> RuntimeGraph:
        for graph in self.graphs:
            if graph.bucket_frames == bucket_frames:
                return graph
        raise KeyError(bucket_frames)

    def validate(self) -> None:
        if not self.graphs:
            raise RuntimeIRError("RuntimeBundle must contain at least one graph")
        _require_uint32("format_version", self.format_version)
        if self.format_version == 0:
            raise RuntimeIRError("format_version must be greater than zero")
        if len(self.graphs) > 1 and any(
            graph.bucket_frames is None for graph in self.graphs
        ):
            raise RuntimeIRError(
                "every graph in a multi-graph bundle must define bucket_frames"
            )
        buckets = [
            graph.bucket_frames
            for graph in self.graphs
            if graph.bucket_frames is not None
        ]
        if len(set(buckets)) != len(buckets):
            raise RuntimeIRError("RuntimeBundle contains duplicate bucket sizes")
        self._validate_initializer_scopes()

    def _validate_initializer_scopes(self) -> None:
        """Hold SHARED initializers byte-identical and let BUCKET_LOCAL vary.

        A trained parameter that silently changed between buckets would slip
        through a plain "may differ" rule, so the scope - not the observed
        difference - decides what is allowed.
        """

        reference = {item.name: item for item in self.graphs[0].initializers}
        for graph in self.graphs[1:]:
            current = {item.name: item for item in graph.initializers}
            if set(current) != set(reference):
                missing = sorted(set(reference) - set(current))
                added = sorted(set(current) - set(reference))
                raise RuntimeIRError(
                    "every graph in a RuntimeBundle must carry the same "
                    f"initializer names (missing {missing[:3]}, extra {added[:3]})"
                )
            for name, item in current.items():
                expected = reference[name]
                if item.scope is not expected.scope:
                    raise RuntimeIRError(
                        f"initializer {name!r} is {expected.scope.value} in one "
                        f"graph and {item.scope.value} in another"
                    )
                if item.scope is not InitializerScope.SHARED:
                    continue
                if (
                    item.dtype != expected.dtype
                    or item.shape != expected.shape
                    or item.raw_data != expected.raw_data
                ):
                    raise RuntimeIRError(
                        f"SHARED initializer {name!r} differs between buckets; "
                        "a trained parameter must not depend on the bucket"
                    )


__all__ = [
    "AttributeScalar",
    "AttributeValue",
    "InitializerScope",
    "RuntimeBundle",
    "RuntimeGraph",
    "RuntimeInitializer",
    "RuntimeIRError",
    "RuntimeOperator",
    "RuntimeTensor",
]
