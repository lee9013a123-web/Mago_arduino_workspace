"""Plan channel-last padded activation storage without changing logical shapes.

The CAM++ graph keeps ONNX logical dimensions (N,C,T) or (N,C,H,W), while
byte strides describe physical [N,T,C_padded] / [N,H,W,C_padded] storage.
Kernels therefore exclude padding by iterating logical dimensions, and Dense
slab aliases keep sharing one backing allocation with channel-byte offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ..format.binary_format_schema import OperatorCode, TensorStorageType
from ..runtime_ir import RuntimeGraph, RuntimeOperator, RuntimeTensor


class CacheLayoutPlanningError(ValueError):
    """The graph cannot safely be represented by the requested layout."""


@dataclass(frozen=True, slots=True)
class CacheLayoutRewriteResult:
    graph: RuntimeGraph
    channel_block: int
    removed_transpose_operator_ids: tuple[int, ...]
    removed_descriptor_copy_operator_ids: tuple[int, ...]
    channels_last_tensor_count: int
    padded_tensor_count: int
    activation_storage_bytes_before: int
    activation_storage_bytes_after: int

    @property
    def padding_bytes(self) -> int:
        return self.activation_storage_bytes_after - self.activation_storage_bytes_before

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket_frames": self.graph.bucket_frames,
            "logical_layout": "NCT/NCHW",
            "physical_layout": "NTC_padded/NHWC_padded",
            "channel_block": self.channel_block,
            "removed_transpose_operator_ids": list(
                self.removed_transpose_operator_ids
            ),
            "removed_descriptor_copy_operator_ids": list(
                self.removed_descriptor_copy_operator_ids
            ),
            "removed_layout_copy_count": len(
                self.removed_descriptor_copy_operator_ids
            ),
            "channels_last_tensor_count": self.channels_last_tensor_count,
            "padded_tensor_count": self.padded_tensor_count,
            "activation_storage_bytes_before": self.activation_storage_bytes_before,
            "activation_storage_bytes_after": self.activation_storage_bytes_after,
            "padding_bytes": self.padding_bytes,
        }


def _aligned(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _element_size(tensor: RuntimeTensor) -> int:
    return tensor.byte_size // math.prod(tensor.shape)


def _is_channel_tensor(tensor: RuntimeTensor) -> bool:
    return (
        len(tensor.shape) in (3, 4)
        and tensor.storage_type is not TensorStorageType.CONSTANT
    )


def _channels_last_strides(
    shape: tuple[int, ...], channel_capacity: int, element_size: int
) -> tuple[int, ...]:
    if len(shape) == 3:
        _, _, width = shape
        return (
            width * channel_capacity * element_size,
            element_size,
            channel_capacity * element_size,
        )
    if len(shape) == 4:
        _, _, height, width = shape
        return (
            height * width * channel_capacity * element_size,
            element_size,
            width * channel_capacity * element_size,
            channel_capacity * element_size,
        )
    raise CacheLayoutPlanningError(f"unsupported channel rank: {len(shape)}")


def _required_span(
    shape: tuple[int, ...], strides: tuple[int, ...], element_size: int
) -> int:
    return element_size + sum(
        (dimension - 1) * stride for dimension, stride in zip(shape, strides)
    )


def _channel_offset(view: RuntimeTensor, element_size: int) -> int:
    # A previously rewritten NTC view stores a channel-byte offset directly.
    if len(view.strides) >= 2 and view.strides[1] == element_size:
        if view.view_byte_offset % element_size:
            raise CacheLayoutPlanningError(
                f"VIEW {view.name!r} has a non-element-aligned offset"
            )
        return view.view_byte_offset // element_size

    plane_bytes = math.prod(view.shape[2:]) * element_size
    if plane_bytes == 0 or view.view_byte_offset % plane_bytes:
        raise CacheLayoutPlanningError(
            f"Dense VIEW {view.name!r} offset is not channel aligned"
        )
    return view.view_byte_offset // plane_bytes


def _rebuild_links(
    tensors: tuple[RuntimeTensor, ...], operators: tuple[RuntimeOperator, ...]
) -> tuple[RuntimeTensor, ...]:
    producers: dict[int, int] = {}
    consumers: dict[int, list[int]] = {tensor.tensor_id: [] for tensor in tensors}
    for operator in operators:
        for tensor_id in operator.input_tensor_ids:
            if not consumers[tensor_id] or consumers[tensor_id][-1] != operator.operator_id:
                consumers[tensor_id].append(operator.operator_id)
        for tensor_id in operator.output_tensor_ids:
            producers[tensor_id] = operator.operator_id
    return tuple(
        replace(
            tensor,
            producer=producers.get(tensor.tensor_id),
            consumers=tuple(consumers[tensor.tensor_id]),
        )
        for tensor in tensors
    )


def _axes(value: object, rank: int) -> tuple[int, ...] | None:
    if isinstance(value, int) and not isinstance(value, bool):
        raw = (value,)
    elif isinstance(value, tuple) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raw = value
    else:
        return None
    normalized = tuple(item + rank if item < 0 else item for item in raw)
    if len(set(normalized)) != len(normalized) or any(
        item < 0 or item >= rank for item in normalized
    ):
        return None
    return tuple(sorted(normalized))


def _descriptor_relation(
    graph: RuntimeGraph, operator: RuntimeOperator, *, remove_input_transpose: bool
) -> tuple[RuntimeTensor, RuntimeTensor, tuple[int | None, ...]] | None:
    source = graph.tensor(operator.input_tensor_ids[0])
    output = graph.tensor(operator.output_tensor_ids[0])
    if source.dtype != output.dtype or math.prod(source.shape) != math.prod(output.shape):
        return None
    if source.storage_type not in (
        TensorStorageType.INPUT,
        TensorStorageType.ACTIVATION,
        TensorStorageType.OUTPUT,
        TensorStorageType.VIEW,
    ):
        return None

    if operator.opcode is OperatorCode.TRANSPOSE:
        perm = operator.attributes.get("perm")
        if (
            not remove_input_transpose
            or source.storage_type is not TensorStorageType.INPUT
            or perm != (0, 2, 1)
            or output.shape != tuple(source.shape[index] for index in perm)
        ):
            return None
        return source, output, tuple(perm)

    if operator.opcode is OperatorCode.UNSQUEEZE:
        inserted = _axes(operator.attributes.get("axes"), len(output.shape))
        if inserted is None or len(output.shape) != len(source.shape) + len(inserted):
            return None
        relation: list[int | None] = []
        source_axis = 0
        for output_axis in range(len(output.shape)):
            if output_axis in inserted:
                if output.shape[output_axis] != 1:
                    return None
                relation.append(None)
            else:
                if output.shape[output_axis] != source.shape[source_axis]:
                    return None
                relation.append(source_axis)
                source_axis += 1
        return source, output, tuple(relation)

    if operator.opcode is OperatorCode.SQUEEZE:
        removed = _axes(operator.attributes.get("axes"), len(source.shape))
        if removed is None:
            return None
        kept = tuple(axis for axis in range(len(source.shape)) if axis not in removed)
        if any(source.shape[axis] != 1 for axis in removed) or output.shape != tuple(
            source.shape[axis] for axis in kept
        ):
            return None
        return source, output, kept

    if operator.opcode is OperatorCode.RESHAPE:
        kept_list: list[int] = []
        output_axis = 0
        for source_axis, dimension in enumerate(source.shape):
            if (
                output_axis < len(output.shape)
                and dimension == output.shape[output_axis]
            ):
                kept_list.append(source_axis)
                output_axis += 1
            elif dimension != 1:
                return None
        kept = tuple(kept_list)
        if output_axis != len(output.shape):
            return None
        return source, output, kept
    return None


def _remove_descriptor_copies(
    graph: RuntimeGraph, *, remove_input_transpose: bool
) -> tuple[
    tuple[RuntimeTensor, ...],
    tuple[RuntimeOperator, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, int, tuple[int | None, ...]], ...],
]:
    overrides: dict[int, RuntimeTensor] = {}
    removed: set[int] = set()
    removed_transposes: set[int] = set()
    relations: list[tuple[int, int, tuple[int | None, ...]]] = []
    for operator in graph.operators:
        if not operator.input_tensor_ids or len(operator.output_tensor_ids) != 1:
            continue
        relation = _descriptor_relation(
            graph, operator, remove_input_transpose=remove_input_transpose
        )
        if relation is None:
            continue
        source, output, axis_map = relation
        source = overrides.get(source.tensor_id, source)
        if source.storage_type is TensorStorageType.VIEW:
            assert source.alias_of_tensor_id is not None
            base = graph.tensor(source.alias_of_tensor_id)
            byte_offset = source.view_byte_offset
        else:
            base = source
            byte_offset = 0
        if base.storage_type not in (
            TensorStorageType.INPUT,
            TensorStorageType.ACTIVATION,
            TensorStorageType.OUTPUT,
        ):
            continue
        strides = tuple(
            0 if source_axis is None else source.strides[source_axis]
            for source_axis in axis_map
        )
        available = (base.storage_span_bytes or base.byte_size) - byte_offset
        if available < output.byte_size:
            raise CacheLayoutPlanningError(
                f"descriptor VIEW {output.name!r} exceeds backing Tensor"
            )
        overrides[output.tensor_id] = replace(
            output,
            strides=strides,
            storage_type=TensorStorageType.VIEW,
            storage_span_bytes=available,
            alias_of_tensor_id=base.tensor_id,
            view_byte_offset=byte_offset,
        )
        removed.add(operator.operator_id)
        if operator.opcode is OperatorCode.TRANSPOSE:
            removed_transposes.add(operator.operator_id)
        relations.append((output.tensor_id, source.tensor_id, axis_map))

    operators = tuple(
        replace(operator, operator_id=new_id)
        for new_id, operator in enumerate(
            operator
            for operator in graph.operators
            if operator.operator_id not in removed
        )
    )
    tensors = tuple(overrides.get(tensor.tensor_id, tensor) for tensor in graph.tensors)
    return (
        _rebuild_links(tensors, operators),
        operators,
        tuple(sorted(removed_transposes)),
        tuple(sorted(removed)),
        tuple(relations),
    )


def rewrite_cache_friendly_activations(
    graph: RuntimeGraph, *, channel_block: int = 4, remove_input_transpose: bool = True
) -> CacheLayoutRewriteResult:
    """Rewrite arena Tensor strides to channels-last padded physical storage."""

    if channel_block <= 0 or channel_block & (channel_block - 1):
        raise CacheLayoutPlanningError("channel_block must be a positive power of two")

    (
        tensors,
        operators,
        removed_transposes,
        removed_descriptor_copies,
        descriptor_relations,
    ) = _remove_descriptor_copies(
        graph, remove_input_transpose=remove_input_transpose
    )
    relation_tensor_ids = {relation[0] for relation in descriptor_relations}

    by_id = {tensor.tensor_id: tensor for tensor in tensors}
    capacities: dict[int, int] = {}
    for tensor in tensors:
        if (
            _is_channel_tensor(tensor)
            and tensor.storage_type not in (TensorStorageType.INPUT, TensorStorageType.VIEW)
        ):
            capacities[tensor.tensor_id] = tensor.shape[1]

    for view in tensors:
        if view.storage_type is not TensorStorageType.VIEW:
            continue
        if view.tensor_id in relation_tensor_ids:
            continue
        assert view.alias_of_tensor_id is not None
        base = by_id[view.alias_of_tensor_id]
        if base.tensor_id not in capacities or not _is_channel_tensor(view):
            continue
        element_size = _element_size(view)
        offset_channels = _channel_offset(view, element_size)
        capacities[base.tensor_id] = max(
            capacities[base.tensor_id], offset_channels + view.shape[1]
        )

    capacities = {
        tensor_id: _aligned(channels, channel_block)
        for tensor_id, channels in capacities.items()
    }
    overrides: dict[int, RuntimeTensor] = {}
    padded_count = 0
    before = 0
    after = 0
    channels_last_count = 0

    for tensor in tensors:
        if tensor.tensor_id in capacities:
            capacity = capacities[tensor.tensor_id]
            element_size = _element_size(tensor)
            strides = _channels_last_strides(tensor.shape, capacity, element_size)
            span = math.prod((tensor.shape[0], *tensor.shape[2:])) * capacity * element_size
            overrides[tensor.tensor_id] = replace(
                tensor, strides=strides, storage_span_bytes=span
            )
            channels_last_count += 1
            padded_count += capacity != tensor.shape[1]
            before += tensor.storage_span_bytes or tensor.byte_size
            after += span
            continue

        if (
            tensor.storage_type is TensorStorageType.VIEW
            and _is_channel_tensor(tensor)
            and tensor.tensor_id not in relation_tensor_ids
        ):
            assert tensor.alias_of_tensor_id is not None
            base = by_id[tensor.alias_of_tensor_id]
            capacity = capacities.get(base.tensor_id)
            if capacity is None:
                # Input transpose view already has [T,C] physical strides.
                continue
            element_size = _element_size(tensor)
            channel_offset = _channel_offset(tensor, element_size)
            strides = _channels_last_strides(tensor.shape, capacity, element_size)
            span = _required_span(tensor.shape, strides, element_size)
            overrides[tensor.tensor_id] = replace(
                tensor,
                strides=strides,
                storage_span_bytes=span,
                view_byte_offset=channel_offset * element_size,
            )
            channels_last_count += 1

    tensors = tuple(overrides.get(tensor.tensor_id, tensor) for tensor in tensors)
    by_id = {tensor.tensor_id: tensor for tensor in tensors}
    for output_id, source_id, axis_map in descriptor_relations:
        source = by_id[source_id]
        output = by_id[output_id]
        element_size = _element_size(output)
        strides = tuple(
            0 if source_axis is None else source.strides[source_axis]
            for source_axis in axis_map
        )
        output = replace(
            output,
            strides=strides,
            storage_span_bytes=_required_span(output.shape, strides, element_size),
        )
        by_id[output_id] = output
    tensors = tuple(by_id[tensor.tensor_id] for tensor in tensors)
    tensors = _rebuild_links(tensors, operators)
    rewritten = RuntimeGraph(
        tensors=tensors,
        operators=operators,
        initializers=graph.initializers,
        input_tensor_ids=graph.input_tensor_ids,
        output_tensor_ids=graph.output_tensor_ids,
        name=graph.name,
        bucket_frames=graph.bucket_frames,
    )
    return CacheLayoutRewriteResult(
        graph=rewritten,
        channel_block=channel_block,
        removed_transpose_operator_ids=removed_transposes,
        removed_descriptor_copy_operator_ids=removed_descriptor_copies,
        channels_last_tensor_count=channels_last_count,
        padded_tensor_count=padded_count,
        activation_storage_bytes_before=before,
        activation_storage_bytes_after=after,
    )


__all__ = [
    "CacheLayoutPlanningError",
    "CacheLayoutRewriteResult",
    "rewrite_cache_friendly_activations",
]
