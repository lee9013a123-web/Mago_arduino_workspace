"""Rewrite DenseNet-style cumulative Concat chains as slab-backed views.

The compiled execution plan already contains the information needed to
recognise these chains (opcode, Tensor IDs, shapes, strides and Concat axis).
The exporter performs the rewrite on :class:`RuntimeGraph`, however, so the
normal writers can regenerate checksums, lifetimes and arena offsets instead
of patching a binary plan in place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from ...format.binary_format_schema import OperatorCode, TensorStorageType
from ...runtime_ir import RuntimeGraph, RuntimeOperator, RuntimeTensor


class DenseSlabPlanningError(ValueError):
    """A candidate Dense Concat chain is ambiguous or unsafe to rewrite."""


@dataclass(frozen=True, slots=True)
class CompiledDenseSlabBlock:
    """Dense chain discovered directly in an existing ``plan_*.bin``."""

    backing_tensor_id: int
    concat_operator_ids: tuple[int, ...]
    feature_tensor_ids: tuple[int, ...]
    prefix_tensor_ids: tuple[int, ...]
    slab_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DenseSlabBlock:
    block_index: int
    backing_tensor_id: int
    backing_tensor_name: str
    slab_shape: tuple[int, ...]
    slab_bytes: int
    concat_operator_ids: tuple[int, ...]
    concat_operator_names: tuple[str, ...]
    feature_tensor_ids: tuple[int, ...]
    prefix_tensor_ids: tuple[int, ...]
    feature_offsets: tuple[int, ...]
    copied_bytes_before: int

    def to_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "backing_tensor_id": self.backing_tensor_id,
            "backing_tensor_name": self.backing_tensor_name,
            "slab_shape": list(self.slab_shape),
            "slab_bytes": self.slab_bytes,
            "concat_operator_ids": list(self.concat_operator_ids),
            "concat_operator_names": list(self.concat_operator_names),
            "feature_tensor_ids": list(self.feature_tensor_ids),
            "prefix_tensor_ids": list(self.prefix_tensor_ids),
            "feature_offsets": list(self.feature_offsets),
            "copied_bytes_before": self.copied_bytes_before,
            "copied_bytes_after": 0,
        }


@dataclass(frozen=True, slots=True)
class DenseSlabRewriteResult:
    graph: RuntimeGraph
    blocks: tuple[DenseSlabBlock, ...]
    original_operator_count: int

    @property
    def removed_concat_count(self) -> int:
        return sum(len(block.concat_operator_ids) for block in self.blocks)

    @property
    def copied_bytes_before(self) -> int:
        return sum(block.copied_bytes_before for block in self.blocks)

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket_frames": self.graph.bucket_frames,
            "layout": "NCT",
            "block_count": len(self.blocks),
            "original_operator_count": self.original_operator_count,
            "optimized_operator_count": len(self.graph.operators),
            "removed_dense_concat_count": self.removed_concat_count,
            "dense_concat_copied_bytes_before": self.copied_bytes_before,
            "dense_concat_copied_bytes_after": 0,
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(frozen=True, slots=True)
class _ConcatCandidate:
    operator: RuntimeOperator
    prefix_input: RuntimeTensor
    feature_input: RuntimeTensor
    output: RuntimeTensor


def _concat_axis(operator: RuntimeOperator) -> int | None:
    value = operator.attributes.get("axis")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, tuple) and len(value) == 1:
        item = value[0]
        if isinstance(item, int) and not isinstance(item, bool):
            return item
    return None


def _candidate(graph: RuntimeGraph, operator: RuntimeOperator) -> _ConcatCandidate | None:
    if operator.opcode is not OperatorCode.CONCAT:
        return None
    if len(operator.input_tensor_ids) != 2 or _concat_axis(operator) != 1:
        return None

    prefix = graph.tensor(operator.input_tensor_ids[0])
    feature = graph.tensor(operator.input_tensor_ids[1])
    output = graph.tensor(operator.output_tensor_ids[0])
    if not (len(prefix.shape) == len(feature.shape) == len(output.shape) == 3):
        return None
    if not (prefix.dtype == feature.dtype == output.dtype):
        return None
    if prefix.shape[0] != feature.shape[0] or prefix.shape[0] != output.shape[0]:
        return None
    if prefix.shape[2] != feature.shape[2] or prefix.shape[2] != output.shape[2]:
        return None
    if output.shape[1] != prefix.shape[1] + feature.shape[1]:
        return None
    if output.byte_size != prefix.byte_size + feature.byte_size:
        return None
    if feature.storage_type not in (
        TensorStorageType.ACTIVATION,
        TensorStorageType.OUTPUT,
    ):
        return None
    return _ConcatCandidate(operator, prefix, feature, output)


def _find_chains(
    graph: RuntimeGraph, *, minimum_chain_length: int
) -> tuple[tuple[_ConcatCandidate, ...], ...]:
    candidates = tuple(
        candidate
        for operator in graph.operators
        if (candidate := _candidate(graph, operator)) is not None
    )
    by_output = {item.output.tensor_id: item for item in candidates}
    successors: dict[int, list[_ConcatCandidate]] = {}
    for item in candidates:
        successors.setdefault(item.prefix_input.tensor_id, []).append(item)

    roots = tuple(
        item for item in candidates if item.prefix_input.tensor_id not in by_output
    )
    chains: list[tuple[_ConcatCandidate, ...]] = []
    visited: set[int] = set()
    for root in roots:
        chain: list[_ConcatCandidate] = []
        current = root
        while True:
            operator_id = current.operator.operator_id
            if operator_id in visited:
                raise DenseSlabPlanningError(
                    f"Dense Concat chain contains a cycle at Operator {operator_id}"
                )
            visited.add(operator_id)
            chain.append(current)
            next_items = successors.get(current.output.tensor_id, [])
            if len(next_items) > 1:
                raise DenseSlabPlanningError(
                    f"Dense prefix Tensor {current.output.tensor_id} feeds multiple "
                    "Concat successors"
                )
            if not next_items:
                break
            current = next_items[0]
        if len(chain) >= minimum_chain_length:
            chains.append(tuple(chain))

    return tuple(chains)


def inspect_compiled_dense_concats(
    loaded_plan: object, *, minimum_chain_length: int = 2
) -> tuple[CompiledDenseSlabBlock, ...]:
    """Read Dense chains from a parsed execution plan without ONNX names.

    ``loaded_plan`` is the object returned by
    :func:`writer.execution_plan_writer.read_execution_plan`.  Keeping the
    dependency structural avoids a builder-to-writer import cycle.
    """

    if minimum_chain_length < 2:
        raise DenseSlabPlanningError("minimum_chain_length must be at least 2")
    operators = loaded_plan.operators
    tensors = loaded_plan.tensors
    candidates: dict[int, tuple[object, object, object, object]] = {}
    by_output: dict[int, int] = {}
    successors: dict[int, list[int]] = {}
    for operator in operators:
        if operator.opcode != int(OperatorCode.CONCAT) or operator.input_count != 2:
            continue
        axis = loaded_plan.attributes_of(operator.operator_id).get("axis")
        if axis not in (1, (1,)):
            continue
        prefix = tensors[operator.input_tensor_ids[0]]
        feature = tensors[operator.input_tensor_ids[1]]
        output = tensors[operator.output_tensor_ids[0]]
        prefix_shape = tuple(prefix.dimensions[: prefix.rank])
        feature_shape = tuple(feature.dimensions[: feature.rank])
        output_shape = tuple(output.dimensions[: output.rank])
        if not (prefix.rank == feature.rank == output.rank == 3):
            continue
        if not (prefix.dtype == feature.dtype == output.dtype):
            continue
        if (
            output_shape[0] != prefix_shape[0]
            or output_shape[0] != feature_shape[0]
            or output_shape[2] != prefix_shape[2]
            or output_shape[2] != feature_shape[2]
            or output_shape[1] != prefix_shape[1] + feature_shape[1]
            or output.logical_byte_size
            != prefix.logical_byte_size + feature.logical_byte_size
        ):
            continue
        candidates[operator.operator_id] = (operator, prefix, feature, output)
        by_output[output.tensor_id] = operator.operator_id
        successors.setdefault(prefix.tensor_id, []).append(operator.operator_id)

    roots = [
        operator_id
        for operator_id, (_, prefix, _, _) in candidates.items()
        if prefix.tensor_id not in by_output
    ]
    blocks: list[CompiledDenseSlabBlock] = []
    for root in roots:
        chain: list[tuple[object, object, object, object]] = []
        current = root
        while True:
            item = candidates[current]
            chain.append(item)
            output = item[3]
            next_items = successors.get(output.tensor_id, [])
            if len(next_items) > 1:
                raise DenseSlabPlanningError(
                    f"compiled prefix Tensor {output.tensor_id} has multiple "
                    "Concat successors"
                )
            if not next_items:
                break
            current = next_items[0]
        if len(chain) < minimum_chain_length:
            continue
        blocks.append(
            CompiledDenseSlabBlock(
                backing_tensor_id=chain[0][1].tensor_id,
                concat_operator_ids=tuple(item[0].operator_id for item in chain),
                feature_tensor_ids=tuple(item[2].tensor_id for item in chain),
                prefix_tensor_ids=tuple(item[3].tensor_id for item in chain),
                slab_shape=tuple(
                    chain[-1][3].dimensions[: chain[-1][3].rank]
                ),
            )
        )
    return tuple(blocks)


def _rebuild_links(
    tensors: tuple[RuntimeTensor, ...], operators: tuple[RuntimeOperator, ...]
) -> tuple[RuntimeTensor, ...]:
    producers: dict[int, int] = {}
    consumers: dict[int, list[int]] = {tensor.tensor_id: [] for tensor in tensors}
    for operator in operators:
        for tensor_id in operator.input_tensor_ids:
            bucket = consumers[tensor_id]
            if not bucket or bucket[-1] != operator.operator_id:
                bucket.append(operator.operator_id)
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


def rewrite_dense_concats_as_slabs(
    graph: RuntimeGraph,
    *,
    minimum_chain_length: int = 2,
    expected_block_count: int | None = None,
    expected_concat_count: int | None = None,
) -> DenseSlabRewriteResult:
    """Remove cumulative channel Concat operators and create direct slab views.

    The first prefix Tensor in each chain remains an arena-owned backing Tensor.
    Its storage span is enlarged to the final prefix size.  Every feature output
    becomes a direct slice view and every removed Concat output becomes a prefix
    view at offset zero.  Tensor IDs and names remain stable; only operator IDs
    after removed Concat instructions are compacted.
    """

    if minimum_chain_length < 2:
        raise DenseSlabPlanningError("minimum_chain_length must be at least 2")
    chains = _find_chains(graph, minimum_chain_length=minimum_chain_length)
    concat_count = sum(len(chain) for chain in chains)
    if expected_block_count is not None and len(chains) != expected_block_count:
        raise DenseSlabPlanningError(
            f"expected {expected_block_count} Dense blocks, found {len(chains)}"
        )
    if expected_concat_count is not None and concat_count != expected_concat_count:
        raise DenseSlabPlanningError(
            f"expected {expected_concat_count} Dense Concats, found {concat_count}"
        )
    if not chains:
        raise DenseSlabPlanningError("no cumulative Dense Concat chain was found")

    tensor_overrides: dict[int, RuntimeTensor] = {}
    removed_operator_ids: set[int] = set()
    blocks: list[DenseSlabBlock] = []
    for block_index, chain in enumerate(chains, start=1):
        base = chain[0].prefix_input
        final_prefix = chain[-1].output
        if base.storage_type not in (
            TensorStorageType.ACTIVATION,
            TensorStorageType.OUTPUT,
        ):
            raise DenseSlabPlanningError(
                f"Dense block {block_index} backing Tensor {base.name!r} is "
                f"{base.storage_type.name}, not arena-owned"
            )
        if base.tensor_id in tensor_overrides:
            raise DenseSlabPlanningError(
                f"Tensor {base.tensor_id} is the backing storage of multiple blocks"
            )

        tensor_overrides[base.tensor_id] = replace(
            base, storage_span_bytes=final_prefix.byte_size
        )
        feature_offsets: list[int] = []
        for item in chain:
            feature_offset = item.prefix_input.byte_size
            feature_offsets.append(feature_offset)
            tensor_overrides[item.feature_input.tensor_id] = replace(
                item.feature_input,
                storage_type=TensorStorageType.VIEW,
                storage_span_bytes=item.feature_input.byte_size,
                alias_of_tensor_id=base.tensor_id,
                view_byte_offset=feature_offset,
            )
            tensor_overrides[item.output.tensor_id] = replace(
                item.output,
                storage_type=TensorStorageType.VIEW,
                storage_span_bytes=item.output.byte_size,
                alias_of_tensor_id=base.tensor_id,
                view_byte_offset=0,
            )
            removed_operator_ids.add(item.operator.operator_id)

        blocks.append(
            DenseSlabBlock(
                block_index=block_index,
                backing_tensor_id=base.tensor_id,
                backing_tensor_name=base.name,
                slab_shape=final_prefix.shape,
                slab_bytes=final_prefix.byte_size,
                concat_operator_ids=tuple(
                    item.operator.operator_id for item in chain
                ),
                concat_operator_names=tuple(item.operator.name for item in chain),
                feature_tensor_ids=tuple(
                    item.feature_input.tensor_id for item in chain
                ),
                prefix_tensor_ids=tuple(item.output.tensor_id for item in chain),
                feature_offsets=tuple(feature_offsets),
                copied_bytes_before=sum(item.output.byte_size for item in chain),
            )
        )

    operators = tuple(
        replace(operator, operator_id=new_id)
        for new_id, operator in enumerate(
            operator
            for operator in graph.operators
            if operator.operator_id not in removed_operator_ids
        )
    )
    tensors = tuple(
        tensor_overrides.get(tensor.tensor_id, tensor) for tensor in graph.tensors
    )
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
    return DenseSlabRewriteResult(
        graph=rewritten,
        blocks=tuple(blocks),
        original_operator_count=len(graph.operators),
    )


__all__ = [
    "CompiledDenseSlabBlock",
    "DenseSlabBlock",
    "DenseSlabPlanningError",
    "DenseSlabRewriteResult",
    "inspect_compiled_dense_concats",
    "rewrite_dense_concats_as_slabs",
]
