"""Rewrite safe CAM++ operator subgraphs as independently selectable kernels.

The pass runs after Dense-slab and cache-layout rewriting.  It never patches an
existing binary plan: operators and dead intermediate tensors are rebuilt,
then the normal arena and execution-plan writers regenerate lifetimes, aliases,
offsets and checksums from the resulting graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Iterable, Mapping

from ..format.binary_format_schema import OperatorCode, TensorStorageType
from ..runtime_ir import (
    RuntimeGraph,
    RuntimeInitializer,
    RuntimeOperator,
    RuntimeTensor,
)


FUSION_BN_RELU_QUANT_KERNEL_ID = 2
FUSION_QUANT_QCONV_KERNEL_ID = 3
FUSION_EPILOGUE_KERNEL_ID = 4
FUSION_STATS_POOLING_KERNEL_ID = 5
FUSION_QDQ_ELEMENTWISE_KERNEL_ID = 6


class FusionPlanningError(ValueError):
    """The requested graph fusion is ambiguous or violates plan constraints."""


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Every family is independent so exporter A/B plans are reproducible."""

    conv_bias_act: bool = True
    bn_relu_quant: bool = True
    pool_cam: bool = True
    qdq_elementwise: bool = True
    stats_pooling: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "FUSION_CONV_BIAS_ACT": self.conv_bias_act,
            "FUSION_BN_RELU_QUANT": self.bn_relu_quant,
            "FUSION_POOL_CAM": self.pool_cam,
            "FUSION_QDQ_ELEMENTWISE": self.qdq_elementwise,
            "FUSION_STATS_POOLING": self.stats_pooling,
        }


@dataclass(frozen=True, slots=True)
class FusionRecord:
    family: str
    pattern: str
    original_operator_ids: tuple[int, ...]
    original_operator_names: tuple[str, ...]
    fused_operator_id: int
    fused_operator_name: str
    kernel_id: int
    eliminated_tensor_ids: tuple[int, ...]
    eliminated_tensor_names: tuple[str, ...]
    tensor_read_bytes_before: int
    tensor_write_bytes_before: int
    tensor_read_bytes_after: int
    tensor_write_bytes_after: int
    scratch_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "pattern": self.pattern,
            "original_operator_ids": list(self.original_operator_ids),
            "original_operator_names": list(self.original_operator_names),
            "fused_operator_id": self.fused_operator_id,
            "fused_operator_name": self.fused_operator_name,
            "kernel_id": self.kernel_id,
            "eliminated_tensor_ids": list(self.eliminated_tensor_ids),
            "eliminated_tensor_names": list(self.eliminated_tensor_names),
            "tensor_read_bytes_before": self.tensor_read_bytes_before,
            "tensor_write_bytes_before": self.tensor_write_bytes_before,
            "tensor_read_bytes_after": self.tensor_read_bytes_after,
            "tensor_write_bytes_after": self.tensor_write_bytes_after,
            "scratch_bytes": self.scratch_bytes,
        }


@dataclass(frozen=True, slots=True)
class FusionRewriteResult:
    graph: RuntimeGraph
    config: FusionConfig
    kernel_ids: Mapping[int, int]
    records: tuple[FusionRecord, ...]
    tensor_id_map: Mapping[int, int]
    original_operator_count: int
    original_tensor_count: int
    original_tensor_read_bytes: int
    original_tensor_write_bytes: int
    original_explicit_copy_bytes: int
    optimized_tensor_read_bytes: int
    optimized_tensor_write_bytes: int
    optimized_explicit_copy_bytes: int
    maximum_scratch_bytes: int
    batch_norm_folded_count: int = 0

    @property
    def removed_operator_count(self) -> int:
        return self.original_operator_count - len(self.graph.operators)

    @property
    def eliminated_tensor_count(self) -> int:
        return self.original_tensor_count - len(self.graph.tensors)

    def to_dict(self) -> dict[str, object]:
        family_counts: dict[str, int] = {}
        pattern_counts: dict[str, int] = {}
        for record in self.records:
            family_counts[record.family] = family_counts.get(record.family, 0) + 1
            pattern_counts[record.pattern] = pattern_counts.get(record.pattern, 0) + 1
        return {
            "bucket_frames": self.graph.bucket_frames,
            "flags": self.config.to_dict(),
            "original_operator_count": self.original_operator_count,
            "optimized_operator_count": len(self.graph.operators),
            "removed_operator_count": self.removed_operator_count,
            "original_tensor_count": self.original_tensor_count,
            "optimized_tensor_count": len(self.graph.tensors),
            "eliminated_tensor_count": self.eliminated_tensor_count,
            "batch_norm_folded_count": self.batch_norm_folded_count,
            "batch_norm_fold_reason": (
                "No hybrid-graph BatchNorm is safe to fold into quantized weights "
                "without changing an existing quantization boundary."
            ),
            "family_counts": family_counts,
            "pattern_counts": pattern_counts,
            "tensor_read_bytes_before": self.original_tensor_read_bytes,
            "tensor_read_bytes_after": self.optimized_tensor_read_bytes,
            "tensor_read_bytes_saved": (
                self.original_tensor_read_bytes - self.optimized_tensor_read_bytes
            ),
            "tensor_write_bytes_before": self.original_tensor_write_bytes,
            "tensor_write_bytes_after": self.optimized_tensor_write_bytes,
            "tensor_write_bytes_saved": (
                self.original_tensor_write_bytes - self.optimized_tensor_write_bytes
            ),
            "explicit_copy_bytes_before": self.original_explicit_copy_bytes,
            "explicit_copy_bytes_after": self.optimized_explicit_copy_bytes,
            "maximum_scratch_bytes": self.maximum_scratch_bytes,
            "kernel_ids": {
                str(operator_id): kernel_id
                for operator_id, kernel_id in self.kernel_ids.items()
            },
            "tensor_id_map": {
                str(old_id): new_id for old_id, new_id in self.tensor_id_map.items()
            },
            "fusions": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class _Group:
    family: str
    pattern: str
    operator_ids: tuple[int, ...]
    input_tensor_ids: tuple[int, ...]
    output_tensor_id: int
    opcode: OperatorCode
    attributes: Mapping[str, object]
    kernel_id: int
    eliminated_tensor_ids: tuple[int, ...]
    scratch_bytes: int = 0

    @property
    def insertion_operator_id(self) -> int:
        return min(self.operator_ids)


_COPY_OPS = frozenset(
    {
        OperatorCode.CONCAT,
        OperatorCode.EXPAND,
        OperatorCode.RESHAPE,
        OperatorCode.TRANSPOSE,
    }
)


def _element_count(tensor: RuntimeTensor) -> int:
    return math.prod(tensor.shape)


def _is_scalar(tensor: RuntimeTensor) -> bool:
    return _element_count(tensor) == 1


def _traffic(
    graph: RuntimeGraph, operators: Iterable[RuntimeOperator]
) -> tuple[int, int, int]:
    read_bytes = 0
    write_bytes = 0
    copy_bytes = 0
    for operator in operators:
        read_bytes += sum(
            graph.tensor(tensor_id).byte_size
            for tensor_id in operator.input_tensor_ids
        )
        output_bytes = sum(
            graph.tensor(tensor_id).byte_size
            for tensor_id in operator.output_tensor_ids
        )
        write_bytes += output_bytes
        if operator.opcode in _COPY_OPS:
            copy_bytes += output_bytes
    return read_bytes, write_bytes, copy_bytes


def _sole_consumer(
    graph: RuntimeGraph, operator: RuntimeOperator, opcode: OperatorCode
) -> RuntimeOperator | None:
    output = graph.tensor(operator.output_tensor_ids[0])
    if len(output.consumers) != 1:
        return None
    consumer = graph.operator(output.consumers[0])
    return consumer if consumer.opcode is opcode else None


def _producer(
    graph: RuntimeGraph, tensor_id: int, opcode: OperatorCode
) -> RuntimeOperator | None:
    producer_id = graph.tensor(tensor_id).producer
    if producer_id is None:
        return None
    operator = graph.operator(producer_id)
    return operator if operator.opcode is opcode else None


def _internal_outputs(
    graph: RuntimeGraph, operator_ids: tuple[int, ...], final_tensor_id: int
) -> tuple[int, ...] | None:
    group = set(operator_ids)
    outputs = tuple(
        tensor_id
        for operator_id in operator_ids
        for tensor_id in graph.operator(operator_id).output_tensor_ids
        if tensor_id != final_tensor_id
    )
    aliases = {
        tensor.alias_of_tensor_id
        for tensor in graph.tensors
        if tensor.storage_type is TensorStorageType.VIEW
    }
    for tensor_id in outputs:
        tensor = graph.tensor(tensor_id)
        if tensor_id in aliases or any(item not in group for item in tensor.consumers):
            return None
    return outputs


def _group_available_at_insertion(graph: RuntimeGraph, group: _Group) -> bool:
    insertion = group.insertion_operator_id
    internal_outputs = {
        tensor_id
        for operator_id in group.operator_ids
        for tensor_id in graph.operator(operator_id).output_tensor_ids
    }
    for tensor_id in group.input_tensor_ids:
        if tensor_id in internal_outputs:
            return False
        producer = graph.tensor(tensor_id).producer
        if producer is not None and producer >= insertion:
            return False
    return True


def _bn_relu_quant_groups(graph: RuntimeGraph) -> list[_Group]:
    result: list[_Group] = []
    for bn in graph.operators:
        if bn.opcode is not OperatorCode.BATCH_NORMALIZATION or len(bn.input_tensor_ids) != 5:
            continue
        relu = _sole_consumer(graph, bn, OperatorCode.RELU)
        quant = _sole_consumer(graph, relu, OperatorCode.QUANTIZE_LINEAR) if relu else None
        if quant is None or len(quant.input_tensor_ids) != 3:
            continue
        if not all(_is_scalar(graph.tensor(item)) for item in quant.input_tensor_ids[1:]):
            continue
        ids = (bn.operator_id, relu.operator_id, quant.operator_id)
        dead = _internal_outputs(graph, ids, quant.output_tensor_ids[0])
        if dead is None:
            continue
        attributes = dict(bn.attributes)
        if "axis" in quant.attributes:
            attributes["axis"] = quant.attributes["axis"]
        group = _Group(
            family="FUSION_BN_RELU_QUANT",
            pattern="BatchNormalization-Relu-QuantizeLinear",
            operator_ids=ids,
            input_tensor_ids=bn.input_tensor_ids + quant.input_tensor_ids[1:],
            output_tensor_id=quant.output_tensor_ids[0],
            opcode=OperatorCode.BATCH_NORMALIZATION,
            attributes=attributes,
            kernel_id=FUSION_BN_RELU_QUANT_KERNEL_ID,
            eliminated_tensor_ids=dead,
        )
        if _group_available_at_insertion(graph, group):
            result.append(group)
    return result


def _dq_relu_quant_groups(graph: RuntimeGraph) -> list[_Group]:
    result: list[_Group] = []
    for dq in graph.operators:
        if dq.opcode is not OperatorCode.DEQUANTIZE_LINEAR:
            continue
        relu = _sole_consumer(graph, dq, OperatorCode.RELU)
        quant = _sole_consumer(graph, relu, OperatorCode.QUANTIZE_LINEAR) if relu else None
        if quant is None or len(dq.input_tensor_ids) != 3 or len(quant.input_tensor_ids) != 3:
            continue
        parameters = dq.input_tensor_ids[1:] + quant.input_tensor_ids[1:]
        if not all(_is_scalar(graph.tensor(item)) for item in parameters):
            continue
        ids = (dq.operator_id, relu.operator_id, quant.operator_id)
        dead = _internal_outputs(graph, ids, quant.output_tensor_ids[0])
        if dead is None:
            continue
        group = _Group(
            family="FUSION_CONV_BIAS_ACT",
            pattern="DequantizeLinear-Relu-QuantizeLinear",
            operator_ids=ids,
            input_tensor_ids=dq.input_tensor_ids + quant.input_tensor_ids[1:],
            output_tensor_id=quant.output_tensor_ids[0],
            opcode=OperatorCode.DEQUANTIZE_LINEAR,
            attributes=quant.attributes,
            kernel_id=FUSION_EPILOGUE_KERNEL_ID,
            eliminated_tensor_ids=dead,
        )
        if _group_available_at_insertion(graph, group):
            result.append(group)
    return result


def _quant_qconv_groups(graph: RuntimeGraph) -> list[_Group]:
    result: list[_Group] = []
    for quant in graph.operators:
        if quant.opcode is not OperatorCode.QUANTIZE_LINEAR or len(quant.input_tensor_ids) != 3:
            continue
        conv = _sole_consumer(graph, quant, OperatorCode.QLINEAR_CONV)
        if conv is None or len(conv.input_tensor_ids) not in (8, 9):
            continue
        if tuple(conv.input_tensor_ids[1:3]) != tuple(quant.input_tensor_ids[1:3]):
            continue
        if not all(_is_scalar(graph.tensor(item)) for item in quant.input_tensor_ids[1:]):
            continue
        ids = (quant.operator_id, conv.operator_id)
        dead = _internal_outputs(graph, ids, conv.output_tensor_ids[0])
        if dead is None:
            continue
        source = graph.tensor(quant.input_tensor_ids[0])
        if source.storage_span_bytes is None or source.storage_span_bytes % 4:
            continue
        if any(stride % 4 for stride in source.strides):
            continue
        group = _Group(
            family="FUSION_CONV_BIAS_ACT",
            pattern="QuantizeLinear-QLinearConvInputPack",
            operator_ids=ids,
            input_tensor_ids=(quant.input_tensor_ids[0],) + conv.input_tensor_ids[1:],
            output_tensor_id=conv.output_tensor_ids[0],
            opcode=OperatorCode.QLINEAR_CONV,
            attributes=conv.attributes,
            kernel_id=FUSION_QUANT_QCONV_KERNEL_ID,
            eliminated_tensor_ids=dead,
            scratch_bytes=source.storage_span_bytes // 4,
        )
        if _group_available_at_insertion(graph, group):
            result.append(group)
    return result


def _dq_sigmoid_mul_groups(graph: RuntimeGraph) -> list[_Group]:
    result: list[_Group] = []
    for dq in graph.operators:
        if dq.opcode is not OperatorCode.DEQUANTIZE_LINEAR or len(dq.input_tensor_ids) != 3:
            continue
        sigmoid = _sole_consumer(graph, dq, OperatorCode.SIGMOID)
        mul = _sole_consumer(graph, sigmoid, OperatorCode.MUL) if sigmoid else None
        if mul is None or len(mul.input_tensor_ids) != 2:
            continue
        other = next(
            (item for item in mul.input_tensor_ids if item != sigmoid.output_tensor_ids[0]),
            None,
        )
        if other is None or not all(_is_scalar(graph.tensor(item)) for item in dq.input_tensor_ids[1:]):
            continue
        ids = (dq.operator_id, sigmoid.operator_id, mul.operator_id)
        dead = _internal_outputs(graph, ids, mul.output_tensor_ids[0])
        if dead is None:
            continue
        group = _Group(
            family="FUSION_POOL_CAM",
            pattern="DequantizeLinear-Sigmoid-Mul",
            operator_ids=ids,
            input_tensor_ids=dq.input_tensor_ids + (other,),
            output_tensor_id=mul.output_tensor_ids[0],
            opcode=OperatorCode.MUL,
            attributes={},
            kernel_id=FUSION_EPILOGUE_KERNEL_ID,
            eliminated_tensor_ids=dead,
        )
        if _group_available_at_insertion(graph, group):
            result.append(group)
    return result


def _qdq_elementwise_groups(graph: RuntimeGraph) -> list[_Group]:
    result: list[_Group] = []
    for dq in graph.operators:
        if dq.opcode is not OperatorCode.DEQUANTIZE_LINEAR or len(dq.input_tensor_ids) != 3:
            continue
        output = graph.tensor(dq.output_tensor_ids[0])
        if len(output.consumers) != 1:
            continue
        binary = graph.operator(output.consumers[0])
        if binary.opcode not in (OperatorCode.ADD, OperatorCode.MUL) or len(binary.input_tensor_ids) != 2:
            continue
        quant = _sole_consumer(graph, binary, OperatorCode.QUANTIZE_LINEAR)
        if quant is None or len(quant.input_tensor_ids) != 3:
            continue
        other = next(
            (item for item in binary.input_tensor_ids if item != dq.output_tensor_ids[0]),
            None,
        )
        parameters = dq.input_tensor_ids[1:] + quant.input_tensor_ids[1:]
        if other is None or not all(_is_scalar(graph.tensor(item)) for item in parameters):
            continue
        ids = (dq.operator_id, binary.operator_id, quant.operator_id)
        dead = _internal_outputs(graph, ids, quant.output_tensor_ids[0])
        if dead is None:
            continue
        group = _Group(
            family="FUSION_QDQ_ELEMENTWISE",
            pattern=f"DequantizeLinear-{binary.opcode.name}-QuantizeLinear",
            operator_ids=ids,
            input_tensor_ids=dq.input_tensor_ids + (other,) + quant.input_tensor_ids[1:],
            output_tensor_id=quant.output_tensor_ids[0],
            opcode=binary.opcode,
            attributes={},
            kernel_id=FUSION_QDQ_ELEMENTWISE_KERNEL_ID,
            eliminated_tensor_ids=dead,
        )
        if _group_available_at_insertion(graph, group):
            result.append(group)
    return result


def _other_input(operator: RuntimeOperator, tensor_id: int) -> int | None:
    if len(operator.input_tensor_ids) != 2:
        return None
    if operator.input_tensor_ids[0] == tensor_id:
        return operator.input_tensor_ids[1]
    if operator.input_tensor_ids[1] == tensor_id:
        return operator.input_tensor_ids[0]
    return None


def _stats_pooling_groups(graph: RuntimeGraph) -> list[_Group]:
    result: list[_Group] = []
    for concat in graph.operators:
        if concat.opcode is not OperatorCode.CONCAT or len(concat.input_tensor_ids) != 2:
            continue
        mean = _producer(graph, concat.input_tensor_ids[0], OperatorCode.REDUCE_MEAN)
        sqrt = _producer(graph, concat.input_tensor_ids[1], OperatorCode.SQRT)
        if mean is None or sqrt is None:
            continue
        div = _producer(graph, sqrt.input_tensor_ids[0], OperatorCode.DIV)
        scaled = _producer(graph, div.input_tensor_ids[0], OperatorCode.MUL) if div else None
        variance = _producer(graph, scaled.input_tensor_ids[0], OperatorCode.REDUCE_MEAN) if scaled else None
        square = _producer(graph, variance.input_tensor_ids[0], OperatorCode.MUL) if variance else None
        sub = _producer(graph, square.input_tensor_ids[0], OperatorCode.SUB) if square else None
        if None in (div, scaled, variance, square, sub):
            continue
        assert div is not None and scaled is not None and variance is not None
        assert square is not None and sub is not None
        if square.input_tensor_ids[0] != square.input_tensor_ids[1]:
            continue
        centered = square.input_tensor_ids[0]
        if sub.output_tensor_ids[0] != centered:
            continue
        source = sub.input_tensor_ids[0]
        kept_mean = _producer(graph, sub.input_tensor_ids[1], OperatorCode.REDUCE_MEAN)
        if kept_mean is None or mean.input_tensor_ids[0] != source or kept_mean.input_tensor_ids[0] != source:
            continue
        mul_constant = _other_input(scaled, variance.output_tensor_ids[0])
        div_constant = _other_input(div, scaled.output_tensor_ids[0])
        if mul_constant is None or div_constant is None:
            continue
        if not (_is_scalar(graph.tensor(mul_constant)) and _is_scalar(graph.tensor(div_constant))):
            continue
        ids = tuple(
            sorted(
                {
                    mean.operator_id,
                    kept_mean.operator_id,
                    sub.operator_id,
                    square.operator_id,
                    variance.operator_id,
                    scaled.operator_id,
                    div.operator_id,
                    sqrt.operator_id,
                    concat.operator_id,
                }
            )
        )
        if ids != tuple(range(ids[0], ids[-1] + 1)):
            continue
        dead = _internal_outputs(graph, ids, concat.output_tensor_ids[0])
        if dead is None:
            continue
        group = _Group(
            family="FUSION_STATS_POOLING",
            pattern="StatisticsPoolingMeanStd",
            operator_ids=ids,
            input_tensor_ids=(source, mul_constant, div_constant),
            output_tensor_id=concat.output_tensor_ids[0],
            opcode=OperatorCode.CONCAT,
            attributes={},
            kernel_id=FUSION_STATS_POOLING_KERNEL_ID,
            eliminated_tensor_ids=dead,
        )
        if _group_available_at_insertion(graph, group):
            result.append(group)
    return result


def _select_groups(graph: RuntimeGraph, config: FusionConfig) -> tuple[_Group, ...]:
    candidates: list[_Group] = []
    if config.stats_pooling:
        candidates.extend(_stats_pooling_groups(graph))
    if config.pool_cam:
        candidates.extend(_dq_sigmoid_mul_groups(graph))
    if config.bn_relu_quant:
        candidates.extend(_bn_relu_quant_groups(graph))
    if config.qdq_elementwise:
        candidates.extend(_qdq_elementwise_groups(graph))
    if config.conv_bias_act:
        candidates.extend(_dq_relu_quant_groups(graph))
        candidates.extend(_quant_qconv_groups(graph))

    selected: list[_Group] = []
    used: set[int] = set()
    for candidate in candidates:
        if used.isdisjoint(candidate.operator_ids):
            selected.append(candidate)
            used.update(candidate.operator_ids)
    return tuple(sorted(selected, key=lambda item: item.insertion_operator_id))


def _compact_graph(
    graph: RuntimeGraph, operators: tuple[RuntimeOperator, ...]
) -> tuple[RuntimeGraph, Mapping[int, int]]:
    referenced = {
        tensor_id
        for operator in operators
        for tensor_id in (*operator.input_tensor_ids, *operator.output_tensor_ids)
    }
    referenced.update(graph.input_tensor_ids or ())
    referenced.update(graph.output_tensor_ids or ())
    referenced.update(
        tensor.tensor_id
        for tensor in graph.tensors
        if tensor.storage_type is TensorStorageType.CONSTANT
    )
    changed = True
    while changed:
        changed = False
        for tensor in graph.tensors:
            if tensor.tensor_id not in referenced or tensor.storage_type is not TensorStorageType.VIEW:
                continue
            assert tensor.alias_of_tensor_id is not None
            if tensor.alias_of_tensor_id not in referenced:
                referenced.add(tensor.alias_of_tensor_id)
                changed = True

    retained = tuple(tensor for tensor in graph.tensors if tensor.tensor_id in referenced)
    tensor_map = MappingProxyType(
        {tensor.tensor_id: new_id for new_id, tensor in enumerate(retained)}
    )
    remapped_operators = tuple(
        replace(
            operator,
            operator_id=new_id,
            input_tensor_ids=tuple(tensor_map[item] for item in operator.input_tensor_ids),
            output_tensor_ids=tuple(tensor_map[item] for item in operator.output_tensor_ids),
        )
        for new_id, operator in enumerate(operators)
    )

    producers: dict[int, int] = {}
    consumers: dict[int, list[int]] = {tensor_map[item.tensor_id]: [] for item in retained}
    for operator in remapped_operators:
        for tensor_id in operator.input_tensor_ids:
            bucket = consumers[tensor_id]
            if not bucket or bucket[-1] != operator.operator_id:
                bucket.append(operator.operator_id)
        for tensor_id in operator.output_tensor_ids:
            producers[tensor_id] = operator.operator_id
    tensors = tuple(
        replace(
            tensor,
            tensor_id=tensor_map[tensor.tensor_id],
            producer=producers.get(tensor_map[tensor.tensor_id]),
            consumers=tuple(consumers[tensor_map[tensor.tensor_id]]),
            alias_of_tensor_id=(
                tensor_map[tensor.alias_of_tensor_id]
                if tensor.alias_of_tensor_id is not None
                else None
            ),
        )
        for tensor in retained
    )
    initializers = tuple(
        replace(initializer, tensor_id=tensor_map[initializer.tensor_id])
        for initializer in graph.initializers
    )
    rewritten = RuntimeGraph(
        tensors=tensors,
        operators=remapped_operators,
        initializers=initializers,
        input_tensor_ids=tuple(tensor_map[item] for item in graph.input_tensor_ids or ()),
        output_tensor_ids=tuple(tensor_map[item] for item in graph.output_tensor_ids or ()),
        name=graph.name,
        bucket_frames=graph.bucket_frames,
    )
    return rewritten, tensor_map


def rewrite_operator_fusions(
    graph: RuntimeGraph, *, config: FusionConfig | None = None,
    default_kernel_id: int = 1,
) -> FusionRewriteResult:
    """Apply non-overlapping safe fusions and compact dead Tensor IDs."""

    selected_config = config or FusionConfig()
    groups = _select_groups(graph, selected_config)
    by_start = {group.insertion_operator_id: group for group in groups}
    removed = {operator_id for group in groups for operator_id in group.operator_ids}
    provisional: list[RuntimeOperator] = []
    provisional_kernel_ids: list[int] = []
    group_for_new_id: dict[int, _Group] = {}
    for operator in graph.operators:
        group = by_start.get(operator.operator_id)
        if group is not None:
            new_id = len(provisional)
            provisional.append(
                RuntimeOperator(
                    name=f"fused/{group.pattern}/{group.insertion_operator_id}",
                    operator_id=new_id,
                    opcode=group.opcode,
                    input_tensor_ids=group.input_tensor_ids,
                    output_tensor_ids=(group.output_tensor_id,),
                    attributes=group.attributes,
                )
            )
            provisional_kernel_ids.append(group.kernel_id)
            group_for_new_id[new_id] = group
        elif operator.operator_id not in removed:
            provisional.append(replace(operator, operator_id=len(provisional)))
            provisional_kernel_ids.append(default_kernel_id)

    compacted, tensor_map = _compact_graph(graph, tuple(provisional))
    kernel_ids = MappingProxyType(
        {operator_id: value for operator_id, value in enumerate(provisional_kernel_ids)}
    )
    records: list[FusionRecord] = []
    for fused_id, group in group_for_new_id.items():
        original_ops = tuple(graph.operator(item) for item in group.operator_ids)
        before_read, before_write, _ = _traffic(graph, original_ops)
        after_read = sum(graph.tensor(item).byte_size for item in group.input_tensor_ids)
        after_write = graph.tensor(group.output_tensor_id).byte_size
        records.append(
            FusionRecord(
                family=group.family,
                pattern=group.pattern,
                original_operator_ids=group.operator_ids,
                original_operator_names=tuple(item.name for item in original_ops),
                fused_operator_id=fused_id,
                fused_operator_name=compacted.operator(fused_id).name,
                kernel_id=group.kernel_id,
                eliminated_tensor_ids=group.eliminated_tensor_ids,
                eliminated_tensor_names=tuple(
                    graph.tensor(item).name for item in group.eliminated_tensor_ids
                ),
                tensor_read_bytes_before=before_read,
                tensor_write_bytes_before=before_write,
                tensor_read_bytes_after=after_read,
                tensor_write_bytes_after=after_write,
                scratch_bytes=group.scratch_bytes,
            )
        )
    original_read, original_write, original_copy = _traffic(graph, graph.operators)
    optimized_read, optimized_write, _ = _traffic(
        compacted, compacted.operators
    )
    optimized_copy = sum(
        sum(compacted.tensor(item).byte_size for item in operator.output_tensor_ids)
        for operator in compacted.operators
        if operator.opcode in _COPY_OPS
        and kernel_ids[operator.operator_id] == default_kernel_id
    )
    return FusionRewriteResult(
        graph=compacted,
        config=selected_config,
        kernel_ids=kernel_ids,
        records=tuple(records),
        tensor_id_map=tensor_map,
        original_operator_count=len(graph.operators),
        original_tensor_count=len(graph.tensors),
        original_tensor_read_bytes=original_read,
        original_tensor_write_bytes=original_write,
        original_explicit_copy_bytes=original_copy,
        optimized_tensor_read_bytes=optimized_read,
        optimized_tensor_write_bytes=optimized_write,
        optimized_explicit_copy_bytes=optimized_copy,
        maximum_scratch_bytes=max((item.scratch_bytes for item in groups), default=0),
    )


__all__ = [
    "FUSION_BN_RELU_QUANT_KERNEL_ID",
    "FUSION_EPILOGUE_KERNEL_ID",
    "FUSION_QDQ_ELEMENTWISE_KERNEL_ID",
    "FUSION_QUANT_QCONV_KERNEL_ID",
    "FUSION_STATS_POOLING_KERNEL_ID",
    "FusionConfig",
    "FusionPlanningError",
    "FusionRecord",
    "FusionRewriteResult",
    "rewrite_operator_fusions",
]
