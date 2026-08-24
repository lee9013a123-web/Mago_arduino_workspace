"""Offline O4I4 packing for the actual INT8 QLinearConv graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ..format.binary_format_schema import OperatorCode, TensorDType
from ..runtime_ir import RuntimeGraph, RuntimeInitializer, RuntimeTensor


OUTPUT_BLOCK = 4
INPUT_BLOCK = 4


class WeightPackingError(ValueError):
    """QLinearConv weights or quantization metadata cannot be packed safely."""


@dataclass(frozen=True, slots=True)
class PackedWeightRecord:
    tensor_id: int
    name: str
    logical_shape: tuple[int, ...]
    group: int
    original_bytes: int
    packed_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "name": self.name,
            "logical_shape": list(self.logical_shape),
            "layout": "G_O4_K_I4_OL_IL",
            "group": self.group,
            "output_block": OUTPUT_BLOCK,
            "input_block": INPUT_BLOCK,
            "original_bytes": self.original_bytes,
            "packed_bytes": self.packed_bytes,
            "padding_bytes": self.packed_bytes - self.original_bytes,
        }


@dataclass(frozen=True, slots=True)
class WeightPackingResult:
    graph: RuntimeGraph
    records: tuple[PackedWeightRecord, ...]
    qlinear_conv_count: int

    @property
    def original_bytes(self) -> int:
        return sum(record.original_bytes for record in self.records)

    @property
    def packed_bytes(self) -> int:
        return sum(record.packed_bytes for record in self.records)

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket_frames": self.graph.bucket_frames,
            "layout": "[group][output_block][kernel][input_block]"
            "[output_lane][input_lane]",
            "output_block": OUTPUT_BLOCK,
            "input_block": INPUT_BLOCK,
            "qlinear_conv_count": self.qlinear_conv_count,
            "packed_weight_count": len(self.records),
            "original_weight_bytes": self.original_bytes,
            "packed_weight_bytes": self.packed_bytes,
            "weight_padding_bytes": self.packed_bytes - self.original_bytes,
            "runtime_packing_count": 0,
            "weights": [record.to_dict() for record in self.records],
        }


def _group(attributes: object) -> int:
    value = attributes.get("group", 1)
    if isinstance(value, tuple):
        if len(value) != 1:
            raise WeightPackingError(f"invalid group attribute: {value!r}")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WeightPackingError(f"invalid group attribute: {value!r}")
    return value


def _quantized_values(initializer: RuntimeInitializer) -> tuple[int, ...]:
    if initializer.dtype is TensorDType.UINT8:
        return tuple(initializer.raw_data)
    if initializer.dtype is TensorDType.INT8:
        return tuple(value if value < 128 else value - 256 for value in initializer.raw_data)
    raise WeightPackingError(
        f"zero point {initializer.name!r} is not INT8/UINT8"
    )


def _stored_byte(value: int, dtype: TensorDType) -> int:
    if dtype is TensorDType.UINT8:
        if not 0 <= value <= 255:
            raise WeightPackingError(f"UINT8 padding value out of range: {value}")
        return value
    if dtype is TensorDType.INT8:
        if not -128 <= value <= 127:
            raise WeightPackingError(f"INT8 padding value out of range: {value}")
        return value & 0xFF
    raise WeightPackingError(f"unsupported packed dtype: {dtype.name}")


def pack_qconv_o4i4(
    raw_data: bytes,
    shape: tuple[int, ...],
    group: int,
    weight_zero_points: tuple[int, ...],
    dtype: TensorDType,
) -> bytes:
    """Pack ONNX [O,I,K...] bytes into G/O4/K/I4/O-lane/I-lane order."""

    if len(shape) not in (3, 4):
        raise WeightPackingError(f"QConv weight rank must be 3 or 4: {shape}")
    output_channels, input_channels = shape[:2]
    kernel_elements = math.prod(shape[2:])
    if output_channels % group:
        raise WeightPackingError("output channels are not divisible by group")
    if len(raw_data) != math.prod(shape):
        raise WeightPackingError("weight payload does not match logical shape")
    if len(weight_zero_points) not in (1, output_channels):
        raise WeightPackingError(
            "weight zero point must be scalar or per-output-channel"
        )

    outputs_per_group = output_channels // group
    output_blocks = (outputs_per_group + OUTPUT_BLOCK - 1) // OUTPUT_BLOCK
    input_blocks = (input_channels + INPUT_BLOCK - 1) // INPUT_BLOCK
    packed = bytearray(
        group * output_blocks * kernel_elements * input_blocks
        * OUTPUT_BLOCK * INPUT_BLOCK
    )
    cursor = 0
    for group_index in range(group):
        for output_block in range(output_blocks):
            for kernel_index in range(kernel_elements):
                for input_block in range(input_blocks):
                    for output_lane in range(OUTPUT_BLOCK):
                        output_in_group = output_block * OUTPUT_BLOCK + output_lane
                        output_channel = (
                            group_index * outputs_per_group + output_in_group
                        )
                        valid_output = output_in_group < outputs_per_group
                        zero_point = (
                            weight_zero_points[
                                0 if len(weight_zero_points) == 1 else output_channel
                            ]
                            if valid_output
                            else 0
                        )
                        for input_lane in range(INPUT_BLOCK):
                            input_channel = input_block * INPUT_BLOCK + input_lane
                            if valid_output and input_channel < input_channels:
                                logical_index = (
                                    (output_channel * input_channels + input_channel)
                                    * kernel_elements
                                    + kernel_index
                                )
                                packed[cursor] = raw_data[logical_index]
                            else:
                                packed[cursor] = _stored_byte(zero_point, dtype)
                            cursor += 1
    return bytes(packed)


def unpack_qconv_o4i4(
    packed: bytes, shape: tuple[int, ...], group: int
) -> bytes:
    """Restore logical bytes; intended for exporter verification and tests."""

    output_channels, input_channels = shape[:2]
    kernel_elements = math.prod(shape[2:])
    outputs_per_group = output_channels // group
    output_blocks = (outputs_per_group + OUTPUT_BLOCK - 1) // OUTPUT_BLOCK
    input_blocks = (input_channels + INPUT_BLOCK - 1) // INPUT_BLOCK
    expected = (
        group * output_blocks * kernel_elements * input_blocks
        * OUTPUT_BLOCK * INPUT_BLOCK
    )
    if len(packed) != expected:
        raise WeightPackingError(
            f"packed payload is {len(packed)} bytes, expected {expected}"
        )
    restored = bytearray(math.prod(shape))
    cursor = 0
    for group_index in range(group):
        for output_block in range(output_blocks):
            for kernel_index in range(kernel_elements):
                for input_block in range(input_blocks):
                    for output_lane in range(OUTPUT_BLOCK):
                        output_in_group = output_block * OUTPUT_BLOCK + output_lane
                        output_channel = group_index * outputs_per_group + output_in_group
                        for input_lane in range(INPUT_BLOCK):
                            input_channel = input_block * INPUT_BLOCK + input_lane
                            if (
                                output_in_group < outputs_per_group
                                and input_channel < input_channels
                            ):
                                logical_index = (
                                    (output_channel * input_channels + input_channel)
                                    * kernel_elements
                                    + kernel_index
                                )
                                restored[logical_index] = packed[cursor]
                            cursor += 1
    return bytes(restored)


def pack_qlinearconv_weights(graph: RuntimeGraph) -> WeightPackingResult:
    """Pack every unique QLinearConv weight initializer offline."""

    tensors = {tensor.tensor_id: tensor for tensor in graph.tensors}
    initializers = {item.tensor_id: item for item in graph.initializers}
    tensor_overrides: dict[int, RuntimeTensor] = {}
    initializer_overrides: dict[int, RuntimeInitializer] = {}
    records: dict[int, PackedWeightRecord] = {}
    specifications: dict[int, tuple[int, int]] = {}
    qconv_count = 0

    for operator in graph.operators:
        if operator.opcode is not OperatorCode.QLINEAR_CONV:
            continue
        qconv_count += 1
        if len(operator.input_tensor_ids) not in (8, 9):
            raise WeightPackingError(
                f"QLinearConv {operator.operator_id} has invalid input count"
            )
        weight_id = operator.input_tensor_ids[3]
        zero_id = operator.input_tensor_ids[5]
        weight_tensor = tensors[weight_id]
        weight = initializers[weight_id]
        zero = initializers[zero_id]
        group = _group(operator.attributes)
        spec = (group, zero_id)
        if weight_id in specifications:
            if specifications[weight_id] != spec:
                raise WeightPackingError(
                    f"weight Tensor {weight_id} is reused with incompatible metadata"
                )
            continue
        specifications[weight_id] = spec
        if weight_tensor.packed_qconv_o4i4:
            raise WeightPackingError(f"weight {weight.name!r} is already packed")
        zero_points = _quantized_values(zero)
        packed = pack_qconv_o4i4(
            weight.raw_data, weight.shape, group, zero_points, weight.dtype
        )
        if unpack_qconv_o4i4(packed, weight.shape, group) != weight.raw_data:
            raise WeightPackingError(f"pack round-trip failed for {weight.name!r}")
        tensor_overrides[weight_id] = replace(
            weight_tensor,
            storage_span_bytes=len(packed),
            packed_qconv_o4i4=True,
        )
        initializer_overrides[weight_id] = replace(
            weight,
            raw_data=packed,
            storage_span_bytes=len(packed),
            packed_qconv_o4i4=True,
        )
        records[weight_id] = PackedWeightRecord(
            tensor_id=weight_id,
            name=weight.name,
            logical_shape=weight.shape,
            group=group,
            original_bytes=weight.logical_byte_size,
            packed_bytes=len(packed),
        )

    rewritten = RuntimeGraph(
        tensors=tuple(
            tensor_overrides.get(tensor.tensor_id, tensor)
            for tensor in graph.tensors
        ),
        operators=graph.operators,
        initializers=tuple(
            initializer_overrides.get(item.tensor_id, item)
            for item in graph.initializers
        ),
        input_tensor_ids=graph.input_tensor_ids,
        output_tensor_ids=graph.output_tensor_ids,
        name=graph.name,
        bucket_frames=graph.bucket_frames,
    )
    return WeightPackingResult(
        graph=rewritten,
        records=tuple(records[tensor_id] for tensor_id in sorted(records)),
        qlinear_conv_count=qconv_count,
    )


__all__ = [
    "INPUT_BLOCK",
    "OUTPUT_BLOCK",
    "PackedWeightRecord",
    "WeightPackingError",
    "WeightPackingResult",
    "pack_qconv_o4i4",
    "pack_qlinearconv_weights",
    "unpack_qconv_o4i4",
]
