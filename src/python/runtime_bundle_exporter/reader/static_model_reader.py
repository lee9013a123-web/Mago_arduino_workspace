"""bucket별 정적 ONNX를 읽어 Runtime IR로 옮기는 모듈.

``results/static/campp_static_{98,298,498,998}.onnx``는 shape 도메인을 전부
접어 둔 실행 가능한 그래프다. 이 모듈은 그 안의 node, attribute, initializer,
quantization parameter를 ONNX protobuf 밖으로 꺼내고, graph IR과 대조가 끝나면
:class:`~.runtime_ir.RuntimeGraph`로 변환한다.

여기서는 어떤 최적화도 하지 않는다. fusion, 메모리 배치, ID 채번 규칙 변경은
모두 뒤쪽 builder와 writer의 몫이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper

from ..format.binary_format_schema import (
    OPERATOR_INPUT_CAPACITY,
    TENSOR_MAX_RANK,
    OperatorCode,
    TensorDType,
    TensorStorageType,
)
from .graph_ir_reader import DTYPE_BYTE_SIZE, GraphIRView, cross_validate
from ..runtime_ir import (
    AttributeValue,
    InitializerScope,
    RuntimeGraph,
    RuntimeInitializer,
    RuntimeOperator,
    RuntimeTensor,
)


# CAM++ 정적 그래프에 남는 20개 op가 Runtime opcode와 1:1로 대응한다.
ONNX_OP_TO_OPCODE: Mapping[str, OperatorCode] = MappingProxyType(
    {
        "QLinearConv": OperatorCode.QLINEAR_CONV,
        "QuantizeLinear": OperatorCode.QUANTIZE_LINEAR,
        "DequantizeLinear": OperatorCode.DEQUANTIZE_LINEAR,
        "BatchNormalization": OperatorCode.BATCH_NORMALIZATION,
        "Relu": OperatorCode.RELU,
        "Sigmoid": OperatorCode.SIGMOID,
        "AveragePool": OperatorCode.AVERAGE_POOL,
        "ReduceMean": OperatorCode.REDUCE_MEAN,
        "Add": OperatorCode.ADD,
        "Mul": OperatorCode.MUL,
        "Sub": OperatorCode.SUB,
        "Div": OperatorCode.DIV,
        "Sqrt": OperatorCode.SQRT,
        "Concat": OperatorCode.CONCAT,
        "Expand": OperatorCode.EXPAND,
        "Slice": OperatorCode.SLICE,
        "Reshape": OperatorCode.RESHAPE,
        "Transpose": OperatorCode.TRANSPOSE,
        "Squeeze": OperatorCode.SQUEEZE,
        "Unsqueeze": OperatorCode.UNSQUEEZE,
    }
)

ONNX_ELEM_TYPE_TO_DTYPE: Mapping[int, TensorDType] = MappingProxyType(
    {
        TensorProto.FLOAT: TensorDType.FLOAT32,
        TensorProto.FLOAT16: TensorDType.FLOAT16,
        TensorProto.UINT8: TensorDType.UINT8,
        TensorProto.INT8: TensorDType.INT8,
        TensorProto.INT32: TensorDType.INT32,
        TensorProto.INT64: TensorDType.INT64,
        TensorProto.BOOL: TensorDType.BOOL,
    }
)

# raw_data를 만들 때 쓰는 little-endian numpy dtype. C Runtime이 그대로 읽는다.
_NUMPY_DTYPE: Mapping[TensorDType, str] = MappingProxyType(
    {
        TensorDType.FLOAT32: "<f4",
        TensorDType.FLOAT16: "<f2",
        TensorDType.UINT8: "|u1",
        TensorDType.INT8: "|i1",
        TensorDType.INT32: "<i4",
        TensorDType.INT64: "<i8",
        TensorDType.BOOL: "|b1",
    }
)

# QLinearConv의 고정 입력 순서: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [B]
_QLINEAR_CONV_ARITY = (8, 9)


class StaticModelError(ValueError):
    """정적 ONNX가 Runtime이 감당할 수 있는 형태가 아닐 때 발생한다."""


@dataclass(frozen=True, slots=True)
class StaticTensorInfo:
    """정적 ONNX가 알고 있는 Tensor 하나의 dtype과 shape."""

    name: str
    dtype: TensorDType
    shape: tuple[int, ...]
    is_initializer: bool

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_size(self) -> int:
        return self.element_count * DTYPE_BYTE_SIZE[self.dtype]


@dataclass(frozen=True, slots=True)
class StaticInitializer:
    """상수 Tensor의 little-endian 원본 바이트."""

    name: str
    dtype: TensorDType
    shape: tuple[int, ...]
    raw_data: bytes

    @property
    def byte_size(self) -> int:
        return len(self.raw_data)

    def to_numpy(self) -> np.ndarray:
        """저장된 바이트를 shape 그대로 되살린다."""

        array = np.frombuffer(self.raw_data, dtype=np.dtype(_NUMPY_DTYPE[self.dtype]))
        return array.reshape(self.shape)

    def values(self) -> tuple[float | int | bool, ...]:
        """스칼라/1차원 parameter를 파이썬 값으로 펼친다."""

        return tuple(self.to_numpy().ravel().tolist())


@dataclass(frozen=True, slots=True)
class StaticNode:
    """실행 계획에 남은 node 하나. ``index``는 정적 ONNX 기준 실행 순서다."""

    index: int
    name: str
    op_type: str
    opcode: OperatorCode
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, AttributeValue]


@dataclass(frozen=True, slots=True)
class QuantizationBinding:
    """양자화된 Tensor 하나와 그 scale/zero_point Tensor 이름."""

    tensor: str
    scale: str
    zero_point: str


@dataclass(frozen=True, slots=True)
class QLinearConvBinding:
    """QLinearConv 한 개가 참조하는 모든 양자화 parameter와 bias."""

    x: QuantizationBinding
    w: QuantizationBinding
    y: QuantizationBinding
    bias: str | None


@dataclass(frozen=True, slots=True)
class StaticModelView:
    """정적 ONNX 한 개를 읽기 전용으로 노출한다."""

    source_path: str
    opset: int
    ir_version: int
    producer_name: str
    producer_version: str
    graph_inputs: tuple[StaticTensorInfo, ...]
    graph_outputs: tuple[StaticTensorInfo, ...]
    nodes: tuple[StaticNode, ...]
    tensors: Mapping[str, StaticTensorInfo]
    initializers: Mapping[str, StaticInitializer]
    producer_of: Mapping[str, int] = field(init=False, repr=False, compare=False)
    consumers_of: Mapping[str, tuple[int, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "graph_inputs", tuple(self.graph_inputs))
        object.__setattr__(self, "graph_outputs", tuple(self.graph_outputs))
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
        object.__setattr__(
            self, "initializers", MappingProxyType(dict(self.initializers))
        )

        producer_of: dict[str, int] = {}
        consumers_of: dict[str, list[int]] = {}
        for node in self.nodes:
            for name in node.outputs:
                if name in producer_of:
                    raise StaticModelError(
                        f"Tensor {name!r}를 두 개 이상의 node가 생성한다"
                    )
                producer_of[name] = node.index
            for name in node.inputs:
                consumers = consumers_of.setdefault(name, [])
                if not consumers or consumers[-1] != node.index:
                    consumers.append(node.index)

        object.__setattr__(self, "producer_of", MappingProxyType(producer_of))
        object.__setattr__(
            self,
            "consumers_of",
            MappingProxyType(
                {name: tuple(indices) for name, indices in consumers_of.items()}
            ),
        )
        self.validate()

    # -- 조회 --------------------------------------------------------------

    def tensor(self, name: str) -> StaticTensorInfo:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise StaticModelError(f"정적 ONNX에 없는 Tensor: {name!r}") from exc

    def initializer(self, name: str) -> StaticInitializer:
        try:
            return self.initializers[name]
        except KeyError as exc:
            raise StaticModelError(f"initializer가 아닌 Tensor: {name!r}") from exc

    def scale_values(self, name: str) -> tuple[float, ...]:
        """scale Tensor를 float로 읽는다. per-tensor면 길이 1이다."""

        initializer = self.initializer(name)
        if initializer.dtype not in (TensorDType.FLOAT32, TensorDType.FLOAT16):
            raise StaticModelError(
                f"scale {name!r}의 dtype이 부동소수가 아니다: {initializer.dtype.name}"
            )
        return tuple(float(value) for value in initializer.values())

    def zero_point_values(self, name: str) -> tuple[int, ...]:
        """zero_point Tensor를 정수로 읽는다. per-tensor면 길이 1이다."""

        initializer = self.initializer(name)
        if initializer.dtype not in (
            TensorDType.UINT8,
            TensorDType.INT8,
            TensorDType.INT32,
        ):
            raise StaticModelError(
                f"zero_point {name!r}의 dtype이 정수가 아니다: "
                f"{initializer.dtype.name}"
            )
        return tuple(int(value) for value in initializer.values())

    def qlinear_conv_binding(self, node: StaticNode) -> QLinearConvBinding:
        """QLinearConv의 입력 슬롯을 이름으로 풀어 준다."""

        if node.op_type != "QLinearConv":
            raise StaticModelError(f"QLinearConv가 아니다: {node.name!r}")
        if len(node.inputs) not in _QLINEAR_CONV_ARITY:
            raise StaticModelError(
                f"QLinearConv {node.name!r}의 입력이 8개나 9개가 아니다: "
                f"{len(node.inputs)}"
            )
        return QLinearConvBinding(
            x=QuantizationBinding(node.inputs[0], node.inputs[1], node.inputs[2]),
            w=QuantizationBinding(node.inputs[3], node.inputs[4], node.inputs[5]),
            y=QuantizationBinding(node.outputs[0], node.inputs[6], node.inputs[7]),
            bias=node.inputs[8] if len(node.inputs) == 9 else None,
        )

    def quantization_binding(self, node: StaticNode) -> QuantizationBinding:
        """QuantizeLinear/DequantizeLinear의 scale과 zero_point를 풀어 준다."""

        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            raise StaticModelError(
                f"QuantizeLinear/DequantizeLinear가 아니다: {node.name!r}"
            )
        if len(node.inputs) != 3:
            raise StaticModelError(
                f"{node.op_type} {node.name!r}의 입력이 3개가 아니다: "
                f"{len(node.inputs)}"
            )
        quantized = (
            node.outputs[0] if node.op_type == "QuantizeLinear" else node.inputs[0]
        )
        return QuantizationBinding(quantized, node.inputs[1], node.inputs[2])

    # -- 검증 --------------------------------------------------------------

    def validate(self) -> None:
        """Runtime binary 형식이 감당할 수 있는 그래프인지 검사한다."""

        if not self.nodes:
            raise StaticModelError("정적 ONNX에 node가 없다")
        if not self.graph_inputs:
            raise StaticModelError("정적 ONNX에 graph input이 없다")
        if not self.graph_outputs:
            raise StaticModelError("정적 ONNX에 graph output이 없다")

        for node in self.nodes:
            if not node.inputs:
                raise StaticModelError(f"node {node.name!r}에 입력이 없다")
            if len(node.inputs) > OPERATOR_INPUT_CAPACITY:
                raise StaticModelError(
                    f"node {node.name!r}의 입력 {len(node.inputs)}개가 형식 한계 "
                    f"{OPERATOR_INPUT_CAPACITY}개를 넘는다"
                )
            if len(node.outputs) != 1:
                raise StaticModelError(
                    f"node {node.name!r}의 출력이 1개가 아니다: {len(node.outputs)}"
                )
            for name in (*node.inputs, *node.outputs):
                if name not in self.tensors:
                    raise StaticModelError(
                        f"node {node.name!r}가 shape를 알 수 없는 Tensor "
                        f"{name!r}를 참조한다"
                    )

        for info in self.tensors.values():
            if len(info.shape) > TENSOR_MAX_RANK:
                raise StaticModelError(
                    f"Tensor {info.name!r}의 rank {len(info.shape)}가 형식 한계 "
                    f"{TENSOR_MAX_RANK}를 넘는다"
                )
            if any(dimension <= 0 for dimension in info.shape):
                raise StaticModelError(
                    f"Tensor {info.name!r}의 shape가 정적이지 않다: {list(info.shape)}"
                )

        for name, initializer in self.initializers.items():
            info = self.tensors[name]
            if initializer.dtype != info.dtype or initializer.shape != info.shape:
                raise StaticModelError(
                    f"initializer {name!r}의 dtype/shape가 Tensor 정보와 다르다"
                )
            if initializer.byte_size != info.byte_size:
                raise StaticModelError(
                    f"initializer {name!r}가 {initializer.byte_size}바이트인데 "
                    f"dtype/shape 계산은 {info.byte_size}바이트다"
                )


# ---------------------------------------------------------------------------
# ONNX -> StaticModelView
# ---------------------------------------------------------------------------


def _dtype_of(elem_type: int, owner: str) -> TensorDType:
    dtype = ONNX_ELEM_TYPE_TO_DTYPE.get(elem_type)
    if dtype is None:
        raise StaticModelError(
            f"{owner}의 dtype {TensorProto.DataType.Name(elem_type)}은(는) "
            "Runtime이 지원하지 않는다"
        )
    return dtype


def _value_info_to_tensor(value_info: onnx.ValueInfoProto) -> StaticTensorInfo:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise StaticModelError(f"Tensor {value_info.name!r}에 shape 정보가 없다")

    shape: list[int] = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise StaticModelError(
                f"Tensor {value_info.name!r}에 심볼릭 차원이 남아 있다 "
                f"({dimension.dim_param!r})"
            )
        shape.append(dimension.dim_value)

    return StaticTensorInfo(
        name=value_info.name,
        dtype=_dtype_of(tensor_type.elem_type, f"Tensor {value_info.name!r}"),
        shape=tuple(shape),
        is_initializer=False,
    )


def _read_initializer(proto: TensorProto) -> StaticInitializer:
    if proto.data_location == TensorProto.EXTERNAL:
        raise StaticModelError(
            f"initializer {proto.name!r}가 외부 파일에 있다. "
            "onnx.load(..., load_external_data=True)로 읽어야 한다"
        )

    dtype = _dtype_of(proto.data_type, f"initializer {proto.name!r}")
    # to_array는 raw_data / float_data / int32_data 중 어디에 담겼든 정규화해 준다.
    array = numpy_helper.to_array(proto)
    array = np.ascontiguousarray(array, dtype=np.dtype(_NUMPY_DTYPE[dtype]))

    return StaticInitializer(
        name=proto.name,
        dtype=dtype,
        shape=tuple(int(dimension) for dimension in proto.dims),
        raw_data=array.tobytes(),
    )


def _attribute_value(attribute: AttributeProto, owner: str) -> AttributeValue:
    if attribute.type == AttributeProto.INT:
        return int(attribute.i)
    if attribute.type == AttributeProto.FLOAT:
        return float(attribute.f)
    if attribute.type == AttributeProto.STRING:
        return attribute.s.decode("utf-8")
    if attribute.type == AttributeProto.INTS:
        return tuple(int(value) for value in attribute.ints)
    if attribute.type == AttributeProto.FLOATS:
        return tuple(float(value) for value in attribute.floats)
    raise StaticModelError(
        f"{owner}의 attribute {attribute.name!r} 타입 "
        f"{AttributeProto.AttributeType.Name(attribute.type)}은(는) 지원하지 않는다"
    )


def _apply_attribute_defaults(
    op_type: str,
    name: str,
    attributes: dict[str, AttributeValue],
    input_shapes: Sequence[tuple[int, ...]],
) -> None:
    """ONNX가 생략을 허용하는 attribute를 명시값으로 채운다.

    C Runtime이 기본값을 다시 추론하지 않도록, plan에는 항상 완전한 값을 싣는다.
    """

    auto_pad = attributes.get("auto_pad", "NOTSET")
    if auto_pad != "NOTSET":
        raise StaticModelError(
            f"node {name!r}의 auto_pad={auto_pad!r}는 지원하지 않는다. "
            "명시적 pads로 export해야 한다"
        )
    attributes.pop("auto_pad", None)

    if op_type in ("QLinearConv", "AveragePool"):
        kernel_shape = attributes.get("kernel_shape")
        if not isinstance(kernel_shape, tuple) or not kernel_shape:
            raise StaticModelError(f"node {name!r}에 kernel_shape가 없다")
        spatial = len(kernel_shape)
        attributes.setdefault("strides", (1,) * spatial)
        attributes.setdefault("pads", (0,) * (2 * spatial))
        if op_type == "QLinearConv":
            attributes.setdefault("dilations", (1,) * spatial)
            attributes.setdefault("group", 1)
        else:
            attributes.setdefault("ceil_mode", 0)
            attributes.setdefault("count_include_pad", 0)

    elif op_type == "BatchNormalization":
        attributes.setdefault("epsilon", 1e-5)
        attributes.setdefault("momentum", 0.9)

    elif op_type == "ReduceMean":
        attributes.setdefault("keepdims", 1)
        if "axes" not in attributes and input_shapes:
            attributes["axes"] = tuple(range(len(input_shapes[0])))

    elif op_type == "Transpose":
        if "perm" not in attributes and input_shapes:
            attributes["perm"] = tuple(reversed(range(len(input_shapes[0]))))

    elif op_type == "Concat":
        if "axis" not in attributes:
            raise StaticModelError(f"Concat {name!r}에 axis가 없다")


def _read_node(
    index: int, proto: onnx.NodeProto, tensors: Mapping[str, StaticTensorInfo]
) -> StaticNode:
    opcode = ONNX_OP_TO_OPCODE.get(proto.op_type)
    if opcode is None:
        raise StaticModelError(
            f"node {proto.name or index}의 op_type {proto.op_type}은(는) "
            "Runtime opcode에 없다"
        )

    name = proto.name or f"{proto.op_type}_{index}"
    attributes = {
        attribute.name: _attribute_value(attribute, f"node {name!r}")
        for attribute in proto.attribute
    }
    inputs = tuple(item for item in proto.input if item)
    outputs = tuple(item for item in proto.output if item)
    input_shapes = [tensors[item].shape for item in inputs if item in tensors]
    _apply_attribute_defaults(proto.op_type, name, attributes, input_shapes)

    return StaticNode(
        index=index,
        name=name,
        op_type=proto.op_type,
        opcode=opcode,
        inputs=inputs,
        outputs=outputs,
        attributes=MappingProxyType(attributes),
    )


def read_static_model(path: Path | str) -> StaticModelView:
    """정적 ONNX 한 개를 읽어 검증된 :class:`StaticModelView`로 만든다."""

    source = Path(path)
    if not source.exists():
        raise StaticModelError(f"정적 ONNX 파일이 없다: {source}")
    model = onnx.load(str(source))
    graph = model.graph

    initializers = {
        proto.name: _read_initializer(proto) for proto in graph.initializer
    }

    tensors: dict[str, StaticTensorInfo] = {}
    for name, initializer in initializers.items():
        tensors[name] = StaticTensorInfo(
            name=name,
            dtype=initializer.dtype,
            shape=initializer.shape,
            is_initializer=True,
        )
    for value_info in (*graph.input, *graph.output, *graph.value_info):
        if value_info.name in initializers:
            continue
        tensors[value_info.name] = _value_info_to_tensor(value_info)

    graph_inputs = tuple(
        tensors[value_info.name]
        for value_info in graph.input
        if value_info.name not in initializers
    )
    graph_outputs = tuple(tensors[value_info.name] for value_info in graph.output)

    nodes = tuple(
        _read_node(index, proto, tensors) for index, proto in enumerate(graph.node)
    )
    opset = next(
        (
            entry.version
            for entry in model.opset_import
            if entry.domain in ("", "ai.onnx")
        ),
        0,
    )

    return StaticModelView(
        source_path=str(source),
        opset=opset,
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        nodes=nodes,
        tensors=tensors,
        initializers=initializers,
    )


# ---------------------------------------------------------------------------
# StaticModelView + GraphIRView -> RuntimeGraph
# ---------------------------------------------------------------------------


def _contiguous_byte_strides(
    shape: tuple[int, ...], element_size: int
) -> tuple[int, ...]:
    """C 순서(row-major) 연속 배치의 축별 byte stride를 만든다."""

    strides = [0] * len(shape)
    running = element_size
    for axis in reversed(range(len(shape))):
        strides[axis] = running
        running *= shape[axis]
    return tuple(strides)


def _initializer_scope(name: str, ir: GraphIRView) -> InitializerScope:
    """initializer 하나가 bucket 사이에서 공유되는 값인지 IR 구조로 판정한다.

    reader는 한 번에 bucket 하나만 보므로 "값이 다르면 bucket-local"이라는 규칙은
    쓸 수 없다. 대신 그 상수가 어디서 왔는지로 나눈다. 원본 ONNX가 들고 있던
    initializer는 학습 parameter이므로 ``SHARED``이고, ``export_static``이
    shape 도메인 node를 접어 만든 상수는 frame 수에 따라 달라질 수 있으므로
    ``BUCKET_LOCAL``이다.
    """

    tensor = ir.tensor(name)
    if tensor.is_initializer and tensor.producer is None:
        return InitializerScope.SHARED
    if tensor.producer is not None and ir.node(tensor.producer).is_static:
        return InitializerScope.BUCKET_LOCAL
    raise StaticModelError(
        f"initializer {name!r}의 출처를 IR에서 판정할 수 없다 "
        f"(is_initializer={tensor.is_initializer}, producer={tensor.producer!r}). "
        "학습 weight도 shape 도메인 상수도 아니면 분류 규칙을 먼저 정해야 한다"
    )


def build_runtime_graph(
    static: StaticModelView,
    ir: GraphIRView,
    *,
    validate: bool = True,
    name: str = "",
) -> RuntimeGraph:
    """정적 ONNX와 graph IR을 하나의 :class:`RuntimeGraph`로 합친다.

    Tensor ID는 ``graph input -> initializer -> 실행 순서상의 activation`` 순으로
    0부터 조밀하게 매긴다. ``validate``가 참이면 변환 전에
    :func:`~.graph_ir_reader.cross_validate`로 두 입력이 같은 그래프인지 확인하고,
    다르면 :class:`~.graph_ir_reader.GraphIRMismatchError`를 던져 중단한다.
    """

    if validate:
        cross_validate(static, ir)

    output_names = {info.name for info in static.graph_outputs}
    input_names = {info.name for info in static.graph_inputs}

    tensor_ids: dict[str, int] = {}
    ordered: list[tuple[StaticTensorInfo, TensorStorageType]] = []

    def register(info: StaticTensorInfo, storage: TensorStorageType) -> None:
        if info.name in tensor_ids:
            raise StaticModelError(f"Tensor {info.name!r}에 ID를 두 번 매겼다")
        tensor_ids[info.name] = len(ordered)
        ordered.append((info, storage))

    for info in static.graph_inputs:
        register(info, TensorStorageType.INPUT)
    for tensor_name in static.initializers:
        register(static.tensor(tensor_name), TensorStorageType.CONSTANT)
    for node in static.nodes:
        for tensor_name in node.outputs:
            storage = (
                TensorStorageType.OUTPUT
                if tensor_name in output_names
                else TensorStorageType.ACTIVATION
            )
            register(static.tensor(tensor_name), storage)

    unreferenced = set(static.tensors) - set(tensor_ids)
    if unreferenced:
        raise StaticModelError(
            f"ID를 받지 못한 Tensor가 있다: {sorted(unreferenced)[:5]}"
        )
    for tensor_name in output_names:
        if tensor_name in input_names:
            raise StaticModelError(
                f"Tensor {tensor_name!r}가 graph input이자 output이다"
            )

    operators: list[RuntimeOperator] = []
    producers: dict[int, int] = {}
    consumers: dict[int, list[int]] = {index: [] for index in range(len(ordered))}

    for operator_id, node in enumerate(static.nodes):
        input_ids = tuple(tensor_ids[item] for item in node.inputs)
        output_ids = tuple(tensor_ids[item] for item in node.outputs)
        for tensor_id in input_ids:
            bucket = consumers[tensor_id]
            if not bucket or bucket[-1] != operator_id:
                bucket.append(operator_id)
        for tensor_id in output_ids:
            producers[tensor_id] = operator_id
        operators.append(
            RuntimeOperator(
                name=node.name,
                operator_id=operator_id,
                opcode=node.opcode,
                input_tensor_ids=input_ids,
                output_tensor_ids=output_ids,
                attributes=node.attributes,
            )
        )

    tensors: list[RuntimeTensor] = []
    for tensor_id, (info, storage) in enumerate(ordered):
        element_size = DTYPE_BYTE_SIZE[info.dtype]
        tensors.append(
            RuntimeTensor(
                name=info.name,
                tensor_id=tensor_id,
                dtype=info.dtype,
                shape=info.shape,
                strides=_contiguous_byte_strides(info.shape, element_size),
                byte_size=info.byte_size,
                storage_type=storage,
                producer=producers.get(tensor_id),
                consumers=tuple(consumers[tensor_id]),
            )
        )

    initializers = tuple(
        RuntimeInitializer(
            name=initializer.name,
            tensor_id=tensor_ids[initializer.name],
            dtype=initializer.dtype,
            shape=initializer.shape,
            raw_data=initializer.raw_data,
            scope=_initializer_scope(initializer.name, ir),
        )
        for initializer in static.initializers.values()
    )

    return RuntimeGraph(
        tensors=tuple(tensors),
        operators=tuple(operators),
        initializers=initializers,
        name=name or Path(static.source_path).stem,
        bucket_frames=ir.frames,
    )


__all__ = [
    "ONNX_ELEM_TYPE_TO_DTYPE",
    "ONNX_OP_TO_OPCODE",
    "QLinearConvBinding",
    "QuantizationBinding",
    "StaticInitializer",
    "StaticModelError",
    "StaticModelView",
    "StaticNode",
    "StaticTensorInfo",
    "build_runtime_graph",
    "read_static_model",
]
