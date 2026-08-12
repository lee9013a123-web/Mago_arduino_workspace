"""bucket별 graph IR(``ir_*.json``)을 읽고 정적 ONNX와 대조하는 모듈.

``scripts/2_graph/build_graph.py``가 남긴 IR은 Tensor shape, producer/consumer,
static/runtime 구분과 topological 실행 순서를 이미 확정해 둔 분석 산출물이다.
이 모듈은 그 JSON을 ONNX 의존성 없는 값 객체로 옮기고, 같은 bucket의 정적
ONNX가 IR과 같은 그래프인지 검증한다. 하나라도 어긋나면 변환을 중단한다.

IR은 원본 dynamic 모델 기준이라 static node와 shape 도메인 Tensor까지 모두
들고 있다. 정적 ONNX는 그중 runtime node만 남기고 shape 도메인을 initializer로
접은 결과이므로, 대조는 "정적 ONNX가 참조하는 모든 이름"을 기준으로 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from .binary_format_schema import TensorDType

if TYPE_CHECKING:  # pragma: no cover - 순환 import와 onnx 의존을 피한다
    from .static_model_reader import StaticModelView


# IR은 dtype을 onnx.TensorProto.DataType 이름으로 기록한다.
IR_DTYPE_NAMES: Mapping[str, TensorDType] = MappingProxyType(
    {
        "FLOAT": TensorDType.FLOAT32,
        "FLOAT16": TensorDType.FLOAT16,
        "UINT8": TensorDType.UINT8,
        "INT8": TensorDType.INT8,
        "INT32": TensorDType.INT32,
        "INT64": TensorDType.INT64,
        "BOOL": TensorDType.BOOL,
    }
)

DTYPE_BYTE_SIZE: Mapping[TensorDType, int] = MappingProxyType(
    {
        TensorDType.FLOAT32: 4,
        TensorDType.FLOAT16: 2,
        TensorDType.UINT8: 1,
        TensorDType.INT8: 1,
        TensorDType.INT32: 4,
        TensorDType.INT64: 8,
        TensorDType.BOOL: 1,
    }
)

# 불일치가 수백 개 쏟아져도 메시지는 앞쪽만 보여준다.
MAX_REPORTED_MISMATCHES: int = 12

_REQUIRED_IR_KEYS = (
    "model_path",
    "batch",
    "frames",
    "seconds",
    "opset",
    "inputs",
    "outputs",
    "nodes",
    "tensors",
)


class GraphIRError(ValueError):
    """graph IR JSON이 기대한 구조나 불변식을 지키지 않았을 때 발생한다."""


class GraphIRMismatchError(GraphIRError):
    """정적 ONNX와 graph IR이 서로 다른 그래프를 가리킬 때 발생한다."""


@dataclass(frozen=True, slots=True)
class IRTensor:
    """IR이 기록한 Tensor 하나.

    ``dtype``/``shape``는 IR이 끝내 확정하지 못한 Tensor에서 ``None``일 수 있다.
    정적 ONNX가 실제로 참조하는 Tensor는 전부 확정되어 있어야 하며, 그 검사는
    :func:`cross_validate`가 담당한다.
    """

    name: str
    dtype: TensorDType | None
    shape: tuple[int, ...] | None
    byte_size: int | None
    is_initializer: bool
    is_static: bool
    producer: int | None
    consumers: tuple[int, ...]

    @property
    def is_resolved(self) -> bool:
        return self.dtype is not None and self.shape is not None


@dataclass(frozen=True, slots=True)
class IRNode:
    """IR이 기록한 node 하나. ``index``는 원본 dynamic 그래프 기준이다."""

    index: int
    name: str
    op_type: str
    scope: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    is_static: bool
    kind: str


@dataclass(frozen=True, slots=True)
class GraphIRView:
    """한 bucket의 graph IR을 읽기 전용으로 노출한다."""

    source_path: str
    model_path: str
    batch: int
    frames: int
    seconds: float
    opset: int
    graph_inputs: tuple[str, ...]
    graph_outputs: tuple[str, ...]
    nodes: tuple[IRNode, ...]
    tensors: Mapping[str, IRTensor]
    runtime_nodes: tuple[IRNode, ...] = field(init=False, repr=False, compare=False)
    static_nodes: tuple[IRNode, ...] = field(init=False, repr=False, compare=False)
    _execution_position: Mapping[int, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))
        object.__setattr__(self, "graph_inputs", tuple(self.graph_inputs))
        object.__setattr__(self, "graph_outputs", tuple(self.graph_outputs))

        runtime = tuple(node for node in self.nodes if not node.is_static)
        static = tuple(node for node in self.nodes if node.is_static)
        object.__setattr__(self, "runtime_nodes", runtime)
        object.__setattr__(self, "static_nodes", static)
        object.__setattr__(
            self,
            "_execution_position",
            MappingProxyType(
                {node.index: position for position, node in enumerate(runtime)}
            ),
        )
        self.validate()

    # -- 조회 --------------------------------------------------------------

    def tensor(self, name: str) -> IRTensor:
        """이름으로 Tensor를 찾는다. 없으면 :class:`GraphIRError`."""

        try:
            return self.tensors[name]
        except KeyError as exc:
            raise GraphIRError(f"graph IR에 없는 Tensor: {name!r}") from exc

    def node(self, index: int) -> IRNode:
        """원본 그래프 기준 index로 node를 찾는다."""

        if not 0 <= index < len(self.nodes):
            raise GraphIRError(f"graph IR node index 범위를 벗어남: {index}")
        return self.nodes[index]

    def execution_position(self, index: int) -> int:
        """원본 node index를 runtime 실행 순서(0부터)로 바꾼다."""

        try:
            return self._execution_position[index]
        except KeyError as exc:
            raise GraphIRError(
                f"node index {index}는 runtime node가 아니라 실행 순서가 없다"
            ) from exc

    def is_runtime_node(self, index: int) -> bool:
        return index in self._execution_position

    # -- 검증 --------------------------------------------------------------

    def validate(self) -> None:
        """IR 자체의 불변식(그래프 I/O, producer/consumer, 실행 순서)을 검사한다."""

        if not self.nodes:
            raise GraphIRError("graph IR에 node가 없다")
        if not self.runtime_nodes:
            raise GraphIRError("graph IR에 runtime node가 하나도 없다")
        if self.frames <= 0:
            raise GraphIRError(f"frames는 0보다 커야 한다: {self.frames}")

        for name in (*self.graph_inputs, *self.graph_outputs):
            if name not in self.tensors:
                raise GraphIRError(f"graph I/O Tensor {name!r}가 tensors에 없다")

        seen_names: set[str] = set()
        for node in self.nodes:
            if node.name in seen_names:
                raise GraphIRError(f"graph IR에 중복된 node 이름: {node.name!r}")
            seen_names.add(node.name)

        self._validate_producer_consumer()
        self._validate_execution_order()

    def _validate_producer_consumer(self) -> None:
        for node in self.nodes:
            for name in node.outputs:
                tensor = self.tensor(name)
                if tensor.producer != node.index:
                    raise GraphIRError(
                        f"Tensor {name!r}의 producer가 {tensor.producer!r}인데 "
                        f"node {node.index}가 출력한다"
                    )
            for name in node.inputs:
                tensor = self.tensor(name)
                if node.index not in tensor.consumers:
                    raise GraphIRError(
                        f"Tensor {name!r}의 consumers에 node {node.index}가 없다"
                    )

    def _validate_execution_order(self) -> None:
        """runtime node가 위상 순서대로 나열되어 있는지 확인한다.

        runtime Tensor는 반드시 앞선 runtime node가 만들어 두었어야 한다.
        static Tensor는 정적 ONNX에서 initializer로 접히므로 언제든 읽어도 된다.
        """

        produced: set[str] = set()
        for node in self.runtime_nodes:
            for name in node.inputs:
                tensor = self.tensor(name)
                if tensor.is_static or name in self.graph_inputs:
                    continue
                if name not in produced:
                    raise GraphIRError(
                        f"node {node.name!r}가 아직 생성되지 않은 Tensor "
                        f"{name!r}를 읽는다 (실행 순서가 위상 정렬이 아니다)"
                    )
            produced.update(node.outputs)


@dataclass(frozen=True, slots=True)
class CrossValidationReport:
    """정적 ONNX와 graph IR 대조가 통과했을 때의 요약."""

    frames: int
    runtime_node_count: int
    tensor_count: int
    initializer_count: int
    graph_inputs: tuple[str, ...]
    graph_outputs: tuple[str, ...]


def _parse_shape(raw: object, name: str) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise GraphIRError(f"Tensor {name!r}의 shape가 리스트가 아니다: {raw!r}")
    for dimension in raw:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise GraphIRError(f"Tensor {name!r}의 shape에 정수가 아닌 값: {raw!r}")
        if dimension < 0:
            return None  # 심볼릭 차원이 남아 있으면 미확정으로 본다
    return tuple(raw)


def _parse_tensor(name: str, raw: Mapping[str, object]) -> IRTensor:
    dtype = IR_DTYPE_NAMES.get(str(raw.get("dtype", "UNDEFINED")))
    shape = _parse_shape(raw.get("shape"), name)

    byte_size: int | None = None
    if dtype is not None and shape is not None:
        byte_size = DTYPE_BYTE_SIZE[dtype]
        for dimension in shape:
            byte_size *= dimension

    recorded = raw.get("bytes")
    if byte_size is not None and isinstance(recorded, int) and recorded != byte_size:
        raise GraphIRError(
            f"Tensor {name!r}의 byte 크기가 IR 기록({recorded})과 "
            f"dtype/shape 계산({byte_size})에서 다르다"
        )

    producer = raw.get("producer")
    if producer is not None and not isinstance(producer, int):
        raise GraphIRError(f"Tensor {name!r}의 producer가 정수가 아니다: {producer!r}")

    consumers = raw.get("consumers") or []
    if not isinstance(consumers, list):
        raise GraphIRError(f"Tensor {name!r}의 consumers가 리스트가 아니다")

    return IRTensor(
        name=name,
        dtype=dtype,
        shape=shape,
        byte_size=byte_size,
        is_initializer=bool(raw.get("is_initializer", False)),
        is_static=bool(raw.get("is_static", False)),
        producer=producer,
        consumers=tuple(consumers),
    )


def _parse_node(raw: Mapping[str, object]) -> IRNode:
    try:
        index = int(raw["index"])  # type: ignore[arg-type]
        name = str(raw["name"])
        op_type = str(raw["op_type"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GraphIRError(f"graph IR node 항목이 손상되었다: {raw!r}") from exc

    return IRNode(
        index=index,
        name=name,
        op_type=op_type,
        scope=str(raw.get("scope", "")),
        inputs=tuple(str(item) for item in raw.get("inputs", []) if item),
        outputs=tuple(str(item) for item in raw.get("outputs", []) if item),
        is_static=bool(raw.get("is_static", False)),
        kind=str(raw.get("kind", "other")),
    )


def read_graph_ir(path: Path | str) -> GraphIRView:
    """``ir_*.json`` 한 개를 읽어 검증된 :class:`GraphIRView`로 만든다."""

    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GraphIRError(f"graph IR 파일이 없다: {source}") from exc
    except json.JSONDecodeError as exc:
        raise GraphIRError(f"graph IR JSON을 읽을 수 없다: {source}: {exc}") from exc

    if not isinstance(document, dict):
        raise GraphIRError(f"graph IR 최상위가 객체가 아니다: {source}")
    missing = [key for key in _REQUIRED_IR_KEYS if key not in document]
    if missing:
        raise GraphIRError(f"graph IR에 필수 항목이 없다: {', '.join(missing)}")

    raw_tensors = document["tensors"]
    if not isinstance(raw_tensors, dict):
        raise GraphIRError("graph IR의 tensors가 객체가 아니다")
    raw_nodes = document["nodes"]
    if not isinstance(raw_nodes, list):
        raise GraphIRError("graph IR의 nodes가 배열이 아니다")

    tensors = {name: _parse_tensor(name, raw) for name, raw in raw_tensors.items()}
    nodes = tuple(_parse_node(raw) for raw in raw_nodes)
    for position, node in enumerate(nodes):
        if node.index != position:
            raise GraphIRError(
                f"graph IR node index가 순서와 다르다: {position} 자리에 "
                f"{node.index}"
            )

    return GraphIRView(
        source_path=str(source),
        model_path=str(document["model_path"]),
        batch=int(document["batch"]),
        frames=int(document["frames"]),
        seconds=float(document["seconds"]),
        opset=int(document["opset"]),
        graph_inputs=tuple(str(name) for name in document["inputs"]),
        graph_outputs=tuple(str(name) for name in document["outputs"]),
        nodes=nodes,
        tensors=tensors,
    )


class _MismatchLog:
    """대조 실패를 모아 두었다가 한 번에 보고한다."""

    def __init__(self) -> None:
        self._messages: list[str] = []

    def add(self, message: str) -> None:
        self._messages.append(message)

    def raise_if_any(self, static_path: str, ir_path: str) -> None:
        if not self._messages:
            return
        shown = self._messages[:MAX_REPORTED_MISMATCHES]
        hidden = len(self._messages) - len(shown)
        lines = [
            f"정적 ONNX와 graph IR이 일치하지 않아 변환을 중단한다 "
            f"({len(self._messages)}건)",
            f"  ONNX: {static_path}",
            f"  IR  : {ir_path}",
            *(f"  - {message}" for message in shown),
        ]
        if hidden:
            lines.append(f"  ... 그 외 {hidden}건")
        raise GraphIRMismatchError("\n".join(lines))


def _check_runtime_nodes(
    static: "StaticModelView", ir: GraphIRView, log: _MismatchLog
) -> None:
    """runtime node 수, 이름, 실행 순서, op_type, 연결 관계를 대조한다."""

    if len(static.nodes) != len(ir.runtime_nodes):
        log.add(
            f"runtime node 수가 다르다: ONNX {len(static.nodes)}개, "
            f"IR {len(ir.runtime_nodes)}개"
        )

    static_names = {node.name for node in static.nodes}
    for node in ir.static_nodes:
        if node.name in static_names:
            log.add(
                f"IR이 static으로 판정한 node {node.name!r}가 실행 계획에 남아 있다"
            )

    for position, (onnx_node, ir_node) in enumerate(
        zip(static.nodes, ir.runtime_nodes)
    ):
        if onnx_node.name != ir_node.name:
            log.add(
                f"실행 순서 {position}의 node 이름이 다르다: "
                f"ONNX {onnx_node.name!r}, IR {ir_node.name!r}"
            )
            continue
        if onnx_node.op_type != ir_node.op_type:
            log.add(
                f"node {onnx_node.name!r}의 op_type이 다르다: "
                f"ONNX {onnx_node.op_type}, IR {ir_node.op_type}"
            )
        if onnx_node.inputs != ir_node.inputs:
            log.add(
                f"node {onnx_node.name!r}의 입력 Tensor가 다르다: "
                f"ONNX {list(onnx_node.inputs)}, IR {list(ir_node.inputs)}"
            )
        if onnx_node.outputs != ir_node.outputs:
            log.add(
                f"node {onnx_node.name!r}의 출력 Tensor가 다르다: "
                f"ONNX {list(onnx_node.outputs)}, IR {list(ir_node.outputs)}"
            )


def _check_tensors(
    static: "StaticModelView", ir: GraphIRView, log: _MismatchLog
) -> None:
    """정적 ONNX가 참조하는 모든 Tensor의 존재, dtype, shape를 대조한다."""

    for name, info in static.tensors.items():
        ir_tensor = ir.tensors.get(name)
        if ir_tensor is None:
            log.add(f"ONNX Tensor {name!r}가 IR에 없다")
            continue
        if not ir_tensor.is_resolved:
            log.add(f"IR이 Tensor {name!r}의 dtype/shape를 확정하지 못했다")
            continue
        if info.dtype != ir_tensor.dtype:
            log.add(
                f"Tensor {name!r}의 dtype이 다르다: ONNX {info.dtype.name}, "
                f"IR {ir_tensor.dtype.name}"
            )
        if info.shape != ir_tensor.shape:
            log.add(
                f"Tensor {name!r}의 shape가 다르다: ONNX {list(info.shape)}, "
                f"IR {list(ir_tensor.shape)}"
            )

    # 정적 ONNX의 initializer는 IR에서 반드시 컴파일 타임에 확정된 값이어야 한다.
    for name in static.initializers:
        ir_tensor = ir.tensors.get(name)
        if ir_tensor is not None and not ir_tensor.is_static:
            log.add(
                f"initializer {name!r}를 IR은 runtime Tensor로 판정했다 "
                "(상수로 접을 수 없는 값이다)"
            )


def _check_graph_io(
    static: "StaticModelView", ir: GraphIRView, log: _MismatchLog
) -> None:
    onnx_inputs = tuple(info.name for info in static.graph_inputs)
    onnx_outputs = tuple(info.name for info in static.graph_outputs)
    if onnx_inputs != ir.graph_inputs:
        log.add(
            f"graph input이 다르다: ONNX {list(onnx_inputs)}, "
            f"IR {list(ir.graph_inputs)}"
        )
    if onnx_outputs != ir.graph_outputs:
        log.add(
            f"graph output이 다르다: ONNX {list(onnx_outputs)}, "
            f"IR {list(ir.graph_outputs)}"
        )


def _check_topology(
    static: "StaticModelView", ir: GraphIRView, log: _MismatchLog
) -> None:
    """producer/consumer 관계를 runtime node 범위로 좁혀 대조한다.

    IR의 producer/consumer는 원본 그래프 index라서 static node까지 가리킨다.
    정적 ONNX에 남은 node만 골라 이름 기준으로 비교한다.
    """

    for name, producer_index in static.producer_of.items():
        ir_tensor = ir.tensors.get(name)
        if ir_tensor is None or ir_tensor.producer is None:
            log.add(f"ONNX가 생성하는 Tensor {name!r}에 IR producer가 없다")
            continue
        ir_producer = ir.node(ir_tensor.producer)
        onnx_producer = static.nodes[producer_index]
        if ir_producer.name != onnx_producer.name:
            log.add(
                f"Tensor {name!r}의 producer가 다르다: "
                f"ONNX {onnx_producer.name!r}, IR {ir_producer.name!r}"
            )

    for name, consumer_indices in static.consumers_of.items():
        ir_tensor = ir.tensors.get(name)
        if ir_tensor is None:
            continue  # 존재 여부는 _check_tensors가 이미 보고했다
        onnx_consumers = {static.nodes[index].name for index in consumer_indices}
        ir_consumers = {
            ir.node(index).name
            for index in ir_tensor.consumers
            if ir.is_runtime_node(index)
        }
        if onnx_consumers != ir_consumers:
            log.add(
                f"Tensor {name!r}의 consumer 집합이 다르다: "
                f"ONNX만 {sorted(onnx_consumers - ir_consumers)}, "
                f"IR만 {sorted(ir_consumers - onnx_consumers)}"
            )


def cross_validate(
    static: "StaticModelView", ir: GraphIRView
) -> CrossValidationReport:
    """정적 ONNX와 graph IR이 같은 그래프인지 확인한다.

    검사 항목은 runtime node 수와 실행 순서, Tensor 이름/dtype/shape,
    graph input과 output, producer/consumer topology, 그리고 static node가
    실행 계획에 섞여 들어가지 않았는지다. 하나라도 어긋나면
    :class:`GraphIRMismatchError`를 던져 변환을 중단시킨다.
    """

    log = _MismatchLog()
    _check_runtime_nodes(static, ir, log)
    _check_tensors(static, ir, log)
    _check_graph_io(static, ir, log)
    _check_topology(static, ir, log)
    log.raise_if_any(static.source_path, ir.source_path)

    return CrossValidationReport(
        frames=ir.frames,
        runtime_node_count=len(static.nodes),
        tensor_count=len(static.tensors),
        initializer_count=len(static.initializers),
        graph_inputs=ir.graph_inputs,
        graph_outputs=ir.graph_outputs,
    )


__all__ = [
    "CrossValidationReport",
    "DTYPE_BYTE_SIZE",
    "GraphIRError",
    "GraphIRMismatchError",
    "GraphIRView",
    "IRNode",
    "IRTensor",
    "IR_DTYPE_NAMES",
    "MAX_REPORTED_MISMATCHES",
    "cross_validate",
    "read_graph_ir",
]
