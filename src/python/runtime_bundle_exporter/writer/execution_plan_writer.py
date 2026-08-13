# bucket별 Tensor table과 Operator table을 plan_*.bin으로 저장하는 모듈이다.
#
# 출력:
# - plan_98.bin
# - plan_298.bin
# - plan_498.bin
# - plan_998.bin
#
# 모든 정수 필드는 binary_format_schema.py 규칙에 따라 명시적으로 직렬화한다.
# Python 객체나 C 구조체의 메모리 표현을 그대로 파일에 쓰지 않는다.

"""RuntimeGraph 하나를 bucket 전용 execution plan 파일로 굳힌다.

파일 구조는 header가 가리키는 세 구역이 이어진 형태다::

    [0, 80)                      PlanHeader
    tensor_table_offset          TensorDescriptor  x tensor_count
    operator_table_offset        OperatorDescriptor x operator_count
    attribute_section_offset     attribute block들

descriptor 크기가 각각 80바이트와 64바이트라서 세 offset은 별도 패딩 없이
8바이트 정렬을 만족한다. header 뒤 전체의 SHA-256을 header에 적어 두므로,
loader는 한 번의 해시로 파일이 온전한지 확인할 수 있다.

bucket 전용 상수는 이 파일에 넣지 않는다. weights.bin이 bucket마다 따로
사본을 들고 있고 plan의 Tensor descriptor가 자기 bucket offset을 가리키므로,
plan에는 상수 구역이 필요 없다. 그래서 PlanHeader의 구역이 셋으로 충분하다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

from ..format.binary_format_schema import (
    OPERATOR_DESCRIPTOR_SIZE,
    PLAN_HEADER_SIZE,
    PLAN_SECTION_ALIGNMENT,
    TENSOR_DESCRIPTOR_SIZE,
    BackendId,
    BinaryFormatError,
    DEFAULT_KERNEL_ID,
    OperatorDescriptor,
    PlanHeader,
    TensorDescriptor,
    TensorDType,
    TensorStorageType,
    decode_attribute_block,
)
from ..builder.operator_table_builder import build_operator_table
from ..runtime_ir import RuntimeGraph
from ..builder.tensor_table_builder import WeightBlobLayout, build_tensor_table
from ..builder.planner.tensor_arena_planner import TensorArenaLayout


PLAN_FILE_NAME_TEMPLATE: str = "plan_{frames}.bin"
EXECUTION_PLAN_DIR_NAME: str = "execution_plans"


class ExecutionPlanError(ValueError):
    """plan 파일을 쓰거나 되읽을 수 없을 때 발생한다."""


def plan_file_name(bucket_frames: int) -> str:
    return PLAN_FILE_NAME_TEMPLATE.format(frames=bucket_frames)


@dataclass(frozen=True, slots=True)
class ExecutionPlanResult:
    """plan 파일 하나를 쓴 결과."""

    path: Path
    bucket_frames: int
    byte_size: int
    sha256: str
    tensor_count: int
    operator_count: int
    attribute_section_bytes: int
    attribute_blocks: int
    arena_size: int | None = None
    arena_alignment: int | None = None


@dataclass(frozen=True, slots=True)
class LoadedPlan:
    """되읽은 plan 파일. C loader가 보게 될 값과 같은 것만 담는다."""

    header: PlanHeader
    tensors: tuple[TensorDescriptor, ...]
    operators: tuple[OperatorDescriptor, ...]
    attribute_section: bytes

    def attributes_of(self, operator_id: int) -> dict:
        if not 0 <= operator_id < len(self.operators):
            raise ExecutionPlanError(f"Operator ID 범위를 벗어남: {operator_id}")
        descriptor = self.operators[operator_id]
        if descriptor.attribute_size == 0:
            return {}
        return decode_attribute_block(
            self.attribute_section, descriptor.attribute_offset
        )


def build_plan_bytes(
    graph: RuntimeGraph,
    layout: WeightBlobLayout,
    *,
    backend_id: BackendId = BackendId.AUTO,
    kernel_id: int = DEFAULT_KERNEL_ID,
    arena_layout: TensorArenaLayout | None = None,
) -> tuple[bytes, int, int]:
    """plan 파일 내용을 메모리에서 만든다.

    ``(bytes, attribute_section_bytes, attribute_blocks)``를 돌려준다.
    """

    if graph.bucket_frames is None:
        raise ExecutionPlanError("plan을 쓰려면 graph에 bucket_frames가 있어야 한다")

    tensor_table = build_tensor_table(
        graph, layout, arena_layout=arena_layout
    )
    operator_table = build_operator_table(
        graph, backend_id=backend_id, kernel_id=kernel_id
    )

    tensor_table_offset = PLAN_HEADER_SIZE
    operator_table_offset = tensor_table_offset + tensor_table.byte_size
    attribute_section_offset = operator_table_offset + operator_table.byte_size
    for name, offset in (
        ("tensor_table_offset", tensor_table_offset),
        ("operator_table_offset", operator_table_offset),
        ("attribute_section_offset", attribute_section_offset),
    ):
        if offset % PLAN_SECTION_ALIGNMENT != 0:
            raise ExecutionPlanError(f"{name}이 정렬되지 않았다: {offset}")

    payload = b"".join(
        (
            tensor_table.pack(),
            operator_table.pack(),
            operator_table.attribute_section,
        )
    )
    header = PlanHeader.create(
        bucket_frames=graph.bucket_frames,
        tensor_count=len(tensor_table),
        operator_count=len(operator_table),
        tensor_table_offset=tensor_table_offset,
        operator_table_offset=operator_table_offset,
        attribute_section_offset=attribute_section_offset,
        payload=payload,
    )
    return (
        header.pack() + payload,
        operator_table.attribute_section_size,
        operator_table.attribute_blocks,
    )


def write_execution_plan(
    graph: RuntimeGraph,
    layout: WeightBlobLayout,
    path: Path | str,
    *,
    backend_id: BackendId = BackendId.AUTO,
    kernel_id: int = DEFAULT_KERNEL_ID,
    arena_layout: TensorArenaLayout | None = None,
) -> ExecutionPlanResult:
    """한 bucket의 plan 파일을 쓴다."""

    target = Path(path)
    data, attribute_bytes, attribute_blocks = build_plan_bytes(
        graph,
        layout,
        backend_id=backend_id,
        kernel_id=kernel_id,
        arena_layout=arena_layout,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    return ExecutionPlanResult(
        path=target,
        bucket_frames=graph.bucket_frames,
        byte_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        tensor_count=len(graph.tensors),
        operator_count=len(graph.operators),
        attribute_section_bytes=attribute_bytes,
        attribute_blocks=attribute_blocks,
        arena_size=(arena_layout.arena_size if arena_layout is not None else None),
        arena_alignment=(
            arena_layout.alignment if arena_layout is not None else None
        ),
    )


def read_execution_plan(path: Path | str) -> LoadedPlan:
    """plan 파일을 읽고 header checksum까지 확인한다."""

    source = Path(path)
    if not source.exists():
        raise ExecutionPlanError(f"plan 파일이 없다: {source}")
    data = source.read_bytes()

    try:
        header = PlanHeader.unpack_from(data)
    except BinaryFormatError as exc:
        raise ExecutionPlanError(f"{source}의 header가 형식을 위반한다: {exc}") from exc

    payload = data[PLAN_HEADER_SIZE:]
    if not header.verify_payload_checksum(payload):
        raise ExecutionPlanError(f"{source}의 checksum이 맞지 않는다")

    tensors = tuple(
        TensorDescriptor.unpack_from(
            data, header.tensor_table_offset + index * TENSOR_DESCRIPTOR_SIZE
        )
        for index in range(header.tensor_count)
    )
    operators = tuple(
        OperatorDescriptor.unpack_from(
            data, header.operator_table_offset + index * OPERATOR_DESCRIPTOR_SIZE
        )
        for index in range(header.operator_count)
    )
    attribute_section = data[header.attribute_section_offset :]

    return LoadedPlan(
        header=header,
        tensors=tensors,
        operators=operators,
        attribute_section=attribute_section,
    )


def _normalized_attributes(attributes: Mapping[str, object]) -> dict[str, tuple]:
    """스칼라를 원소 1개짜리 튜플로 맞춘다. 형식이 둘을 구분하지 않는다."""

    normalized: dict[str, tuple] = {}
    for name, value in attributes.items():
        normalized[name] = (
            tuple(value) if isinstance(value, (tuple, list)) else (value,)
        )
    return normalized


def verify_plan(
    loaded: LoadedPlan,
    graph: RuntimeGraph,
    layout: WeightBlobLayout,
    *,
    arena_layout: TensorArenaLayout | None = None,
) -> None:
    """되읽은 plan이 원본 RuntimeGraph와 같은지 항목별로 확인한다.

    plan에는 이름이 실리지 않으므로 이름은 비교 대상이 아니다. 그 외에 plan이
    담은 모든 값 - dtype, shape, byte stride, storage, weights offset, 수명,
    opcode, Tensor ID 배선, attribute - 이 원본과 일치해야 한다.
    """

    if loaded.header.bucket_frames != graph.bucket_frames:
        raise ExecutionPlanError(
            f"bucket_frames가 다르다: plan {loaded.header.bucket_frames}, "
            f"graph {graph.bucket_frames}"
        )
    if len(loaded.tensors) != len(graph.tensors):
        raise ExecutionPlanError(
            f"Tensor 수가 다르다: plan {len(loaded.tensors)}, "
            f"graph {len(graph.tensors)}"
        )
    if len(loaded.operators) != len(graph.operators):
        raise ExecutionPlanError(
            f"Operator 수가 다르다: plan {len(loaded.operators)}, "
            f"graph {len(graph.operators)}"
        )

    expected_tensor_table = build_tensor_table(
        graph, layout, arena_layout=arena_layout
    )
    descriptor_fields = (
        "tensor_id",
        "dtype",
        "rank",
        "storage_type",
        "flags",
        "dimensions",
        "byte_strides",
        "data_offset",
        "logical_byte_size",
        "storage_span_bytes",
        "alias_of_tensor_id",
        "quantization_index",
        "first_use",
        "last_use",
    )
    for descriptor, expected, tensor in zip(
        loaded.tensors, expected_tensor_table.descriptors, graph.tensors
    ):
        mismatches = []
        for field_name in descriptor_fields:
            if getattr(descriptor, field_name) != getattr(expected, field_name):
                mismatches.append(field_name)
        if mismatches:
            raise ExecutionPlanError(
                f"Tensor {tensor.tensor_id} {tensor.name!r}가 plan과 다르다: "
                f"{', '.join(mismatches)}"
            )

    for descriptor, operator in zip(loaded.operators, graph.operators):
        used = descriptor.input_tensor_ids[: descriptor.input_count]
        mismatches = []
        if descriptor.operator_id != operator.operator_id:
            mismatches.append("operator_id")
        if descriptor.opcode != int(operator.opcode):
            mismatches.append("opcode")
        if descriptor.input_count != len(operator.input_tensor_ids):
            mismatches.append("input_count")
        elif used != tuple(operator.input_tensor_ids):
            mismatches.append("input_tensor_ids")
        if tuple(descriptor.output_tensor_ids) != tuple(operator.output_tensor_ids):
            mismatches.append("output_tensor_ids")
        if mismatches:
            raise ExecutionPlanError(
                f"Operator {operator.operator_id} {operator.name!r}가 plan과 "
                f"다르다: {', '.join(mismatches)}"
            )

        expected_attributes = _normalized_attributes(operator.attributes)
        stored = loaded.attributes_of(operator.operator_id)
        if stored != expected_attributes:
            raise ExecutionPlanError(
                f"Operator {operator.operator_id} {operator.name!r}의 attribute가 "
                f"다르다: plan {stored}, graph {expected_attributes}"
            )


_DTYPE_NAMES: Mapping[int, str] = {
    TensorDType.FLOAT32: "FLOAT32",
    TensorDType.UINT8: "UINT8",
    TensorDType.INT8: "INT8",
    TensorDType.INT32: "INT32",
    TensorDType.INT64: "INT64",
    TensorDType.BOOL: "BOOL",
    TensorDType.FLOAT16: "FLOAT16",
}

_STORAGE_NAMES: Mapping[int, str] = {
    TensorStorageType.INPUT: "INPUT",
    TensorStorageType.OUTPUT: "OUTPUT",
    TensorStorageType.CONSTANT: "CONSTANT",
    TensorStorageType.ACTIVATION: "ACTIVATION",
    TensorStorageType.VIEW: "VIEW",
}


def format_tensor_dump(loaded: LoadedPlan) -> str:
    """C의 ``--dump-tensors`` 출력과 문자 단위로 같은 문자열을 만든다.

    C Runtime이 같은 plan을 같은 값으로 읽는지 확인하는 기준이다. 형식을 바꾸면
    ``campp_compiled_model_inspect.c``의 출력도 함께 바꿔야 한다.
    """

    lines = []
    for descriptor in loaded.tensors:
        dimensions = ",".join(str(value) for value in descriptor.dimensions)
        strides = ",".join(str(value) for value in descriptor.byte_strides)
        lines.append(
            f"T {descriptor.tensor_id} "
            f"{_DTYPE_NAMES[descriptor.dtype]} "
            f"rank={descriptor.rank} "
            f"{_STORAGE_NAMES[descriptor.storage_type]} "
            f"flags=0x{descriptor.flags:02x} "
            f"dims={dimensions} strides={strides} "
            f"offset={descriptor.data_offset} "
            f"logical={descriptor.logical_byte_size} "
            f"span={descriptor.storage_span_bytes} "
            f"alias={descriptor.alias_of_tensor_id} "
            f"quant={descriptor.quantization_index} "
            f"first={descriptor.first_use} last={descriptor.last_use}"
        )
    return "\n".join(lines) + "\n"


def format_operator_dump(loaded: LoadedPlan) -> str:
    """C의 ``--dump-operators`` 출력과 문자 단위로 같은 문자열을 만든다."""

    lines = []
    for descriptor in loaded.operators:
        used = descriptor.input_tensor_ids[: descriptor.input_count]
        inputs = ",".join(str(value) for value in used)
        lines.append(
            f"O {descriptor.operator_id} "
            f"opcode={descriptor.opcode} "
            f"backend={descriptor.backend_id} "
            f"kernel={descriptor.kernel_id} "
            f"in={inputs} "
            f"out={descriptor.output_tensor_ids[0]} "
            f"attr={descriptor.attribute_size}@{descriptor.attribute_offset}"
        )
    return "\n".join(lines) + "\n"


def format_summary(loaded: LoadedPlan, weights_size: int) -> str:
    """C가 인자 없이 실행됐을 때의 요약 출력과 같은 문자열을 만든다."""

    def shape_of(descriptor: TensorDescriptor) -> str:
        return "[" + ", ".join(
            str(value) for value in descriptor.dimensions[: descriptor.rank]
        ) + "]"

    constant_bytes = sum(
        descriptor.logical_byte_size
        for descriptor in loaded.tensors
        if descriptor.storage_type == TensorStorageType.CONSTANT
    )
    lines = [
        f"Loaded plan: {loaded.header.bucket_frames} frames",
        f"Tensor count: {loaded.header.tensor_count}",
        f"Operator count: {loaded.header.operator_count}",
        f"Weight bytes: {weights_size}",
        f"Constant bytes in use: {constant_bytes}",
        f"Attribute section bytes: {len(loaded.attribute_section)}",
    ]
    for descriptor in loaded.tensors:
        if descriptor.storage_type == TensorStorageType.INPUT:
            lines.append(
                f"Input tensor: id {descriptor.tensor_id} "
                f"{_DTYPE_NAMES[descriptor.dtype]} {shape_of(descriptor)} "
                f"{descriptor.logical_byte_size} bytes"
            )
    for descriptor in loaded.tensors:
        if descriptor.storage_type == TensorStorageType.OUTPUT:
            lines.append(
                f"Output tensor: id {descriptor.tensor_id} "
                f"{_DTYPE_NAMES[descriptor.dtype]} {shape_of(descriptor)} "
                f"{descriptor.logical_byte_size} bytes"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "EXECUTION_PLAN_DIR_NAME",
    "ExecutionPlanError",
    "ExecutionPlanResult",
    "LoadedPlan",
    "PLAN_FILE_NAME_TEMPLATE",
    "build_plan_bytes",
    "format_operator_dump",
    "format_summary",
    "format_tensor_dump",
    "plan_file_name",
    "read_execution_plan",
    "verify_plan",
    "write_execution_plan",
]
