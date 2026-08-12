"""RuntimeGraph의 operator를 plan의 실행 명령 표로 굳히는 모듈.

1,438개 runtime node는 1,438개의 C 함수가 아니라 1,438개의 명령이 된다.
같은 opcode는 같은 kernel 하나를 반복 호출할 뿐이다::

    225개 QLinearConv node -> opcode QLINEAR_CONV 명령 225개 -> C 함수 1개를 225번 호출

표는 두 부분으로 나뉜다. 고정 64바이트 descriptor가 opcode와 Tensor ID를
담고, 가변 길이 attribute block이 ``perm``이나 ``pads`` 같은 값을 담는다.
attribute block은 내용이 같으면 하나만 기록하고 여러 operator가 같은 offset을
가리킨다. CAM++는 같은 conv 설정이 수백 번 반복되므로 이 공유가 크게 먹힌다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..format.binary_format_schema import (
    DEFAULT_KERNEL_ID,
    INVALID_TENSOR_ID,
    OPERATOR_DESCRIPTOR_SIZE,
    OPERATOR_INPUT_CAPACITY,
    OPERATOR_OUTPUT_CAPACITY,
    PLAN_SECTION_ALIGNMENT,
    BackendId,
    OperatorDescriptor,
    TensorStorageType,
    decode_attribute_block,
    encode_attribute_block,
)
from ..runtime_ir import RuntimeGraph


class OperatorTableError(ValueError):
    """Operator table을 만들 수 없거나 실행 순서가 성립하지 않을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class OperatorTable:
    """한 bucket의 plan에 실릴 Operator descriptor 표와 attribute section."""

    descriptors: tuple[OperatorDescriptor, ...]
    attribute_section: bytes
    backend_id: BackendId
    attribute_blocks: int
    names: tuple[str, ...] = ()
    _ids: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        object.__setattr__(self, "names", tuple(self.names))
        if self.names and len(self.names) != len(self.descriptors):
            raise OperatorTableError("descriptor 수와 이름 수가 다르다")
        object.__setattr__(
            self,
            "_ids",
            MappingProxyType({name: index for index, name in enumerate(self.names)}),
        )

    def __len__(self) -> int:
        return len(self.descriptors)

    @property
    def byte_size(self) -> int:
        return len(self.descriptors) * OPERATOR_DESCRIPTOR_SIZE

    @property
    def attribute_section_size(self) -> int:
        return len(self.attribute_section)

    def id_of(self, name: str) -> int:
        """디버깅과 ORT 대조에만 쓴다. plan에는 이름이 실리지 않는다."""

        try:
            return self._ids[name]
        except KeyError as exc:
            raise OperatorTableError(f"Operator table에 없는 이름: {name!r}") from exc

    def attributes_of(self, operator_id: int) -> dict[str, tuple]:
        """기록된 attribute block을 되읽는다. C loader와 같은 경로를 탄다."""

        if not 0 <= operator_id < len(self.descriptors):
            raise OperatorTableError(f"Operator ID 범위를 벗어남: {operator_id}")
        descriptor = self.descriptors[operator_id]
        if descriptor.attribute_size == 0:
            return {}
        return decode_attribute_block(
            self.attribute_section, descriptor.attribute_offset
        )

    def pack(self) -> bytes:
        """표 전체를 plan에 실을 바이트열로 직렬화한다."""

        return b"".join(descriptor.pack() for descriptor in self.descriptors)


def _padded_input_ids(operator_id: int, input_ids: tuple[int, ...]) -> tuple[int, ...]:
    if not 1 <= len(input_ids) <= OPERATOR_INPUT_CAPACITY:
        raise OperatorTableError(
            f"Operator {operator_id}의 입력 {len(input_ids)}개가 형식 한계 "
            f"1..{OPERATOR_INPUT_CAPACITY}를 벗어난다"
        )
    padding = OPERATOR_INPUT_CAPACITY - len(input_ids)
    return input_ids + (INVALID_TENSOR_ID,) * padding


def build_operator_table(
    graph: RuntimeGraph,
    *,
    backend_id: BackendId = BackendId.AUTO,
    kernel_id: int = DEFAULT_KERNEL_ID,
) -> OperatorTable:
    """RuntimeGraph 하나를 실행 명령 표로 굳힌다.

    attribute가 같은 operator는 attribute section의 같은 block을 가리킨다.

    ``backend_id``는 기본이 ``AUTO``다. backend는 모델의 성질이 아니라 배포할 때
    고르는 것이므로, plan 하나가 어느 backend에서든 실행되게 두고 실제 선택은
    Runtime이 한다. 특정 backend로 못박으면 그 plan은 다른 backend에서 거부되고,
    같은 plan으로 backend만 바꿔 값을 대조하는 검증 경로가 막힌다.
    """

    validate_execution_order(graph)

    section = bytearray()
    block_offsets: dict[bytes, int] = {}
    descriptors: list[OperatorDescriptor] = []
    names: list[str] = []

    for operator in graph.operators:
        block = encode_attribute_block(operator.attributes)
        if block:
            offset = block_offsets.get(block)
            if offset is None:
                remainder = len(section) % PLAN_SECTION_ALIGNMENT
                if remainder:
                    section.extend(b"\x00" * (PLAN_SECTION_ALIGNMENT - remainder))
                offset = len(section)
                section.extend(block)
                block_offsets[block] = offset
            attribute_size = len(block)
        else:
            offset = 0
            attribute_size = 0

        if len(operator.output_tensor_ids) != OPERATOR_OUTPUT_CAPACITY:
            raise OperatorTableError(
                f"Operator {operator.name!r}의 출력이 "
                f"{OPERATOR_OUTPUT_CAPACITY}개가 아니다"
            )

        descriptors.append(
            OperatorDescriptor(
                operator_id=operator.operator_id,
                opcode=int(operator.opcode),
                input_count=len(operator.input_tensor_ids),
                output_count=len(operator.output_tensor_ids),
                input_tensor_ids=_padded_input_ids(
                    operator.operator_id, operator.input_tensor_ids
                ),
                output_tensor_ids=tuple(operator.output_tensor_ids),
                attribute_offset=offset,
                attribute_size=attribute_size,
                backend_id=int(backend_id),
                kernel_id=kernel_id,
            )
        )
        names.append(operator.name)

    return OperatorTable(
        descriptors=tuple(descriptors),
        attribute_section=bytes(section),
        backend_id=BackendId(backend_id),
        attribute_blocks=len(block_offsets),
        names=tuple(names),
    )


def validate_execution_order(graph: RuntimeGraph) -> None:
    """표를 위에서 아래로 읽었을 때 실행이 성립하는지 확인한다.

    Operator를 순서대로 훑으면서, 입력 Tensor가 이미 준비되어 있는지만 본다.
    준비되었다는 것은 graph input이거나 상수이거나 앞선 operator가 생산했다는
    뜻이다. RuntimeGraph의 producer/consumer 불변식과는 별개로, 표 자체를
    독립적으로 검사한다.
    """

    ready = {
        tensor.tensor_id
        for tensor in graph.tensors
        if tensor.storage_type
        in (TensorStorageType.INPUT, TensorStorageType.CONSTANT)
    }
    produced: set[int] = set()

    for position, operator in enumerate(graph.operators):
        if operator.operator_id != position:
            raise OperatorTableError(
                f"Operator ID가 실행 순서와 다르다: {position} 자리에 "
                f"{operator.operator_id}"
            )
        for tensor_id in operator.input_tensor_ids:
            if tensor_id not in ready:
                producer = graph.tensor(tensor_id).producer
                raise OperatorTableError(
                    f"Operator {operator.operator_id} {operator.name!r}가 아직 "
                    f"생산되지 않은 Tensor {tensor_id} "
                    f"{graph.tensor(tensor_id).name!r}를 읽는다 "
                    f"(producer={producer!r})"
                )
        for tensor_id in operator.output_tensor_ids:
            if tensor_id in produced:
                raise OperatorTableError(
                    f"Tensor {tensor_id}를 두 개 이상의 Operator가 생산한다"
                )
            produced.add(tensor_id)
            ready.add(tensor_id)

    for tensor in graph.tensors:
        if tensor.storage_type in (
            TensorStorageType.INPUT,
            TensorStorageType.CONSTANT,
        ):
            continue
        if tensor.tensor_id not in produced:
            raise OperatorTableError(
                f"Tensor {tensor.name!r}를 아무 Operator도 생산하지 않는다"
            )


__all__ = [
    "OperatorTable",
    "OperatorTableError",
    "build_operator_table",
    "validate_execution_order",
]
