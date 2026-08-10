"""CAM++ execution-plan binary의 공통 디스크 형식.

이 모듈은 Python exporter가 plan header를 직렬화하고, C Runtime이 같은
바이트 배열을 해석할 수 있도록 ABI를 고정한다. 모든 다중 바이트 정수는
little-endian이며 header 뒤의 payload SHA-256을 checksum으로 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum, IntFlag
import hashlib
import hmac
import struct
from typing import Final


PLAN_MAGIC: Final[bytes] = b"CAMPLAN\x00" #PLAN_MAGIC: 해당 파일이 CAM++ 실행 계획 파일인지 확인하는 8바이트 값
PLAN_FORMAT_VERSION: Final[int] = 1 #binary 형식 버전
PLAN_CHECKSUM_SIZE: Final[int] = hashlib.sha256().digest_size #항상 32바이트
PLAN_SECTION_ALIGNMENT: Final[int] = 8 #각 데이터 구역은 8바이트에서 시작

# 8s: magic
# 4I: format_version, bucket_frames, tensor_count, operator_count
# 3Q: tensor_table_offset, operator_table_offset, attribute_section_offset
# 32s: header 뒤 payload의 SHA-256
PLAN_HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<8sIIIIQQQ32s")
PLAN_HEADER_SIZE: Final[int] = PLAN_HEADER_STRUCT.size

# I + 4B + 4I + 4I + 3Q + 4I = 80 bytes
TENSOR_DESCRIPTOR_STRUCT: Final[struct.Struct] = struct.Struct(
    "<IBBBB4I4IQQQIIII"
)
TENSOR_DESCRIPTOR_SIZE: Final[int] = TENSOR_DESCRIPTOR_STRUCT.size
TENSOR_MAX_RANK: Final[int] = 4

# I + H + 2B + 9I + I + Q + I + 2H = 64 bytes
OPERATOR_DESCRIPTOR_STRUCT: Final[struct.Struct] = struct.Struct(
    "<IHBB9IIQIHH"
)
OPERATOR_DESCRIPTOR_SIZE: Final[int] = OPERATOR_DESCRIPTOR_STRUCT.size
OPERATOR_INPUT_CAPACITY: Final[int] = 9
OPERATOR_OUTPUT_CAPACITY: Final[int] = 1

UINT32_MAX: Final[int] = (1 << 32) - 1
UINT64_MAX: Final[int] = (1 << 64) - 1
UINT16_MAX: Final[int] = (1 << 16) - 1
UINT8_MAX: Final[int] = (1 << 8) - 1
EMPTY_SHA256: Final[bytes] = hashlib.sha256(b"").digest()

INVALID_TENSOR_ID: Final[int] = UINT32_MAX
INVALID_QUANTIZATION_INDEX: Final[int] = UINT32_MAX
INVALID_OPERATOR_INDEX: Final[int] = UINT32_MAX
INVALID_DATA_OFFSET: Final[int] = UINT64_MAX
DEFAULT_KERNEL_ID: Final[int] = 0


class TensorDType(IntEnum):
    """C CamppTensorDType과 숫자값을 공유한다."""

    INVALID = 0
    FLOAT32 = 1
    UINT8 = 2
    INT8 = 3
    INT32 = 4
    INT64 = 5
    BOOL = 6
    FLOAT16 = 7


class TensorStorageType(IntEnum):
    """Tensor data_offset이 기준으로 삼을 저장 영역을 나타낸다."""

    INVALID = 0
    INPUT = 1
    OUTPUT = 2
    CONSTANT = 3
    ACTIVATION = 4
    VIEW = 5


class TensorFlags(IntFlag):
    """하나의 uint8 필드에 조합해 저장하는 Tensor 속성이다."""

    NONE = 0
    READ_ONLY = 1 << 0
    CONTIGUOUS = 1 << 1
    EXTERNAL = 1 << 2
    ALIASED = 1 << 3
    DENSE_SLAB = 1 << 4


ALL_TENSOR_FLAGS: Final[int] = int(
    TensorFlags.READ_ONLY
    | TensorFlags.CONTIGUOUS
    | TensorFlags.EXTERNAL
    | TensorFlags.ALIASED
    | TensorFlags.DENSE_SLAB
)


class OperatorCode(IntEnum):
    """현재 동결된 CAM++ Runtime operator의 고정 opcode다."""

    INVALID = 0
    QLINEAR_CONV = 1
    QUANTIZE_LINEAR = 2
    DEQUANTIZE_LINEAR = 3
    BATCH_NORMALIZATION = 4
    RELU = 5
    SIGMOID = 6
    AVERAGE_POOL = 7
    REDUCE_MEAN = 8
    ADD = 9
    MUL = 10
    SUB = 11
    DIV = 12
    SQRT = 13
    CONCAT = 14
    EXPAND = 15
    SLICE = 16
    RESHAPE = 17
    TRANSPOSE = 18
    SQUEEZE = 19
    UNSQUEEZE = 20


class BackendId(IntEnum):
    """Operator kernel을 제공하는 backend의 고정 ID다."""

    AUTO = 0
    CPU_REFERENCE = 1
    CPU_AARCH64 = 2
    GPU_OPENCL = 3
    GPU_VULKAN = 4


class BinaryFormatError(ValueError):
    """Execution-plan binary가 고정 ABI를 위반했을 때 발생한다."""


def _require_uint(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BinaryFormatError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise BinaryFormatError(f"{name} is outside 0..{maximum}: {value}")


def _require_aligned(name: str, value: int) -> None:
    if value % PLAN_SECTION_ALIGNMENT != 0:
        raise BinaryFormatError(
            f"{name} must be {PLAN_SECTION_ALIGNMENT}-byte aligned: {value}"
        )


def _require_enum(name: str, value: int, enum_type: type[IntEnum]) -> None:
    _require_uint(name, int(value), UINT16_MAX)
    try:
        enum_type(value)
    except ValueError as exc:
        raise BinaryFormatError(f"unsupported {name}: {value}") from exc


def _require_uint_tuple(
    name: str, values: tuple[int, ...], length: int, maximum: int
) -> None:
    if not isinstance(values, tuple):
        raise BinaryFormatError(f"{name} must be a tuple")
    if len(values) != length:
        raise BinaryFormatError(f"{name} must contain {length} values")
    for index, value in enumerate(values):
        _require_uint(f"{name}[{index}]", value, maximum)


def _require_record_available(
    name: str,
    data: bytes | bytearray | memoryview,
    offset: int,
    record_size: int,
) -> None:
    _require_uint("offset", offset, UINT64_MAX)
    if offset + record_size > len(data):
        raise BinaryFormatError(
            f"truncated {name}: need {record_size} bytes at {offset}, "
            f"have {len(data) - offset}"
        )


def payload_sha256(payload: bytes | bytearray | memoryview) -> bytes:
    """Header 뒤 payload에 기록할 32바이트 SHA-256을 계산한다."""

    return hashlib.sha256(payload).digest()


@dataclass(frozen=True, slots=True)
class PlanHeader:
    """80바이트 execution-plan header의 host-side 표현."""

    magic: bytes
    format_version: int
    bucket_frames: int
    tensor_count: int
    operator_count: int
    tensor_table_offset: int
    operator_table_offset: int
    attribute_section_offset: int
    checksum: bytes

    @classmethod
    def create(
        cls,
        *,
        bucket_frames: int,
        tensor_count: int,
        operator_count: int,
        tensor_table_offset: int,
        operator_table_offset: int,
        attribute_section_offset: int,
        payload: bytes | bytearray | memoryview,
    ) -> "PlanHeader":
        """현재 format version과 payload checksum으로 header를 생성한다."""

        header = cls(
            magic=PLAN_MAGIC,
            format_version=PLAN_FORMAT_VERSION,
            bucket_frames=bucket_frames,
            tensor_count=tensor_count,
            operator_count=operator_count,
            tensor_table_offset=tensor_table_offset,
            operator_table_offset=operator_table_offset,
            attribute_section_offset=attribute_section_offset,
            checksum=payload_sha256(payload),
        )
        header.validate()
        return header

    def validate(self, *, require_current_version: bool = True) -> None:
        """필드 범위, section 순서와 정렬을 검사한다."""

        if self.magic != PLAN_MAGIC:
            raise BinaryFormatError(f"invalid plan magic: {self.magic!r}")

        _require_uint("format_version", self.format_version, UINT32_MAX)
        if require_current_version and self.format_version != PLAN_FORMAT_VERSION:
            raise BinaryFormatError(
                f"unsupported format version: {self.format_version} "
                f"(expected {PLAN_FORMAT_VERSION})"
            )

        _require_uint("bucket_frames", self.bucket_frames, UINT32_MAX)
        _require_uint("tensor_count", self.tensor_count, UINT32_MAX)
        _require_uint("operator_count", self.operator_count, UINT32_MAX)
        if self.bucket_frames == 0:
            raise BinaryFormatError("bucket_frames must be greater than zero")
        if self.tensor_count == 0:
            raise BinaryFormatError("tensor_count must be greater than zero")
        if self.operator_count == 0:
            raise BinaryFormatError("operator_count must be greater than zero")

        offsets = (
            ("tensor_table_offset", self.tensor_table_offset),
            ("operator_table_offset", self.operator_table_offset),
            ("attribute_section_offset", self.attribute_section_offset),
        )
        for name, value in offsets:
            _require_uint(name, value, UINT64_MAX)
            _require_aligned(name, value)

        if self.tensor_table_offset < PLAN_HEADER_SIZE:
            raise BinaryFormatError(
                "tensor_table_offset must start at or after the plan header"
            )
        if self.operator_table_offset < self.tensor_table_offset:
            raise BinaryFormatError(
                "operator_table_offset must not precede tensor_table_offset"
            )
        if self.attribute_section_offset < self.operator_table_offset:
            raise BinaryFormatError(
                "attribute_section_offset must not precede operator_table_offset"
            )

        if not isinstance(self.checksum, bytes):
            raise BinaryFormatError("checksum must be bytes")
        if len(self.checksum) != PLAN_CHECKSUM_SIZE:
            raise BinaryFormatError(
                f"checksum must be {PLAN_CHECKSUM_SIZE} bytes: {len(self.checksum)}"
            )

    def pack(self) -> bytes:
        """Header를 정확히 80바이트 little-endian 배열로 직렬화한다."""

        self.validate()
        packed = PLAN_HEADER_STRUCT.pack(
            self.magic,
            self.format_version,
            self.bucket_frames,
            self.tensor_count,
            self.operator_count,
            self.tensor_table_offset,
            self.operator_table_offset,
            self.attribute_section_offset,
            self.checksum,
        )
        if len(packed) != PLAN_HEADER_SIZE:
            raise AssertionError("internal plan-header size mismatch")
        return packed

    @classmethod
    def unpack_from(
        cls,
        data: bytes | bytearray | memoryview,
        offset: int = 0,
        *,
        require_current_version: bool = True,
    ) -> "PlanHeader":
        """바이트 배열의 지정 위치에서 header를 읽고 즉시 검증한다."""

        _require_record_available("plan header", data, offset, PLAN_HEADER_SIZE)

        header = cls(*PLAN_HEADER_STRUCT.unpack_from(data, offset))
        header.validate(require_current_version=require_current_version)
        return header

    def with_payload_checksum(
        self, payload: bytes | bytearray | memoryview
    ) -> "PlanHeader":
        """나머지 필드는 유지하고 payload checksum만 다시 계산한다."""

        return replace(self, checksum=payload_sha256(payload))

    def verify_payload_checksum(
        self, payload: bytes | bytearray | memoryview
    ) -> bool:
        """저장된 checksum과 payload SHA-256을 timing-safe 방식으로 비교한다."""

        return hmac.compare_digest(self.checksum, payload_sha256(payload))


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    """80바이트 Tensor descriptor의 host-side 표현."""

    tensor_id: int
    dtype: int
    rank: int
    storage_type: int
    flags: int
    dimensions: tuple[int, int, int, int]
    byte_strides: tuple[int, int, int, int]
    data_offset: int
    logical_byte_size: int
    storage_span_bytes: int
    alias_of_tensor_id: int
    quantization_index: int
    first_use: int
    last_use: int

    def validate(self) -> None:
        """고정 길이, enum, 수명과 view 참조 규칙을 검사한다."""

        _require_uint("tensor_id", self.tensor_id, UINT32_MAX)
        _require_enum("dtype", self.dtype, TensorDType)
        if int(self.dtype) == TensorDType.INVALID:
            raise BinaryFormatError("dtype must not be INVALID")

        _require_uint("rank", self.rank, UINT8_MAX)
        if self.rank > TENSOR_MAX_RANK:
            raise BinaryFormatError(
                f"rank exceeds CAM++ maximum {TENSOR_MAX_RANK}: {self.rank}"
            )

        _require_enum("storage_type", self.storage_type, TensorStorageType)
        if int(self.storage_type) == TensorStorageType.INVALID:
            raise BinaryFormatError("storage_type must not be INVALID")

        _require_uint("flags", self.flags, UINT8_MAX)
        if self.flags & ~ALL_TENSOR_FLAGS:
            raise BinaryFormatError(f"unknown Tensor flags: 0x{self.flags:02x}")

        _require_uint_tuple("dimensions", self.dimensions, TENSOR_MAX_RANK, UINT32_MAX)
        _require_uint_tuple(
            "byte_strides", self.byte_strides, TENSOR_MAX_RANK, UINT32_MAX
        )
        for axis, dimension in enumerate(self.dimensions):
            if axis < self.rank and dimension == 0:
                raise BinaryFormatError(f"dimensions[{axis}] must be greater than zero")
            if axis >= self.rank and dimension != 1:
                raise BinaryFormatError(
                    f"unused dimensions[{axis}] must be 1, got {dimension}"
                )
        for axis in range(self.rank, TENSOR_MAX_RANK):
            if self.byte_strides[axis] != 0:
                raise BinaryFormatError(
                    f"unused byte_strides[{axis}] must be 0, "
                    f"got {self.byte_strides[axis]}"
                )

        _require_uint("data_offset", self.data_offset, UINT64_MAX)
        _require_uint("logical_byte_size", self.logical_byte_size, UINT64_MAX)
        _require_uint("storage_span_bytes", self.storage_span_bytes, UINT64_MAX)
        if self.logical_byte_size == 0:
            raise BinaryFormatError("logical_byte_size must be greater than zero")
        if self.storage_span_bytes == 0:
            raise BinaryFormatError("storage_span_bytes must be greater than zero")

        _require_uint("alias_of_tensor_id", self.alias_of_tensor_id, UINT32_MAX)
        _require_uint("quantization_index", self.quantization_index, UINT32_MAX)
        _require_uint("first_use", self.first_use, UINT32_MAX)
        _require_uint("last_use", self.last_use, UINT32_MAX)

        is_view = int(self.storage_type) == TensorStorageType.VIEW
        is_aliased = bool(self.flags & TensorFlags.ALIASED)
        if is_view and self.alias_of_tensor_id == INVALID_TENSOR_ID:
            raise BinaryFormatError("VIEW Tensor must reference alias_of_tensor_id")
        if is_view and not is_aliased:
            raise BinaryFormatError("VIEW Tensor must set the ALIASED flag")
        if not is_view and self.alias_of_tensor_id != INVALID_TENSOR_ID:
            raise BinaryFormatError(
                "non-VIEW Tensor must use INVALID_TENSOR_ID for alias_of_tensor_id"
            )

        if (
            self.first_use != INVALID_OPERATOR_INDEX
            and self.last_use != INVALID_OPERATOR_INDEX
            and self.first_use > self.last_use
        ):
            raise BinaryFormatError("first_use must not be greater than last_use")

    def pack(self) -> bytes:
        """Descriptor를 정확히 80바이트 little-endian 배열로 직렬화한다."""

        self.validate()
        return TENSOR_DESCRIPTOR_STRUCT.pack(
            self.tensor_id,
            int(self.dtype),
            self.rank,
            int(self.storage_type),
            int(self.flags),
            *self.dimensions,
            *self.byte_strides,
            self.data_offset,
            self.logical_byte_size,
            self.storage_span_bytes,
            self.alias_of_tensor_id,
            self.quantization_index,
            self.first_use,
            self.last_use,
        )

    @classmethod
    def unpack_from(
        cls, data: bytes | bytearray | memoryview, offset: int = 0
    ) -> "TensorDescriptor":
        """바이트 배열에서 Tensor descriptor를 읽고 검증한다."""

        _require_record_available(
            "Tensor descriptor", data, offset, TENSOR_DESCRIPTOR_SIZE
        )
        values = TENSOR_DESCRIPTOR_STRUCT.unpack_from(data, offset)
        descriptor = cls(
            tensor_id=values[0],
            dtype=values[1],
            rank=values[2],
            storage_type=values[3],
            flags=values[4],
            dimensions=tuple(values[5:9]),
            byte_strides=tuple(values[9:13]),
            data_offset=values[13],
            logical_byte_size=values[14],
            storage_span_bytes=values[15],
            alias_of_tensor_id=values[16],
            quantization_index=values[17],
            first_use=values[18],
            last_use=values[19],
        )
        descriptor.validate()
        return descriptor


@dataclass(frozen=True, slots=True)
class OperatorDescriptor:
    """64바이트 Operator descriptor의 host-side 표현."""

    operator_id: int
    opcode: int
    input_count: int
    output_count: int
    input_tensor_ids: tuple[int, ...]
    output_tensor_ids: tuple[int, ...]
    attribute_offset: int
    attribute_size: int
    backend_id: int
    kernel_id: int

    def validate(self) -> None:
        """Opcode, Tensor ID 슬롯, attribute와 backend 규칙을 검사한다."""

        _require_uint("operator_id", self.operator_id, UINT32_MAX)
        _require_enum("opcode", self.opcode, OperatorCode)
        if int(self.opcode) == OperatorCode.INVALID:
            raise BinaryFormatError("opcode must not be INVALID")

        _require_uint("input_count", self.input_count, UINT8_MAX)
        _require_uint("output_count", self.output_count, UINT8_MAX)
        if not 1 <= self.input_count <= OPERATOR_INPUT_CAPACITY:
            raise BinaryFormatError(
                f"input_count must be 1..{OPERATOR_INPUT_CAPACITY}: "
                f"{self.input_count}"
            )
        if self.output_count != OPERATOR_OUTPUT_CAPACITY:
            raise BinaryFormatError(
                f"format v1 requires exactly {OPERATOR_OUTPUT_CAPACITY} output"
            )

        _require_uint_tuple(
            "input_tensor_ids",
            self.input_tensor_ids,
            OPERATOR_INPUT_CAPACITY,
            UINT32_MAX,
        )
        _require_uint_tuple(
            "output_tensor_ids",
            self.output_tensor_ids,
            OPERATOR_OUTPUT_CAPACITY,
            UINT32_MAX,
        )
        for index, tensor_id in enumerate(self.input_tensor_ids):
            if index < self.input_count and tensor_id == INVALID_TENSOR_ID:
                raise BinaryFormatError(
                    f"used input_tensor_ids[{index}] must not be invalid"
                )
            if index >= self.input_count and tensor_id != INVALID_TENSOR_ID:
                raise BinaryFormatError(
                    f"unused input_tensor_ids[{index}] must be invalid"
                )
        if self.output_tensor_ids[0] == INVALID_TENSOR_ID:
            raise BinaryFormatError("output_tensor_ids[0] must not be invalid")

        _require_uint("attribute_offset", self.attribute_offset, UINT64_MAX)
        _require_uint("attribute_size", self.attribute_size, UINT32_MAX)
        if self.attribute_size == 0 and self.attribute_offset != 0:
            raise BinaryFormatError(
                "attribute_offset must be zero when attribute_size is zero"
            )
        if self.attribute_size > 0:
            _require_aligned("attribute_offset", self.attribute_offset)

        _require_enum("backend_id", self.backend_id, BackendId)
        _require_uint("kernel_id", self.kernel_id, UINT16_MAX)

    def pack(self) -> bytes:
        """Descriptor를 정확히 64바이트 little-endian 배열로 직렬화한다."""

        self.validate()
        return OPERATOR_DESCRIPTOR_STRUCT.pack(
            self.operator_id,
            int(self.opcode),
            self.input_count,
            self.output_count,
            *self.input_tensor_ids,
            *self.output_tensor_ids,
            self.attribute_offset,
            self.attribute_size,
            int(self.backend_id),
            self.kernel_id,
        )

    @classmethod
    def unpack_from(
        cls, data: bytes | bytearray | memoryview, offset: int = 0
    ) -> "OperatorDescriptor":
        """바이트 배열에서 Operator descriptor를 읽고 검증한다."""

        _require_record_available(
            "Operator descriptor", data, offset, OPERATOR_DESCRIPTOR_SIZE
        )
        values = OPERATOR_DESCRIPTOR_STRUCT.unpack_from(data, offset)
        descriptor = cls(
            operator_id=values[0],
            opcode=values[1],
            input_count=values[2],
            output_count=values[3],
            input_tensor_ids=tuple(values[4:13]),
            output_tensor_ids=(values[13],),
            attribute_offset=values[14],
            attribute_size=values[15],
            backend_id=values[16],
            kernel_id=values[17],
        )
        descriptor.validate()
        return descriptor


if PLAN_HEADER_SIZE != 80:
    raise AssertionError(f"PlanHeader ABI changed unexpectedly: {PLAN_HEADER_SIZE}")
if TENSOR_DESCRIPTOR_SIZE != 80:
    raise AssertionError(
        f"TensorDescriptor ABI changed unexpectedly: {TENSOR_DESCRIPTOR_SIZE}"
    )
if OPERATOR_DESCRIPTOR_SIZE != 64:
    raise AssertionError(
        f"OperatorDescriptor ABI changed unexpectedly: {OPERATOR_DESCRIPTOR_SIZE}"
    )
