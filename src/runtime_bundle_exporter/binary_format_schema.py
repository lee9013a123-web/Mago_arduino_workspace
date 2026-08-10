"""CAM++ execution-plan binary의 공통 디스크 형식.

이 모듈은 Python exporter가 plan header를 직렬화하고, C Runtime이 같은
바이트 배열을 해석할 수 있도록 ABI를 고정한다. 모든 다중 바이트 정수는
little-endian이며 header 뒤의 payload SHA-256을 checksum으로 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import hmac
import struct
from typing import Final


PLAN_MAGIC: Final[bytes] = b"CAMPLAN\x00"
PLAN_FORMAT_VERSION: Final[int] = 1
PLAN_CHECKSUM_SIZE: Final[int] = hashlib.sha256().digest_size
PLAN_SECTION_ALIGNMENT: Final[int] = 8

# 8s: magic
# 4I: format_version, bucket_frames, tensor_count, operator_count
# 3Q: tensor_table_offset, operator_table_offset, attribute_section_offset
# 32s: header 뒤 payload의 SHA-256
PLAN_HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<8sIIIIQQQ32s")
PLAN_HEADER_SIZE: Final[int] = PLAN_HEADER_STRUCT.size

UINT32_MAX: Final[int] = (1 << 32) - 1
UINT64_MAX: Final[int] = (1 << 64) - 1
EMPTY_SHA256: Final[bytes] = hashlib.sha256(b"").digest()


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

        _require_uint("offset", offset, UINT64_MAX)
        if offset + PLAN_HEADER_SIZE > len(data):
            raise BinaryFormatError(
                f"truncated plan header: need {PLAN_HEADER_SIZE} bytes at {offset}, "
                f"have {len(data) - offset}"
            )

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


if PLAN_HEADER_SIZE != 80:
    raise AssertionError(f"PlanHeader ABI changed unexpectedly: {PLAN_HEADER_SIZE}")
