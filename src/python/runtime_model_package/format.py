"""Binary container for a fixed CAM++ execution plan and its contracts.

The container does not implement audio preprocessing or speaker scoring.  It
stores the executable embedding graph together with versioned contracts that a
future audio-to-score pipeline must obey.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import hmac
import json
from pathlib import Path
import struct
from typing import Final, Mapping, Sequence


MODEL_PACKAGE_MAGIC: Final[bytes] = b"CAMPMDL\x00"
MODEL_PACKAGE_FORMAT_VERSION: Final[int] = 1
MODEL_PACKAGE_ALIGNMENT: Final[int] = 64
MODEL_PACKAGE_FLAG_LITTLE_ENDIAN: Final[int] = 1
MODEL_PACKAGE_MAX_SECTIONS: Final[int] = 32

# magic, version, header size, count, flags, bucket, reserved,
# file size, table offset, payload offset, payload sha256, reserved
MODEL_PACKAGE_HEADER_STRUCT: Final[struct.Struct] = struct.Struct(
    "<8sIIIIIIQQQ32sQ"
)
MODEL_PACKAGE_HEADER_SIZE: Final[int] = MODEL_PACKAGE_HEADER_STRUCT.size

# type, flags, offset, size, sha256, reserved
MODEL_PACKAGE_SECTION_STRUCT: Final[struct.Struct] = struct.Struct(
    "<IIQQ32sQ"
)
MODEL_PACKAGE_SECTION_SIZE: Final[int] = MODEL_PACKAGE_SECTION_STRUCT.size

assert MODEL_PACKAGE_HEADER_SIZE == 96
assert MODEL_PACKAGE_SECTION_SIZE == 64


class ModelPackageError(RuntimeError):
    """The model package is malformed or violates its deployment contract."""


class ModelPackageSectionType(IntEnum):
    EXECUTION_PLAN = 1
    PACKED_WEIGHTS = 2
    MODEL_METADATA_JSON = 3
    FRONTEND_CONTRACT_JSON = 4
    POSTPROCESS_CONTRACT_JSON = 5


REQUIRED_SECTION_TYPES: Final[frozenset[ModelPackageSectionType]] = frozenset(
    ModelPackageSectionType
)


@dataclass(frozen=True, slots=True)
class ModelPackageSection:
    section_type: ModelPackageSectionType
    payload: bytes


@dataclass(frozen=True, slots=True)
class LoadedModelPackage:
    bucket_frames: int
    flags: int
    file_size: int
    sections: Mapping[ModelPackageSectionType, bytes]

    def json_section(self, section_type: ModelPackageSectionType) -> dict:
        if section_type not in (
            ModelPackageSectionType.MODEL_METADATA_JSON,
            ModelPackageSectionType.FRONTEND_CONTRACT_JSON,
            ModelPackageSectionType.POSTPROCESS_CONTRACT_JSON,
        ):
            raise ModelPackageError(f"section is not JSON: {section_type.name}")
        try:
            value = json.loads(self.sections[section_type].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelPackageError(
                f"invalid JSON section: {section_type.name}"
            ) from exc
        if not isinstance(value, dict):
            raise ModelPackageError(
                f"JSON section root is not an object: {section_type.name}"
            )
        return value


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _align(value: int, alignment: int = MODEL_PACKAGE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sha256(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def build_model_package(
    *, bucket_frames: int, sections: Sequence[ModelPackageSection],
) -> bytes:
    if not 0 < bucket_frames < (1 << 32):
        raise ModelPackageError("bucket_frames must fit uint32 and be positive")
    if not sections or len(sections) > MODEL_PACKAGE_MAX_SECTIONS:
        raise ModelPackageError("invalid model-package section count")

    by_type: dict[ModelPackageSectionType, bytes] = {}
    for section in sections:
        if section.section_type in by_type:
            raise ModelPackageError(
                f"duplicate section: {section.section_type.name}"
            )
        payload = bytes(section.payload)
        if not payload:
            raise ModelPackageError(
                f"empty section: {section.section_type.name}"
            )
        by_type[section.section_type] = payload
    missing = REQUIRED_SECTION_TYPES - set(by_type)
    if missing:
        names = ", ".join(item.name for item in sorted(missing))
        raise ModelPackageError(f"missing required sections: {names}")

    ordered = sorted(by_type.items(), key=lambda item: int(item[0]))
    table_offset = MODEL_PACKAGE_HEADER_SIZE
    table_end = table_offset + len(ordered) * MODEL_PACKAGE_SECTION_SIZE
    payload_offset = _align(table_end)
    cursor = payload_offset
    entries: list[tuple[ModelPackageSectionType, int, bytes]] = []
    for section_type, payload in ordered:
        cursor = _align(cursor)
        entries.append((section_type, cursor, payload))
        cursor += len(payload)
    file_size = cursor

    body = bytearray(file_size - MODEL_PACKAGE_HEADER_SIZE)
    for index, (section_type, offset, payload) in enumerate(entries):
        entry = MODEL_PACKAGE_SECTION_STRUCT.pack(
            int(section_type),
            0,
            offset,
            len(payload),
            _sha256(payload),
            0,
        )
        start = table_offset - MODEL_PACKAGE_HEADER_SIZE
        start += index * MODEL_PACKAGE_SECTION_SIZE
        body[start : start + MODEL_PACKAGE_SECTION_SIZE] = entry
        payload_start = offset - MODEL_PACKAGE_HEADER_SIZE
        body[payload_start : payload_start + len(payload)] = payload

    header = MODEL_PACKAGE_HEADER_STRUCT.pack(
        MODEL_PACKAGE_MAGIC,
        MODEL_PACKAGE_FORMAT_VERSION,
        MODEL_PACKAGE_HEADER_SIZE,
        len(entries),
        MODEL_PACKAGE_FLAG_LITTLE_ENDIAN,
        bucket_frames,
        0,
        file_size,
        table_offset,
        payload_offset,
        _sha256(bytes(body)),
        0,
    )
    return header + bytes(body)


def read_model_package(payload: bytes) -> LoadedModelPackage:
    if len(payload) < MODEL_PACKAGE_HEADER_SIZE:
        raise ModelPackageError("model package is shorter than its header")
    (
        magic,
        version,
        header_size,
        section_count,
        flags,
        bucket_frames,
        reserved,
        file_size,
        table_offset,
        payload_offset,
        payload_checksum,
        reserved_tail,
    ) = MODEL_PACKAGE_HEADER_STRUCT.unpack_from(payload)
    if magic != MODEL_PACKAGE_MAGIC:
        raise ModelPackageError("invalid model-package magic")
    if version != MODEL_PACKAGE_FORMAT_VERSION:
        raise ModelPackageError(f"unsupported model-package version: {version}")
    if header_size != MODEL_PACKAGE_HEADER_SIZE:
        raise ModelPackageError("invalid model-package header size")
    if reserved != 0 or reserved_tail != 0:
        raise ModelPackageError("non-zero reserved model-package field")
    if flags != MODEL_PACKAGE_FLAG_LITTLE_ENDIAN:
        raise ModelPackageError(f"unsupported model-package flags: {flags}")
    if file_size != len(payload):
        raise ModelPackageError("model-package file size mismatch")
    if not 0 < bucket_frames < (1 << 32):
        raise ModelPackageError("invalid model-package bucket")
    if not 0 < section_count <= MODEL_PACKAGE_MAX_SECTIONS:
        raise ModelPackageError("invalid model-package section count")
    if table_offset != MODEL_PACKAGE_HEADER_SIZE:
        raise ModelPackageError("invalid model-package section-table offset")
    table_end = table_offset + section_count * MODEL_PACKAGE_SECTION_SIZE
    if payload_offset != _align(table_end) or payload_offset > file_size:
        raise ModelPackageError("invalid model-package payload offset")
    if not hmac.compare_digest(
        hashlib.sha256(payload[header_size:]).digest(), payload_checksum
    ):
        raise ModelPackageError("model-package payload checksum mismatch")

    sections: dict[ModelPackageSectionType, bytes] = {}
    occupied: list[tuple[int, int]] = []
    for index in range(section_count):
        entry_offset = table_offset + index * MODEL_PACKAGE_SECTION_SIZE
        raw_type, section_flags, offset, size, digest, entry_reserved = (
            MODEL_PACKAGE_SECTION_STRUCT.unpack_from(payload, entry_offset)
        )
        try:
            section_type = ModelPackageSectionType(raw_type)
        except ValueError as exc:
            raise ModelPackageError(f"unknown section type: {raw_type}") from exc
        if section_type in sections:
            raise ModelPackageError(f"duplicate section: {section_type.name}")
        if section_flags != 0 or entry_reserved != 0:
            raise ModelPackageError(
                f"unsupported section flags: {section_type.name}"
            )
        end = offset + size
        if size == 0:
            raise ModelPackageError(f"empty section: {section_type.name}")
        if offset % MODEL_PACKAGE_ALIGNMENT != 0:
            raise ModelPackageError(f"unaligned section: {section_type.name}")
        if offset < payload_offset or end < offset or end > file_size:
            raise ModelPackageError(f"section is out of bounds: {section_type.name}")
        if any(offset < prior_end and prior_offset < end
               for prior_offset, prior_end in occupied):
            raise ModelPackageError(f"overlapping section: {section_type.name}")
        section_payload = payload[offset:end]
        if not hmac.compare_digest(_sha256(section_payload), digest):
            raise ModelPackageError(f"section checksum mismatch: {section_type.name}")
        sections[section_type] = section_payload
        occupied.append((offset, end))

    missing = REQUIRED_SECTION_TYPES - set(sections)
    if missing:
        names = ", ".join(item.name for item in sorted(missing))
        raise ModelPackageError(f"missing required sections: {names}")
    loaded = LoadedModelPackage(
        bucket_frames=bucket_frames,
        flags=flags,
        file_size=file_size,
        sections=sections,
    )
    for section_type in (
        ModelPackageSectionType.MODEL_METADATA_JSON,
        ModelPackageSectionType.FRONTEND_CONTRACT_JSON,
        ModelPackageSectionType.POSTPROCESS_CONTRACT_JSON,
    ):
        loaded.json_section(section_type)
    return loaded


def verify_model_package(path: Path | str) -> LoadedModelPackage:
    target = Path(path)
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise ModelPackageError(f"cannot read model package: {target}") from exc
    return read_model_package(payload)
