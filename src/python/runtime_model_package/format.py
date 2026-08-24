"""Binary container for a fixed CAM++ execution plan and its contracts.

The container does not implement audio preprocessing or speaker scoring.  It
stores the executable embedding graph together with versioned contracts that a
future audio-to-score pipeline must obey.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import hmac
import json
from pathlib import Path
import struct
from typing import Final, Mapping, Sequence


MODEL_PACKAGE_MAGIC: Final[bytes] = b"CAMPMDL\x00"
MODEL_PACKAGE_FORMAT_VERSION: Final[int] = 2
# v1은 bucket 하나만 담았다.  v2는 EXECUTION_PLAN을 bucket마다 하나씩 두고
# section flags에 그 bucket frame 수를 넣는다.  weights는 4개 bucket이 공유하므로
# 단일 PACKED_WEIGHTS로 남는다 -- bucket별 파일을 따로 만들면 7.5 MB weights가
# 그만큼 복제된다.  읽기는 v1도 계속 지원한다.
MODEL_PACKAGE_MIN_READ_VERSION: Final[int] = 1
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
    # v2 선택 섹션.  weight windowing과 layer-hybrid dispatch를 모델에 담는다.
    STREAMING_WEIGHTS = 6        # page 정렬된 windowing용 weight 배치
    WEIGHT_STREAM_SCHEDULE = 7   # block별 prefetch/evict operator 표
    KERNEL_DISPATCH_TABLE = 8    # operator별 커널 선택 (layer-hybrid)


REQUIRED_SECTION_TYPES: Final[frozenset[ModelPackageSectionType]] = frozenset({
    ModelPackageSectionType.EXECUTION_PLAN,
    ModelPackageSectionType.PACKED_WEIGHTS,
    ModelPackageSectionType.MODEL_METADATA_JSON,
    ModelPackageSectionType.FRONTEND_CONTRACT_JSON,
    ModelPackageSectionType.POSTPROCESS_CONTRACT_JSON,
})

# bucket마다 하나씩 올 수 있는 섹션.  section flags에 bucket frame 수를 담는다.
PER_BUCKET_SECTION_TYPES: Final[frozenset[ModelPackageSectionType]] = frozenset({
    ModelPackageSectionType.EXECUTION_PLAN,
    ModelPackageSectionType.WEIGHT_STREAM_SCHEDULE,
    ModelPackageSectionType.KERNEL_DISPATCH_TABLE,
})


@dataclass(frozen=True, slots=True)
class ModelPackageSection:
    section_type: ModelPackageSectionType
    payload: bytes


@dataclass(frozen=True, slots=True)
class LoadedModelPackage:
    bucket_frames: int          # v1 호환: v2에서는 첫 bucket
    flags: int
    file_size: int
    sections: Mapping[ModelPackageSectionType, bytes]
    version: int = MODEL_PACKAGE_FORMAT_VERSION
    # bucket frame 수 -> execution plan.  v1은 항목이 하나다.
    plans_by_bucket: Mapping[int, bytes] = field(default_factory=dict)
    # 선택 섹션.  {section_type: {bucket: payload}}
    per_bucket_sections: Mapping[
        ModelPackageSectionType, Mapping[int, bytes]] = field(
            default_factory=dict)

    def optional_section(
        self, section_type: ModelPackageSectionType, bucket_frames: int,
    ) -> bytes | None:
        """없으면 None.  streaming/dispatch는 모델마다 선택적이다."""
        return self.per_bucket_sections.get(section_type, {}).get(
            bucket_frames)

    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(sorted(self.plans_by_bucket))

    def plan_for(self, bucket_frames: int) -> bytes:
        try:
            return self.plans_by_bucket[bucket_frames]
        except KeyError as exc:
            raise ModelPackageError(
                f"package has no plan for bucket {bucket_frames}; "
                f"available: {list(self.buckets)}"
            ) from exc

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
    *, bucket_frames: int | None = None,
    sections: Sequence[ModelPackageSection] = (),
    plans_by_bucket: Mapping[int, bytes] | None = None,
    per_bucket_sections: Mapping[
        ModelPackageSectionType, Mapping[int, bytes]] | None = None,
) -> bytes:
    """모델 패키지를 만든다.

    `plans_by_bucket`을 주면 bucket마다 EXECUTION_PLAN을 하나씩 넣는다.  각 plan
    section은 flags에 자기 bucket frame 수를 담는다.  weights는 bucket이 공유하므로
    하나만 들어간다.  단일 bucket이면 `bucket_frames` + EXECUTION_PLAN section을
    쓰는 기존 방식도 그대로 동작한다.
    """
    plan_map: dict[int, bytes] = dict(plans_by_bucket or {})
    other = [s for s in sections
             if s.section_type != ModelPackageSectionType.EXECUTION_PLAN]
    if not plan_map:
        for s in sections:
            if s.section_type == ModelPackageSectionType.EXECUTION_PLAN:
                if bucket_frames is None:
                    raise ModelPackageError(
                        "single-plan package needs bucket_frames")
                plan_map[int(bucket_frames)] = bytes(s.payload)
    if not plan_map:
        raise ModelPackageError("model package needs at least one plan")
    for bucket in plan_map:
        if not 0 < bucket < (1 << 32):
            raise ModelPackageError(
                "bucket_frames must fit uint32 and be positive")
    if bucket_frames is None:
        bucket_frames = min(plan_map)

    by_type: dict[ModelPackageSectionType, bytes] = {}
    for section in other:
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
    for bucket, payload in plan_map.items():
        if not payload:
            raise ModelPackageError(f"empty execution plan for bucket {bucket}")
    missing = REQUIRED_SECTION_TYPES - (
        set(by_type) | {ModelPackageSectionType.EXECUTION_PLAN})
    if missing:
        names = ", ".join(item.name for item in sorted(missing))
        raise ModelPackageError(f"missing required sections: {names}")

    # (type, flags, payload) 순서로 펼친다.  plan은 bucket 오름차순이다.
    ordered: list[tuple[ModelPackageSectionType, int, bytes]] = [
        (ModelPackageSectionType.EXECUTION_PLAN, bucket, plan_map[bucket])
        for bucket in sorted(plan_map)
    ]
    for section_type, by_bucket in sorted(
            (per_bucket_sections or {}).items(), key=lambda kv: int(kv[0])):
        if section_type not in PER_BUCKET_SECTION_TYPES:
            raise ModelPackageError(
                f"{section_type.name} is not a per-bucket section")
        for bucket in sorted(by_bucket):
            payload = bytes(by_bucket[bucket])
            if not payload:
                raise ModelPackageError(
                    f"empty {section_type.name} for bucket {bucket}")
            if not 0 < bucket < (1 << 32):
                raise ModelPackageError("bucket must fit uint32")
            ordered.append((section_type, bucket, payload))
    ordered += [(t, 0, payload)
                for t, payload in sorted(by_type.items(),
                                         key=lambda item: int(item[0]))]
    if len(ordered) > MODEL_PACKAGE_MAX_SECTIONS:
        raise ModelPackageError("invalid model-package section count")
    table_offset = MODEL_PACKAGE_HEADER_SIZE
    table_end = table_offset + len(ordered) * MODEL_PACKAGE_SECTION_SIZE
    payload_offset = _align(table_end)
    cursor = payload_offset
    entries: list[tuple[ModelPackageSectionType, int, int, bytes]] = []
    for section_type, section_flags, payload in ordered:
        cursor = _align(cursor)
        entries.append((section_type, section_flags, cursor, payload))
        cursor += len(payload)
    file_size = cursor

    body = bytearray(file_size - MODEL_PACKAGE_HEADER_SIZE)
    for index, (section_type, section_flags, offset, payload) in enumerate(
            entries):
        entry = MODEL_PACKAGE_SECTION_STRUCT.pack(
            int(section_type),
            section_flags,
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
    if not MODEL_PACKAGE_MIN_READ_VERSION <= version <= (
            MODEL_PACKAGE_FORMAT_VERSION):
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
    plans_by_bucket: dict[int, bytes] = {}
    per_bucket: dict[ModelPackageSectionType, dict[int, bytes]] = {}
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
        is_plan = section_type == ModelPackageSectionType.EXECUTION_PLAN
        is_per_bucket = section_type in PER_BUCKET_SECTION_TYPES
        if section_type in sections and not is_per_bucket:
            raise ModelPackageError(f"duplicate section: {section_type.name}")
        if entry_reserved != 0:
            raise ModelPackageError(
                f"non-zero reserved field: {section_type.name}")
        # v2는 plan section flags에 bucket frame 수를 담는다.  그 외에는 0이다.
        if section_flags != 0 and not (is_per_bucket and version >= 2):
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
        if is_per_bucket:
            slot = section_flags if version >= 2 else bucket_frames
            table = per_bucket.setdefault(section_type, {})
            if slot in table:
                raise ModelPackageError(
                    f"duplicate {section_type.name} for bucket {slot}")
            table[slot] = section_payload
            if is_plan:
                plans_by_bucket[slot] = section_payload
        sections.setdefault(section_type, section_payload)
        occupied.append((offset, end))

    missing = REQUIRED_SECTION_TYPES - set(sections)
    if missing:
        names = ", ".join(item.name for item in sorted(missing))
        raise ModelPackageError(f"missing required sections: {names}")
    if not plans_by_bucket:
        raise ModelPackageError("model package has no execution plan")
    if version >= 2 and bucket_frames not in plans_by_bucket:
        raise ModelPackageError(
            f"header bucket {bucket_frames} has no matching plan")
    loaded = LoadedModelPackage(
        bucket_frames=bucket_frames,
        flags=flags,
        file_size=file_size,
        sections=sections,
        version=version,
        plans_by_bucket=plans_by_bucket,
        per_bucket_sections={k: dict(v) for k, v in per_bucket.items()},
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
