"""layer-hybrid dispatch 테이블을 C 소스에서 읽어 모델에 담을 형태로 만든다.

`conv_layer_hybrid_plan.c`의 `static const uint32_t ..._IDS_{bucket}[]` 배열은
결국 operator_id 목록일 뿐이다.  이 값을 모델에 담아두면 배포된 모델과 런타임
바이너리가 같은 선택을 하는지 대조할 수 있다.

**커널 구현체는 담기지 않는다.**  v5/v4 NEON 코드는 기계어라 런타임 바이너리에
남는다.  ONNX가 그래프만 담고 커널은 ORT에 두는 것과 같은 구조다.

바이너리 포맷 (little endian):

    magic   8B  "CAMPPDSP"
    version 4B  = 1
    bucket  4B
    count   4B  (엔트리 수)
    reserved 4B
    엔트리 count개, 각 8B:  kernel_kind 4B, operator_id 4B
"""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Final, Mapping

DISPATCH_MAGIC: Final[bytes] = b"CAMPPDSP"
DISPATCH_VERSION: Final[int] = 1
_HEADER = struct.Struct("<8sIIII")
_ENTRY = struct.Struct("<II")

# C 배열 이름 접두사 -> 커널 종류 코드.  런타임 enum과 짝을 맞춘다.
KERNEL_KIND: Final[Mapping[str, int]] = {
    "CAMPP_QCONV_MAC_FIXED_IDS": 1,     # qconv mac_fixed
    "CAMPP_QCONV_V5_IDS": 2,            # qconv v5
    "CAMPP_FUSED_QCONV_FIXED_IDS": 3,   # fused combined_fixed
    "CAMPP_FUSED_QCONV_V5_IDS": 4,      # fused combined_v5
}
KIND_NAME: Final[Mapping[int, str]] = {
    1: "qconv_mac_fixed", 2: "qconv_v5",
    3: "fused_combined_fixed", 4: "fused_combined_v5",
}


class DispatchTableError(RuntimeError):
    """dispatch 테이블을 읽거나 쓸 수 없었다."""


def parse_dispatch_source(source: Path | str) -> dict[int, dict[str, list[int]]]:
    """C 소스에서 bucket별 {배열 이름: operator_id 목록}을 뽑는다."""
    text = Path(source).read_text(encoding="utf-8")
    out: dict[int, dict[str, list[int]]] = {}
    pattern = re.compile(
        r"static\s+const\s+uint32_t\s+(" + "|".join(KERNEL_KIND) +
        r")_(\d+)\[\]\s*=\s*\{(.*?)\}\s*;", re.S)
    for match in pattern.finditer(text):
        name, bucket_text, body = match.groups()
        bucket = int(bucket_text)
        ids = [int(value) for value in re.findall(r"(\d+)u", body)]
        if ids != sorted(ids):
            raise DispatchTableError(
                f"{name}_{bucket} is not sorted; the runtime binary-searches it")
        out.setdefault(bucket, {})[name] = ids
    if not out:
        raise DispatchTableError(f"no dispatch table found in {source}")
    return out


def encode_dispatch_table(bucket_frames: int,
                          tables: Mapping[str, list[int]]) -> bytes:
    """한 bucket의 테이블을 바이너리로 만든다.  (kind, operator_id) 오름차순."""
    entries: list[tuple[int, int]] = []
    for name, ids in tables.items():
        kind = KERNEL_KIND.get(name)
        if kind is None:
            raise DispatchTableError(f"unknown dispatch array: {name}")
        entries.extend((kind, operator_id) for operator_id in ids)
    entries.sort()
    body = b"".join(_ENTRY.pack(kind, operator_id) for kind, operator_id in entries)
    return _HEADER.pack(
        DISPATCH_MAGIC, DISPATCH_VERSION, bucket_frames, len(entries), 0) + body


def decode_dispatch_table(payload: bytes) -> tuple[int, list[tuple[int, int]]]:
    if len(payload) < _HEADER.size:
        raise DispatchTableError("dispatch table is shorter than its header")
    magic, version, bucket, count, reserved = _HEADER.unpack_from(payload)
    if magic != DISPATCH_MAGIC:
        raise DispatchTableError("invalid dispatch-table magic")
    if version != DISPATCH_VERSION:
        raise DispatchTableError(f"unsupported dispatch version: {version}")
    if reserved != 0:
        raise DispatchTableError("non-zero reserved dispatch field")
    expected = _HEADER.size + count * _ENTRY.size
    if len(payload) != expected:
        raise DispatchTableError(
            f"dispatch table size {len(payload)} != expected {expected}")
    entries = [_ENTRY.unpack_from(payload, _HEADER.size + i * _ENTRY.size)
               for i in range(count)]
    return bucket, entries


def summarise(tables: Mapping[str, list[int]]) -> dict[str, int]:
    """사람이 읽을 요약.  배열 이름 대신 커널 이름을 쓴다."""
    return {KIND_NAME[KERNEL_KIND[name]]: len(ids)
            for name, ids in sorted(tables.items())}
