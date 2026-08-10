from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from runtime_bundle_exporter.binary_format_schema import (  # noqa: E402
    BinaryFormatError,
    PLAN_FORMAT_VERSION,
    PLAN_HEADER_SIZE,
    PLAN_MAGIC,
    PlanHeader,
)


class PlanHeaderTests(unittest.TestCase):
    def make_header(self, payload: bytes = b"payload") -> PlanHeader:
        return PlanHeader.create(
            bucket_frames=298,
            tensor_count=1439,
            operator_count=1438,
            tensor_table_offset=80,
            operator_table_offset=160,
            attribute_section_offset=240,
            payload=payload,
        )

    def test_layout_is_exactly_80_bytes(self) -> None:
        packed = self.make_header().pack()
        self.assertEqual(PLAN_HEADER_SIZE, 80)
        self.assertEqual(len(packed), 80)
        self.assertEqual(packed[:8], PLAN_MAGIC)
        self.assertEqual(int.from_bytes(packed[8:12], "little"), PLAN_FORMAT_VERSION)
        self.assertEqual(int.from_bytes(packed[12:16], "little"), 298)
        self.assertEqual(int.from_bytes(packed[16:20], "little"), 1439)
        self.assertEqual(int.from_bytes(packed[20:24], "little"), 1438)
        self.assertEqual(int.from_bytes(packed[24:32], "little"), 80)
        self.assertEqual(int.from_bytes(packed[32:40], "little"), 160)
        self.assertEqual(int.from_bytes(packed[40:48], "little"), 240)
        self.assertEqual(packed[48:80], hashlib.sha256(b"payload").digest())

    def test_round_trip_preserves_every_field(self) -> None:
        expected = self.make_header()
        actual = PlanHeader.unpack_from(expected.pack())
        self.assertEqual(actual, expected)

    def test_payload_checksum_detects_change(self) -> None:
        header = self.make_header(b"original")
        self.assertTrue(header.verify_payload_checksum(b"original"))
        self.assertFalse(header.verify_payload_checksum(b"changed"))

    def test_rejects_truncated_header(self) -> None:
        with self.assertRaisesRegex(BinaryFormatError, "truncated"):
            PlanHeader.unpack_from(b"\x00" * (PLAN_HEADER_SIZE - 1))

    def test_rejects_invalid_magic(self) -> None:
        packed = bytearray(self.make_header().pack())
        packed[0] ^= 0xFF
        with self.assertRaisesRegex(BinaryFormatError, "magic"):
            PlanHeader.unpack_from(packed)

    def test_rejects_unsupported_version(self) -> None:
        packed = bytearray(self.make_header().pack())
        packed[8:12] = (PLAN_FORMAT_VERSION + 1).to_bytes(4, "little")
        with self.assertRaisesRegex(BinaryFormatError, "version"):
            PlanHeader.unpack_from(packed)

    def test_rejects_unaligned_section_offset(self) -> None:
        header = self.make_header()
        invalid = PlanHeader(
            **{
                **{name: getattr(header, name) for name in header.__dataclass_fields__},
                "operator_table_offset": 161,
            }
        )
        with self.assertRaisesRegex(BinaryFormatError, "aligned"):
            invalid.pack()

    def test_c_header_constants_match_python_schema(self) -> None:
        header_path = (
            ROOT
            / "src"
            / "runtime"
            / "include"
            / "campp_runtime"
            / "model_binary_format.h"
        )
        text = header_path.read_text(encoding="utf-8")

        expected_macros = {
            "CAMPP_PLAN_MAGIC_SIZE": 8,
            "CAMPP_PLAN_CHECKSUM_SIZE": 32,
            "CAMPP_PLAN_FORMAT_VERSION": PLAN_FORMAT_VERSION,
            "CAMPP_PLAN_HEADER_SIZE": PLAN_HEADER_SIZE,
            "CAMPP_PLAN_CHECKSUM_OFFSET": 48,
        }
        for name, expected in expected_macros.items():
            match = re.search(rf"#define\s+{name}\s+(\d+)u", text)
            self.assertIsNotNone(match, name)
            self.assertEqual(int(match.group(1)), expected, name)


if __name__ == "__main__":
    unittest.main()
