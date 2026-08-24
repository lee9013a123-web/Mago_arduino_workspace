from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from runtime_bundle_exporter.format.binary_format_schema import (  # noqa: E402
    ATTRIBUTE_BLOCK_HEADER_SIZE,
    ATTRIBUTE_KEY_NAMES,
    ATTRIBUTE_MAX_VALUE_COUNT,
    ATTRIBUTE_RECORD_HEADER_SIZE,
    ATTRIBUTE_VALUE_SIZE,
    AttributeKey,
    AttributeRecord,
    AttributeValueType,
    BackendId,
    BinaryFormatError,
    decode_attribute_block,
    encode_attribute_block,
    DEFAULT_KERNEL_ID,
    INVALID_DATA_OFFSET,
    INVALID_OPERATOR_INDEX,
    INVALID_QUANTIZATION_INDEX,
    INVALID_TENSOR_ID,
    OPERATOR_DESCRIPTOR_SIZE,
    OPERATOR_INPUT_CAPACITY,
    OPERATOR_OUTPUT_CAPACITY,
    OperatorCode,
    OperatorDescriptor,
    PLAN_FORMAT_VERSION,
    PLAN_HEADER_SIZE,
    PLAN_MAGIC,
    PlanHeader,
    TENSOR_DESCRIPTOR_SIZE,
    TENSOR_MAX_RANK,
    TensorDType,
    TensorDescriptor,
    TensorFlags,
    TensorStorageType,
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
            / "c"
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


class TensorDescriptorTests(unittest.TestCase):
    def make_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            tensor_id=0,
            dtype=TensorDType.FLOAT32,
            rank=3,
            storage_type=TensorStorageType.INPUT,
            flags=int(TensorFlags.EXTERNAL | TensorFlags.CONTIGUOUS),
            dimensions=(1, 298, 80, 1),
            byte_strides=(95_360, 320, 4, 0),
            data_offset=INVALID_DATA_OFFSET,
            logical_byte_size=95_360,
            storage_span_bytes=95_360,
            alias_of_tensor_id=INVALID_TENSOR_ID,
            quantization_index=INVALID_QUANTIZATION_INDEX,
            first_use=0,
            last_use=0,
        )

    def test_layout_is_exactly_80_bytes(self) -> None:
        packed = self.make_descriptor().pack()
        self.assertEqual(TENSOR_DESCRIPTOR_SIZE, 80)
        self.assertEqual(TENSOR_MAX_RANK, 4)
        self.assertEqual(len(packed), 80)
        self.assertEqual(int.from_bytes(packed[0:4], "little"), 0)
        self.assertEqual(packed[4], TensorDType.FLOAT32)
        self.assertEqual(packed[5], 3)
        self.assertEqual(packed[6], TensorStorageType.INPUT)
        self.assertEqual(
            int.from_bytes(packed[8:12], "little"), 1
        )  # dimension[0]
        self.assertEqual(
            int.from_bytes(packed[12:16], "little"), 298
        )  # dimension[1]
        self.assertEqual(
            int.from_bytes(packed[40:48], "little"), INVALID_DATA_OFFSET
        )
        self.assertEqual(int.from_bytes(packed[48:56], "little"), 95_360)
        self.assertEqual(int.from_bytes(packed[64:68], "little"), INVALID_TENSOR_ID)

    def test_round_trip_preserves_every_field(self) -> None:
        expected = self.make_descriptor()
        actual = TensorDescriptor.unpack_from(expected.pack())
        self.assertEqual(actual, expected)

    def test_view_requires_alias_id_and_flag(self) -> None:
        source = self.make_descriptor()
        invalid = TensorDescriptor(
            tensor_id=1,
            dtype=TensorDType.FLOAT32,
            rank=3,
            storage_type=TensorStorageType.VIEW,
            flags=int(TensorFlags.CONTIGUOUS),
            dimensions=source.dimensions,
            byte_strides=source.byte_strides,
            data_offset=0,
            logical_byte_size=source.logical_byte_size,
            storage_span_bytes=source.storage_span_bytes,
            alias_of_tensor_id=INVALID_TENSOR_ID,
            quantization_index=INVALID_QUANTIZATION_INDEX,
            first_use=1,
            last_use=2,
        )
        with self.assertRaisesRegex(BinaryFormatError, "alias"):
            invalid.pack()

    def test_rejects_rank_above_campp_limit(self) -> None:
        source = self.make_descriptor()
        invalid = TensorDescriptor(
            tensor_id=source.tensor_id,
            dtype=source.dtype,
            rank=5,
            storage_type=source.storage_type,
            flags=source.flags,
            dimensions=source.dimensions,
            byte_strides=source.byte_strides,
            data_offset=source.data_offset,
            logical_byte_size=source.logical_byte_size,
            storage_span_bytes=source.storage_span_bytes,
            alias_of_tensor_id=source.alias_of_tensor_id,
            quantization_index=source.quantization_index,
            first_use=source.first_use,
            last_use=source.last_use,
        )
        with self.assertRaisesRegex(BinaryFormatError, "rank"):
            invalid.pack()


class OperatorDescriptorTests(unittest.TestCase):
    def make_descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            operator_id=11,
            opcode=OperatorCode.QLINEAR_CONV,
            input_count=9,
            output_count=1,
            input_tensor_ids=tuple(range(1, 10)),
            output_tensor_ids=(10,),
            attribute_offset=16,
            attribute_size=24,
            backend_id=BackendId.CPU_REFERENCE,
            kernel_id=DEFAULT_KERNEL_ID,
        )

    def test_layout_is_exactly_64_bytes(self) -> None:
        packed = self.make_descriptor().pack()
        self.assertEqual(OPERATOR_DESCRIPTOR_SIZE, 64)
        self.assertEqual(OPERATOR_INPUT_CAPACITY, 9)
        self.assertEqual(OPERATOR_OUTPUT_CAPACITY, 1)
        self.assertEqual(len(packed), 64)
        self.assertEqual(int.from_bytes(packed[0:4], "little"), 11)
        self.assertEqual(int.from_bytes(packed[4:6], "little"), OperatorCode.QLINEAR_CONV)
        self.assertEqual(packed[6], 9)
        self.assertEqual(packed[7], 1)
        self.assertEqual(int.from_bytes(packed[8:12], "little"), 1)
        self.assertEqual(int.from_bytes(packed[40:44], "little"), 9)
        self.assertEqual(int.from_bytes(packed[44:48], "little"), 10)
        self.assertEqual(int.from_bytes(packed[48:56], "little"), 16)
        self.assertEqual(int.from_bytes(packed[56:60], "little"), 24)
        self.assertEqual(int.from_bytes(packed[60:62], "little"), BackendId.CPU_REFERENCE)

    def test_round_trip_preserves_every_field(self) -> None:
        expected = self.make_descriptor()
        actual = OperatorDescriptor.unpack_from(expected.pack())
        self.assertEqual(actual, expected)

    def test_unused_input_slots_must_be_invalid(self) -> None:
        invalid = OperatorDescriptor(
            operator_id=12,
            opcode=OperatorCode.RELU,
            input_count=1,
            output_count=1,
            input_tensor_ids=(1, 2) + (INVALID_TENSOR_ID,) * 7,
            output_tensor_ids=(3,),
            attribute_offset=0,
            attribute_size=0,
            backend_id=BackendId.CPU_REFERENCE,
            kernel_id=DEFAULT_KERNEL_ID,
        )
        with self.assertRaisesRegex(BinaryFormatError, "unused"):
            invalid.pack()

    def test_nonempty_attribute_must_be_aligned(self) -> None:
        source = self.make_descriptor()
        invalid = OperatorDescriptor(
            operator_id=source.operator_id,
            opcode=source.opcode,
            input_count=source.input_count,
            output_count=source.output_count,
            input_tensor_ids=source.input_tensor_ids,
            output_tensor_ids=source.output_tensor_ids,
            attribute_offset=3,
            attribute_size=source.attribute_size,
            backend_id=source.backend_id,
            kernel_id=source.kernel_id,
        )
        with self.assertRaisesRegex(BinaryFormatError, "aligned"):
            invalid.pack()


class AttributeBlockTests(unittest.TestCase):
    def test_round_trip_preserves_values(self) -> None:
        attributes = {
            "kernel_shape": (3, 3),
            "pads": (1, 1, 1, 1),
            "strides": (1, 1),
            "dilations": (1, 1),
            "group": 1,
        }
        block = encode_attribute_block(attributes)
        decoded = decode_attribute_block(block)
        self.assertEqual(decoded["kernel_shape"], (3, 3))
        self.assertEqual(decoded["pads"], (1, 1, 1, 1))
        # 스칼라는 원소 1개짜리 튜플로 돌아온다. opcode가 arity를 알고 있다.
        self.assertEqual(decoded["group"], (1,))

    def test_empty_attributes_produce_no_block(self) -> None:
        self.assertEqual(encode_attribute_block({}), b"")

    def test_block_size_matches_the_header(self) -> None:
        block = encode_attribute_block({"perm": (0, 2, 1)})
        expected = (
            ATTRIBUTE_BLOCK_HEADER_SIZE
            + ATTRIBUTE_RECORD_HEADER_SIZE
            + 3 * ATTRIBUTE_VALUE_SIZE
        )
        self.assertEqual(len(block), expected)
        self.assertEqual(len(block) % 8, 0)

    def test_identical_attributes_encode_identically(self) -> None:
        first = encode_attribute_block({"strides": (1, 1), "group": 1})
        second = encode_attribute_block({"group": 1, "strides": (1, 1)})
        self.assertEqual(first, second)

    def test_float_type_comes_from_the_key_not_the_value(self) -> None:
        """epsilon이 정수값이어도 FLOAT으로 기록되어야 한다."""

        block = encode_attribute_block({"epsilon": 1.0})
        record = AttributeRecord.unpack_from(block, ATTRIBUTE_BLOCK_HEADER_SIZE)
        self.assertIs(record.key, AttributeKey.EPSILON)
        self.assertIs(record.value_type, AttributeValueType.FLOAT)
        self.assertEqual(decode_attribute_block(block)["epsilon"], (1.0,))

    def test_negative_axis_survives(self) -> None:
        decoded = decode_attribute_block(encode_attribute_block({"axis": -1}))
        self.assertEqual(decoded["axis"], (-1,))

    def test_rejects_unknown_attribute_name(self) -> None:
        with self.assertRaises(BinaryFormatError):
            encode_attribute_block({"auto_pad": "SAME_UPPER"})

    def test_rejects_truncated_block(self) -> None:
        block = encode_attribute_block({"perm": (0, 2, 1)})
        with self.assertRaises(BinaryFormatError):
            decode_attribute_block(block[:-8])


class CHeaderParityTests(unittest.TestCase):
    @staticmethod
    def read_header(name: str) -> str:
        return (
            ROOT
            / "src"
            / "c"
            / "runtime"
            / "include"
            / "campp_runtime"
            / name
        ).read_text(encoding="utf-8")

    def assert_macro(self, text: str, name: str, expected: int) -> None:
        match = re.search(rf"#define\s+{name}\s+(\d+)u", text)
        self.assertIsNotNone(match, name)
        self.assertEqual(int(match.group(1)), expected, name)

    def test_tensor_header_matches_python_schema(self) -> None:
        text = self.read_header("tensor_descriptor.h")
        self.assert_macro(text, "CAMPP_TENSOR_MAX_RANK", TENSOR_MAX_RANK)
        self.assert_macro(
            text, "CAMPP_TENSOR_DESCRIPTOR_SIZE", TENSOR_DESCRIPTOR_SIZE
        )
        self.assertRegex(text, r"CAMPP_DTYPE_FLOAT32\s*=\s*1")
        self.assertRegex(text, r"CAMPP_TENSOR_STORAGE_VIEW\s*=\s*5")

    def test_operator_header_matches_python_schema(self) -> None:
        text = self.read_header("operator_descriptor.h")
        self.assert_macro(
            text, "CAMPP_OPERATOR_INPUT_CAPACITY", OPERATOR_INPUT_CAPACITY
        )
        self.assert_macro(
            text, "CAMPP_OPERATOR_OUTPUT_CAPACITY", OPERATOR_OUTPUT_CAPACITY
        )
        self.assert_macro(
            text, "CAMPP_OPERATOR_DESCRIPTOR_SIZE", OPERATOR_DESCRIPTOR_SIZE
        )
        self.assertRegex(text, r"CAMPP_OP_QLINEAR_CONV\s*=\s*1")
        self.assertRegex(text, r"CAMPP_OP_UNSQUEEZE\s*=\s*20")
        self.assertRegex(text, r"CAMPP_BACKEND_CPU_REFERENCE\s*=\s*1")

    def test_attribute_section_matches_python_schema(self) -> None:
        text = self.read_header("operator_descriptor.h")
        self.assert_macro(
            text, "CAMPP_ATTRIBUTE_BLOCK_HEADER_SIZE", ATTRIBUTE_BLOCK_HEADER_SIZE
        )
        self.assert_macro(
            text, "CAMPP_ATTRIBUTE_RECORD_HEADER_SIZE", ATTRIBUTE_RECORD_HEADER_SIZE
        )
        self.assert_macro(text, "CAMPP_ATTRIBUTE_VALUE_SIZE", ATTRIBUTE_VALUE_SIZE)
        self.assert_macro(
            text, "CAMPP_ATTRIBUTE_MAX_VALUE_COUNT", ATTRIBUTE_MAX_VALUE_COUNT
        )
        self.assertRegex(text, r"CAMPP_ATTR_VALUE_INT\s*=\s*1")
        self.assertRegex(text, r"CAMPP_ATTR_VALUE_FLOAT\s*=\s*2")
        # 모든 attribute key가 같은 숫자값으로 양쪽에 존재해야 한다.
        for name, key in ATTRIBUTE_KEY_NAMES.items():
            with self.subTest(attribute=name):
                self.assertRegex(
                    text, rf"CAMPP_ATTR_{key.name}\s*=\s*{int(key)}\b"
                )


if __name__ == "__main__":
    unittest.main()
