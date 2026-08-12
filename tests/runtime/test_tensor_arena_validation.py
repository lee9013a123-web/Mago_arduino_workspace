from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "3_runtime" / "08_validate_tensor_arena.py"
SPEC = importlib.util.spec_from_file_location("validate_tensor_arena", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_dump(prefix: Path, layout: str, payload: bytes) -> None:
    prefix.with_suffix(".bin").write_bytes(payload)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "mode": "full_graph",
                "memory_layout": layout,
                "activation_bytes": 128 if layout == "reference" else 64,
                "bucket_frames": 98,
                "operator_count": 1,
                "tensors": [
                    {
                        "operator_id": 0,
                        "tensor_id": 2,
                        "dtype": 3,
                        "rank": 1,
                        "shape": [len(payload)],
                        "offset": 0,
                        "byte_size": len(payload),
                    }
                ],
                "executed": 1,
            }
        ),
        encoding="utf-8",
    )


class TensorArenaValidationTests(unittest.TestCase):
    def test_bitwise_identical_dump_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            arena = root / "arena"
            _write_dump(reference, "reference", b"\x01\x02\x03")
            _write_dump(arena, "tensor_arena", b"\x01\x02\x03")

            result = MODULE.compare_dumps(reference, arena)

            self.assertTrue(result["bitwise_identical"])
            self.assertEqual(result["saved_activation_bytes"], 64)

    def test_first_payload_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            arena = root / "arena"
            _write_dump(reference, "reference", b"\x01\x02\x03")
            _write_dump(arena, "tensor_arena", b"\x01\xff\x03")

            result = MODULE.compare_dumps(reference, arena)

            self.assertFalse(result["bitwise_identical"])
            self.assertEqual(result["first_mismatch"]["kind"], "payload")
            self.assertEqual(
                result["first_mismatch"]["byte_offset_in_tensor"], 1
            )


if __name__ == "__main__":
    unittest.main()
