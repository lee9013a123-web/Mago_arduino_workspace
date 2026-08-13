from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "3_runtime" / "11_validate_dense_slab.py"
SPEC = importlib.util.spec_from_file_location("validate_dense_slab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_dump(prefix: Path, tensors: list[tuple[int, int, bytes]]) -> None:
    payload = bytearray()
    entries = []
    for operator_id, tensor_id, data in tensors:
        entries.append(
            {
                "operator_id": operator_id,
                "tensor_id": tensor_id,
                "dtype": "FLOAT32",
                "rank": 1,
                "shape": [len(data)],
                "offset": len(payload),
                "byte_size": len(data),
            }
        )
        payload.extend(data)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".bin").write_bytes(payload)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "memory_layout": "tensor_arena",
                "activation_bytes": 64,
                "bucket_frames": 98,
                "operator_count": len(tensors),
                "tensors": entries,
            }
        ),
        encoding="utf-8",
    )


class DenseSlabValidationTests(unittest.TestCase):
    def test_removed_prefix_is_skipped_but_remaining_tensor_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            dense = root / "dense"
            report = root / "report.json"
            _write_dump(
                reference,
                [(5, 10, b"prefix"), (6, 20, b"feature")],
            )
            _write_dump(dense, [(5, 20, b"feature")])
            report.write_text(
                json.dumps(
                    {
                        "removed_dense_concat_count": 1,
                        "dense_concat_copied_bytes_before": 6,
                        "blocks": [{"prefix_tensor_ids": [10]}],
                    }
                ),
                encoding="utf-8",
            )

            result = MODULE.compare_dumps(reference, dense, report)

        self.assertTrue(result["bitwise_identical"])
        self.assertEqual(result["compared_tensor_count"], 1)
        self.assertEqual(result["removed_prefix_tensor_count"], 1)

    def test_payload_mismatch_reports_tensor_and_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            dense = root / "dense"
            report = root / "report.json"
            _write_dump(reference, [(0, 20, b"abc")])
            _write_dump(dense, [(0, 20, b"axc")])
            report.write_text(
                json.dumps(
                    {
                        "removed_dense_concat_count": 0,
                        "dense_concat_copied_bytes_before": 0,
                        "blocks": [],
                    }
                ),
                encoding="utf-8",
            )

            result = MODULE.compare_dumps(reference, dense, report)

        self.assertFalse(result["bitwise_identical"])
        self.assertEqual(result["first_mismatch"]["tensor_id"], 20)
        self.assertEqual(result["first_mismatch"]["byte_offset_in_tensor"], 1)


if __name__ == "__main__":
    unittest.main()
