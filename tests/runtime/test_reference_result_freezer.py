from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "src" / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from runtime_bundle_exporter.reporting.reference_result_freezer import (
    EXPECTED_BUCKETS,
    _implemented_opcodes,
    _summarize_operator_bucket,
    freeze_reference_results,
)


class ReferenceResultFreezerUnitTests(unittest.TestCase):
    def test_registry_contains_all_twenty_reference_opcodes(self) -> None:
        registry = (
            ROOT
            / "src"
            / "c"
            / "runtime"
            / "backends"
            / "cpu_reference"
            / "reference_backend.c"
        )
        opcodes = _implemented_opcodes(registry)
        self.assertEqual(len(opcodes), 20)
        self.assertIn("QLINEAR_CONV", opcodes)
        self.assertIn("TRANSPOSE", opcodes)

    def test_bucket_summary_keeps_a_numerical_failure_visible(self) -> None:
        comparisons = [
            {
                "operator_id": 0,
                "opcode": "RELU",
                "tensor_id": 10,
                "tensor_name": "hidden",
                "status": "exact",
                "max_abs_error": 0.0,
            },
            {
                "operator_id": 1,
                "opcode": "BATCH_NORMALIZATION",
                "tensor_id": 11,
                "tensor_name": "embedding",
                "status": "float_mismatch",
                "max_abs_error": 0.1,
                "cosine_similarity": 0.998,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compare_998.json"
            path.write_text(json.dumps(comparisons), encoding="utf-8")
            summary, _ = _summarize_operator_bucket(
                path, 998, 2, 0.999999, ROOT
            )
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["failed_tensors"], 1)
        self.assertEqual(summary["first_failure"]["operator_id"], 1)
        self.assertFalse(summary["embedding_cosine_passed"])


class ReferenceResultFreezerIntegrationTests(unittest.TestCase):
    def test_current_evidence_generates_a_stable_baseline(self) -> None:
        bundle = ROOT / "models" / "compiled" / "reference"
        validation = ROOT / "results" / "runtime"
        required = [bundle / "manifest.json", bundle / "weights.bin"]
        required.extend(validation / f"compare_{bucket}.json" for bucket in EXPECTED_BUCKETS)
        if not all(path.is_file() for path in required):
            self.skipTest("ignored binary/reference evidence is not present in this checkout")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arguments = {
                "repository_root": ROOT,
                "bundle_dir": bundle,
                "validation_dir": validation,
                "config_path": ROOT / "configs" / "runtime" / "reference.json",
                "registry_path": ROOT
                / "src"
                / "c"
                / "runtime"
                / "backends"
                / "cpu_reference"
                / "reference_backend.c",
                "output_dir": output,
            }
            first = freeze_reference_results(**arguments)
            first_payloads = {
                path.name: path.read_bytes() for path in first["output_paths"].values()
            }
            second = freeze_reference_results(**arguments)
            second_payloads = {
                path.name: path.read_bytes() for path in second["output_paths"].values()
            }

        self.assertTrue(first["baseline_frozen"])
        self.assertEqual(first["baseline_id"], second["baseline_id"])
        self.assertEqual(first_payloads, second_payloads)
        self.assertEqual(first["failed_buckets"], [998])
        self.assertFalse(first["phase4_ready"])


if __name__ == "__main__":
    unittest.main()
