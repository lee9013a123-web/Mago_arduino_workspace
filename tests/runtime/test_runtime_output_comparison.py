from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/3_runtime/05_compare_runtime_outputs.py"
SPEC = importlib.util.spec_from_file_location("runtime_output_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RuntimeOutputComparisonTests(unittest.TestCase):
    def test_embedding_gate_keeps_retained_mismatch_as_diagnostic(self) -> None:
        comparisons = [
            {"tensor_name": "hidden", "status": "integer_mismatch"},
            {
                "tensor_name": "embedding",
                "status": "float_mismatch",
                "cosine_similarity": 0.995,
            },
        ]

        result = MODULE.evaluate_comparison_gate(
            comparisons, gate="embedding-cosine", embedding_cosine_min=0.99
        )

        self.assertTrue(result["gate_passed"])
        self.assertFalse(result["retained_tensor_gate_passed"])
        self.assertEqual(len(result["failures"]), 2)

    def test_embedding_gate_rejects_non_finite_embedding(self) -> None:
        result = MODULE.evaluate_comparison_gate(
            [
                {
                    "tensor_name": "embedding",
                    "status": "non_finite",
                    "cosine_similarity": float("nan"),
                }
            ],
            gate="embedding-cosine",
            embedding_cosine_min=0.99,
        )

        self.assertFalse(result["embedding_finite"])
        self.assertFalse(result["gate_passed"])


if __name__ == "__main__":
    unittest.main()
