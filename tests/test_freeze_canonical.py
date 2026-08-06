from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "graph" / "freeze_canonical.py"
SPEC = importlib.util.spec_from_file_location("freeze_canonical", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


class FreezeCanonicalTests(unittest.TestCase):
    def test_parse_bucket(self) -> None:
        self.assertEqual(freeze.parse_bucket("3:298"), (3.0, 298))

    def test_synthetic_cases_are_reproducible(self) -> None:
        first = freeze.synthetic_cases(np, 98, [0, 1])
        second = freeze.synthetic_cases(np, 98, [0, 1])
        self.assertEqual([name for name, _ in first], [name for name, _ in second])
        for (_, left), (_, right) in zip(first, second):
            self.assertTrue(np.array_equal(left, right))

    def test_exact_output_metrics(self) -> None:
        output = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
        metrics = freeze.output_metrics(np, output, output.copy())
        self.assertTrue(metrics["shape_match"])
        self.assertTrue(metrics["exact"])
        self.assertEqual(metrics["max_abs_diff"], 0.0)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)

    def test_changed_output_fails_exact(self) -> None:
        reference = np.array([[1.0, 2.0]], dtype=np.float32)
        candidate = np.array([[1.0, 2.5]], dtype=np.float32)
        metrics = freeze.output_metrics(np, reference, candidate)
        self.assertFalse(metrics["exact"])
        self.assertEqual(metrics["max_abs_diff"], 0.5)


if __name__ == "__main__":
    unittest.main()
