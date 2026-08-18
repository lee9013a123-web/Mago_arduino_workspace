from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "4_profill" / "profile_statistics.py"
SPEC = importlib.util.spec_from_file_location("profile_statistics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProfileStatisticsTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(MODULE.percentile([1, 2, 3], 50.0), 2.0)
        self.assertAlmostEqual(MODULE.percentile([1, 2, 3], 95.0), 2.9)

    def test_top_set_uses_end_to_end_denominator(self) -> None:
        ranked, summary = MODULE.rank_operator_samples(
            {
                0: [60_000_000, 60_000_000],
                1: [20_000_000, 20_000_000],
                2: [10_000_000, 10_000_000],
            },
            [100_000_000, 100_000_000],
        )

        self.assertTrue(summary["complete"])
        self.assertEqual(summary["operator_ids"], [0, 1])
        self.assertEqual(summary["kernel_accounted_end_to_end_pct"], 90.0)
        self.assertEqual([item["operator_id"] for item in ranked], [0, 1, 2])
        self.assertAlmostEqual(ranked[1]["cumulative_end_to_end_share_pct"], 80.0)
        self.assertFalse(ranked[2]["in_top_bottleneck_set"])

    def test_incomplete_top_set_is_reported_without_renormalizing(self) -> None:
        _, summary = MODULE.rank_operator_samples(
            {0: [10_000_000], 1: [5_000_000]},
            [100_000_000],
        )

        self.assertFalse(summary["complete"])
        self.assertEqual(summary["kernel_accounted_end_to_end_pct"], 15.0)
        self.assertIn("reason", summary)


if __name__ == "__main__":
    unittest.main()
