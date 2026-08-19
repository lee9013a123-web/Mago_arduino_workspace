from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "4_profill" / "profile_statistics.py"
SPEC = importlib.util.spec_from_file_location("profile_statistics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PROFILE_PATH = ROOT / "scripts" / "4_profill" / "02_profile_e7.py"
PROFILE_SPEC = importlib.util.spec_from_file_location("profile_e7", PROFILE_PATH)
assert PROFILE_SPEC is not None and PROFILE_SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(PROFILE_SPEC)
sys.modules[PROFILE_SPEC.name] = PROFILE
PROFILE_SPEC.loader.exec_module(PROFILE)


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

    def test_quick_mode_reduces_profile_and_baseline_repeats(self) -> None:
        config = SimpleNamespace(
            buckets={98: 1.0},
            input_ids=PROFILE.EXPECTED_INPUT_IDS,
            threads=1,
            warmup=20,
            repeat=100,
            environment=SimpleNamespace(affinity=(0,)),
        )

        protocol = PROFILE._resolve_protocol(config, "quick")

        self.assertEqual(protocol["warmup"], 5)
        self.assertEqual(protocol["repeat"], 20)
        self.assertEqual(protocol["baseline_repeat"], 5)
        self.assertFalse(protocol["official"])

    def test_overhead_compares_means_when_repeat_counts_differ(self) -> None:
        overhead = PROFILE._overhead_document(
            [
                {
                    "baseline_total_ns": 500,
                    "baseline_sample_count": 5,
                    "profiled_total_ns": 2_010,
                    "profiled_sample_count": 20,
                }
            ],
            preliminary=True,
        )

        self.assertAlmostEqual(overhead["overhead_pct"], 0.5)
        self.assertTrue(overhead["passes"])
        self.assertTrue(overhead["preliminary"])

    def test_quick_estimate_is_about_fifteen_minutes(self) -> None:
        protocol = {
            "warmup": 5,
            "repeat": 20,
            "baseline_repeat": 5,
        }

        estimated = PROFILE._estimated_minutes(protocol, 3, "stock")

        self.assertAlmostEqual(estimated, 14.7)

    def test_final_quick_estimate_uses_promoted_baseline(self) -> None:
        protocol = {
            "warmup": 5,
            "repeat": 20,
            "baseline_repeat": 5,
        }

        estimated = PROFILE._estimated_minutes(protocol, 3, "final")

        self.assertAlmostEqual(estimated, 0.9140750934)

    def test_final_estimate_matches_canonical_baseline(self) -> None:
        baseline_path = ROOT / "results" / "profiling" / "e7_98" / "baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        measured_seconds = baseline["metrics"]["non_instrumented_same_suite"][
            "rtf"
        ]["mean"]

        self.assertAlmostEqual(
            PROFILE.FINAL_E7_BASELINE_INFERENCE_SECONDS, measured_seconds
        )

    def test_profiler_payload_rejects_mixed_optimization_suite(self) -> None:
        with self.assertRaises(PROFILE.ProfileError):
            PROFILE._validate_profiler_payload(
                {"optimization_suite": "stock"}, None, 20, "final"
            )

    def test_profiler_payload_rejects_mixed_final_suite_config(self) -> None:
        with self.assertRaises(PROFILE.ProfileError):
            PROFILE._validate_profiler_payload(
                {
                    "optimization_suite": "final",
                    "optimization_suite_config": "old_final",
                },
                None,
                20,
                "final",
                "new_final",
            )

    def test_analysis_aggregates_kernel_share(self) -> None:
        operators = [
            {
                "rank": 1,
                "operator_id": 1,
                "operator_type": "QLINEAR_CONV",
                "kernel_id": 1,
                "kernel_name": "qconv",
                "fusion_family": None,
                "call_count": 60,
                "total_exclusive_ms": 80.0,
                "mean_ms": 1.0,
                "p50_ms": 1.0,
                "p95_ms": 1.1,
                "end_to_end_share_pct": 80.0,
                "cumulative_end_to_end_share_pct": 80.0,
            },
            {
                "rank": 2,
                "operator_id": 2,
                "operator_type": "ADD",
                "kernel_id": 1,
                "kernel_name": "add",
                "fusion_family": None,
                "call_count": 60,
                "total_exclusive_ms": 10.0,
                "mean_ms": 0.1,
                "p50_ms": 0.1,
                "p95_ms": 0.2,
                "end_to_end_share_pct": 10.0,
                "cumulative_end_to_end_share_pct": 90.0,
            },
        ]
        timing = {
            "total_end_to_end_ms": 100.0,
            "kernel_accounted_end_to_end_pct": 90.0,
        }

        analysis = PROFILE._analysis_document(
            operators, timing, {"preliminary": True}
        )

        self.assertEqual(
            analysis["primary_bottleneck"]["kernel"]["name"], "qconv"
        )
        self.assertEqual(
            analysis["kernel_breakdown"][0]["end_to_end_share_pct"], 80.0
        )


if __name__ == "__main__":
    unittest.main()
