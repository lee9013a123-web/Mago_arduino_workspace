from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "4_profill" / "optimization"
    / "14_compare_final_v2_v3.py"
)
SPEC = importlib.util.spec_from_file_location("final_v2_v3_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARE
SPEC.loader.exec_module(COMPARE)


def summary(mean_ms: float, suite: str) -> dict:
    return {
        "configuration": {
            "protocol_mode": "quick",
            "bucket_frames": 98,
            "threads": 1,
            "cpu_affinity": [0],
            "warmup": 5,
            "repeat": 20,
            "overhead_baseline_repeat": 5,
            "input_ids": ["0000", "0005", "0006"],
            "optimization_suite_config": suite,
        },
        "artifacts": {
            name: {"sha256": f"same-{name}"}
            for name in ("plan", "weights", "fusion_plan")
        },
        "timing": {
            "end_to_end": {
                "mean_ms": mean_ms,
                "p50_ms": mean_ms - 1.0,
                "p95_ms": mean_ms + 2.0,
            }
        },
        "validity": {"profile_valid": True},
    }


def profile(kernel: str, mean_ms: float) -> dict:
    return {
        "operators": [{
            "operator_id": 2,
            "operator_type": "QLINEAR_CONV",
            "kernel_name": kernel,
            "mean_ms": mean_ms,
        }]
    }


class FinalV2V3ComparisonTests(unittest.TestCase):
    def test_builds_e2e_and_operator_delta(self) -> None:
        result = COMPARE.build_comparison(
            summary(400.0, "v2"), summary(360.0, "v3"),
            profile("qconv_v4", 20.0), profile("qconv_hybrid_v3", 15.0),
            {2: "v5"},
        )
        self.assertAlmostEqual(result["e2e"]["speedup_ratio"], 400.0 / 360.0)
        self.assertAlmostEqual(result["e2e"]["latency_reduction_pct"], 10.0)
        self.assertEqual(result["e2e"]["rtf_v3"], 0.36)
        self.assertEqual(result["largest_operator_deltas"][0]["delta_ms"], -5.0)
        self.assertEqual(
            result["largest_operator_deltas"][0]["v3_selected_mode"], "v5"
        )

    def test_rejects_different_measurement_protocol(self) -> None:
        v3 = summary(360.0, "v3")
        v3["configuration"]["repeat"] = 100
        with self.assertRaises(COMPARE.FinalComparisonError):
            COMPARE.build_comparison(
                summary(400.0, "v2"), v3,
                profile("qconv_v4", 20.0),
                profile("qconv_hybrid_v3", 15.0),
            )


if __name__ == "__main__":
    unittest.main()
