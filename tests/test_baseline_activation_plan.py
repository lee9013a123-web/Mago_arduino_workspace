from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "experiments" / "baseline_activation_plan" / "measure_activation_plan.py"
)
SPEC = importlib.util.spec_from_file_location("measure_activation_plan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def native_result(events: tuple[int, int], allocated: tuple[int, int]) -> dict:
    runs = []
    for index in range(2):
        runs.append(
            {
                "session_run": index + 1,
                "latency_ms": 10.0,
                "output_hash_fnv1a64": "abc123",
                "output_bytes": 768,
                "memory_before": {
                    "current_rss_bytes": 90_000_000,
                    "peak_rss_bytes": 90_000_000,
                },
                "memory_after_run": {
                    "current_rss_bytes": 100_000_000,
                    "peak_rss_bytes": 100_000_000,
                },
                "memory_after_output_release": {
                    "current_rss_bytes": 100_000_000,
                    "peak_rss_bytes": 100_000_000,
                },
                "allocator_run_delta": {
                    "total_allocated_bytes": allocated[index],
                    "num_allocs": events[index],
                    "num_reserves": 0,
                    "num_arena_extensions": 0,
                    "in_use_bytes": 0,
                },
            }
        )
    return {"inferences": runs}


class ActivationPlanAnalysisTest(unittest.TestCase):
    def test_confirms_second_run_allocation_collapse_and_decomposes_rss(self) -> None:
        pattern_on = native_result((100, 2), (8_000_000, 2_007_808))
        pattern_off = native_result((100, 100), (8_000_000, 8_000_000))

        result = MODULE.analyze_pair(pattern_on, pattern_off, 2_007_040)

        self.assertIs(result["activation_plan_running"], True)
        memory = result["memory_decomposition"]
        self.assertEqual(memory["activation_plan_estimate_bytes"], 2_007_040)
        self.assertEqual(memory["non_activation_rss_estimate_bytes"], 97_992_960)
        self.assertAlmostEqual(memory["activation_share_of_native_rss"], 0.0200704)

    def test_missing_allocator_stats_is_inconclusive(self) -> None:
        pattern_on = native_result((100, 2), (8_000_000, 2_007_808))
        pattern_off = native_result((100, 100), (8_000_000, 8_000_000))
        pattern_on["inferences"][1]["allocator_run_delta"] = None

        result = MODULE.analyze_pair(pattern_on, pattern_off, 2_007_040)

        self.assertIsNone(result["activation_plan_running"])
        self.assertIsNone(
            result["memory_decomposition"]["activation_plan_estimate_bytes"]
        )


if __name__ == "__main__":
    unittest.main()
