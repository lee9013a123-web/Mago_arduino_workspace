from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "4_profill" / "optimization"
    / "07_compare_qconv_spill.py"
)
SPEC = importlib.util.spec_from_file_location("compare_qconv_spill", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPARE
SPEC.loader.exec_module(COMPARE)


def measurement(
    compiler: str, mode: str, means: tuple[float, float], spills: tuple[float, float]
) -> dict:
    return {
        "compiler": compiler,
        "mode": mode,
        "cases": [
            {
                "case_name": name,
                "mean_ms": mean,
                "output_hashes": [f"{name}-0", f"{name}-1", f"{name}-2"],
                "stack_spill_share_of_annotated_pct": spill,
            }
            for name, mean, spill in zip(
                ("qconv_3x3", "qconv_1x1"), means, spills
            )
        ],
    }


class QConvSpillComparisonTests(unittest.TestCase):
    def test_selects_fastest_bitwise_compiler(self) -> None:
        measurements = [
            measurement("gcc", "mac", (40.0, 24.0), (45.0, 21.0)),
            measurement("gcc", "mac_fixed", (32.0, 20.0), (8.0, 4.0)),
            measurement("clang", "mac", (39.0, 23.0), (40.0, 18.0)),
            measurement("clang", "mac_fixed", (28.0, 18.0), (3.0, 2.0)),
        ]
        decision = COMPARE.build_decision(measurements)
        self.assertTrue(decision["ready"])
        self.assertEqual(decision["selected_compiler"], "clang")
        self.assertFalse(decision["assembly_required"])
        self.assertEqual(decision["next_mode"], "mac_fixed")

    def test_requests_assembly_when_spill_remains(self) -> None:
        measurements = [
            measurement("gcc", "mac", (40.0, 24.0), (45.0, 21.0)),
            measurement("gcc", "mac_fixed", (30.0, 20.0), (12.0, 7.0)),
        ]
        decision = COMPARE.build_decision(measurements, spill_limit_pct=5.0)
        self.assertTrue(decision["assembly_required"])
        self.assertEqual(decision["next_mode"], "mac_asm")

    def test_rejects_latency_regression(self) -> None:
        measurements = [
            measurement("gcc", "mac", (40.0, 24.0), (45.0, 21.0)),
            measurement("gcc", "mac_fixed", (41.0, 24.0), (0.0, 0.0)),
        ]
        decision = COMPARE.build_decision(measurements)
        self.assertFalse(decision["ready"])


if __name__ == "__main__":
    unittest.main()
