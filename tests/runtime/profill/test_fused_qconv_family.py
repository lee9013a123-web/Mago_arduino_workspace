from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "4_profill" / "optimization"
    / "08_benchmark_fused_qconv_family.py"
)
SPEC = importlib.util.spec_from_file_location("fused_qconv_family", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FAMILY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FAMILY
SPEC.loader.exec_module(FAMILY)


class FusedQconvFamilyTests(unittest.TestCase):
    def test_summary_selects_fastest_candidate_that_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, passed, latency in (
                ("mac_fixed", True, 10.0),
                ("quant_neon", False, 80.0),
                ("combined_fixed", True, 8.0),
            ):
                document = {
                    "candidate_microbench_gate_passed": passed,
                    "all_output_hashes_bitwise_identical": True,
                    "no_case_regressed_over_1pct": passed,
                    "family_operator_sum": {
                        "operator_count": 2,
                        "baseline_mean_ms": 100.0,
                        "candidate_mean_ms": latency,
                        "mean_speedup_ratio": 100.0 / latency,
                    },
                }
                (root / f"{mode}_comparison.json").write_text(
                    json.dumps(document), encoding="utf-8"
                )
            summary = FAMILY.build_summary(FAMILY.DEFAULT_MODES, root)
            self.assertTrue(summary["production_gate_ready"])
            self.assertEqual(summary["winner"], "combined_fixed")
            self.assertEqual(len(summary["comparisons"]), 3)


if __name__ == "__main__":
    unittest.main()
