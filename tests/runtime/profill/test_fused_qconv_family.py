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
    def test_selects_final_v3_profile_name_as_fused_family(self) -> None:
        operator = {
            "operator_id": 10,
            "kernel_id": 8,
            "kernel_name": "fused_quant_qlinear_conv_o4i4_layer_hybrid_v3",
            "operator_type": "QLINEAR_CONV",
            "weight_shapes": [[64, 128, 1]],
            "mean_ms": 1.0,
            "end_to_end_share_pct": 1.0,
        }
        cases = FAMILY.select_family_cases({"operators": [operator]})
        self.assertEqual(cases[0]["operator_id"], 10)
        self.assertEqual(
            cases[0]["kernel_name"], "fused_quant_qlinear_conv_o4i4"
        )

    def test_summary_selects_fastest_candidate_that_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, passed, latency in (
                ("mac_fixed", True, 10.0),
                ("quant_neon", False, 80.0),
                ("combined_fixed", True, 8.0),
                ("combined_v4", True, 6.0),
                ("combined_hybrid", True, 5.0),
                ("combined_v5", True, 4.0),
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
            summary = FAMILY.build_summary(FAMILY.FULL_DIAGNOSTIC_MODES, root)
            self.assertTrue(summary["production_gate_ready"])
            self.assertEqual(summary["winner"], "combined_v5")
            self.assertEqual(len(summary["comparisons"]), 6)
            direct = summary["mac_fixed_vs_combined_v5"]
            self.assertIsNotNone(direct)
            self.assertEqual(direct["baseline_mode"], "mac_fixed")
            self.assertEqual(direct["candidate_mode"], "combined_v5")
            self.assertAlmostEqual(direct["speedup_ratio"], 2.5)
            self.assertTrue(direct["both_gates_passed"])
            self.assertTrue(direct["both_bitwise_identical"])

    def test_quick_defaults_to_baseline_and_final_candidate(self) -> None:
        self.assertEqual(
            FAMILY.DEFAULT_MODES,
            (
                "baseline", "combined_fixed", "combined_v4",
                "combined_hybrid", "combined_v5",
            ),
        )

    def test_batch_payload_builds_comparison_compatible_documents(self) -> None:
        case = {
            "case_name": "fused_quant_qconv_op_10",
            "operator_id": 10,
            "kernel_id": 3,
            "kernel_name": "fused_quant_qlinear_conv_o4i4",
            "operator_type": "QLINEAR_CONV",
            "weight_shape": [32, 32, 3, 3],
            "profile_mean_ms": 10.0,
            "profile_share_pct": 1.0,
        }

        def payload(offset: int) -> dict[str, object]:
            return {
                "mode": "fused_qconv_family_batch",
                "clock": "CLOCK_MONOTONIC_RAW",
                "configuration": {
                    "effective_threads": 1,
                    "repeat": 2,
                    "graph_traversals": 1,
                },
                "model": {"bucket_frames": 98, "operator_count": 1},
                "cases": [
                    {
                        "operator_id": 10,
                        "kernel_id": 3,
                        "kernel_name": "fused_quant_qlinear_conv_o4i4",
                        "modes": [
                            {
                                "name": "baseline",
                                "samples_ns": [100 + offset, 110 + offset],
                                "output_hash": "abc",
                                "matches_baseline": True,
                            },
                            {
                                "name": "combined_fixed",
                                "samples_ns": [10 + offset, 11 + offset],
                                "output_hash": "abc",
                                "matches_baseline": True,
                            },
                            {
                                "name": "combined_v4",
                                "samples_ns": [8 + offset, 9 + offset],
                                "output_hash": "abc",
                                "matches_baseline": True,
                            },
                            {
                                "name": "combined_hybrid",
                                "samples_ns": [7 + offset, 8 + offset],
                                "output_hash": "abc",
                                "matches_baseline": True,
                            },
                            {
                                "name": "combined_v5",
                                "samples_ns": [6 + offset, 7 + offset],
                                "output_hash": "abc",
                                "matches_baseline": True,
                            },
                        ],
                    }
                ],
            }

        modes = FAMILY.DEFAULT_MODES
        payloads = [
            (Path(f"input_{index}.f32"), payload(index))
            for index in range(3)
        ]
        for _, item in payloads:
            FAMILY.validate_batch_payload(item, [case], modes, 2)
        documents = FAMILY.build_mode_documents(
            [case], payloads, modes,
            warmup=1, repeat=2, elapsed_seconds=1.0,
            artifacts={}, bucket_frames=498,
        )
        self.assertEqual(documents["baseline"]["case_count"], 1)
        self.assertEqual(
            documents["baseline"]["configuration"]["bucket_frames"], 498
        )
        self.assertEqual(
            documents["combined_fixed"]["cases"][0]["wall"]["call_count"],
            6,
        )
        self.assertTrue(
            documents["combined_fixed"]["optimization_gate"]["ready"]
        )
        self.assertTrue(
            documents["combined_v4"]["optimization_gate"]["ready"]
        )
        self.assertTrue(
            documents["combined_hybrid"]["optimization_gate"]["ready"]
        )
        self.assertTrue(
            documents["combined_v5"]["optimization_gate"]["ready"]
        )


if __name__ == "__main__":
    unittest.main()
