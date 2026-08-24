from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "4_profill" / "optimization"
    / "09_profile_fused_qconv_spill.py"
)
SPEC = importlib.util.spec_from_file_location("profile_fused_qconv_spill", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SPILL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SPILL
SPEC.loader.exec_module(SPILL)


class FusedQconvSpillTests(unittest.TestCase):
    def test_selects_highest_profile_operator_per_shape(self) -> None:
        cases = [
            {"operator_id": 10, "weight_shape": [32, 128, 3], "profile_mean_ms": 5.0},
            {"operator_id": 20, "weight_shape": [32, 128, 3], "profile_mean_ms": 7.0},
            {"operator_id": 0, "weight_shape": [32, 1, 3, 3], "profile_mean_ms": 4.0},
        ]
        selected = SPILL.select_shape_representatives(cases)
        self.assertEqual([case["operator_id"] for case in selected], [0, 20])

    @staticmethod
    def _input(path: str, mac: float, quant: float) -> dict[str, object]:
        return {
            "execution_path": path,
            "mac_annotate_symbol": (
                "campp_qconv_mac_4x8_intrinsics_raw"
                if path.startswith("fixed")
                else "campp_qconv_mac_neon_tile_v2"
            ),
            "fixed_microkernel_sample_count": 3 if path.startswith("fixed") else 0,
            "spill": {
                "annotated_percent_total": 100.0,
                "stack_spill_share_of_annotated_pct": mac,
            },
            "quantize_spill": {
                "annotated_percent_total": 100.0,
                "stack_spill_share_of_annotated_pct": quant,
            },
            "input": f"{path}.f32",
            "output_hash": "abc",
        }

    def test_aggregate_passes_stable_fixed_path_below_limit(self) -> None:
        case = {
            "case_name": "fused_quant_qconv_op_10",
            "operator_id": 10,
            "kernel_id": 3,
            "kernel_name": "fused_quant_qlinear_conv_o4i4",
            "weight_shape": [32, 32, 3, 3],
            "profile_mean_ms": 10.0,
        }
        result = SPILL.aggregate_case(
            case,
            [
                self._input("fixed_4x8_intrinsics", 2.0, 1.0),
                self._input("fixed_4x8_intrinsics", 3.0, 1.5),
                self._input("fixed_4x8_intrinsics", 4.0, 2.0),
            ],
            spill_limit_pct=5.0,
            quantize_expected=True,
        )
        self.assertTrue(result["spill_gate"]["ready"])
        self.assertEqual(result["fixed_microkernel_sample_count"], 9)
        self.assertAlmostEqual(result["mac_spill"]["max_pct"], 4.0)

    def test_aggregate_rejects_missing_quantize_annotation(self) -> None:
        case = {
            "case_name": "fused_quant_qconv_op_0",
            "operator_id": 0,
            "kernel_id": 3,
            "kernel_name": "fused_quant_qlinear_conv_o4i4",
            "weight_shape": [32, 1, 3, 3],
            "profile_mean_ms": 70.0,
        }
        inputs = [self._input("fallback_v2", 2.0, 1.0) for _ in range(3)]
        inputs[1]["quantize_spill"] = None
        result = SPILL.aggregate_case(
            case, inputs, spill_limit_pct=5.0, quantize_expected=True
        )
        self.assertEqual(result["execution_path"], "fallback_v2")
        self.assertFalse(result["spill_gate"]["ready"])
        self.assertIn("sample period", result["spill_gate"]["next_action"])


if __name__ == "__main__":
    unittest.main()
