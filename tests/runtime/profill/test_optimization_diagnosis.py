from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "4_profill" / "optimization" / "02_diagnose_top4.py"
SPEC = importlib.util.spec_from_file_location("diagnose_top4", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DIAGNOSE
SPEC.loader.exec_module(DIAGNOSE)
COMPARE_SCRIPT = (
    ROOT / "scripts" / "4_profill" / "optimization" / "03_compare_candidate.py"
)
COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_optimization_candidate", COMPARE_SCRIPT
)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(COMPARE_SPEC)
sys.modules[COMPARE_SPEC.name] = COMPARE
COMPARE_SPEC.loader.exec_module(COMPARE)


def _operator(
    operator_id: int,
    kernel_name: str,
    total_ns: int,
    shape: list[int],
) -> dict:
    return {
        "operator_id": operator_id,
        "kernel_id": 1 if kernel_name == "qlinear_conv_o4i4_neon" else 3,
        "kernel_name": kernel_name,
        "operator_type": "QLINEAR_CONV",
        "weight_shapes": [[], shape],
        "mean_ms": total_ns / 1_000_000,
        "total_exclusive_ns": total_ns,
        "end_to_end_share_pct": float(total_ns) / 100.0,
    }


def _payload(case: dict, *, pmu_available: bool = True) -> dict:
    stage_names = [
        "qconv_setup",
        "qconv_mac_address",
        "qconv_requant_write",
        "fused_input_quantize",
        "fused_qconv",
        "bn_setup",
        "bn_elementwise",
        "dequant_setup",
        "dequant_elementwise",
    ]
    stage_values = {
        "qconv_setup": [2, 2],
        "qconv_mac_address": [70, 70],
        "qconv_requant_write": [8, 8],
        "fused_input_quantize": [20, 20],
        "fused_qconv": [70, 70],
        "bn_setup": [1, 1],
        "bn_elementwise": [90, 90],
        "dequant_setup": [1, 1],
        "dequant_elementwise": [90, 90],
    }
    events = {
        "cpu_cycles": [200, 200],
        "instructions": [300, 300],
        "cache_references": [20, 20],
        "cache_misses": [2, 2],
        "branches": [50, 50],
        "branch_misses": [1, 1],
    }
    return {
        "mode": "operator_microbench",
        "clock": "CLOCK_MONOTONIC_RAW",
        "qconv_candidate": "baseline",
        "fused_qconv_candidate": "baseline",
        "bn_candidate": "baseline",
        "dequant_candidate": "baseline",
        "operator": {
            "operator_id": case["operator_id"],
            "kernel_id": case["kernel_id"],
            "kernel_name": case["kernel_name"],
        },
        "samples_ns": [100, 100],
        "output_hash": "0123456789abcdef",
        "output_hash_matches": True,
        "stages": [
            {"name": name, "samples_ns": stage_values[name]}
            for name in stage_names
        ],
        "pmu": {
            "available": pmu_available,
            "unavailable_errno": 0 if pmu_available else 13,
            "events": [
                {"name": name, "samples": values}
                for name, values in events.items()
            ],
        },
    }


class OptimizationDiagnosisTests(unittest.TestCase):
    def test_selects_qconv_shape_cases_and_four_kernel_families(self) -> None:
        operators = [
            _operator(2, "qlinear_conv_o4i4_neon", 500, [32, 32, 3, 3]),
            _operator(825, "qlinear_conv_o4i4_neon", 300, [512, 1024, 1]),
            _operator(10, "fused_quant_qlinear_conv_o4i4", 400, [32, 32, 3, 3]),
            _operator(824, "fused_bn_relu_quant", 100, [1024]),
            _operator(5, "dequantize_linear_stride", 90, []),
        ]
        cases = DIAGNOSE.select_representative_cases(operators)
        self.assertEqual(
            [case["case_name"] for case in cases],
            [
                "qconv_3x3",
                "qconv_1x1",
                "fused_quant_qconv",
                "fused_bn_relu_quant",
                "dequantize_linear",
            ],
        )
        self.assertEqual(cases[0]["operator_id"], 2)
        self.assertEqual(cases[1]["operator_id"], 825)

    def test_selects_every_fused_qconv_operator_for_family_run(self) -> None:
        operators = [
            _operator(10, "fused_quant_qlinear_conv_o4i4", 400, [32, 32, 3, 3]),
            _operator(25, "fused_quant_qlinear_conv_o4i4", 300, [32, 128, 3]),
            _operator(2, "qlinear_conv_o4i4_neon", 500, [32, 32, 3, 3]),
        ]
        cases = DIAGNOSE.select_fused_qconv_family_cases(operators)
        self.assertEqual([case["operator_id"] for case in cases], [10, 25])
        self.assertEqual(cases[0]["case_name"], "fused_quant_qconv_op_10")

    def test_aggregate_reports_dominant_stage_and_pmu_ratios(self) -> None:
        case = {
            "case_name": "qconv_3x3",
            "operator_id": 2,
            "kernel_id": 1,
            "kernel_name": "qlinear_conv_o4i4_neon",
            "operator_type": "QLINEAR_CONV",
            "weight_shape": [32, 32, 3, 3],
            "profile_mean_ms": 0.1,
            "profile_share_pct": 1.0,
        }
        result = DIAGNOSE._aggregate_case(case, [_payload(case), _payload(case)])
        self.assertEqual(result["dominant_stage"], "qconv_mac_address")
        self.assertAlmostEqual(result["dominant_stage_wall_share_pct"], 70.0)
        self.assertAlmostEqual(result["pmu"]["instructions_per_cycle"], 1.5)
        self.assertAlmostEqual(result["pmu"]["cache_miss_pct"], 10.0)
        self.assertAlmostEqual(result["pmu"]["branch_miss_pct"], 2.0)

    def test_payload_validation_rejects_hash_mismatch(self) -> None:
        case = {
            "case_name": "dequantize_linear",
            "operator_id": 5,
            "kernel_id": 3,
            "kernel_name": "dequantize_linear_stride",
        }
        payload = _payload(case)
        payload["output_hash_matches"] = False
        with self.assertRaises(DIAGNOSE.DiagnosisError):
            DIAGNOSE._validate_payload(payload, case, 2)

    def test_qconv_candidate_is_forwarded_and_validated(self) -> None:
        command = DIAGNOSE._command(
            Path("microbench"),
            plan=Path("plan.bin"),
            weights=Path("weights.bin"),
            feature=Path("input.f32"),
            operator_id=2,
            warmup=5,
            repeat=20,
            qconv_candidate="combined",
        )
        self.assertEqual(
            command[command.index("--qconv-candidate") + 1], "combined"
        )
        case = {
            "case_name": "qconv_3x3",
            "operator_id": 2,
            "kernel_id": 1,
            "kernel_name": "qlinear_conv_o4i4_neon",
        }
        payload = _payload(case)
        payload["qconv_candidate"] = "combined"
        DIAGNOSE._validate_payload(payload, case, 2, "combined")
        with self.assertRaises(DIAGNOSE.DiagnosisError):
            DIAGNOSE._validate_payload(payload, case, 2, "address")

    def test_bn_candidate_is_forwarded_and_validated(self) -> None:
        command = DIAGNOSE._command(
            Path("microbench"),
            plan=Path("plan.bin"),
            weights=Path("weights.bin"),
            feature=Path("input.f32"),
            operator_id=824,
            warmup=5,
            repeat=20,
            bn_candidate="combined",
        )
        self.assertEqual(command[command.index("--bn-candidate") + 1], "combined")
        case = {
            "case_name": "fused_bn_relu_quant",
            "operator_id": 824,
            "kernel_id": 2,
            "kernel_name": "fused_bn_relu_quant",
        }
        payload = _payload(case)
        payload["bn_candidate"] = "combined"
        DIAGNOSE._validate_payload(payload, case, 2, "baseline", "combined")
        with self.assertRaises(DIAGNOSE.DiagnosisError):
            DIAGNOSE._validate_payload(payload, case, 2, "baseline", "affine")

    def test_fused_qconv_candidate_is_forwarded_and_validated(self) -> None:
        command = DIAGNOSE._command(
            Path("microbench"),
            plan=Path("plan.bin"),
            weights=Path("weights.bin"),
            feature=Path("input.f32"),
            operator_id=10,
            warmup=5,
            repeat=20,
            fused_qconv_candidate="combined_fixed",
        )
        self.assertEqual(
            command[command.index("--fused-qconv-candidate") + 1],
            "combined_fixed",
        )
        case = {
            "case_name": "fused_quant_qconv",
            "operator_id": 10,
            "kernel_id": 3,
            "kernel_name": "fused_quant_qlinear_conv_o4i4",
        }
        payload = _payload(case)
        payload["fused_qconv_candidate"] = "combined_fixed"
        DIAGNOSE._validate_payload(
            payload, case, 2, "baseline", "baseline", "combined_fixed"
        )
        with self.assertRaises(DIAGNOSE.DiagnosisError):
            DIAGNOSE._validate_payload(
                payload, case, 2, "baseline", "baseline", "mac"
            )

    def test_dequant_candidate_is_forwarded_and_validated(self) -> None:
        command = DIAGNOSE._command(
            Path("microbench"),
            plan=Path("plan.bin"),
            weights=Path("weights.bin"),
            feature=Path("input.f32"),
            operator_id=5,
            warmup=5,
            repeat=20,
            dequant_candidate="neon_combined",
        )
        self.assertEqual(
            command[command.index("--dequant-candidate") + 1],
            "neon_combined",
        )
        case = {
            "case_name": "dequantize_linear",
            "operator_id": 5,
            "kernel_id": 1,
            "kernel_name": "dequantize_linear_stride",
        }
        payload = _payload(case)
        payload["dequant_candidate"] = "neon_combined"
        DIAGNOSE._validate_payload(
            payload, case, 2, "baseline", "baseline", "baseline",
            "neon_combined",
        )
        with self.assertRaises(DIAGNOSE.DiagnosisError):
            DIAGNOSE._validate_payload(
                payload, case, 2, "baseline", "baseline", "baseline",
                "address",
            )

    def test_candidate_comparison_requires_hashes_and_speedup(self) -> None:
        case = {
            "case_name": "qconv_3x3",
            "operator_id": 2,
            "kernel_id": 1,
            "kernel_name": "qlinear_conv_o4i4_neon",
            "weight_shape": [32, 32, 3, 3],
            "output_hashes": {"input_0": "0123456789abcdef"},
            "wall": {"mean_ms": 10.0, "p50_ms": 10.0, "p95_ms": 11.0},
        }
        candidate = {
            **case,
            "wall": {"mean_ms": 8.0, "p50_ms": 8.0, "p95_ms": 9.0},
        }
        result = COMPARE.compare_diagnoses(
            {"cases": [case]}, {"cases": [candidate]}
        )
        self.assertTrue(result["candidate_microbench_gate_passed"])
        self.assertAlmostEqual(result["cases"][0]["speedup_ratio"], 1.25)
        self.assertAlmostEqual(
            result["cases"][0]["p50_speedup_ratio"], 1.25
        )
        self.assertAlmostEqual(
            result["family_operator_sum"]["mean_speedup_ratio"], 1.25
        )

        candidate["output_hashes"] = {"input_0": "different"}
        result = COMPARE.compare_diagnoses(
            {"cases": [case]}, {"cases": [candidate]}
        )
        self.assertFalse(result["candidate_microbench_gate_passed"])


if __name__ == "__main__":
    unittest.main()
