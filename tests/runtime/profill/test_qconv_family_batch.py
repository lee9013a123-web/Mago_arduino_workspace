from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "4_profill" / "optimization"
    / "12_benchmark_qconv_family.py"
)
SPEC = importlib.util.spec_from_file_location("qconv_family_batch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FAMILY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FAMILY
SPEC.loader.exec_module(FAMILY)


class QconvFamilyBatchTests(unittest.TestCase):
    def test_selects_final_v3_profile_name_as_ordinary_family(self) -> None:
        operator = {
            "operator_id": 7,
            "kernel_id": 4,
            "kernel_name": "qlinear_conv_o4i4_layer_hybrid_v3",
            "operator_type": "QLINEAR_CONV",
            "weight_shapes": [[32, 64, 1]],
            "mean_ms": 1.0,
            "end_to_end_share_pct": 1.0,
        }
        cases = FAMILY.select_qconv_family_cases({"operators": [operator]})
        self.assertEqual(cases[0]["operator_id"], 7)
        self.assertEqual(cases[0]["kernel_name"], "qlinear_conv_o4i4_neon")

    def test_selects_final_v3_profile_name_as_ordinary_family(self) -> None:
        operator = {
            "operator_id": 7,
            "kernel_id": 4,
            "kernel_name": "qlinear_conv_o4i4_layer_hybrid_v3",
            "operator_type": "QLINEAR_CONV",
            "weight_shapes": [[32, 64, 1]],
            "mean_ms": 1.0,
            "end_to_end_share_pct": 1.0,
        }
        cases = FAMILY.select_qconv_family_cases({"operators": [operator]})
        self.assertEqual(cases[0]["operator_id"], 7)
        self.assertEqual(cases[0]["kernel_name"], "qlinear_conv_o4i4_neon")

    def test_batch_payload_builds_comparison_documents(self) -> None:
        case = {
            "case_name": "qconv_op_2",
            "operator_id": 2,
            "kernel_id": 1,
            "kernel_name": "qlinear_conv_o4i4_neon",
            "operator_type": "QLINEAR_CONV",
            "weight_shape": [32, 32, 3, 3],
            "profile_mean_ms": 10.0,
            "profile_share_pct": 1.0,
        }

        def payload(offset: int) -> dict[str, object]:
            modes = []
            for name, samples in (
                ("baseline", [100 + offset, 110 + offset]),
                ("mac_fixed", [10 + offset, 11 + offset]),
                ("v4", [8 + offset, 9 + offset]),
                ("v5", [7 + offset, 8 + offset]),
            ):
                modes.append(
                    {
                        "name": name,
                        "samples_ns": samples,
                        "output_hash": "abc",
                        "matches_baseline": True,
                    }
                )
            return {
                "mode": "qconv_family_batch",
                "clock": "CLOCK_MONOTONIC_RAW",
                "configuration": {
                    "effective_threads": 1,
                    "repeat": 2,
                    "graph_traversals": 1,
                },
                "model": {"bucket_frames": 98, "operator_count": 1},
                "cases": [
                    {
                        "operator_id": 2,
                        "kernel_id": 1,
                        "kernel_name": "qlinear_conv_o4i4_neon",
                        "modes": modes,
                    }
                ],
            }

        payloads = [
            (Path(f"input_{index}.f32"), payload(index))
            for index in range(3)
        ]
        for _, item in payloads:
            FAMILY.validate_batch_payload(
                item, [case], FAMILY.DEFAULT_MODES, 2
            )
        documents = FAMILY.build_mode_documents(
            [case], payloads, FAMILY.DEFAULT_MODES,
            warmup=1, repeat=2, elapsed_seconds=1.0, artifacts={},
            bucket_frames=298,
        )

        self.assertEqual(documents["baseline"]["case_count"], 1)
        self.assertEqual(
            documents["baseline"]["configuration"]["bucket_frames"], 298
        )
        self.assertEqual(
            documents["v5"]["cases"][0]["wall"]["call_count"], 6
        )
        self.assertTrue(documents["mac_fixed"]["optimization_gate"]["ready"])
        self.assertTrue(documents["v4"]["optimization_gate"]["ready"])
        self.assertTrue(documents["v5"]["optimization_gate"]["ready"])

    def test_rejects_missing_operator_or_mode(self) -> None:
        case = {
            "operator_id": 2,
            "kernel_id": 1,
            "kernel_name": "qlinear_conv_o4i4_neon",
        }
        payload = {
            "mode": "qconv_family_batch",
            "clock": "CLOCK_MONOTONIC_RAW",
            "configuration": {
                "effective_threads": 1,
                "repeat": 1,
                "graph_traversals": 1,
            },
            "model": {"bucket_frames": 98, "operator_count": 1},
            "cases": [],
        }
        with self.assertRaises(FAMILY.QconvFamilyError):
            FAMILY.validate_batch_payload(
                payload, [case], FAMILY.DEFAULT_MODES, 1
            )


if __name__ == "__main__":
    unittest.main()
