from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "4_profill" / "optimization" / "05_profile_bn_hotspot.py"
)
SPEC = importlib.util.spec_from_file_location("profile_bn_hotspot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
HOTSPOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HOTSPOT
SPEC.loader.exec_module(HOTSPOT)


class BnHotspotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_map = HOTSPOT.build_source_map()

    def test_source_map_tracks_current_bn_source(self) -> None:
        source_map = self.source_map
        self.assertLess(source_map["loop_begin"], source_map["loop_end"])
        self.assertLess(source_map["index_span"][0], source_map["parameter_span"][0])
        self.assertLess(source_map["parameter_span"][0], source_map["affine_span"][0])
        self.assertLess(source_map["affine_span"][0], source_map["relu_quant_span"][0])
        self.assertLess(source_map["relu_quant_span"][0], source_map["output_span"][0])

    def test_classifies_bn_sections_and_inlined_tensor_view(self) -> None:
        source = "/tmp/bn_relu_quant_conv.c"
        source_map = self.source_map
        samples = "\n".join(
            [
                f" 100 ffff campp_fused_bn_relu_quant {source}:{source_map['index_span'][0]}",
                f" 200 fffe campp_fused_bn_relu_quant {source}:{source_map['parameter_span'][0]}",
                f" 300 fffd sqrtf /lib/libm.so:0",
                f" 400 fffc nearbyintf /lib/libm.so:0",
                f" 500 fffb campp_reference_write_quantized /tmp/reference_kernel_utils.c:1",
                " 600 fffa campp_fused_bn_relu_quant /tmp/tensor_view.h:117",
            ]
        )
        result = HOTSPOT.classify_perf_script(samples, source_map)
        self.assertEqual(result["parsed_sample_count"], 6)
        self.assertEqual(result["category_periods"]["bn_index_address"], 700)
        self.assertEqual(result["category_periods"]["bn_parameter_load"], 200)
        self.assertEqual(result["category_periods"]["bn_sqrt_affine"], 300)
        self.assertEqual(result["category_periods"]["bn_relu_quant"], 400)
        self.assertEqual(result["category_periods"]["bn_output_write"], 500)
        self.assertEqual(result["category_periods"]["unclassified"], 0)

    def test_gate_requires_stable_winner_and_low_unclassified(self) -> None:
        case = {
            "case_name": "fused_bn_relu_quant",
            "operator_id": 824,
            "kernel_id": 2,
            "kernel_name": "fused_bn_relu_quant",
        }

        def result(address: int, affine: int, unclassified: int = 0) -> dict:
            periods = {name: 0 for name in HOTSPOT.CATEGORIES}
            periods["bn_index_address"] = address
            periods["bn_sqrt_affine"] = affine
            periods["unclassified"] = unclassified
            total = sum(periods.values())
            return {
                "category_periods": periods,
                "category_share_pct": {
                    name: value / total * 100.0 for name, value in periods.items()
                },
            }

        stable = HOTSPOT._aggregate_case(
            case, [result(70, 30), result(65, 35), result(75, 25)]
        )
        self.assertTrue(stable["decision"]["ready"])
        self.assertEqual(stable["decision"]["winner"], "bn_index_address")
        self.assertEqual(
            stable["aggregate"]["top_bottleneck_set"],
            ["bn_index_address", "bn_sqrt_affine"],
        )

        unclassified = HOTSPOT._aggregate_case(
            case, [result(60, 20, 20), result(60, 20, 20), result(59, 20, 21)]
        )
        self.assertFalse(unclassified["decision"]["ready"])

    def test_rejects_empty_perf_text(self) -> None:
        with self.assertRaises(HOTSPOT.BnHotspotError):
            HOTSPOT.classify_perf_script("# no samples", self.source_map)


if __name__ == "__main__":
    unittest.main()
