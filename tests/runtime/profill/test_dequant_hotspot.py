from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "scripts"
    / "4_profill"
    / "optimization"
    / "06_profile_dequant_hotspot.py"
)
SPEC = importlib.util.spec_from_file_location("profile_dequant_hotspot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
HOTSPOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HOTSPOT
SPEC.loader.exec_module(HOTSPOT)


class DequantHotspotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_map = HOTSPOT.build_source_map()

    def test_source_map_tracks_current_dequant_source(self) -> None:
        source_map = self.source_map
        self.assertLess(source_map["loop_begin"], source_map["loop_end"])
        self.assertLess(source_map["index_span"][0], source_map["input_span"][0])
        self.assertLess(
            source_map["input_span"][0], source_map["parameter_span"][0]
        )
        self.assertLess(
            source_map["parameter_span"][0], source_map["convert_span"][0]
        )
        self.assertLess(
            source_map["convert_span"][0], source_map["output_span"][0]
        )

    def test_classifies_sections_and_inlined_helpers(self) -> None:
        source = "/tmp/quantization_operators.c"
        source_map = self.source_map
        samples = "\n".join(
            [
                f" 100 ffff {HOTSPOT.TARGET_SYMBOL} "
                f"{source}:{source_map['index_span'][0]}",
                f" 200 fffe {HOTSPOT.TARGET_SYMBOL} "
                f"{source}:{source_map['input_span'][0]}",
                f" 300 fffd {HOTSPOT.TARGET_SYMBOL} "
                f"{source}:{source_map['parameter_span'][0]}",
                f" 400 fffc {HOTSPOT.TARGET_SYMBOL} "
                f"{source}:{source_map['convert_span'][0]}",
                f" 500 fffb {HOTSPOT.TARGET_SYMBOL} "
                f"{source}:{source_map['output_span'][0]}",
                " 600 fffa campp_reference_offset_for_linear "
                "/tmp/reference_kernel_utils.c:64",
                " 700 fff9 campp_reference_read_f32 "
                "/tmp/reference_kernel_utils.c:117",
                " 800 fff8 campp_reference_read_quantized "
                "/tmp/reference_kernel_utils.c:145",
                f" 900 fff7 {HOTSPOT.TARGET_SYMBOL} /tmp/tensor_view.h:117",
            ]
        )
        result = HOTSPOT.classify_perf_script(samples, source_map)
        self.assertEqual(result["parsed_sample_count"], 9)
        self.assertEqual(
            result["category_periods"]["dequant_index_address"], 1600
        )
        self.assertEqual(
            result["category_periods"]["dequant_input_load"], 1000
        )
        self.assertEqual(
            result["category_periods"]["dequant_parameter_load"], 1000
        )
        self.assertEqual(
            result["category_periods"]["dequant_convert_mul"], 400
        )
        self.assertEqual(
            result["category_periods"]["dequant_output_store"], 500
        )
        self.assertEqual(result["category_periods"]["unclassified"], 0)

    def test_gate_requires_stable_winner_and_low_unclassified(self) -> None:
        case = {
            "case_name": "dequantize_linear",
            "operator_id": 5,
            "kernel_id": 1,
            "kernel_name": "dequantize_linear_stride",
        }

        def result(address: int, parameter: int, unclassified: int = 0) -> dict:
            periods = {name: 0 for name in HOTSPOT.CATEGORIES}
            periods["dequant_index_address"] = address
            periods["dequant_parameter_load"] = parameter
            periods["unclassified"] = unclassified
            total = sum(periods.values())
            return {
                "category_periods": periods,
                "category_share_pct": {
                    name: value / total * 100.0
                    for name, value in periods.items()
                },
            }

        stable = HOTSPOT._aggregate_case(
            case, [result(70, 30), result(65, 35), result(75, 25)]
        )
        self.assertTrue(stable["decision"]["ready"])
        self.assertEqual(
            stable["decision"]["winner"], "dequant_index_address"
        )
        self.assertEqual(
            stable["aggregate"]["top_bottleneck_set"],
            ["dequant_index_address", "dequant_parameter_load"],
        )

        unclassified = HOTSPOT._aggregate_case(
            case,
            [result(60, 20, 20), result(60, 20, 20), result(59, 20, 21)],
        )
        self.assertFalse(unclassified["decision"]["ready"])

    def test_rejects_empty_perf_text(self) -> None:
        with self.assertRaises(HOTSPOT.DequantHotspotError):
            HOTSPOT.classify_perf_script("# no samples", self.source_map)


if __name__ == "__main__":
    unittest.main()
