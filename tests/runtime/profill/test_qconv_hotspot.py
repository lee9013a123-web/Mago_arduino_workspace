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
    / "04_profile_qconv_hotspot.py"
)
SPEC = importlib.util.spec_from_file_location("profile_qconv_hotspot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
HOTSPOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HOTSPOT
SPEC.loader.exec_module(HOTSPOT)


class QConvHotspotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_map = HOTSPOT.build_source_map()

    def test_source_map_tracks_current_qconv_source(self) -> None:
        source_map = self.source_map
        self.assertLess(source_map["dot_span"][0], source_map["dot_span"][1])
        self.assertLess(source_map["core_begin"], source_map["mac_loop_begin"])
        self.assertLess(source_map["mac_loop_begin"], source_map["core_end"])
        self.assertLessEqual(source_map["requant_begin"], source_map["requant_end"])

    def test_parses_and_classifies_perf_source_lines(self) -> None:
        source_map = self.source_map
        source = "/tmp/qlinear_convolution_neon.c"
        samples = "\n".join(
            [
                f" 100000 ffff campp_aarch64_qlinear_conv_o4i4 "
                f"{source}:{source_map['core_begin']}",
                f" 200000 fffe campp_dot4_i16 {source}:{source_map['dot_span'][0]}",
                f" 50000 fffd campp_qconv_write {source}:{source_map['requant_span'][0]}",
            ]
        )
        result = HOTSPOT.classify_perf_script(samples, source_map)
        self.assertEqual(result["parsed_sample_count"], 3)
        self.assertEqual(
            result["category_periods"]["address_load_control"], 100000
        )
        self.assertEqual(result["category_periods"]["mac_reduction"], 200000)
        self.assertEqual(result["category_periods"]["requant_write"], 50000)
        self.assertAlmostEqual(
            result["address_vs_mac"]["mac_pct"], 200000 / 300000 * 100.0
        )

    def test_aggregate_requires_same_winner_on_all_inputs(self) -> None:
        case = {
            "case_name": "qconv_3x3",
            "operator_id": 2,
            "kernel_id": 1,
            "kernel_name": "qlinear_conv_o4i4_neon",
        }

        def input_result(address: int, mac: int, unclassified: int = 0) -> dict:
            periods = {category: 0 for category in HOTSPOT.CATEGORIES}
            periods["address_load_control"] = address
            periods["mac_reduction"] = mac
            periods["unclassified"] = unclassified
            total = sum(periods.values())
            core = address + mac
            return {
                "category_periods": periods,
                "category_share_pct": {
                    category: value / total * 100.0
                    for category, value in periods.items()
                },
                "address_vs_mac": {
                    "address_pct": address / core * 100.0,
                    "mac_pct": mac / core * 100.0,
                },
            }

        stable = HOTSPOT._aggregate_case(
            case,
            [input_result(70, 30), input_result(75, 25), input_result(65, 35)],
        )
        self.assertTrue(stable["decision"]["stable_across_inputs"])
        self.assertEqual(stable["decision"]["winner"], "address_load_control")

        mixed = HOTSPOT._aggregate_case(
            case,
            [input_result(70, 30), input_result(30, 70), input_result(65, 35)],
        )
        self.assertFalse(mixed["decision"]["stable_across_inputs"])
        self.assertEqual(mixed["decision"]["winner"], "mixed")

    def test_rejects_perf_text_without_attributable_samples(self) -> None:
        with self.assertRaises(HOTSPOT.HotspotError):
            HOTSPOT.classify_perf_script("# no samples", self.source_map)


if __name__ == "__main__":
    unittest.main()
