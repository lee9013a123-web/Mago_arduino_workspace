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

    def test_all_scope_selects_every_1x1_and_3x3_operator(self) -> None:
        def operator(operator_id: int, shape: list[int], mean_ms: float) -> dict:
            return {
                "operator_id": operator_id,
                "kernel_id": 1,
                "kernel_name": "qlinear_conv_o4i4_neon",
                "operator_type": "QLINEAR_CONV",
                "weight_shapes": [shape],
                "mean_ms": mean_ms,
                "end_to_end_share_pct": mean_ms,
            }

        cases = HOTSPOT.select_qconv_cases(
            [
                operator(2, [32, 32, 3, 3], 5.0),
                operator(10, [64, 32, 3, 3], 4.0),
                operator(825, [512, 1024, 1], 3.0),
            ],
            case_scope="all",
        )
        self.assertEqual([case["operator_id"] for case in cases], [2, 10, 825])
        self.assertEqual([case["shape"] for case in cases], ["3x3", "3x3", "1x1"])

    def test_representative10_selects_five_hot_operators_per_shape(self) -> None:
        def operator(operator_id: int, shape: list[int], mean_ms: float) -> dict:
            return {
                "operator_id": operator_id,
                "kernel_id": 1,
                "kernel_name": "qlinear_conv_o4i4_neon",
                "operator_type": "QLINEAR_CONV",
                "weight_shapes": [shape],
                "mean_ms": mean_ms,
                "end_to_end_share_pct": mean_ms,
            }

        operators = [
            operator(index, [32, 32, 3, 3], float(index))
            for index in range(1, 7)
        ] + [
            operator(100 + index, [64, 128, 1], float(index))
            for index in range(1, 7)
        ]
        cases = HOTSPOT.select_qconv_cases(
            operators, case_scope="representative10"
        )
        self.assertEqual(len(cases), 10)
        self.assertEqual(
            [case["operator_id"] for case in cases[:5]], [6, 5, 4, 3, 2]
        )
        self.assertEqual(
            [case["operator_id"] for case in cases[5:]],
            [106, 105, 104, 103, 102],
        )
        self.assertEqual(
            [case["shape"] for case in cases], ["3x3"] * 5 + ["1x1"] * 5
        )

    def test_parses_and_classifies_perf_source_lines(self) -> None:
        source_map = self.source_map
        source = "/tmp/qlinear_convolution_neon.c"
        requant_source = "/tmp/requantization_neon.c"
        samples = "\n".join(
            [
                f" 100000 ffff campp_aarch64_qlinear_conv_o4i4 "
                f"{source}:{source_map['core_begin']}",
                f" 200000 fffe campp_dot4_i16 {source}:{source_map['dot_span'][0]}",
                " 50000 fffd campp_aarch64_qconv_requantize_store4 "
                f"{requant_source}:{source_map['requant_span'][0]}",
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

    def test_fixed_and_assembly_symbols_are_mac_samples(self) -> None:
        source_map = HOTSPOT.build_v2_source_map()
        for symbol, source in (
            (
                "campp_qconv_mac_4x8_intrinsics_raw",
                "/tmp/qconv_mac_4x8_intrinsics.c",
            ),
            (
                "campp_qconv_mac_4x8_aarch64_raw",
                "/tmp/qconv_mac_4x8_aarch64.S",
            ),
        ):
            category = HOTSPOT.classify_v2_sample(
                {"symbol": symbol, "source": source, "line": 1},
                source_map,
            )
            self.assertEqual(category, "v2_mac_smlal")
        parsed = HOTSPOT.classify_v2_perf_script(
            " 100000 ffff campp_qconv_mac_4x8_aarch64_raw "
            "/tmp/qconv_mac_4x8_aarch64.S:80",
            source_map,
        )
        self.assertEqual(parsed["fixed_microkernel_sample_count"], 1)

    def test_fixed_microkernel_is_split_into_load_weight_mac_and_store(self) -> None:
        source_map = HOTSPOT.build_v2_source_map()
        source = "/tmp/qconv_mac_4x8_intrinsics.c"
        rows = (
            (
                source_map["fixed_input_calls_span"][0],
                "address_input_load_transform",
            ),
            (
                source_map["fixed_weight_call_lines"][0],
                "mac_weight_load_transform",
            ),
            (source_map["fixed_accumulate_call_lines"][0], "mac_accumulate"),
            (
                source_map["fixed_accumulator_store_span"][0],
                "mac_accumulator_store",
            ),
        )
        for line, expected in rows:
            detail = HOTSPOT.classify_v2_detail_sample(
                {
                    "symbol": HOTSPOT.FIXED_INTRINSICS_SYMBOL,
                    "source": source,
                    "line": line,
                },
                source_map,
            )
            self.assertEqual(detail, expected)

    def test_candidate_breakdown_reports_address_mac_and_requant_areas(self) -> None:
        source_map = HOTSPOT.build_v2_source_map()
        rows = (
            (
                "campp_qconv_candidate_run_v2",
                "/tmp/qconv_candidate.c",
                source_map["cand_point_table_span"][0],
                100,
                "address_point_table",
            ),
            (
                HOTSPOT.FIXED_INTRINSICS_SYMBOL,
                "/tmp/qconv_mac_4x8_intrinsics.c",
                source_map["fixed_accumulate_call_lines"][0],
                300,
                "mac_accumulate",
            ),
            (
                "campp_qconv_requantize_neon4",
                "/tmp/requantization_neon.c",
                source_map["requantize_neon4_scale_span"][0],
                200,
                "requant_scale_multiply",
            ),
        )
        text = "\n".join(
            f" {period} ffff {symbol} {source}:{line}"
            for symbol, source, line, period, _ in rows
        )
        result = HOTSPOT.classify_v2_perf_script(text, source_map)
        for _, _, _, period, detail in rows:
            self.assertEqual(result["detail_category_periods"][detail], period)
        self.assertEqual(result["area_periods"]["address"], 100)
        self.assertEqual(result["area_periods"]["mac"], 300)
        self.assertEqual(result["area_periods"]["requant"], 200)
        self.assertAlmostEqual(
            result["area_share_of_classified_pct"]["mac"], 50.0
        )

        case = {
            "case_name": "qconv_1x1",
            "operator_id": 825,
            "kernel_id": 1,
            "kernel_name": "qlinear_conv_o4i4_neon",
        }
        aggregate = HOTSPOT._aggregate_v2_case(case, [result, result, result])
        self.assertEqual(aggregate["shape"], "1x1")
        self.assertEqual(aggregate["decision"]["area_winner"], "mac")
        self.assertEqual(
            aggregate["decision"]["detail_winner"], "mac_accumulate"
        )
        self.assertTrue(aggregate["decision"]["detail_stable_across_inputs"])

    def test_v4_sources_are_split_into_address_mac_requant_and_store(self) -> None:
        source_map = HOTSPOT.build_v2_source_map()
        rows = (
            (
                "campp_qconv_v4_plan_3x3_interior",
                "/tmp/qconv_v4_tile_plan.c",
                source_map["v4_3x3_padding_span"][0],
                "address_padding_bounds",
            ),
            (
                HOTSPOT.FIXED_INTRINSICS_SYMBOL,
                "/tmp/qconv_mac_4x8_intrinsics.c",
                source_map["fixed_accumulate_call_lines"][0],
                "mac_accumulate",
            ),
            (
                "campp_qconv_v4_requantize4",
                "/tmp/qconv_requant_neon8.c",
                source_map["v4_requant4_scale_span"][0],
                "requant_scale_multiply",
            ),
            (
                "campp_qconv_v4_store_channel_packed",
                "/tmp/qconv_store_channel_packed.c",
                source_map["v4_store_write_span"][0],
                "requant_output_store",
            ),
        )
        for symbol, source, line, expected in rows:
            detail = HOTSPOT.classify_v2_detail_sample(
                {"symbol": symbol, "source": source, "line": line},
                source_map,
            )
            self.assertEqual(detail, expected)

    def test_v4_annotate_target_reports_specialized_and_tail_paths(self) -> None:
        fixed = HOTSPOT.select_mac_annotate_target(
            " 100000 ffff campp_qconv_mac_4x8_intrinsics_raw "
            "/tmp/qconv_mac_4x8_intrinsics.c:100",
            qconv_candidate="v4",
        )
        self.assertEqual(fixed["execution_path"], "v4_fixed_4x8_intrinsics")
        self.assertEqual(fixed["symbol"], HOTSPOT.FIXED_INTRINSICS_SYMBOL)

        tail = HOTSPOT.select_mac_annotate_target(
            " 100000 ffff campp_qconv_mac_neon_tile_v2 "
            "/tmp/qconv_mac_neon.c:600",
            qconv_candidate="v4",
        )
        self.assertEqual(tail["execution_path"], "v4_generic_or_tail_v2")
        self.assertEqual(tail["symbol"], HOTSPOT.V2_MAC_SYMBOL)

    def test_spill_parser_separates_stack_and_tensor_loads(self) -> None:
        annotate = "\n".join(
            [
                " 12.50 : 10: ldr q0, [sp, #16]",
                "  7.50 : 14: stp q1, q2, [x29, #-32]",
                " 20.00 : 18: ldr q3, [x4]",
                " 60.00 : 1c: smlal v16.4s, v0.4h, v4.h[0]",
            ]
        )
        result = HOTSPOT.measure_spill_share(annotate)
        self.assertAlmostEqual(result["stack_spill_percent"], 20.0)
        self.assertAlmostEqual(result["other_memory_percent"], 20.0)

    def test_fixed_annotate_target_falls_back_when_symbol_did_not_run(self) -> None:
        fixed = HOTSPOT.select_mac_annotate_target(
            " 100000 ffff campp_qconv_mac_4x8_intrinsics_raw "
            "/tmp/qconv_mac_4x8_intrinsics.c:20",
            fused_qconv_candidate="combined_fixed",
        )
        self.assertEqual(fixed["execution_path"], "fixed_4x8_intrinsics")
        self.assertEqual(fixed["symbol"], HOTSPOT.FIXED_INTRINSICS_SYMBOL)

        fallback = HOTSPOT.select_mac_annotate_target(
            " 100000 ffff campp_qconv_mac_neon_tile_v2 "
            "/tmp/qconv_mac_neon.c:580",
            fused_qconv_candidate="combined_fixed",
        )
        self.assertEqual(fallback["execution_path"], "fallback_v2")
        self.assertEqual(fallback["symbol"], HOTSPOT.V2_MAC_SYMBOL)

    def test_microbench_command_accepts_fused_candidate(self) -> None:
        command = HOTSPOT._microbench_command(
            Path("bench"),
            plan=Path("plan.bin"),
            weights=Path("weights.bin"),
            feature=Path("input.f32"),
            operator_id=10,
            warmup=5,
            repeat=20,
            fused_qconv_candidate="combined_fixed",
        )
        index = len(command) - 1 - command[::-1].index(
            "--fused-qconv-candidate"
        )
        self.assertEqual(command[index + 1], "combined_fixed")
        self.assertEqual(command[-1], "--perf-window")


if __name__ == "__main__":
    unittest.main()
