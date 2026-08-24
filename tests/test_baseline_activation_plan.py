from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "experiments" / "baseline_activation_plan" / "measure_activation_plan.py"
)
SPEC = importlib.util.spec_from_file_location("measure_activation_plan", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def memory(rss: int, peak: int | None = None) -> dict:
    value = {
        "current_rss_bytes": rss,
        "peak_rss_bytes": peak if peak is not None else rss,
    }
    for key in MODULE.MEMORY_METRICS:
        value.setdefault(key, rss if key == "current_rss_bytes" else None)
    return value


def inference(
    index: int,
    events: int,
    backing_growth: int,
    before_in_use: int,
    before_max: int,
    after_max: int,
    rss: int,
) -> dict:
    return {
        "session_run": index,
        "latency_ms": 200.0,
        "output_hash_fnv1a64": "abc123",
        "output_bytes": 768,
        "memory_before": memory(rss - 1_000),
        "memory_after_run": memory(rss),
        "memory_after_output_release": memory(rss),
        "allocator_before": {
            "in_use_bytes": before_in_use,
            "max_in_use_bytes": before_max,
        },
        "allocator_after_run": {
            "in_use_bytes": before_in_use + 768,
            "max_in_use_bytes": after_max,
        },
        "allocator_after_output_release": {
            "in_use_bytes": before_in_use,
            "max_in_use_bytes": after_max,
        },
        "allocator_run_delta": {
            "total_allocated_bytes": backing_growth,
            "num_allocs": events,
            "num_reserves": 0,
            "num_arena_extensions": 3 if backing_growth else 0,
            "in_use_bytes": before_in_use,
        },
    }


def native_result(
    events: tuple[int, ...],
    backing: tuple[int, ...],
    session_rss: int = 55_000_000,
    steady_rss: int = 68_000_000,
) -> dict:
    runs = []
    max_in_use = 7_000_000
    for index, (event_count, growth) in enumerate(zip(events, backing), start=1):
        after_max = 9_000_000 if index == 1 else max_in_use
        if index == 1:
            max_in_use = after_max
        runs.append(
            inference(
                index,
                event_count,
                growth,
                before_in_use=6_000_000,
                before_max=7_000_000 if index == 1 else max_in_use,
                after_max=after_max,
                rss=steady_rss,
            )
        )
    return {
        "memory": {
            "start": memory(7_000_000),
            "after_model_load": memory(session_rss - 1_000_000),
            "after_context_create": memory(session_rss),
            "after_warmup": memory(session_rss),
            "after_measurement": memory(steady_rss),
        },
        "inferences": runs,
    }


class ActivationPlanAnalysisTest(unittest.TestCase):
    def test_confirms_reuse_but_does_not_call_zero_growth_activation(self) -> None:
        pattern_on = native_result((399, 239, 239), (14_680_000, 0, 0))
        pattern_off = native_result((399, 399), (14_680_000, 0))

        result = MODULE.analyze_pattern_pair(pattern_on, pattern_off, 1.0)

        self.assertEqual(result["memory_pattern_status"], "confirmed")
        self.assertEqual(
            result["pattern_on_runs"][1]["arena_backing_growth_bytes"], 0
        )
        self.assertNotIn("activation_plan_estimate_bytes", result)

    def test_transient_upper_bound_requires_new_max_in_use(self) -> None:
        first = inference(1, 10, 1_000, 6_000, 7_000, 9_000, 20_000)
        second = inference(2, 5, 0, 6_000, 9_000, 9_000, 20_000)

        first_peak = MODULE.transient_peak_upper_bound(first)
        second_peak = MODULE.transient_peak_upper_bound(second)

        self.assertEqual(first_peak["bytes"], 3_000)
        self.assertTrue(first_peak["observed_new_max"])
        self.assertIsNone(second_peak["bytes"])
        self.assertFalse(second_peak["observed_new_max"])

    def test_null_floor_is_subtracted_at_matched_phases(self) -> None:
        result = MODULE.decompose_metric(
            null_start=7_460_000,
            null_session=26_820_000,
            null_steady=27_420_000,
            cam_session=54_580_000,
            cam_steady=67_960_000,
        )

        self.assertEqual(result["model_load_resident_delta"], 27_760_000)
        self.assertEqual(result["campp_inference_resident_delta"], 12_780_000)
        self.assertEqual(result["model_attributed_steady_delta"], 40_540_000)
        self.assertEqual(result["rounding_residual"], 0)

    def test_bucket_summary_keeps_exact_activation_unknown(self) -> None:
        on = native_result((399, 239, 239), (14_680_000, 0, 0))
        off = native_result((399, 399), (14_680_000, 0))
        analysis = MODULE.analyze_pattern_pair(on, off, 1.0)
        measurement = {
            "speaker": "0000",
            "input": "feature.f32",
            "input_sha256": "hash",
            "analysis": analysis,
            "raw": {"memory_pattern_on": on, "memory_pattern_off": off},
        }
        null = native_result(
            (10, 5, 5), (1_000_000, 0, 0),
            session_rss=27_000_000, steady_rss=28_000_000,
        )
        summary = MODULE.summarize_bucket(
            98,
            1.0,
            {
                "path": "model.onnx",
                "sha256": "hash",
                "file_bytes": 8_000_000,
                "initializer_count": 1,
                "serialized_initializer_payload_bytes": 7_400_000,
                "graph_node_count": 1,
            },
            2_007_040,
            [null],
            [measurement],
        )

        activation = summary["activation_and_transient_memory"]
        self.assertIsNone(activation["exact_ort_activation_tensor_bytes"])
        self.assertEqual(
            activation["offline_custom_c_tensor_arena_bytes"], 2_007_040
        )


if __name__ == "__main__":
    unittest.main()
