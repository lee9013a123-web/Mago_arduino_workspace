from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/5_model/04_evaluate_onnx_crt_multibucket.py"
SPEC = importlib.util.spec_from_file_location("multibucket_model_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _aggregate(runtime: str, policy: str | None = None) -> dict:
    value = {
        "runtime": runtime,
        "latency": {"p50_ms": 100.0, "p95_ms": 102.0, "p99_ms": 103.0},
        "rtf": {"p50_rtf": 0.5},
        "first_inference": {"p50_ms": 110.0},
        "peak_rss_bytes": 10_000,
        "incremental_peak_rss_bytes": 5_000,
        "stability_gate": {
            "all_inputs_passed": True,
            "aggregate_passed": True,
        },
    }
    if policy is not None:
        value["optimization_bucket_policy"] = policy
    return value


class MultibucketModelEvaluationTests(unittest.TestCase):
    def test_full_v3_gate_rejects_missing_bucket_plan(self) -> None:
        capabilities = {"optimization_bucket_plans": [98, 298, 498]}

        with self.assertRaises(MODULE.EvaluationError):
            MODULE.require_compiled_v3_buckets(
                capabilities, [98, 298, 498, 998]
            )

    def test_full_v3_gate_accepts_all_bucket_plans(self) -> None:
        compiled = MODULE.require_compiled_v3_buckets(
            {"optimization_bucket_plans": [98, 298, 498, 998]},
            [98, 298, 498, 998],
        )

        self.assertEqual(compiled, {98, 298, 498, 998})

    def test_matrix_requires_v3_policy_for_98(self) -> None:
        accuracy = [
            {
                "bucket_frames": 98,
                "embedding": {"cosine_similarity": 0.999},
                "embedding_finite": True,
                "gate_passed": True,
            }
        ]
        performance = {
            "accuracy": {
                "buckets": {
                    "98": {
                        "minimum_cosine_similarity": 0.999,
                        "all_finite": True,
                        "all_passed": True,
                    }
                }
            },
            "aggregates": {
                "onnxruntime-cpu": {"98": _aggregate("onnxruntime-cpu")},
                "campp-c-runtime": {
                    "98": _aggregate("campp-c-runtime", "layer_hybrid_v3")
                },
            },
            "comparisons": [
                {
                    "bucket_frames": 98,
                    "c_speedup_over_ort_p50": 2.0,
                    "peak_rss_reduction_ratio": 0.5,
                    "incremental_peak_rss_reduction_ratio": 0.5,
                }
            ],
        }

        report = MODULE.build_frame_matrix(
            accuracy,
            performance,
            [98],
            expected_policies={98: "layer_hybrid_v3"},
            embedding_cosine_min=0.99,
            baseline_98_rtf=0.522322,
            max_regression_pct=3.0,
        )

        self.assertTrue(report["all_passed"])
        self.assertEqual(report["rows"][0]["actual_c_policy"], "layer_hybrid_v3")


if __name__ == "__main__":
    unittest.main()
