from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "1_benchmark" / "benchmark_onnx.py"
SPEC = importlib.util.spec_from_file_location("benchmark_onnx", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkMathTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(benchmark.percentile([1.0, 2.0, 3.0], 50), 2.0)
        self.assertAlmostEqual(benchmark.percentile([0.0, 10.0], 95), 9.5)

    def test_summary_and_cv(self) -> None:
        summary = benchmark.summarize_ms([10.0, 10.0, 10.0, 10.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["p50_ms"], 10.0)
        self.assertEqual(summary["cv_pct"], 0.0)

    def test_cosine_identical(self) -> None:
        embedding = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        metrics = benchmark.cosine_metrics(embedding, embedding.copy())
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)
        self.assertEqual(metrics["max_abs_diff"], 0.0)

    def test_common_bucket_duration(self) -> None:
        self.assertEqual(benchmark.infer_audio_seconds(298, None), 3.0)
        self.assertEqual(benchmark.infer_audio_seconds(998, None), 10.0)

    def test_config_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark.load_config(path)

    def test_raw_f32_input_uses_exact_payload(self) -> None:
        class Input:
            name = "feature"
            shape = [1, 98, 80]

        class Session:
            def get_inputs(self):
                return [Input()]

        expected = np.arange(98 * 80, dtype=np.float32).reshape(1, 98, 80)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.f32"
            path.write_bytes(expected.tobytes(order="C"))

            actual, source = benchmark.load_feature(
                None, path, 98, 0, Session(), "feature"
            )

        self.assertEqual(source, "f32")
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
