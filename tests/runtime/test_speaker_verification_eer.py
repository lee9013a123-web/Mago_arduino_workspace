from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from speaker_verification_evaluation.manifests import (  # noqa: E402
    ManifestError,
    load_embedding_set,
    read_trials,
    score_trials,
)
from speaker_verification_evaluation.metrics import (  # noqa: E402
    MetricError,
    compute_verification_metrics,
)


SCRIPT = ROOT / "scripts/5_model/14_evaluate_eer.py"
SPEC = importlib.util.spec_from_file_location("evaluate_eer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)

COLLECTOR_SCRIPT = ROOT / "scripts/5_model/14_collect_eer_embeddings.py"
COLLECTOR_SPEC = importlib.util.spec_from_file_location(
    "collect_eer_embeddings", COLLECTOR_SCRIPT
)
assert COLLECTOR_SPEC is not None and COLLECTOR_SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(COLLECTOR_SPEC)
sys.modules[COLLECTOR_SPEC.name] = COLLECTOR
COLLECTOR_SPEC.loader.exec_module(COLLECTOR)


class VerificationMetricTests(unittest.TestCase):
    def test_perfect_separation_has_zero_eer_and_min_dcf(self) -> None:
        metrics = compute_verification_metrics(
            [0.90, 0.80, 0.20, 0.10], [1, 1, 0, 0]
        )

        self.assertEqual(metrics.eer_rate, 0.0)
        self.assertEqual(metrics.min_dcf, 0.0)
        self.assertEqual(metrics.min_dcf_normalized, 0.0)

    def test_known_crossing_has_fifty_percent_eer(self) -> None:
        metrics = compute_verification_metrics(
            [0.90, 0.10, 0.80, 0.20], [1, 1, 0, 0]
        )

        self.assertAlmostEqual(metrics.eer_rate, 0.5)
        self.assertAlmostEqual(metrics.eer_threshold, 0.8)

    def test_tied_target_and_non_target_are_grouped(self) -> None:
        forward = compute_verification_metrics([0.5, 0.5], [1, 0])
        reverse = compute_verification_metrics([0.5, 0.5], [0, 1])

        self.assertAlmostEqual(forward.eer_rate, 0.5)
        self.assertEqual(forward.eer_rate, reverse.eer_rate)
        self.assertEqual(forward.eer_threshold, reverse.eer_threshold)

    def test_requires_both_trial_classes(self) -> None:
        with self.assertRaises(MetricError):
            compute_verification_metrics([0.2, 0.3], [1, 1])


class EerEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.trials = self.root / "trials.tsv"
        self.trials.write_text(
            "trial_id\tenroll_id\ttest_id\ttarget\ttrial_type\n"
            "T1\te1\tp1\t1\ttarget\n"
            "T2\te1\tn1\t0\tunregistered_impostor\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifest(self, backend: str, *, omit_negative: bool = False) -> Path:
        vectors = {
            "e1": np.asarray([1.0, 0.0], dtype=np.float32),
            "p1": np.asarray([1.0, 0.0], dtype=np.float32),
            "n1": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        rows = []
        for input_id, vector in vectors.items():
            if omit_negative and input_id == "n1":
                continue
            path = self.root / f"{backend}_{input_id}.npy"
            np.save(path, vector)
            rows.append(
                {
                    "input_id": input_id,
                    "bucket_frames": 98,
                    "path": path.name,
                }
            )
        manifest = self.root / f"{backend}.json"
        manifest.write_text(
            json.dumps(
                {"schema_version": 1, "backend": backend, "embeddings": rows}
            ),
            encoding="utf-8",
        )
        return manifest

    def test_manifest_scores_trials_for_requested_bucket(self) -> None:
        trials = read_trials(self.trials)
        embedding_set = load_embedding_set(
            self._manifest("ort"), backend="ort"
        )

        scored = score_trials(trials, embedding_set, 98)

        self.assertAlmostEqual(scored[0].score, 1.0)
        self.assertAlmostEqual(scored[1].score, 0.0)

    def test_missing_trial_embedding_is_rejected(self) -> None:
        trials = read_trials(self.trials)
        embedding_set = load_embedding_set(
            self._manifest("ort", omit_negative=True), backend="ort"
        )

        with self.assertRaises(ManifestError):
            score_trials(trials, embedding_set, 98)

    def test_cli_writes_bucket_metrics_and_backend_comparison(self) -> None:
        ort = self._manifest("ort")
        crt = self._manifest("crt")
        output = self.root / "results"

        result = EVALUATOR.main(
            [
                "--trials",
                str(self.trials),
                "--embeddings",
                f"ort={ort}",
                "--embeddings",
                f"crt={crt}",
                "--buckets",
                "98",
                "--reference-backend",
                "ort",
                "--output-dir",
                str(output),
            ]
        )

        self.assertEqual(result, 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(len(summary["rows"]), 2)
        self.assertEqual(summary["comparisons"][0]["eer_delta_percentage_points"], 0.0)
        self.assertTrue((output / "scores/scores_crt_98.tsv").is_file())
        self.assertTrue((output / "metrics/metrics_ort_98.json").is_file())

    def test_collector_requires_every_trial_input_for_each_bucket(self) -> None:
        feature_rows = []
        for input_id in ("e1", "p1"):
            feature = self.root / f"{input_id}_98.f32"
            np.zeros((1, 98, 80), dtype=np.float32).tofile(feature)
            feature_rows.append(
                {
                    "input_id": input_id,
                    "bucket_frames": 98,
                    "path": str(feature),
                }
            )
        manifest = self.root / "features.json"
        manifest.write_text(
            json.dumps({"schema_version": 1, "features": feature_rows}),
            encoding="utf-8",
        )

        records = COLLECTOR.load_feature_records(manifest)
        with self.assertRaises(COLLECTOR.CollectionError):
            COLLECTOR.select_accuracy_records(
                records, {"e1", "p1", "n1"}, [98]
            )


if __name__ == "__main__":
    unittest.main()
