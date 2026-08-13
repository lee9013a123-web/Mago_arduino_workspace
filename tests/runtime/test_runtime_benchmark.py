from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "3_runtime" / "09_benchmark_runtime.py"
SPEC = importlib.util.spec_from_file_location("runtime_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config(
    root: Path, *, require_all: bool = True, input_ids: list[str] | None = None
) -> MODULE.BenchmarkConfig:
    document = {
        "schema_version": 1,
        "profile": "test",
        "buckets": {"98": 1.0},
        "threads": 1,
        "warmup": 2,
        "repeat": 3,
        "cold_runs": 1,
        "cv_threshold_pct": 3.0,
        "verify_source_wavs": False,
        "require_all_latency_inputs": require_all,
        "input_ids": input_ids or [],
        "paths": {
            "workspace_root": "runs",
            "canonical_model": "model.onnx",
            "arena_bundle": "bundle",
            "feature_manifest": "features.json",
            "dataset_manifest": "inputs.tsv",
            "c_benchmark": "benchmark",
            "ort_benchmark": "benchmark_onnx.py",
            "graph_dir": "graph",
            "verification_result": "validation.json",
            "tensor_arena_result": "arena.json",
        },
        "environment": {
            "affinity": [],
            "required_governor": None,
            "required_device_model_substring": None,
            "maximum_start_temperature_c": None,
        },
    }
    path = root / "config.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return MODULE.load_config(path, repository_root=root)


class RuntimeBenchmarkTests(unittest.TestCase):
    def test_config_resolves_paths_and_bucket_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)

        self.assertEqual(config.buckets, {98: 1.0})
        self.assertEqual(config.paths.canonical_model, root / "model.onnx")
        self.assertEqual(config.threads, 1)

    def test_config_can_freeze_evaluation_input_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(
                Path(directory), input_ids=["speaker_0000", "speaker_0005"]
            )

        self.assertEqual(config.input_ids, ("speaker_0000", "speaker_0005"))

    def test_feature_payload_size_and_hash_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            payload = bytes(98 * 80 * 4)
            feature_path = root / "feature.f32"
            feature_path.write_bytes(payload)
            checksum = hashlib.sha256(payload).hexdigest()
            feature = MODULE.FeatureInput(
                input_id="sample",
                bucket_frames=98,
                audio_seconds=1.0,
                path=feature_path,
                sha256=checksum,
            )

            selected = {"sample": {"input_id": "sample"}}
            validated = MODULE.validate_features(config, selected, [feature])

            self.assertEqual(validated, [feature])

    def test_required_dataset_matrix_rejects_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _config(root)
            with self.assertRaises(MODULE.BenchmarkError):
                MODULE.validate_features(
                    config,
                    {
                        "first": {"input_id": "first"},
                        "second": {"input_id": "second"},
                    },
                    [],
                )

    def test_summary_reports_interpolated_percentiles(self) -> None:
        summary = MODULE.summarize([1.0, 2.0, 3.0])

        self.assertEqual(summary["p50_ms"], 2.0)
        self.assertAlmostEqual(summary["p95_ms"], 2.9)


if __name__ == "__main__":
    unittest.main()
