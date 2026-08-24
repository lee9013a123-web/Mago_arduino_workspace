from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "egs/runtime_cam_pipeline"
SRC = PIPELINE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from similarity_detect.reporting import format_terminal_report  # noqa: E402
from similarity_detect.scoring import cosine_similarity, load_embedding  # noqa: E402
from voice_embedding.audio import fixed_length_pcm  # noqa: E402
from voice_embedding.enrollment import (  # noqa: E402
    aggregate_embeddings,
    validate_speaker_folder,
)
from voice_embedding.frontend import (  # noqa: E402
    _extract_fbank_numpy,
    normalize_waveform,
)
from voice_embedding.runtime import (  # noqa: E402
    RuntimeMetrics,
    RuntimePipelineError,
    run_embedding,
    select_bucket_assets,
)


class AudioFrontendTest(unittest.TestCase):
    def test_fixed_length_pcm_crops_and_pads(self) -> None:
        cropped = fixed_length_pcm(np.arange(20000, dtype=np.int16), 1)
        self.assertEqual(cropped.size, 16000)
        self.assertEqual(int(cropped[-1]), 15999)
        padded = fixed_length_pcm(np.arange(10, dtype=np.int16), 1)
        self.assertEqual(padded.size, 16000)
        np.testing.assert_array_equal(padded[:10], np.arange(10, dtype=np.int16))
        self.assertTrue(np.all(padded[10:] == 0))

    def test_waveform_normalization_is_finite_and_peak_bounded(self) -> None:
        waveform = normalize_waveform(
            np.array([-32768, -1000, 1000, 32767], dtype=np.int16)
        )
        self.assertTrue(np.isfinite(waveform).all())
        self.assertLessEqual(float(np.max(np.abs(waveform))), 0.950001)
        self.assertAlmostEqual(float(waveform.mean()), 0.0, places=6)

    def test_numpy_fbank_obeys_all_fixed_bucket_shapes(self) -> None:
        for seconds, frames in ((1, 98), (3, 298), (5, 498), (10, 998)):
            with self.subTest(seconds=seconds):
                feature = _extract_fbank_numpy(
                    np.zeros(seconds * 16000, dtype=np.float32)
                )
                self.assertEqual(feature.shape, (frames, 80))
                self.assertTrue(np.isfinite(feature).all())


class RuntimeAssetSelectionTest(unittest.TestCase):
    def test_bucket_selects_one_coherent_asset_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bucket_dir = root / "assets/298"
            bucket_dir.mkdir(parents=True)
            for name in ("plan_298.bin", "weights_298.bin", "schedule_298.bin"):
                (bucket_dir / name).write_bytes(b"x")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "format": "campp-weight-streaming-sidecar-v1",
                "buckets": {
                    "298": {
                        "plan": "assets/298/plan_298.bin",
                        "weights": "assets/298/weights_298.bin",
                        "schedule": "assets/298/schedule_298.bin",
                    }
                },
            }), encoding="utf-8")
            selected = select_bucket_assets(
                repo_root=root,
                manifest_path=manifest,
                bucket_frames=298,
            )
            self.assertEqual(selected.bucket_frames, 298)
            self.assertEqual(selected.audio_seconds, 3)
            self.assertEqual(selected.schedule.name, "schedule_298.bin")

    def test_mixed_bucket_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "format": "campp-weight-streaming-sidecar-v1",
                "buckets": {
                    "98": {
                        "plan": "plan_98.bin",
                        "weights": "weights_298.bin",
                        "schedule": "schedule_98.bin",
                    }
                },
            }), encoding="utf-8")
            with self.assertRaises(RuntimePipelineError):
                select_bucket_assets(
                    repo_root=root,
                    manifest_path=manifest,
                    bucket_frames=98,
                    require_files=False,
                )

    def test_runtime_receives_selected_plan_weight_and_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_dir = root / "assets/98"
            asset_dir.mkdir(parents=True)
            paths = {
                "plan": asset_dir / "plan_98.bin",
                "weights": asset_dir / "weights_98.bin",
                "schedule": asset_dir / "weight_schedule_98.bin",
            }
            for path in paths.values():
                path.write_bytes(b"x")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "format": "campp-weight-streaming-sidecar-v1",
                "buckets": {
                    "98": {key: str(path.relative_to(root))
                           for key, path in paths.items()}
                },
            }), encoding="utf-8")
            runtime = root / "campp_runtime"
            runtime.write_bytes(b"executable placeholder")
            feature = root / "feature.f32"
            np.zeros((98, 80), dtype="<f4").tofile(feature)
            embedding = root / "embedding.f32"
            seen: list[str] = []

            def fake_run(command: list[str], **_: object) -> object:
                seen.extend(command)
                np.ones(192, dtype="<f4").tofile(embedding)
                payload = {
                    "model": {"bucket_frames": 98, "weight_bytes": 8000},
                    "configuration": {"weight_mode": "windowed"},
                    "optimization_bucket_policy": "layer_hybrid_v3",
                    "warm": {"timings_ms": [500.0]},
                    "memory": {
                        "after_measurement": {"peak_rss_bytes": 20000},
                        "activation_bytes": 2000,
                    },
                }
                return type("Completed", (), {
                    "returncode": 0,
                    "stdout": json.dumps(payload),
                    "stderr": "",
                })()

            with patch("voice_embedding.runtime.subprocess.run", fake_run):
                result = run_embedding(
                    repo_root=root,
                    runtime_binary=runtime,
                    asset_manifest=manifest,
                    bucket_frames=98,
                    feature_path=feature,
                    embedding_output=embedding,
                )
            self.assertIn(str(paths["plan"]), seen)
            self.assertIn(str(paths["weights"]), seen)
            self.assertIn(str(paths["schedule"]), seen)
            self.assertIn("windowed", seen)
            self.assertAlmostEqual(result.metrics.rtf, 0.5)


class EmbeddingTest(unittest.TestCase):
    def test_aggregate_and_cosine(self) -> None:
        first = np.zeros(192, dtype=np.float32)
        second = np.zeros(192, dtype=np.float32)
        first[0] = 1.0
        second[0] = 2.0
        mean = aggregate_embeddings([first, second])
        self.assertAlmostEqual(float(mean[0]), 1.0, places=6)
        self.assertAlmostEqual(cosine_similarity(mean, first), 1.0, places=6)

    def test_raw_embedding_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mean_embedding.f32"
            np.ones(192, dtype="<f4").tofile(path)
            value = load_embedding(path)
            self.assertEqual(value.shape, (192,))
            self.assertAlmostEqual(float(np.linalg.norm(value)), 1.0, places=6)

    def test_speaker_folder_rejects_path_traversal(self) -> None:
        with self.assertRaises(RuntimePipelineError):
            validate_speaker_folder("../speaker")
        self.assertEqual(validate_speaker_folder("화자_01"), "화자_01")


class ReportTest(unittest.TestCase):
    def test_required_terminal_fields(self) -> None:
        text = format_terminal_report(
            0.75,
            RuntimeMetrics(
                latency_mean_ms=500.0,
                rtf=0.5,
                peak_rss_bytes=20 * 1024 * 1024,
                weight_bytes=8 * 1024 * 1024,
                activation_bytes=2 * 1024 * 1024,
            ),
        )
        self.assertIn("final score: 0.750000", text)
        self.assertIn("total RAM:", text)
        self.assertIn("weight 점유율:", text)
        self.assertIn("activation 점유율:", text)
        self.assertIn("RTF: 0.500000", text)


if __name__ == "__main__":
    unittest.main()
