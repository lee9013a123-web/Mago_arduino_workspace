from __future__ import annotations

import hashlib
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
from similarity_detect.scoring import SimilarityError  # noqa: E402
from similarity_detect_onnx.scoring_onnx import (  # noqa: E402
    resolve_onnx_speaker_embedding,
)
from voice_embedding_onnx.runtime_onnx import (  # noqa: E402
    OrtPipelineError,
    run_embedding_onnx,
    select_ort_assets,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_manifest(root: Path) -> Path:
    model = root / "models/campplus_int8_static_qop.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx-model")
    frontend = root / "campp_fbank"
    frontend.write_bytes(b"native-fbank")
    manifest = root / "assets.json"
    manifest.write_text(json.dumps({
        "format": "campp-onnx-pipeline-assets-v1",
        "supported_buckets": [98, 298, 498, 998],
        "model": {
            "path": "models/campplus_int8_static_qop.onnx",
            "sha256": _sha256(b"onnx-model"),
        },
        "frontend": {
            "binary": "campp_fbank",
            "sha256": _sha256(b"native-fbank"),
        },
    }), encoding="utf-8")
    return manifest


class OrtAssetTest(unittest.TestCase):
    def test_one_dynamic_model_supports_every_fixed_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = _write_manifest(Path(temporary))
            for bucket in (98, 298, 498, 998):
                assets = select_ort_assets(manifest, bucket)
                self.assertEqual(assets.model.name, "campplus_int8_static_qop.onnx")
                self.assertEqual(assets.frontend.name, "campp_fbank")

    def test_manifest_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root.parent / "outside.onnx"
            manifest = root / "assets.json"
            manifest.write_text(json.dumps({
                "format": "campp-onnx-pipeline-assets-v1",
                "supported_buckets": [98],
                "model": {"path": "../outside.onnx", "sha256": "0" * 64},
                "frontend": {"binary": "campp_fbank", "sha256": "0" * 64},
            }), encoding="utf-8")
            with self.assertRaises(OrtPipelineError):
                select_ort_assets(manifest, 98)

    def test_ort_runner_contract_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _write_manifest(root)
            feature = root / "feature.f32"
            np.zeros((98, 80), dtype="<f4").tofile(feature)
            embedding = root / "embedding.f32"
            seen: list[str] = []

            def fake_run(command: list[str], **_: object) -> object:
                seen.extend(command)
                np.ones(192, dtype="<f4").tofile(embedding)
                return type("Completed", (), {
                    "returncode": 0,
                    "stdout": json.dumps({
                        "backend": "onnxruntime-cpu",
                        "frames": 98,
                        "model": {"file_bytes": 10},
                        "warm": {"timings_ms": [186.0]},
                        "memory": {
                            "after_measurement": {"peak_rss_bytes": 100_000_000},
                            "activation_bytes": None,
                        },
                    }),
                    "stderr": "",
                })()

            with patch("voice_embedding_onnx.runtime_onnx.subprocess.run", fake_run):
                result = run_embedding_onnx(
                    asset_manifest=manifest,
                    bucket_frames=98,
                    feature_path=feature,
                    embedding_output=embedding,
                )
            self.assertIn("--frames", seen)
            self.assertIn("98", seen)
            self.assertEqual(result.embedding.shape, (192,))
            self.assertAlmostEqual(result.metrics.rtf, 0.186)
            self.assertEqual(result.metrics.runtime_peak_label, "ORT process")
            self.assertIsNone(result.metrics.activation_bytes)


class OrtIsolationTest(unittest.TestCase):
    def test_ort_template_is_separate_from_c_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Path(temporary)
            c_template = pipeline / "voice/embedded/lee/mean_embedding.f32"
            c_template.parent.mkdir(parents=True)
            np.ones(192, dtype="<f4").tofile(c_template)
            with self.assertRaises(SimilarityError):
                resolve_onnx_speaker_embedding(pipeline, "lee")
            ort_template = pipeline / "voice_onnx/embedded/lee/mean_embedding.f32"
            ort_template.parent.mkdir(parents=True)
            np.ones(192, dtype="<f4").tofile(ort_template)
            self.assertEqual(
                resolve_onnx_speaker_embedding(pipeline, "lee"),
                ort_template.resolve(),
            )

    def test_ort_report_labels_unknown_activation_honestly(self) -> None:
        from voice_embedding.runtime import RuntimeMetrics

        report = format_terminal_report(
            0.5,
            RuntimeMetrics(
                latency_mean_ms=186.0,
                rtf=0.186,
                peak_rss_bytes=100 * 1024 * 1024,
                weight_bytes=80 * 1024 * 1024,
                activation_bytes=None,
                backend_name="onnxruntime-cpu",
                runtime_peak_label="ORT process",
                weight_label="ONNX model file",
            ),
        )
        self.assertIn("Backend: onnxruntime-cpu", report)
        self.assertIn("ORT process peak: 100.00 MiB", report)
        self.assertIn("ONNX model file: 80.00 MiB", report)
        self.assertIn("logical activation: unavailable", report)


if __name__ == "__main__":
    unittest.main()
