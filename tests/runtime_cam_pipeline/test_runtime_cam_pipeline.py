from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "egs/runtime_cam_pipeline"
SRC = PIPELINE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from similarity_detect.reporting import format_terminal_report  # noqa: E402
from similarity_detect.memory_monitor import (  # noqa: E402
    ProcessTreeMemoryMonitor,
    process_tree_pids,
)
from similarity_detect.scoring import (  # noqa: E402
    SimilarityError,
    cosine_similarity,
    load_embedding,
    resolve_speaker_embedding,
)
from similarity_detect.web_terminal import (  # noqa: E402
    PipelineWebServer,
    WebTerminalError,
    parse_pipeline_command,
)
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
    require_pipeline_local,
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
    def test_pipeline_local_package_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Path(temporary)
            runtime_root = pipeline / "runtime"
            model = runtime_root / "models/campp_sv_298.camppmodel"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"package")
            checksum = hashlib.sha256(b"package").hexdigest()
            manifest = runtime_root / "assets.json"
            manifest.write_text(json.dumps({
                "format": "campp-runtime-pipeline-assets-v1",
                "mode": "package",
                "buckets": {
                    "298": {
                        "model": "models/campp_sv_298.camppmodel",
                        "sha256": checksum,
                    }
                },
            }), encoding="utf-8")
            selected = select_bucket_assets(
                repo_root=pipeline,
                manifest_path=manifest,
                bucket_frames=298,
            )
            self.assertEqual(selected.mode, "package")
            self.assertEqual(selected.model, model)
            self.assertIsNone(selected.schedule)
            require_pipeline_local(pipeline, [manifest, model])

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

    def test_runtime_executes_pipeline_local_camppmodel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Path(temporary)
            runtime_root = pipeline / "runtime"
            model = runtime_root / "models/campp_sv_298.camppmodel"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"package")
            manifest = runtime_root / "assets.json"
            manifest.write_text(json.dumps({
                "format": "campp-runtime-pipeline-assets-v1",
                "mode": "package",
                "buckets": {
                    "298": {
                        "model": "models/campp_sv_298.camppmodel",
                        "sha256": hashlib.sha256(b"package").hexdigest(),
                    }
                },
            }), encoding="utf-8")
            runtime = runtime_root / "campp_runtime"
            runtime.write_bytes(b"executable placeholder")
            feature = pipeline / "feature.f32"
            np.zeros((298, 80), dtype="<f4").tofile(feature)
            embedding = pipeline / "embedding.f32"
            seen: list[str] = []

            def fake_run(command: list[str], **kwargs: object) -> object:
                seen.extend(command)
                self.assertEqual(kwargs.get("cwd"), runtime_root)
                np.ones(192, dtype="<f4").tofile(embedding)
                payload = {
                    "model": {"bucket_frames": 298, "weight_bytes": 8000},
                    "configuration": {"weight_mode": "malloc"},
                    "optimization_bucket_policy": "layer_hybrid_v3",
                    "warm": {"timings_ms": [600.0]},
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
                    repo_root=pipeline,
                    runtime_binary=runtime,
                    asset_manifest=manifest,
                    bucket_frames=298,
                    feature_path=feature,
                    embedding_output=embedding,
                )
            self.assertIn("--model", seen)
            self.assertIn(str(model), seen)
            self.assertNotIn("--plan", seen)
            self.assertAlmostEqual(result.metrics.rtf, 0.2)


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

    def test_external_speaker_embedding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = root / "pipeline"
            pipeline.mkdir()
            external = root / "outside.f32"
            np.ones(192, dtype="<f4").tofile(external)
            with self.assertRaises(SimilarityError):
                resolve_speaker_embedding(pipeline, str(external))


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
                pipeline_total_peak_rss_bytes=50 * 1024 * 1024,
                python_host_peak_rss_bytes=30 * 1024 * 1024,
            ),
        )
        self.assertIn("final score: 0.750000", text)
        self.assertIn("pipeline total peak: 50.00 MiB", text)
        self.assertIn("Python host peak: 30.00 MiB", text)
        self.assertIn("C runtime peak: 20.00 MiB", text)
        self.assertIn("logical weight: 8.00 MiB", text)
        self.assertIn("logical activation: 2.00 MiB", text)
        self.assertIn("RTF: 0.500000", text)


class PipelineMemoryMonitorTest(unittest.TestCase):
    def test_process_tree_walks_descendants_once(self) -> None:
        children = {10: (11, 12), 11: (13,), 12: (13,), 13: ()}
        self.assertEqual(
            process_tree_pids(10, lambda pid: children.get(pid, ())),
            (10, 11, 12, 13),
        )

    def test_monitor_reports_aggregate_and_root_peak(self) -> None:
        current = {10: (1000, 1500), 11: (2000, 2400)}
        monitor = ProcessTreeMemoryMonitor(
            10,
            interval_seconds=1.0,
            status_reader=lambda pid: current[pid],
            children_reader=lambda pid: (11,) if pid == 10 else (),
        )
        monitor.start()
        current[10] = (1200, 1700)
        current[11] = (2500, 2700)
        measurement = monitor.stop()
        self.assertEqual(measurement.pipeline_total_peak_rss_bytes, 3700)
        self.assertEqual(measurement.python_host_peak_rss_bytes, 1700)

    def test_runtime_source_has_no_torch_import(self) -> None:
        offenders = []
        for path in (PIPELINE / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "import torch" in text or "import torchaudio" in text:
                offenders.append(path)
        self.assertEqual(offenders, [])


class WebTerminalTest(unittest.TestCase):
    def test_allows_native_speaker_verification_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Path(temporary)
            binary = pipeline / "runtime/campp_speaker_verify"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"native placeholder")
            parsed = parse_pipeline_command(
                "./runtime/campp_speaker_verify --mic-version "
                "arduino_default --speaker-embedding lee --bucket 298",
                pipeline_root=pipeline,
            )
            self.assertEqual(parsed.argv[0], str(binary.resolve()))
            self.assertNotIn("python", parsed.argv[0].lower())
            self.assertIn("298", parsed.argv)

    def test_rejects_native_verifier_outside_runtime_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = Path(temporary)
            binary = pipeline / "runtime/campp_speaker_verify"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"native placeholder")
            with self.assertRaises(WebTerminalError):
                parse_pipeline_command(
                    "campp_speaker_verify --bucket 298",
                    pipeline_root=pipeline,
                )

    def test_allows_pipeline_verification_command(self) -> None:
        parsed = parse_pipeline_command(
            "python3 script/verify_speaker.py --mic-version "
            "arduino_default --speaker-embedding lee --bucket 298",
            pipeline_root=PIPELINE,
            python_executable="/usr/bin/python3",
        )
        self.assertEqual(parsed.argv[0:2], ["/usr/bin/python3", "-u"])
        self.assertEqual(Path(parsed.argv[2]).name, "verify_speaker.py")
        self.assertIn("298", parsed.argv)

    def test_rejects_arbitrary_shell_command(self) -> None:
        with self.assertRaises(WebTerminalError):
            parse_pipeline_command(
                "rm -rf /",
                pipeline_root=PIPELINE,
            )

    def test_rejects_shell_chaining(self) -> None:
        with self.assertRaises(WebTerminalError):
            parse_pipeline_command(
                "python3 script/list_microphones.py && python3 evil.py",
                pipeline_root=PIPELINE,
            )

    def test_rejects_unlisted_python_script(self) -> None:
        with self.assertRaises(WebTerminalError):
            parse_pipeline_command(
                "python3 script/prepare_runtime.py --force",
                pipeline_root=PIPELINE,
            )

    def test_http_server_streams_existing_cli_output(self) -> None:
        server = PipelineWebServer(
            ("127.0.0.1", 0),
            pipeline_root=PIPELINE,
            index_html=b"<!doctype html><title>test</title>",
            access_log=False,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            with urlopen(f"http://{host}:{port}/api/health", timeout=5) as response:
                health = json.loads(response.read())
            self.assertTrue(health["ready"])
            request = Request(
                f"http://{host}:{port}/api/run",
                data=json.dumps({
                    "command": "python3 script/list_microphones.py",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=10) as response:
                terminal = response.read().decode("utf-8")
            self.assertIn("$ python3 script/list_microphones.py", terminal)
            self.assertIn("[process exited with code", terminal)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
