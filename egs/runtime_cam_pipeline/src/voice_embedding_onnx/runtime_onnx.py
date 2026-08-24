"""Pipeline-local ONNX Runtime asset selection and isolated execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import numpy as np

from voice_embedding.runtime import (
    AUDIO_SECONDS_BY_BUCKET,
    EMBEDDING_DIMENSION,
    FBANK_BINS,
    RuntimeMetrics,
    SUPPORTED_BUCKETS,
)


class OrtPipelineError(RuntimeError):
    """Raised when ORT assets, dependencies, or output violate the contract."""


@dataclass(frozen=True)
class OrtAssets:
    model: Path
    frontend: Path
    manifest: Path
    supported_buckets: tuple[int, ...]


@dataclass(frozen=True)
class OrtEmbeddingResult:
    embedding: np.ndarray
    metrics: RuntimeMetrics
    runtime_payload: dict[str, Any]
    assets: OrtAssets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _confined_path(base: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise OrtPipelineError("ORT asset path is missing")
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()):
        raise OrtPipelineError(f"ORT asset escapes runtime_onnx: {value}")
    return path


def _verify_asset(path: Path, expected: object) -> None:
    if not path.is_file():
        raise OrtPipelineError(f"ORT asset is missing: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise OrtPipelineError(f"ORT asset checksum is invalid: {path}")
    if _sha256(path) != expected:
        raise OrtPipelineError(f"ORT asset checksum mismatch: {path}")


def select_ort_assets(manifest_path: Path, bucket_frames: int) -> OrtAssets:
    if bucket_frames not in SUPPORTED_BUCKETS:
        raise OrtPipelineError(f"unsupported bucket: {bucket_frames}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrtPipelineError(f"cannot read ORT manifest: {manifest_path}") from exc
    if not isinstance(document, dict) or (
        document.get("format") != "campp-onnx-pipeline-assets-v1"
    ):
        raise OrtPipelineError("unsupported ORT asset manifest")
    buckets = document.get("supported_buckets")
    if not isinstance(buckets, list) or any(not isinstance(v, int) for v in buckets):
        raise OrtPipelineError("ORT manifest has invalid supported_buckets")
    supported = tuple(int(value) for value in buckets)
    if bucket_frames not in supported:
        raise OrtPipelineError(f"ORT manifest does not support bucket {bucket_frames}")
    model_row = document.get("model")
    frontend_row = document.get("frontend")
    if not isinstance(model_row, dict) or not isinstance(frontend_row, dict):
        raise OrtPipelineError("ORT manifest model/frontend entry is missing")
    base = manifest_path.parent.resolve()
    model = _confined_path(base, model_row.get("path"))
    frontend = _confined_path(base, frontend_row.get("binary"))
    _verify_asset(model, model_row.get("sha256"))
    _verify_asset(frontend, frontend_row.get("sha256"))
    return OrtAssets(
        model=model,
        frontend=frontend,
        manifest=manifest_path.resolve(),
        supported_buckets=supported,
    )


def describe_ort_assets(assets: OrtAssets, bucket_frames: int) -> dict[str, object]:
    return {
        "backend": "onnxruntime-cpu",
        "model": str(assets.model),
        "frontend": str(assets.frontend),
        "manifest": str(assets.manifest),
        "bucket_frames": bucket_frames,
        "audio_seconds": AUDIO_SECONDS_BY_BUCKET[bucket_frames],
    }


def _runner_path() -> Path:
    return Path(__file__).with_name("ort_runner.py")


def validate_ort_capabilities() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(_runner_path()), "--capabilities"],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1", "ORT_NUM_THREADS": "1"},
    )
    if completed.returncode != 0:
        raise OrtPipelineError(
            "ORT capability check failed: " + completed.stderr.strip()
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise OrtPipelineError("ORT capabilities are not JSON") from exc
    if not isinstance(payload, dict) or (
        payload.get("backend") != "onnxruntime-cpu"
        or payload.get("required_provider") != "CPUExecutionProvider"
        or "CPUExecutionProvider" not in payload.get("available_providers", [])
    ):
        raise OrtPipelineError("ORT CPUExecutionProvider is unavailable")
    return payload


def run_embedding_onnx(
    *, asset_manifest: Path, bucket_frames: int, feature_path: Path,
    embedding_output: Path, warmup: int = 0, repeat: int = 1,
    threads: int = 1,
) -> OrtEmbeddingResult:
    if warmup < 0 or repeat <= 0 or threads <= 0:
        raise OrtPipelineError("warmup/repeat/threads values are invalid")
    assets = select_ort_assets(asset_manifest, bucket_frames)
    expected_bytes = bucket_frames * FBANK_BINS * 4
    if not feature_path.is_file() or feature_path.stat().st_size != expected_bytes:
        raise OrtPipelineError(
            f"feature must be [1,{bucket_frames},{FBANK_BINS}] float32"
        )
    embedding_output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(_runner_path()),
        "--model", str(assets.model),
        "--input", str(feature_path),
        "--frames", str(bucket_frames),
        "--embedding-output", str(embedding_output),
        "--audio-seconds", str(AUDIO_SECONDS_BY_BUCKET[bucket_frames]),
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--threads", str(threads),
    ]
    completed = subprocess.run(
        command,
        cwd=assets.manifest.parent,
        env={**os.environ, "OMP_NUM_THREADS": "1", "ORT_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OrtPipelineError(
            f"ORT runner failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise OrtPipelineError("ORT runner stdout is not JSON") from exc
    if not isinstance(payload, dict) or payload.get("backend") != "onnxruntime-cpu":
        raise OrtPipelineError("ORT runner returned an invalid backend")
    if payload.get("frames") != bucket_frames:
        raise OrtPipelineError("ORT runner used a different bucket")
    memory = payload.get("memory")
    after = memory.get("after_measurement") if isinstance(memory, dict) else None
    model = payload.get("model")
    warm = payload.get("warm")
    timings = warm.get("timings_ms") if isinstance(warm, dict) else None
    if not isinstance(after, dict) or not isinstance(model, dict):
        raise OrtPipelineError("ORT runner memory/model metrics are missing")
    if not isinstance(timings, list) or not timings:
        raise OrtPipelineError("ORT runner timing samples are missing")
    peak = after.get("peak_rss_bytes")
    model_bytes = model.get("file_bytes")
    if not isinstance(peak, int) or not isinstance(model_bytes, int):
        raise OrtPipelineError("ORT runner memory metrics have invalid types")
    if not embedding_output.is_file() or embedding_output.stat().st_size != (
        EMBEDDING_DIMENSION * 4
    ):
        raise OrtPipelineError("ORT runner did not write a 192-float embedding")
    embedding = np.fromfile(embedding_output, dtype="<f4")
    if not np.isfinite(embedding).all():
        raise OrtPipelineError("ORT embedding contains non-finite values")
    latency = statistics.fmean(float(value) for value in timings)
    metrics = RuntimeMetrics(
        latency_mean_ms=latency,
        rtf=latency / (AUDIO_SECONDS_BY_BUCKET[bucket_frames] * 1000.0),
        peak_rss_bytes=peak,
        weight_bytes=model_bytes,
        activation_bytes=None,
        backend_name="onnxruntime-cpu",
        runtime_peak_label="ORT process",
        weight_label="ONNX model file",
        activation_label="logical activation",
    )
    return OrtEmbeddingResult(
        embedding=embedding.astype(np.float32),
        metrics=metrics,
        runtime_payload=payload,
        assets=assets,
    )
