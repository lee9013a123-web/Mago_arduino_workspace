"""Bucket-coherent Final V3 plan/weight/schedule execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any

import numpy as np


SUPPORTED_BUCKETS = (98, 298, 498, 998)
AUDIO_SECONDS_BY_BUCKET = {98: 1, 298: 3, 498: 5, 998: 10}
EMBEDDING_DIMENSION = 192
FBANK_BINS = 80


class RuntimePipelineError(RuntimeError):
    """Raised when runtime assets or output violate the pipeline contract."""


@dataclass(frozen=True)
class BucketAssets:
    bucket_frames: int
    audio_seconds: int
    plan: Path
    weights: Path
    schedule: Path


@dataclass(frozen=True)
class RuntimeMetrics:
    latency_mean_ms: float
    rtf: float
    peak_rss_bytes: int
    weight_bytes: int
    activation_bytes: int


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: np.ndarray
    metrics: RuntimeMetrics
    runtime_payload: dict[str, Any]
    assets: BucketAssets


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePipelineError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimePipelineError(f"JSON root must be an object: {path}")
    return value


def _repo_path(repo_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimePipelineError("runtime asset path is missing")
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def select_bucket_assets(
    *, repo_root: Path, manifest_path: Path, bucket_frames: int,
    require_files: bool = True,
) -> BucketAssets:
    if bucket_frames not in SUPPORTED_BUCKETS:
        raise RuntimePipelineError(
            f"unsupported bucket {bucket_frames}; expected {SUPPORTED_BUCKETS}"
        )
    document = _load_json(manifest_path)
    if document.get("format") != "campp-weight-streaming-sidecar-v1":
        raise RuntimePipelineError("unsupported runtime asset manifest")
    rows = document.get("buckets")
    row = rows.get(str(bucket_frames)) if isinstance(rows, dict) else None
    if not isinstance(row, dict):
        raise RuntimePipelineError(f"bucket {bucket_frames} is absent from manifest")
    assets = BucketAssets(
        bucket_frames=bucket_frames,
        audio_seconds=AUDIO_SECONDS_BY_BUCKET[bucket_frames],
        plan=_repo_path(repo_root, row.get("plan")),
        weights=_repo_path(repo_root, row.get("weights")),
        schedule=_repo_path(repo_root, row.get("schedule")),
    )
    paths = (assets.plan, assets.weights, assets.schedule)
    expected_suffix = f"_{bucket_frames}"
    if any(not path.stem.endswith(expected_suffix) for path in paths):
        raise RuntimePipelineError(
            f"bucket {bucket_frames} assets are not a coherent named set: "
            f"{tuple(path.name for path in paths)}"
        )
    if len({path.parent.resolve() for path in paths}) != 1:
        raise RuntimePipelineError(
            f"bucket {bucket_frames} assets must share one directory"
        )
    if require_files:
        missing = [
            str(path) for path in paths
            if not path.is_file()
        ]
        if missing:
            raise RuntimePipelineError(
                "selected bucket assets are missing: " + ", ".join(missing)
            )
    return assets


def _read_int(mapping: object, key: str) -> int:
    if not isinstance(mapping, dict) or not isinstance(mapping.get(key), int):
        raise RuntimePipelineError(f"runtime result is missing integer {key}")
    return int(mapping[key])


def validate_runtime_capabilities(
    runtime_binary: Path, bucket_frames: int,
) -> dict[str, Any]:
    if not runtime_binary.is_file():
        raise RuntimePipelineError(f"runtime binary is missing: {runtime_binary}")
    completed = subprocess.run(
        [str(runtime_binary), "--capabilities"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimePipelineError(
            f"runtime capability check failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    try:
        capabilities = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimePipelineError("runtime capabilities are not JSON") from exc
    if not isinstance(capabilities, dict):
        raise RuntimePipelineError("runtime capabilities root is not an object")
    modes = capabilities.get("weight_residency_modes")
    if not isinstance(modes, list) or "windowed" not in modes:
        raise RuntimePipelineError("runtime was built without windowed weights")
    if capabilities.get("optimization_suite") != "final":
        raise RuntimePipelineError("runtime does not contain the final suite")
    plans = capabilities.get("optimization_bucket_plans")
    if not isinstance(plans, list) or bucket_frames not in plans:
        raise RuntimePipelineError(
            f"runtime has no compiled V3 hybrid plan for bucket {bucket_frames}"
        )
    return capabilities


def _parse_metrics(payload: dict[str, Any], audio_seconds: int) -> RuntimeMetrics:
    warm = payload.get("warm")
    timings = warm.get("timings_ms") if isinstance(warm, dict) else None
    if not isinstance(timings, list) or not timings:
        raise RuntimePipelineError("runtime result has no timing samples")
    numeric = [float(value) for value in timings]
    memory = payload.get("memory")
    model = payload.get("model")
    after = memory.get("after_measurement") if isinstance(memory, dict) else None
    latency = statistics.fmean(numeric)
    return RuntimeMetrics(
        latency_mean_ms=latency,
        rtf=latency / (float(audio_seconds) * 1000.0),
        peak_rss_bytes=_read_int(after, "peak_rss_bytes"),
        weight_bytes=_read_int(model, "weight_bytes"),
        activation_bytes=_read_int(memory, "activation_bytes"),
    )


def run_embedding(
    *, repo_root: Path, runtime_binary: Path, asset_manifest: Path,
    bucket_frames: int, feature_path: Path, embedding_output: Path,
    warmup: int = 0, repeat: int = 1, threads: int = 1,
) -> EmbeddingResult:
    if not runtime_binary.is_file():
        raise RuntimePipelineError(f"runtime binary is missing: {runtime_binary}")
    if warmup < 0 or repeat <= 0 or threads <= 0:
        raise RuntimePipelineError("warmup/repeat/threads values are invalid")
    assets = select_bucket_assets(
        repo_root=repo_root,
        manifest_path=asset_manifest,
        bucket_frames=bucket_frames,
    )
    expected_feature_bytes = bucket_frames * FBANK_BINS * 4
    if not feature_path.is_file() or feature_path.stat().st_size != expected_feature_bytes:
        raise RuntimePipelineError(
            f"feature must be [1,{bucket_frames},{FBANK_BINS}] float32"
        )
    embedding_output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(runtime_binary),
        "--plan", str(assets.plan),
        "--weights", str(assets.weights),
        "--weight-mode", "windowed",
        "--weight-schedule", str(assets.schedule),
        "--input", str(feature_path),
        "--audio-seconds", str(assets.audio_seconds),
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--threads", str(threads),
        "--embedding-output", str(embedding_output),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env={**os.environ, "OMP_NUM_THREADS": "1", "ORT_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimePipelineError(
            f"C runtime failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimePipelineError("C runtime stdout is not JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimePipelineError("C runtime JSON root is not an object")
    runtime_model = payload.get("model")
    if not isinstance(runtime_model, dict) or (
        runtime_model.get("bucket_frames") != bucket_frames
    ):
        raise RuntimePipelineError("C runtime loaded a different bucket plan")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict) or (
        configuration.get("weight_mode") != "windowed"
    ):
        raise RuntimePipelineError("C runtime did not activate windowed weights")
    if payload.get("optimization_bucket_policy") != "layer_hybrid_v3":
        raise RuntimePipelineError("C runtime did not activate the V3 bucket plan")
    if not embedding_output.is_file() or embedding_output.stat().st_size != (
        EMBEDDING_DIMENSION * 4
    ):
        raise RuntimePipelineError("C runtime did not write a 192-float embedding")
    embedding = np.fromfile(embedding_output, dtype="<f4")
    if not np.isfinite(embedding).all():
        raise RuntimePipelineError("C runtime embedding contains non-finite values")
    return EmbeddingResult(
        embedding=embedding.astype(np.float32),
        metrics=_parse_metrics(payload, assets.audio_seconds),
        runtime_payload=payload,
        assets=assets,
    )
