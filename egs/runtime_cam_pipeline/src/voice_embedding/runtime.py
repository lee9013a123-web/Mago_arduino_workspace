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
    mode: str
    model: Path | None = None
    plan: Path | None = None
    weights: Path | None = None
    schedule: Path | None = None


@dataclass(frozen=True)
class RuntimeMetrics:
    latency_mean_ms: float
    rtf: float
    peak_rss_bytes: int
    weight_bytes: int
    activation_bytes: int
    pipeline_total_peak_rss_bytes: int | None = None
    python_torch_peak_rss_bytes: int | None = None


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


def _asset_path(base: Path, value: object, *, confined: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimePipelineError("runtime asset path is missing")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if confined and not resolved.is_relative_to(base.resolve()):
        raise RuntimePipelineError(
            f"pipeline-local asset escapes its runtime directory: {value}"
        )
    return resolved


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path, expected: object) -> None:
    if expected is None:
        return
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimePipelineError(f"invalid checksum for runtime asset: {path}")
    if path.is_file() and _sha256(path) != expected:
        raise RuntimePipelineError(f"runtime asset checksum mismatch: {path}")


def select_bucket_assets(
    *, repo_root: Path, manifest_path: Path, bucket_frames: int,
    require_files: bool = True,
) -> BucketAssets:
    if bucket_frames not in SUPPORTED_BUCKETS:
        raise RuntimePipelineError(
            f"unsupported bucket {bucket_frames}; expected {SUPPORTED_BUCKETS}"
        )
    document = _load_json(manifest_path)
    manifest_format = document.get("format")
    rows = document.get("buckets")
    row = rows.get(str(bucket_frames)) if isinstance(rows, dict) else None
    if not isinstance(row, dict):
        raise RuntimePipelineError(f"bucket {bucket_frames} is absent from manifest")
    if manifest_format == "campp-runtime-pipeline-assets-v1":
        mode = document.get("mode")
        base = manifest_path.parent
        confined = True
    elif manifest_format == "campp-weight-streaming-sidecar-v1":
        mode = "windowed"
        base = repo_root
        confined = False
    elif manifest_format == "campp-fixed-bucket-model-set-v1":
        mode = "package"
        base = manifest_path.parent
        confined = True
    else:
        raise RuntimePipelineError("unsupported runtime asset manifest")
    if mode == "package":
        model = _asset_path(base, row.get("model"), confined=confined)
        assets = BucketAssets(
            bucket_frames=bucket_frames,
            audio_seconds=AUDIO_SECONDS_BY_BUCKET[bucket_frames],
            mode=mode,
            model=model,
        )
        if not model.stem.endswith(f"_{bucket_frames}"):
            raise RuntimePipelineError(
                f"bucket {bucket_frames} package has an inconsistent name: "
                f"{model.name}"
            )
        if require_files and not model.is_file():
            raise RuntimePipelineError(f"selected model package is missing: {model}")
        _verify_checksum(model, row.get("sha256"))
        return assets
    if mode != "windowed":
        raise RuntimePipelineError(f"unsupported pipeline runtime mode: {mode!r}")
    assets = BucketAssets(
        bucket_frames=bucket_frames,
        audio_seconds=AUDIO_SECONDS_BY_BUCKET[bucket_frames],
        mode=mode,
        plan=_asset_path(base, row.get("plan"), confined=confined),
        weights=_asset_path(base, row.get("weights"), confined=confined),
        schedule=_asset_path(base, row.get("schedule"), confined=confined),
    )
    paths = (assets.plan, assets.weights, assets.schedule)
    if any(path is None for path in paths):
        raise RuntimePipelineError("windowed runtime assets are incomplete")
    complete_paths = tuple(path for path in paths if path is not None)
    expected_suffix = f"_{bucket_frames}"
    if any(not path.stem.endswith(expected_suffix) for path in complete_paths):
        raise RuntimePipelineError(
            f"bucket {bucket_frames} assets are not a coherent named set: "
            f"{tuple(path.name for path in complete_paths)}"
        )
    if len({path.parent.resolve() for path in complete_paths}) != 1:
        raise RuntimePipelineError(
            f"bucket {bucket_frames} assets must share one directory"
        )
    if require_files:
        missing = [
            str(path) for path in complete_paths
            if not path.is_file()
        ]
        if missing:
            raise RuntimePipelineError(
                "selected bucket assets are missing: " + ", ".join(missing)
            )
    for key, path in zip(("plan", "weights", "schedule"), complete_paths):
        _verify_checksum(path, row.get(f"{key}_sha256"))
    return assets


def describe_assets(assets: BucketAssets) -> dict[str, object]:
    return {
        "mode": assets.mode,
        "bucket_frames": assets.bucket_frames,
        "audio_seconds": assets.audio_seconds,
        "model": str(assets.model) if assets.model is not None else None,
        "plan": str(assets.plan) if assets.plan is not None else None,
        "weights": str(assets.weights) if assets.weights is not None else None,
        "schedule": str(assets.schedule) if assets.schedule is not None else None,
    }


def require_pipeline_local(pipeline_root: Path, paths: list[Path]) -> None:
    boundary = pipeline_root.resolve()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(boundary):
            raise RuntimePipelineError(
                f"runtime pipeline path escapes {boundary}: {resolved}"
            )


def _read_int(mapping: object, key: str) -> int:
    if not isinstance(mapping, dict) or not isinstance(mapping.get(key), int):
        raise RuntimePipelineError(f"runtime result is missing integer {key}")
    return int(mapping[key])


def validate_runtime_capabilities(
    runtime_binary: Path, bucket_frames: int, mode: str,
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
    if mode == "package":
        if capabilities.get("model_package_format") != "camppmodel-v1":
            raise RuntimePipelineError("runtime cannot load camppmodel-v1")
    elif mode == "windowed":
        modes = capabilities.get("weight_residency_modes")
        if not isinstance(modes, list) or "windowed" not in modes:
            raise RuntimePipelineError("runtime was built without windowed weights")
    else:
        raise RuntimePipelineError(f"unsupported runtime mode: {mode!r}")
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
    command = [str(runtime_binary)]
    if assets.mode == "package":
        if assets.model is None:
            raise RuntimePipelineError("package selection has no model")
        command.extend(["--model", str(assets.model)])
        expected_weight_mode = "malloc"
    else:
        if assets.plan is None or assets.weights is None or assets.schedule is None:
            raise RuntimePipelineError("windowed selection is incomplete")
        command.extend([
            "--plan", str(assets.plan),
            "--weights", str(assets.weights),
            "--weight-mode", "windowed",
            "--weight-schedule", str(assets.schedule),
        ])
        expected_weight_mode = "windowed"
    command.extend([
        "--input", str(feature_path),
        "--audio-seconds", str(assets.audio_seconds),
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--threads", str(threads),
        "--embedding-output", str(embedding_output),
    ])
    completed = subprocess.run(
        command,
        cwd=runtime_binary.parent,
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
        configuration.get("weight_mode") != expected_weight_mode
    ):
        raise RuntimePipelineError(
            f"C runtime did not activate {expected_weight_mode} weight mode"
        )
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
