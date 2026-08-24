"""Five-utterance enrollment stored separately for ONNX Runtime."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from voice_embedding.audio import MicrophoneProfile, countdown_before_recording, record_wav
from voice_embedding.enrollment import (
    DEFAULT_RECORDINGS,
    ENROLLMENT_BUCKET,
    ENROLLMENT_SECONDS,
    aggregate_embeddings,
    l2_normalize,
    validate_speaker_folder,
)
from voice_embedding.frontend import wav_to_fixed_fbank, write_feature

from .runtime_onnx import describe_ort_assets, run_embedding_onnx, OrtPipelineError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enroll_speaker_onnx(
    *, pipeline_root: Path, profile: MicrophoneProfile, speaker_folder: str,
    native_fbank_binary: Path, asset_manifest: Path,
    recording_count: int = DEFAULT_RECORDINGS, countdown_seconds: int = 3,
    warmup: int = 0, repeat: int = 1, threads: int = 1,
    force: bool = False, output: Callable[[str], None] = print,
) -> dict:
    speaker = validate_speaker_folder(speaker_folder)
    if recording_count <= 0:
        raise OrtPipelineError("recording count must be positive")
    if countdown_seconds < 0:
        raise OrtPipelineError("countdown seconds must not be negative")
    recorded_root = pipeline_root / "voice_onnx/recorded" / speaker
    embedded_root = pipeline_root / "voice_onnx/embedded" / speaker
    run_root = pipeline_root / "runs_onnx/enrollment" / speaker
    template_path = embedded_root / "mean_embedding.f32"
    metadata_path = embedded_root / "enrollment.json"
    if not force and (template_path.exists() or metadata_path.exists()):
        raise OrtPipelineError(
            f"ORT speaker enrollment already exists: {embedded_root}; use --force"
        )
    recorded_root.mkdir(parents=True, exist_ok=True)
    embedded_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    utterances: list[dict] = []
    result = None
    for index in range(1, recording_count + 1):
        stem = f"recording_{index:02d}"
        wav_path = recorded_root / f"{stem}.wav"
        feature_path = run_root / f"{stem}__998.f32"
        embedding_path = embedded_root / f"{stem}.f32"
        output(
            f"[ORT {index}/{recording_count}] Speak naturally for "
            f"{ENROLLMENT_SECONDS} seconds."
        )
        countdown_before_recording(countdown_seconds, output)
        record_wav(profile=profile, output_path=wav_path, seconds=ENROLLMENT_SECONDS)
        feature = wav_to_fixed_fbank(
            wav_path=wav_path,
            native_binary=native_fbank_binary,
            bucket_frames=ENROLLMENT_BUCKET,
            audio_seconds=ENROLLMENT_SECONDS,
        )
        write_feature(feature_path, feature)
        result = run_embedding_onnx(
            asset_manifest=asset_manifest,
            bucket_frames=ENROLLMENT_BUCKET,
            feature_path=feature_path,
            embedding_output=embedding_path,
            warmup=warmup,
            repeat=repeat,
            threads=threads,
        )
        normalized = l2_normalize(result.embedding)
        normalized.astype("<f4").tofile(embedding_path)
        embeddings.append(normalized)
        utterances.append({
            "index": index,
            "wav": str(wav_path.relative_to(pipeline_root)),
            "embedding": str(embedding_path.relative_to(pipeline_root)),
            "embedding_sha256": _sha256(embedding_path),
            "runtime_metrics": asdict(result.metrics),
        })
    if result is None:
        raise OrtPipelineError("ORT enrollment produced no embedding")
    template = aggregate_embeddings(embeddings)
    template.astype("<f4").tofile(template_path)
    metadata = {
        "schema_version": 1,
        "backend": "onnxruntime-cpu",
        "speaker_folder": speaker,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "microphone_version": profile.version,
        "recording_count": recording_count,
        "recording_seconds": ENROLLMENT_SECONDS,
        "bucket_frames": ENROLLMENT_BUCKET,
        "aggregation": "mean_of_l2_normalized_then_l2_normalize",
        "embedding_dimension": int(template.size),
        "mean_embedding": str(template_path.relative_to(pipeline_root)),
        "mean_embedding_sha256": _sha256(template_path),
        "runtime_assets": describe_ort_assets(result.assets, ENROLLMENT_BUCKET),
        "frontend": {
            "backend": "kaldi-native-fbank",
            "binary": str(native_fbank_binary),
            "torch_required": False,
        },
        "utterances": utterances,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata
