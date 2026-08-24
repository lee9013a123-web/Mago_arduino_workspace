"""One-shot microphone capture, embedding inference, and speaker scoring."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from voice_embedding.audio import (
    MicrophoneProfile,
    countdown_before_recording,
    record_wav,
)
from voice_embedding.frontend import wav_to_fixed_fbank, write_feature
from voice_embedding.runtime import (
    AUDIO_SECONDS_BY_BUCKET,
    RuntimePipelineError,
    describe_assets,
    run_embedding,
)

from .reporting import format_terminal_report
from .memory_monitor import ProcessTreeMemoryMonitor
from .scoring import (
    cosine_similarity,
    load_embedding,
    resolve_speaker_embedding,
)


def verify_speaker(
    *, repo_root: Path, pipeline_root: Path, profile: MicrophoneProfile,
    speaker_embedding: str, bucket_frames: int, runtime_binary: Path,
    native_fbank_binary: Path, asset_manifest: Path, warmup: int = 0,
    repeat: int = 1,
    threads: int = 1, countdown_seconds: int = 3,
    output: Callable[[str], None] = print,
) -> dict:
    if bucket_frames not in AUDIO_SECONDS_BY_BUCKET:
        raise RuntimePipelineError(f"unsupported bucket: {bucket_frames}")
    template_path = resolve_speaker_embedding(pipeline_root, speaker_embedding)
    template = load_embedding(template_path)
    seconds = AUDIO_SECONDS_BY_BUCKET[bucket_frames]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = pipeline_root / "runs/inference" / timestamp
    wav_path = run_root / f"query__{bucket_frames}.wav"
    feature_path = run_root / f"query__{bucket_frames}.f32"
    embedding_path = run_root / f"query_embedding__{bucket_frames}.f32"
    report_path = run_root / "report.json"
    run_root.mkdir(parents=True, exist_ok=False)

    memory_monitor = ProcessTreeMemoryMonitor()
    memory_monitor.start()
    try:
        output(
            f"Recording {seconds} seconds with microphone {profile.version!r}..."
        )
        countdown_before_recording(countdown_seconds, output)
        record_wav(profile=profile, output_path=wav_path, seconds=seconds)
        feature = wav_to_fixed_fbank(
            wav_path=wav_path,
            native_binary=native_fbank_binary,
            bucket_frames=bucket_frames,
            audio_seconds=seconds,
        )
        write_feature(feature_path, feature)
        result = run_embedding(
            repo_root=repo_root,
            runtime_binary=runtime_binary,
            asset_manifest=asset_manifest,
            bucket_frames=bucket_frames,
            feature_path=feature_path,
            embedding_output=embedding_path,
            warmup=warmup,
            repeat=repeat,
            threads=threads,
        )
        score = cosine_similarity(template, result.embedding)
    finally:
        pipeline_memory = memory_monitor.stop()
    metrics = replace(
        result.metrics,
        pipeline_total_peak_rss_bytes=(
            pipeline_memory.pipeline_total_peak_rss_bytes
        ),
        python_torch_peak_rss_bytes=(
            pipeline_memory.python_torch_peak_rss_bytes
        ),
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_score": score,
        "threshold": None,
        "threshold_status": "not_calibrated",
        "speaker_embedding": str(template_path),
        "microphone_version": profile.version,
        "bucket_frames": bucket_frames,
        "audio_seconds": seconds,
        "runtime_assets": describe_assets(result.assets),
        "runtime_metrics": asdict(metrics),
        "memory_measurement": asdict(pipeline_memory),
        "frontend": {
            "backend": "kaldi-native-fbank",
            "binary": str(native_fbank_binary),
            "torch_required": False,
        },
        "artifacts": {
            "wav": str(wav_path),
            "feature": str(feature_path),
            "query_embedding": str(embedding_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output(format_terminal_report(score, metrics))
    output(f"report JSON: {report_path}")
    return report
