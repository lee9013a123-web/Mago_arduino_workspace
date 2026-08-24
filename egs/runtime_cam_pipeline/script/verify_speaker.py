#!/usr/bin/env python3
"""Capture one fixed-bucket utterance and print speaker cosine similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from similarity_detect.scoring import (  # noqa: E402
    SimilarityError,
    resolve_speaker_embedding,
)
from similarity_detect.verification import verify_speaker  # noqa: E402
from voice_embedding.audio import (  # noqa: E402
    AudioCaptureError,
    load_microphone_profile,
)
from voice_embedding.runtime import (  # noqa: E402
    AUDIO_SECONDS_BY_BUCKET,
    describe_assets,
    require_pipeline_local,
    RuntimePipelineError,
    select_bucket_assets,
    validate_runtime_capabilities,
)


def _pipeline_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PIPELINE_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mic-version", required=True)
    parser.add_argument(
        "--speaker-embedding", "--enrollment",
        dest="speaker_embedding",
        required=True,
        help="raw/npy embedding path or voice/embedded speaker folder name",
    )
    parser.add_argument(
        "--bucket", type=int, choices=tuple(AUDIO_SECONDS_BY_BUCKET), required=True,
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--microphone-config",
        type=_pipeline_path,
        default=PIPELINE_ROOT / "configs/microphones.json",
    )
    parser.add_argument(
        "--runtime",
        type=_pipeline_path,
        default=(
            PIPELINE_ROOT / "runtime/campp_runtime"
        ),
    )
    parser.add_argument(
        "--asset-manifest",
        type=_pipeline_path,
        default=(
            PIPELINE_ROOT / "runtime/assets.json"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        profile = load_microphone_profile(args.microphone_config, args.mic_version)
        template = resolve_speaker_embedding(PIPELINE_ROOT, args.speaker_embedding)
        assets = select_bucket_assets(
            repo_root=PIPELINE_ROOT,
            manifest_path=args.asset_manifest,
            bucket_frames=args.bucket,
        )
        require_pipeline_local(PIPELINE_ROOT, [
            args.microphone_config,
            args.runtime,
            args.asset_manifest,
            template,
            *[
                path for path in (
                    assets.model, assets.plan, assets.weights, assets.schedule,
                ) if path is not None
            ],
        ])
        capabilities = validate_runtime_capabilities(
            args.runtime, args.bucket, assets.mode,
        )
        if args.dry_run:
            print(json.dumps({
                "ready": True,
                "microphone": {
                    "version": profile.version,
                    "device": profile.device,
                },
                "speaker_embedding": str(template),
                "bucket_frames": args.bucket,
                "audio_seconds": AUDIO_SECONDS_BY_BUCKET[args.bucket],
                "assets": describe_assets(assets),
                "runtime": str(args.runtime),
                "runtime_capabilities": capabilities,
            }, ensure_ascii=False, indent=2))
            return 0
        verify_speaker(
            repo_root=PIPELINE_ROOT,
            pipeline_root=PIPELINE_ROOT,
            profile=profile,
            speaker_embedding=args.speaker_embedding,
            bucket_frames=args.bucket,
            runtime_binary=args.runtime,
            asset_manifest=args.asset_manifest,
            warmup=args.warmup,
            repeat=args.repeat,
            threads=args.threads,
        )
        return 0
    except (
        AudioCaptureError,
        RuntimePipelineError,
        SimilarityError,
        OSError,
        ValueError,
    ) as exc:
        print(f"speaker verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
