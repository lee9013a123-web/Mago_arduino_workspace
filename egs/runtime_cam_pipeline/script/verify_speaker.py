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
from similarity_detect_onnx.scoring_onnx import (  # noqa: E402
    resolve_onnx_speaker_embedding,
)
from similarity_detect_onnx.verification_onnx import (  # noqa: E402
    verify_speaker_onnx,
)
from voice_embedding.frontend import (  # noqa: E402
    FrontendError,
    validate_native_fbank,
)
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
from voice_embedding_onnx.runtime_onnx import (  # noqa: E402
    describe_ort_assets,
    OrtPipelineError,
    select_ort_assets,
    validate_ort_capabilities,
)


def _pipeline_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PIPELINE_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--c", action="store_true", help="use Final V3 C Runtime")
    backend.add_argument("--ort", action="store_true", help="use ONNX Runtime CPU")
    parser.add_argument("--mic-version", required=True)
    parser.add_argument(
        "--speaker-embedding", "--enrollment",
        dest="speaker_embedding",
        required=True,
        help=(
            "raw/npy embedding path or backend-specific voice/embedded "
            "speaker folder name"
        ),
    )
    parser.add_argument(
        "--bucket", type=int, choices=tuple(AUDIO_SECONDS_BY_BUCKET), required=True,
    )
    parser.add_argument("--countdown", type=int, default=3)
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
        "--fbank",
        type=_pipeline_path,
        default=(
            PIPELINE_ROOT / "runtime/campp_fbank"
        ),
    )
    parser.add_argument(
        "--ort-asset-manifest",
        type=_pipeline_path,
        default=(
            PIPELINE_ROOT / "runtime_onnx/assets.json"
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
        if args.ort:
            template = resolve_onnx_speaker_embedding(
                PIPELINE_ROOT, args.speaker_embedding,
            )
            ort_assets = select_ort_assets(args.ort_asset_manifest, args.bucket)
            require_pipeline_local(PIPELINE_ROOT, [
                args.microphone_config,
                args.ort_asset_manifest,
                template,
                ort_assets.model,
                ort_assets.frontend,
            ])
            frontend_capabilities = validate_native_fbank(ort_assets.frontend)
            ort_capabilities = validate_ort_capabilities()
            if args.dry_run:
                print(json.dumps({
                    "ready": True,
                    "backend": "onnxruntime-cpu",
                    "microphone": {
                        "version": profile.version,
                        "device": profile.device,
                    },
                    "speaker_embedding": str(template),
                    "bucket_frames": args.bucket,
                    "audio_seconds": AUDIO_SECONDS_BY_BUCKET[args.bucket],
                    "assets": describe_ort_assets(ort_assets, args.bucket),
                    "frontend_capabilities": frontend_capabilities,
                    "runtime_capabilities": ort_capabilities,
                }, ensure_ascii=False, indent=2))
                return 0
            verify_speaker_onnx(
                pipeline_root=PIPELINE_ROOT,
                profile=profile,
                speaker_embedding=args.speaker_embedding,
                bucket_frames=args.bucket,
                native_fbank_binary=ort_assets.frontend,
                asset_manifest=args.ort_asset_manifest,
                warmup=args.warmup,
                repeat=args.repeat,
                threads=args.threads,
                countdown_seconds=args.countdown,
            )
            return 0
        template = resolve_speaker_embedding(PIPELINE_ROOT, args.speaker_embedding)
        assets = select_bucket_assets(
            repo_root=PIPELINE_ROOT,
            manifest_path=args.asset_manifest,
            bucket_frames=args.bucket,
        )
        require_pipeline_local(PIPELINE_ROOT, [
            args.microphone_config,
            args.runtime,
            args.fbank,
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
        frontend_capabilities = validate_native_fbank(args.fbank)
        if args.dry_run:
            print(json.dumps({
                "ready": True,
                "backend": "campp-c-runtime",
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
                "frontend": str(args.fbank),
                "frontend_capabilities": frontend_capabilities,
            }, ensure_ascii=False, indent=2))
            return 0
        verify_speaker(
            repo_root=PIPELINE_ROOT,
            pipeline_root=PIPELINE_ROOT,
            profile=profile,
            speaker_embedding=args.speaker_embedding,
            bucket_frames=args.bucket,
            runtime_binary=args.runtime,
            native_fbank_binary=args.fbank,
            asset_manifest=args.asset_manifest,
            warmup=args.warmup,
            repeat=args.repeat,
            threads=args.threads,
            countdown_seconds=args.countdown,
        )
        return 0
    except (
        AudioCaptureError,
        FrontendError,
        OrtPipelineError,
        RuntimePipelineError,
        SimilarityError,
        OSError,
        ValueError,
    ) as exc:
        print(f"speaker verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
