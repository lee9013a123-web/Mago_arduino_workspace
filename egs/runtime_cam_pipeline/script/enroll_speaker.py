#!/usr/bin/env python3
"""Record five 10-second utterances and create one speaker template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voice_embedding.audio import (  # noqa: E402
    AudioCaptureError,
    load_microphone_profile,
)
from voice_embedding.enrollment import enroll_speaker  # noqa: E402
from voice_embedding.frontend import (  # noqa: E402
    FrontendError,
    validate_native_fbank,
)
from voice_embedding.runtime import (  # noqa: E402
    describe_assets,
    require_pipeline_local,
    RuntimePipelineError,
    select_bucket_assets,
    validate_runtime_capabilities,
)
from voice_embedding_onnx.enrollment_onnx import (  # noqa: E402
    enroll_speaker_onnx,
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
    parser.add_argument("--speaker-folder", required=True)
    parser.add_argument("--recording-count", type=int, default=5)
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        profile = load_microphone_profile(args.microphone_config, args.mic_version)
        if args.ort:
            ort_assets = select_ort_assets(args.ort_asset_manifest, 998)
            require_pipeline_local(PIPELINE_ROOT, [
                args.microphone_config,
                args.ort_asset_manifest,
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
                    "speaker_folder": args.speaker_folder,
                    "recording_count": args.recording_count,
                    "recording_seconds": 10,
                    "bucket_frames": 998,
                    "assets": describe_ort_assets(ort_assets, 998),
                    "frontend_capabilities": frontend_capabilities,
                    "runtime_capabilities": ort_capabilities,
                }, ensure_ascii=False, indent=2))
                return 0
            metadata = enroll_speaker_onnx(
                pipeline_root=PIPELINE_ROOT,
                profile=profile,
                speaker_folder=args.speaker_folder,
                native_fbank_binary=ort_assets.frontend,
                asset_manifest=args.ort_asset_manifest,
                recording_count=args.recording_count,
                countdown_seconds=args.countdown,
                warmup=args.warmup,
                repeat=args.repeat,
                threads=args.threads,
                force=args.force,
            )
            print(
                "ORT enrollment complete: "
                f"{PIPELINE_ROOT / metadata['mean_embedding']}"
            )
            return 0
        assets = select_bucket_assets(
            repo_root=PIPELINE_ROOT,
            manifest_path=args.asset_manifest,
            bucket_frames=998,
        )
        require_pipeline_local(PIPELINE_ROOT, [
            args.microphone_config,
            args.runtime,
            args.fbank,
            args.asset_manifest,
            *[
                path for path in (
                    assets.model, assets.plan, assets.weights, assets.schedule,
                ) if path is not None
            ],
        ])
        capabilities = validate_runtime_capabilities(
            args.runtime, 998, assets.mode,
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
                "speaker_folder": args.speaker_folder,
                "recording_count": args.recording_count,
                "recording_seconds": 10,
                "bucket_frames": assets.bucket_frames,
                "assets": describe_assets(assets),
                "runtime": str(args.runtime),
                "runtime_capabilities": capabilities,
                "frontend": str(args.fbank),
                "frontend_capabilities": frontend_capabilities,
            }, ensure_ascii=False, indent=2))
            return 0
        metadata = enroll_speaker(
            repo_root=PIPELINE_ROOT,
            pipeline_root=PIPELINE_ROOT,
            profile=profile,
            speaker_folder=args.speaker_folder,
            runtime_binary=args.runtime,
            native_fbank_binary=args.fbank,
            asset_manifest=args.asset_manifest,
            recording_count=args.recording_count,
            countdown_seconds=args.countdown,
            warmup=args.warmup,
            repeat=args.repeat,
            threads=args.threads,
            force=args.force,
        )
        print(
            "enrollment complete: "
            f"{PIPELINE_ROOT / metadata['mean_embedding']}"
        )
        return 0
    except (
        AudioCaptureError,
        FrontendError,
        OrtPipelineError,
        RuntimePipelineError,
        OSError,
        ValueError,
    ) as exc:
        print(f"speaker enrollment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
