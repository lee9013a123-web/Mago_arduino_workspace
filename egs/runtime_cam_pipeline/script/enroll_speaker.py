#!/usr/bin/env python3
"""Record five 10-second utterances and create one speaker template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parents[1]
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voice_embedding.audio import (  # noqa: E402
    AudioCaptureError,
    load_microphone_profile,
)
from voice_embedding.enrollment import enroll_speaker  # noqa: E402
from voice_embedding.runtime import (  # noqa: E402
    RuntimePipelineError,
    select_bucket_assets,
    validate_runtime_capabilities,
)


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mic-version", required=True)
    parser.add_argument("--speaker-folder", required=True)
    parser.add_argument("--recording-count", type=int, default=5)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--microphone-config",
        type=_repo_path,
        default=PIPELINE_ROOT / "configs/microphones.json",
    )
    parser.add_argument(
        "--runtime",
        type=_repo_path,
        default=(
            REPO_ROOT / "build/profill/weight_streaming_98"
            / "campp_runtime_benchmark_final"
        ),
    )
    parser.add_argument(
        "--asset-manifest",
        type=_repo_path,
        default=(
            REPO_ROOT / "runs/models/campplus/final_v3/weight_streaming"
            / "weight_streaming_manifest.json"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        profile = load_microphone_profile(args.microphone_config, args.mic_version)
        assets = select_bucket_assets(
            repo_root=REPO_ROOT,
            manifest_path=args.asset_manifest,
            bucket_frames=998,
        )
        capabilities = validate_runtime_capabilities(args.runtime, 998)
        if args.dry_run:
            print(json.dumps({
                "ready": True,
                "microphone": {
                    "version": profile.version,
                    "device": profile.device,
                },
                "speaker_folder": args.speaker_folder,
                "recording_count": args.recording_count,
                "recording_seconds": 10,
                "bucket_frames": assets.bucket_frames,
                "plan": str(assets.plan),
                "weights": str(assets.weights),
                "schedule": str(assets.schedule),
                "runtime": str(args.runtime),
                "runtime_capabilities": capabilities,
            }, ensure_ascii=False, indent=2))
            return 0
        metadata = enroll_speaker(
            repo_root=REPO_ROOT,
            pipeline_root=PIPELINE_ROOT,
            profile=profile,
            speaker_folder=args.speaker_folder,
            runtime_binary=args.runtime,
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
    except (AudioCaptureError, RuntimePipelineError, OSError, ValueError) as exc:
        print(f"speaker enrollment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
