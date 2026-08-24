#!/usr/bin/env python3
"""Compare native FBank against the offline Torchaudio/Kaldi reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voice_embedding.audio import fixed_length_pcm, read_pcm16_mono  # noqa: E402
from voice_embedding.frontend import (  # noqa: E402
    FBANK_BINS,
    FRAME_LENGTH_MS,
    FRAME_SHIFT_MS,
    SAMPLE_RATE,
    _extract_fbank_native,
    normalize_waveform,
)
from voice_embedding.runtime import AUDIO_SECONDS_BY_BUCKET  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument(
        "--bucket", type=int, choices=tuple(AUDIO_SECONDS_BY_BUCKET), required=True,
    )
    parser.add_argument(
        "--fbank",
        type=Path,
        default=PIPELINE_ROOT / "runtime/campp_fbank",
    )
    parser.add_argument("--atol", type=float, default=1.0e-3)
    args = parser.parse_args()

    # Torch/Torchaudio are validation-only dependencies. Runtime source does
    # not import either package.
    try:
        import torch
        import torchaudio
    except (ImportError, OSError) as exc:
        print(
            "offline validation requires torch and torchaudio: " + str(exc),
            file=sys.stderr,
        )
        return 2

    seconds = AUDIO_SECONDS_BY_BUCKET[args.bucket]
    samples = fixed_length_pcm(read_pcm16_mono(args.wav), seconds)
    waveform = normalize_waveform(samples)
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    reference = torchaudio.compliance.kaldi.fbank(
        tensor,
        num_mel_bins=FBANK_BINS,
        sample_frequency=SAMPLE_RATE,
        frame_length=FRAME_LENGTH_MS,
        frame_shift=FRAME_SHIFT_MS,
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        use_energy=False,
    ).numpy().astype(np.float32)
    reference -= reference.mean(axis=0, keepdims=True)
    candidate = _extract_fbank_native(
        wav_path=args.wav,
        native_binary=args.fbank,
        bucket_frames=args.bucket,
        audio_seconds=seconds,
    )
    difference = np.abs(reference - candidate)
    payload = {
        "ready": True,
        "bucket_frames": args.bucket,
        "shape": list(candidate.shape),
        "reference": "torchaudio.compliance.kaldi.fbank",
        "candidate": "kaldi-native-fbank-1.22.3",
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "atol": args.atol,
        "allclose": bool(np.allclose(reference, candidate, rtol=0.0, atol=args.atol)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["allclose"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
