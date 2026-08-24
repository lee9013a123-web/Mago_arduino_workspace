#!/usr/bin/env python3
"""Build bucket-wise FBank features for the 780-trial EER protocol.

`runtime_features.json`은 지연시간 측정을 위해 화자별 wav를 이어붙여 zero-padding
없이 4개 bucket을 채웠다. EER은 trial마다 발화가 독립이어야 하므로 이어붙일 수
없고, bucket보다 짧은 입력은 zero-padding으로 채운다. 어느 입력이 얼마나 padding
되었는지는 manifest의 `padding` 필드에 남긴다 -- bucket별 EER을 비교할 때 이
값을 반드시 함께 읽어야 한다.

FBank 계약은 `egs/pipeline_experiment/core.py::extract_fbank_torchaudio`와 같다.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
PIPELINE_SRC = ROOT / "egs/runtime_cam_pipeline/src"
for entry in (PYTHON_SRC, PIPELINE_SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402

from speaker_verification_evaluation import read_trials  # noqa: E402
from speaker_verification_evaluation.manifests import (  # noqa: E402
    file_sha256,
    required_input_ids,
)
from voice_embedding.audio import (  # noqa: E402
    SAMPLE_RATE,
    fixed_length_pcm,
    read_pcm16_mono,
)
from voice_embedding.frontend import (  # noqa: E402
    FBANK_BINS,
    extract_fbank,
    normalize_waveform,
)


SUPPORTED_BUCKETS = (98, 298, 498, 998)
AUDIO_SECONDS_BY_BUCKET = {98: 1, 298: 3, 498: 5, 998: 10}


class FeatureBuildError(RuntimeError):
    """Feature generation cannot satisfy the EER protocol."""


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_inputs(path: Path) -> dict[str, dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
    except OSError as exc:
        raise FeatureBuildError(f"cannot read inputs manifest: {path}") from exc
    table: dict[str, dict[str, str]] = {}
    for row in rows:
        input_id = (row.get("input_id") or "").strip()
        if input_id:
            table[input_id] = row
    if not table:
        raise FeatureBuildError(f"inputs manifest is empty: {path}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=_path,
        default=ROOT / "benchmarks/campplus/manifests/trials.tsv",
    )
    parser.add_argument(
        "--inputs", type=_path,
        default=ROOT / "benchmarks/campplus/manifests/inputs.tsv",
    )
    parser.add_argument(
        "--audio-root", type=Path, default=Path("/home/arduino/workspace/data"),
    )
    parser.add_argument(
        "--feature-root", type=_path,
        default=ROOT / "benchmarks/campplus/features/eer",
    )
    parser.add_argument(
        "--output", type=_path,
        default=ROOT / "benchmarks/campplus/manifests/eer_features.json",
    )
    parser.add_argument(
        "--buckets", type=int, nargs="+", choices=SUPPORTED_BUCKETS,
        default=list(SUPPORTED_BUCKETS),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        if args.output.exists() and not args.force:
            raise FeatureBuildError(f"output exists: {args.output}; use --force")
        buckets = tuple(dict.fromkeys(args.buckets))
        trials = read_trials(args.trials)
        needed = required_input_ids(trials)
        inputs = read_inputs(args.inputs)
        missing = sorted(needed - set(inputs))
        if missing:
            raise FeatureBuildError(
                f"inputs manifest lacks {len(missing)} trial ids: {missing[:3]}"
            )

        features: list[dict[str, Any]] = []
        padding_summary: dict[str, dict[str, int]] = {}
        for input_id in sorted(needed):
            row = inputs[input_id]
            wav_path = args.audio_root / row["relative_path"]
            if not wav_path.is_file():
                raise FeatureBuildError(f"source audio missing: {wav_path}")
            samples = read_pcm16_mono(wav_path)
            source_samples = int(samples.size)
            for bucket in buckets:
                seconds = AUDIO_SECONDS_BY_BUCKET[bucket]
                expected = SAMPLE_RATE * seconds
                fixed = fixed_length_pcm(samples, seconds)
                padded = max(0, expected - source_samples)
                feature = extract_fbank(normalize_waveform(fixed))
                if feature.shape != (bucket, FBANK_BINS):
                    raise FeatureBuildError(
                        f"{input_id}/{bucket}: produced {feature.shape}, "
                        f"expected ({bucket}, {FBANK_BINS})"
                    )
                if not np.isfinite(feature).all():
                    raise FeatureBuildError(f"{input_id}/{bucket}: non-finite FBank")
                target = args.feature_root / str(bucket) / f"{input_id}.f32"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    np.asarray(feature, dtype="<f4").tobytes(order="C")
                )
                bucket_key = str(bucket)
                stats = padding_summary.setdefault(
                    bucket_key, {"inputs": 0, "zero_padded_inputs": 0}
                )
                stats["inputs"] += 1
                if padded > 0:
                    stats["zero_padded_inputs"] += 1
                features.append({
                    "input_id": input_id,
                    "bucket_frames": bucket,
                    "audio_seconds": float(seconds),
                    "path": target.resolve().relative_to(ROOT).as_posix(),
                    "sha256": file_sha256(target),
                    "padding": {
                        "source_samples": source_samples,
                        "bucket_samples": expected,
                        "zero_padded_samples": padded,
                        "zero_padded_ratio": round(padded / expected, 6),
                    },
                })
            print(f"features built: {input_id}", flush=True)

        document = {
            "schema_version": 1,
            "protocol": "eer_780_trials",
            "preprocessing": {
                "implementation": (
                    "egs/pipeline_experiment/core.py::extract_fbank_torchaudio "
                    "(torchaudio.compliance.kaldi.fbank)"
                ),
                "source_dataset": str(args.audio_root),
                "concatenation": (
                    "없음. EER trial은 발화 단위가 독립이어야 하므로 "
                    "runtime_features.json과 달리 wav를 이어붙이지 않는다."
                ),
                "crop_padding_policy": (
                    "offset 0에서 bucket 길이(1/3/5/10초)만큼 자르고, 원본이 "
                    "짧으면 뒤를 zero-padding한다. padding 비율은 feature마다 "
                    "`padding` 필드에 기록한다."
                ),
                "waveform_normalization": (
                    "int16 → float32 [-1,1], DC 평균 제거, peak를 0.95로 정규화, "
                    "[-1,1] clip. crop/pad 이후의 파형에 적용한다."
                ),
                "fbank": {
                    "num_mel_bins": FBANK_BINS,
                    "sample_frequency": SAMPLE_RATE,
                    "frame_length_ms": 25.0,
                    "frame_shift_ms": 10.0,
                    "dither": 0.0,
                    "energy_floor": 0.0,
                    "window_type": "hamming",
                    "use_energy": False,
                    "snip_edges": True,
                },
                "cmvn": "time축 평균 감산 (feat - feat.mean(axis=0))",
                "payload_layout": "[1, frames, 80] little-endian float32",
            },
            "padding_summary": padding_summary,
            "features": features,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(json.dumps({
            "ready": True,
            "output": str(args.output),
            "inputs": len(needed),
            "buckets": list(buckets),
            "features": len(features),
            "padding_summary": padding_summary,
        }, ensure_ascii=False, indent=2))
        return 0
    except (FeatureBuildError, OSError, ValueError) as exc:
        print(f"eer feature build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
