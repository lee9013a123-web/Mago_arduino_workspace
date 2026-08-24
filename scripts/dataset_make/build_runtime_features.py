#!/usr/bin/env python3
"""09 벤치마크용 실데이터 FBank payload와 두 manifest를 만든다.

화자 1명당 3개(약 4초)인 `data/multi_speaker`는 클립 하나로는 498/998 frame을
채우지 못한다. 같은 화자의 3개 클립을 파일명 순서로 이어붙이면 중앙값 12초가
되어 1·3·5·10초 bucket을 모두 실제 음성으로 덮을 수 있다. zero-padding은 쓰지
않는다.

FBank는 배포 파이프라인(`egs/pipeline_experiment/core.py`)의 고정 구현을 그대로
쓴다. 이 스크립트는 전처리 정책을 새로 정의하지 않고 그 파라미터를 복제한 뒤
manifest의 `preprocessing`에 기록한다.

산출물:
  data/multi_speaker_concat/<speaker>.wav          이어붙인 source WAV
  benchmarks/campplus/features/<input_id>__<N>.f32 [1,N,80] LE float32
  benchmarks/campplus/manifests/runtime_inputs.tsv
  benchmarks/campplus/manifests/runtime_features.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

# egs/pipeline_experiment/core.py의 PipelineConfig 기본값을 그대로 복제한다.
SAMPLE_RATE = 16000
FBANK_BINS = 80
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0
TARGET_PEAK = 0.95

CORE_IMPLEMENTATION = (
    "egs/pipeline_experiment/core.py::extract_fbank_torchaudio "
    "(torchaudio.compliance.kaldi.fbank)"
)


class FeatureBuildError(RuntimeError):
    """데이터셋이 요구 조건을 만족하지 않을 때 발생한다."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pcm16_mono(path: Path) -> np.ndarray:
    """16 kHz mono 16-bit WAV를 int16 배열로 읽는다."""

    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1:
            raise FeatureBuildError(f"mono가 아니다: {path}")
        if source.getsampwidth() != 2:
            raise FeatureBuildError(f"16-bit이 아니다: {path}")
        if source.getframerate() != SAMPLE_RATE:
            raise FeatureBuildError(f"{SAMPLE_RATE} Hz가 아니다: {path}")
        raw = source.readframes(source.getnframes())
    return np.frombuffer(raw, dtype="<i2")


def write_pcm16_mono(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(SAMPLE_RATE)
        sink.writeframes(samples.astype("<i2").tobytes())


def normalize_waveform(samples: np.ndarray) -> np.ndarray:
    """core.py의 load_audio_16k_mono와 같은 정규화를 적용한다.

    이미 16 kHz mono이므로 resample 분기는 해당되지 않는다. DC 제거 → peak
    정규화 → clip 순서는 core.py와 같다.
    """

    wav = (samples.astype(np.float32) / 32768.0).astype(np.float32)
    if wav.size > 0:
        wav = wav - np.float32(np.mean(wav))
    peak = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
    if peak > 1e-8:
        wav = wav / peak * TARGET_PEAK
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def extract_fbank(wav: np.ndarray) -> np.ndarray:
    """core.py의 extract_fbank_torchaudio와 동일한 FBank + time축 CMVN."""

    import torch
    import torchaudio

    tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    feat = torchaudio.compliance.kaldi.fbank(
        tensor,
        num_mel_bins=FBANK_BINS,
        sample_frequency=SAMPLE_RATE,
        frame_length=FRAME_LENGTH_MS,
        frame_shift=FRAME_SHIFT_MS,
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        use_energy=False,
    )
    feat = feat.numpy().astype(np.float32)
    return (feat - feat.mean(axis=0, keepdims=True)).astype(np.float32)


def eligible_speakers(dataset_root: Path, minimum_seconds: float) -> list[Path]:
    """클립 3개를 이어붙였을 때 가장 긴 bucket을 덮는 화자를 정렬해 돌려준다."""

    chosen: list[Path] = []
    for directory in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        clips = sorted(directory.glob("*.wav"))
        if len(clips) != 3:
            continue
        total = 0
        for clip in clips:
            with wave.open(str(clip), "rb") as source:
                if (
                    source.getnchannels() != 1
                    or source.getsampwidth() != 2
                    or source.getframerate() != SAMPLE_RATE
                ):
                    total = -1
                    break
                total += source.getnframes()
        if total >= 0 and total / SAMPLE_RATE >= minimum_seconds:
            chosen.append(directory)
    return chosen


def load_config(path: Path) -> tuple[dict[int, float], dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    buckets = {int(key): float(value) for key, value in document["buckets"].items()}
    return buckets, document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "benchmark" / "runtime_qrb2210.json",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/arduino/workspace/data/multi_speaker")
    )
    parser.add_argument(
        "--concat-dir",
        type=Path,
        default=Path("/home/arduino/workspace/data/multi_speaker_concat"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/home/arduino/workspace/data")
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=ROOT / "benchmarks" / "campplus" / "features"
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=ROOT / "benchmarks" / "campplus" / "manifests" / "runtime_inputs.tsv",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=ROOT / "benchmarks" / "campplus" / "manifests" / "runtime_features.json",
    )
    parser.add_argument("--speakers", type=int, default=15)
    args = parser.parse_args(argv)

    try:
        buckets, _ = load_config(args.config)
        if not buckets:
            raise FeatureBuildError("config에 bucket이 없다")
        longest_seconds = max(buckets.values())

        if not args.dataset_root.is_dir():
            raise FeatureBuildError(f"데이터셋이 없다: {args.dataset_root}")
        candidates = eligible_speakers(args.dataset_root, longest_seconds)
        if len(candidates) < args.speakers:
            raise FeatureBuildError(
                f"{longest_seconds}초를 덮는 화자가 {len(candidates)}명뿐이다"
                f" (요구 {args.speakers}명)"
            )
        selected = candidates[: args.speakers]

        args.feature_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []
        features: list[dict[str, object]] = []

        for directory in selected:
            speaker = directory.name
            input_id = f"multi__speaker_{speaker}"
            clips = sorted(directory.glob("*.wav"))
            concatenated = np.concatenate([read_pcm16_mono(clip) for clip in clips])

            source_wav = args.concat_dir / f"{speaker}.wav"
            write_pcm16_mono(source_wav, concatenated)
            relative_path = source_wav.relative_to(args.data_root).as_posix()
            rows.append(
                {
                    "input_id": input_id,
                    "split": "multi",
                    "speaker_id": speaker,
                    "relative_path": relative_path,
                    "duration_sec": f"{concatenated.size / SAMPLE_RATE:.6f}",
                    "sample_rate_hz": str(SAMPLE_RATE),
                    "channels": "1",
                    "sample_width_bytes": "2",
                    "sha256": sha256_file(source_wav),
                    "latency_selected": "1",
                }
            )

            waveform = normalize_waveform(concatenated)
            for frames, seconds in sorted(buckets.items()):
                wanted = int(round(seconds * SAMPLE_RATE))
                if waveform.size < wanted:
                    raise FeatureBuildError(
                        f"{input_id}: {seconds}초에 필요한 샘플이 부족하다"
                    )
                feature = extract_fbank(waveform[:wanted])
                if feature.shape != (frames, FBANK_BINS):
                    raise FeatureBuildError(
                        f"{input_id}/{frames}: FBank shape {feature.shape}가 "
                        f"({frames}, {FBANK_BINS})와 다르다"
                    )
                payload = args.feature_dir / f"{input_id}__{frames}.f32"
                feature.reshape(1, frames, FBANK_BINS).astype("<f4").tofile(payload)
                features.append(
                    {
                        "input_id": input_id,
                        "bucket_frames": frames,
                        "audio_seconds": seconds,
                        "path": payload.relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(payload),
                    }
                )
            print(f"  {input_id}: {len(buckets)} bucket", flush=True)

        columns = list(rows[0])
        args.dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.dataset_manifest.open("w", encoding="utf-8", newline="") as sink:
            sink.write("\t".join(columns) + "\n")
            for row in rows:
                sink.write("\t".join(row[name] for name in columns) + "\n")

        document = {
            "schema_version": 1,
            "preprocessing": {
                "implementation": CORE_IMPLEMENTATION,
                "source_dataset": str(args.dataset_root),
                "concatenation": (
                    "화자별 wav 3개를 파일명 오름차순으로 이어붙여 하나의 source "
                    "WAV로 만든다. 원본이 4초 내외라 단일 클립으로는 498/998 "
                    "frame을 덮지 못하기 때문이다."
                ),
                "crop_padding_policy": (
                    "이어붙인 waveform을 정규화한 뒤 offset 0에서 bucket 길이"
                    "(1/3/5/10초)만큼 자른다. bucket은 서로의 prefix이며 "
                    "zero-padding은 쓰지 않는다."
                ),
                "waveform_normalization": (
                    "int16 → float32 [-1,1], DC 평균 제거, peak를 "
                    f"{TARGET_PEAK}로 정규화, [-1,1] clip. 이어붙인 전체 파형에 "
                    "한 번 적용한 뒤 자른다."
                ),
                "fbank": {
                    "num_mel_bins": FBANK_BINS,
                    "sample_frequency": SAMPLE_RATE,
                    "frame_length_ms": FRAME_LENGTH_MS,
                    "frame_shift_ms": FRAME_SHIFT_MS,
                    "dither": 0.0,
                    "energy_floor": 0.0,
                    "window_type": "hamming",
                    "use_energy": False,
                    "snip_edges": True,
                },
                "cmvn": "time축 평균 감산 (feat - feat.mean(axis=0))",
                "payload_layout": "[1, frames, 80] little-endian float32",
            },
            "features": features,
        }
        args.feature_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.feature_manifest.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(f"화자 {len(rows)}명, payload {len(features)}개")
        print(f"  {args.dataset_manifest}")
        print(f"  {args.feature_manifest}")
        return 0
    except (FeatureBuildError, OSError, ValueError, KeyError) as exc:
        print(f"Feature 생성 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
