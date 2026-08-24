"""Torch-free native waveform-to-FBank contract for CAM++ deployment."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json
import os
import subprocess
import tempfile

import numpy as np


FBANK_BINS = 80
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0
SAMPLE_RATE = 16000
TARGET_PEAK = 0.95


class FrontendError(RuntimeError):
    """Raised when frontend dependencies or tensor contracts are invalid."""


def normalize_waveform(samples: np.ndarray) -> np.ndarray:
    waveform = (samples.astype(np.float32) / np.float32(32768.0)).astype(
        np.float32
    )
    if waveform.size:
        waveform -= np.float32(waveform.mean())
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1e-8:
        waveform = waveform / np.float32(peak) * np.float32(TARGET_PEAK)
    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


@lru_cache(maxsize=4)
def validate_native_fbank(binary: Path) -> dict[str, object]:
    """Verify that the pipeline-local frontend has the expected fixed ABI."""

    if not binary.is_file():
        raise FrontendError(
            f"native FBank binary is missing: {binary}; run "
            "script/build_native_fbank.sh and script/prepare_runtime.py"
        )
    # cwd를 바꿔 실행하므로 argv[0]도 절대 경로여야 한다.
    binary = binary.resolve()
    completed = subprocess.run(
        [str(binary), "--version"],
        cwd=binary.parent,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FrontendError(
            f"native FBank capability check failed: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FrontendError("native FBank --version output is not JSON") from exc
    expected = {
        "frontend": "campp-kaldi-native-fbank",
        "api_version": 1,
        "sample_rate": SAMPLE_RATE,
        "mel_bins": FBANK_BINS,
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise FrontendError("native FBank binary has an incompatible ABI")
    return payload


def _extract_fbank_native(
    *, wav_path: Path, native_binary: Path, bucket_frames: int,
    audio_seconds: int,
) -> np.ndarray:
    validate_native_fbank(native_binary)
    if not wav_path.is_file():
        raise FrontendError(f"input WAV is missing: {wav_path}")
    # 아래 subprocess는 cwd를 바이너리 디렉터리로 바꾼다. 상대 경로를 그대로
    # 넘기면 자식 프로세스가 엉뚱한 기준으로 해석해 "cannot open input WAV"로
    # 실패한다 -- 부모에서 한 is_file() 검사는 통과한 뒤라 원인이 드러나지 않는다.
    wav_path = wav_path.resolve()
    native_binary = native_binary.resolve()
    with tempfile.TemporaryDirectory(
        prefix="campp_fbank_", dir=wav_path.parent,
    ) as temporary:
        feature_path = Path(temporary) / "feature.f32"
        completed = subprocess.run(
            [
                str(native_binary),
                "--input", str(wav_path),
                "--output", str(feature_path),
                "--expected-frames", str(bucket_frames),
                "--audio-seconds", str(audio_seconds),
            ],
            cwd=native_binary.parent,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise FrontendError(
                f"native FBank failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        try:
            metadata = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise FrontendError("native FBank output is not JSON") from exc
        if not isinstance(metadata, dict) or (
            metadata.get("backend") != "kaldi-native-fbank"
            or metadata.get("frames") != bucket_frames
            or metadata.get("bins") != FBANK_BINS
            or metadata.get("torch_required") is not False
        ):
            raise FrontendError("native FBank returned incompatible metadata")
        expected_bytes = bucket_frames * FBANK_BINS * 4
        if not feature_path.is_file() or feature_path.stat().st_size != expected_bytes:
            raise FrontendError("native FBank returned an invalid feature file")
        feature = np.fromfile(feature_path, dtype="<f4").reshape(
            bucket_frames, FBANK_BINS,
        )
    if not np.isfinite(feature).all():
        raise FrontendError("native FBank produced non-finite values")
    return feature.astype(np.float32)


def _hz_to_mel(hertz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hertz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int = 512) -> np.ndarray:
    minimum_mel = _hz_to_mel(np.array([20.0], dtype=np.float64))[0]
    maximum_mel = _hz_to_mel(
        np.array([SAMPLE_RATE / 2.0], dtype=np.float64)
    )[0]
    mel_points = np.linspace(minimum_mel, maximum_mel, FBANK_BINS + 2)
    hertz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hertz_points / SAMPLE_RATE).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((FBANK_BINS, n_fft // 2 + 1), dtype=np.float32)
    for mel_index in range(1, FBANK_BINS + 1):
        left = int(bins[mel_index - 1])
        center = min(max(int(bins[mel_index]), left + 1), n_fft // 2)
        right = min(max(int(bins[mel_index + 1]), center + 1), n_fft // 2)
        if center > left:
            filters[mel_index - 1, left:center] = (
                np.arange(left, center) - left
            ) / float(center - left)
        if right > center:
            filters[mel_index - 1, center:right] = (
                right - np.arange(center, right)
            ) / float(right - center)
    return filters


def _extract_fbank_numpy(waveform: np.ndarray) -> np.ndarray:
    """Approximate offline reference retained only for unit/shape tests.

    Production registration and verification never call this approximation.
    The native Kaldi-compatible executable is mandatory instead of silently
    switching feature definitions.
    """

    value = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if value.size >= 2:
        value = np.concatenate((
            value[:1], value[1:] - np.float32(0.97) * value[:-1],
        )).astype(np.float32)
    frame_length = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000.0)
    frame_shift = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000.0)
    if value.size < frame_length:
        value = np.pad(value, (0, frame_length - value.size))
    frame_count = 1 + (value.size - frame_length) // frame_shift
    frames = np.empty((frame_count, frame_length), dtype=np.float32)
    for index in range(frame_count):
        start = index * frame_shift
        frames[index] = value[start:start + frame_length]
    frames *= np.hamming(frame_length).astype(np.float32)
    spectrum = np.fft.rfft(frames, n=512)
    power = (np.abs(spectrum) ** 2) / 512.0
    feature = np.dot(power, _mel_filterbank().T)
    feature = np.log(np.maximum(feature, 1e-10)).astype(np.float32)
    feature -= feature.mean(axis=0, keepdims=True)
    return feature.astype(np.float32)


def wav_to_fixed_fbank(
    *, wav_path: Path, native_binary: Path, bucket_frames: int,
    audio_seconds: int,
) -> np.ndarray:
    """Extract fixed FBank with the required native Kaldi implementation."""

    feature = _extract_fbank_native(
        wav_path=wav_path,
        native_binary=native_binary,
        bucket_frames=bucket_frames,
        audio_seconds=audio_seconds,
    )
    if feature.shape != (bucket_frames, FBANK_BINS):
        raise FrontendError(
            f"frontend produced {feature.shape}; expected "
            f"({bucket_frames}, {FBANK_BINS})"
        )
    if not np.isfinite(feature).all():
        raise FrontendError("frontend produced non-finite values")
    return feature


def write_feature(path: Path, feature: np.ndarray) -> None:
    value = np.asarray(feature, dtype="<f4")
    if value.ndim != 2 or value.shape[1] != FBANK_BINS:
        raise FrontendError(f"invalid FBank shape: {value.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.tobytes(order="C"))
