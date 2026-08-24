"""Exact waveform-to-FBank contract used by the exported CAM++ models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import AudioCaptureError, fixed_length_pcm, read_pcm16_mono


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


def _extract_fbank_torchaudio(waveform: np.ndarray) -> np.ndarray | None:
    try:
        import torch
        import torchaudio
    except (ImportError, OSError):
        return None
    try:
        tensor = torch.from_numpy(
            np.asarray(waveform, dtype=np.float32)
        ).unsqueeze(0)
        feature = torchaudio.compliance.kaldi.fbank(
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
    except Exception:
        return None
    feature -= feature.mean(axis=0, keepdims=True)
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
    """Dependency-free fallback adapted from the validated CAM pipeline.

    The previous dynamic-model fallback used ceil/padding. Fixed buckets use
    Kaldi snip-edges semantics, so this variant deliberately uses floor and
    yields exactly 98/298/498/998 frames for 1/3/5/10 seconds.
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


def extract_fbank(waveform: np.ndarray) -> np.ndarray:
    feature = _extract_fbank_torchaudio(waveform)
    if feature is not None:
        return feature
    return _extract_fbank_numpy(waveform)


def wav_to_fixed_fbank(
    *, wav_path: Path, bucket_frames: int, audio_seconds: int,
) -> np.ndarray:
    try:
        samples = read_pcm16_mono(wav_path)
    except AudioCaptureError as exc:
        raise FrontendError(str(exc)) from exc
    samples = fixed_length_pcm(samples, audio_seconds)
    feature = extract_fbank(normalize_waveform(samples))
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
