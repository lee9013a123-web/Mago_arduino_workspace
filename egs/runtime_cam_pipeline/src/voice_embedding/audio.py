"""ALSA microphone capture and WAV validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable
import wave

import numpy as np


SAMPLE_RATE = 16000


class AudioCaptureError(RuntimeError):
    """Raised when microphone configuration or capture is invalid."""


@dataclass(frozen=True)
class MicrophoneProfile:
    version: str
    backend: str
    device: str
    sample_rate_hz: int = SAMPLE_RATE
    channels: int = 1
    sample_format: str = "S16_LE"


def load_microphone_profile(
    config_path: Path, mic_version: str,
) -> MicrophoneProfile:
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioCaptureError(
            f"cannot read microphone configuration: {config_path}"
        ) from exc
    rows = document.get("microphones") if isinstance(document, dict) else None
    row = rows.get(mic_version) if isinstance(rows, dict) else None
    if not isinstance(row, dict):
        available = sorted(rows) if isinstance(rows, dict) else []
        raise AudioCaptureError(
            f"unknown microphone version {mic_version!r}; available: {available}"
        )
    profile = MicrophoneProfile(
        version=mic_version,
        backend=str(row.get("backend", "")),
        device=str(row.get("device", "")),
        sample_rate_hz=int(row.get("sample_rate_hz", SAMPLE_RATE)),
        channels=int(row.get("channels", 1)),
        sample_format=str(row.get("sample_format", "S16_LE")),
    )
    if profile.backend != "alsa_arecord":
        raise AudioCaptureError(
            f"unsupported microphone backend: {profile.backend!r}"
        )
    if not profile.device:
        raise AudioCaptureError(f"microphone {mic_version!r} has no ALSA device")
    if profile.sample_rate_hz != SAMPLE_RATE or profile.channels != 1:
        raise AudioCaptureError("CAM++ frontend requires 16 kHz mono input")
    if profile.sample_format != "S16_LE":
        raise AudioCaptureError("CAM++ capture requires S16_LE PCM")
    return profile


def list_alsa_devices() -> str:
    executable = shutil.which("arecord")
    if executable is None:
        raise AudioCaptureError("arecord is not installed or not on PATH")
    completed = subprocess.run(
        [executable, "-L"], text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise AudioCaptureError(
            f"arecord -L failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def countdown_before_recording(
    seconds: int, output: Callable[[str], None] = print,
) -> None:
    for remaining in range(seconds, 0, -1):
        output(f"  recording starts in {remaining}...")
        time.sleep(1)


def _report_recording_progress(elapsed: int, seconds: int) -> None:
    message = f"recording! {elapsed}/{seconds}s"

    if sys.stdout.isatty():
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()
    else:
        print(message, flush=True)


def record_wav(
    *, profile: MicrophoneProfile, output_path: Path, seconds: int,
) -> None:
    if seconds <= 0:
        raise AudioCaptureError("recording duration must be positive")
    executable = shutil.which("arecord")
    if executable is None:
        raise AudioCaptureError("arecord is not installed or not on PATH")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    command = [
        executable,
        "-q",
        "-D", profile.device,
        "-t", "wav",
        "-f", profile.sample_format,
        "-r", str(profile.sample_rate_hz),
        "-c", str(profile.channels),
        "-d", str(seconds),
        str(temporary),
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    start = time.monotonic()
    for elapsed in range(1, seconds + 1):
        sleep_for = (start + elapsed) - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        _report_recording_progress(elapsed, seconds)
    _, stderr = process.communicate()
    sys.stdout.write("\n")
    sys.stdout.flush()
    if process.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise AudioCaptureError(
            f"microphone capture failed ({process.returncode}): {stderr.strip()}"
        )
    read_pcm16_mono(temporary)
    temporary.replace(output_path)


def read_pcm16_mono(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            raw = source.readframes(source.getnframes())
    except (OSError, wave.Error) as exc:
        raise AudioCaptureError(f"cannot read WAV: {path}") from exc
    if channels != 1 or width != 2 or rate != SAMPLE_RATE:
        raise AudioCaptureError(
            f"WAV must be mono S16_LE {SAMPLE_RATE} Hz: {path}"
        )
    return np.frombuffer(raw, dtype="<i2").copy()


def fixed_length_pcm(samples: np.ndarray, seconds: int) -> np.ndarray:
    """Crop or zero-pad capture to the exact fixed-bucket sample count."""

    if seconds <= 0:
        raise AudioCaptureError("fixed audio duration must be positive")
    expected = SAMPLE_RATE * seconds
    flat = np.asarray(samples, dtype=np.int16).reshape(-1)
    if flat.size >= expected:
        return flat[:expected].copy()
    output = np.zeros(expected, dtype=np.int16)
    output[:flat.size] = flat
    return output
