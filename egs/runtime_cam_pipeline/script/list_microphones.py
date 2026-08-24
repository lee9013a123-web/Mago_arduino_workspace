#!/usr/bin/env python3
"""List ALSA capture device identifiers usable in microphones.json."""

from __future__ import annotations

from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voice_embedding.audio import AudioCaptureError, list_alsa_devices  # noqa: E402


def main() -> int:
    try:
        print(list_alsa_devices(), end="")
        return 0
    except AudioCaptureError as exc:
        print(f"microphone listing failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
