#!/usr/bin/env python3
"""Run the existing QConv hotspot analyzer with v5 selected."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
HOTSPOT = ROOT / "scripts" / "4_profill" / "optimization" / "04_profile_qconv_hotspot.py"


def main() -> int:
    command = [
        sys.executable,
        str(HOTSPOT),
        "--qconv-candidate",
        "v5",
        "--case-scope",
        "representative10",
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
