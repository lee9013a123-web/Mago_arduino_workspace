#!/usr/bin/env python3
"""Benchmark baseline, mac_fixed, v4, hybrid and v5 with one protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
PROFILE = Path(__file__).with_name("02_profile_qconv_v5_stages.py")
COMPARE = Path(__file__).with_name("04_compare_qconv_v5.py")
MODES = ("baseline", "mac_fixed", "v4", "hybrid", "v5")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    for mode in MODES:
        command = [sys.executable, str(PROFILE), "--mode", mode]
        if args.preflight_only:
            command.append("--preflight-only")
        if args.force:
            command.append("--force")
        run(command)
    if args.preflight_only:
        return 0

    for mode in ("mac_fixed", "v4", "hybrid", "v5"):
        command = [sys.executable, str(COMPARE), "--mode", mode]
        if args.force:
            command.append("--force")
        run(command)
    print(
        "QConv v5 benchmark complete: "
        "results/profiling/e7_98/optimization/qconv_v5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
