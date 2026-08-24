#!/usr/bin/env python3
"""Compare one QConv v5 experiment against its fixed baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
COMPARATOR = (
    ROOT / "scripts" / "4_profill" / "optimization" / "03_compare_candidate.py"
)
RESULT_DIR = ROOT / "results" / "profiling" / "e7_98" / "optimization" / "qconv_v5"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("mac_fixed", "v4", "hybrid", "v5"),
        default="v5"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    command = [
        sys.executable,
        str(COMPARATOR),
        "--baseline",
        str(RESULT_DIR / "baseline.json"),
        "--candidate",
        str(RESULT_DIR / f"{args.mode}.json"),
        "--output",
        str(RESULT_DIR / f"{args.mode}_comparison.json"),
    ]
    if args.force:
        command.append("--force")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
