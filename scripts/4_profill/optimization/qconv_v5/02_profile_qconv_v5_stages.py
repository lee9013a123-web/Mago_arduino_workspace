#!/usr/bin/env python3
"""Profile one ordinary QConv candidate mode with the fixed E7 cases."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "scripts" / "4_profill" / "optimization" / "02_diagnose_top4.py"
MODES = ("baseline", "mac_fixed", "v4", "hybrid", "v5")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="v5")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    command = [
        sys.executable,
        str(RUNNER),
        "--qconv-only",
        "--qconv-candidate",
        args.mode,
        "--runs-dir",
        str(ROOT / "runs" / "profiling" / "e7_98" / "optimization"
            / "qconv_v5" / args.mode),
        "--output",
        str(ROOT / "results" / "profiling" / "e7_98" / "optimization"
            / "qconv_v5" / f"{args.mode}.json"),
    ]
    if args.preflight_only:
        command.append("--preflight-only")
    if args.force:
        command.append("--force")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
