#!/usr/bin/env python3
"""Run split-vs-package retained-tensor validation for every model bucket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SINGLE_VALIDATOR = ROOT / "scripts/5_model/03_validate_model_runtime.py"
SUPPORTED_BUCKETS = (98, 298, 498, 998)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buckets", type=int, nargs="+", choices=SUPPORTED_BUCKETS,
        default=list(SUPPORTED_BUCKETS),
    )
    parser.add_argument(
        "--binary", type=_path,
        default=(
            ROOT / "build/profill/final_v3_hybrid"
            / "campp_runtime_benchmark_final"
        ),
    )
    parser.add_argument(
        "--dump-binary", type=_path,
        default=(
            ROOT / "build/profill/final_v3_hybrid"
            / "campp_reference_dump_final"
        ),
    )
    parser.add_argument(
        "--output", type=_path,
        default=(
            ROOT / "results/model_packages"
            / "campp_sv_multibucket_validation.json"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows: list[dict] = []
    try:
        buckets = tuple(dict.fromkeys(args.buckets))
        if len(buckets) != len(args.buckets):
            raise ValueError("each bucket must be unique")
        for bucket in buckets:
            report_path = (
                ROOT / "results/model_packages"
                / f"campp_sv_{bucket}_validation.json"
            )
            command = [
                sys.executable,
                str(SINGLE_VALIDATOR),
                "--bucket-frames", str(bucket),
                "--binary", str(args.binary),
                "--dump-binary", str(args.dump_binary),
            ]
            if args.preflight_only:
                command.append("--preflight-only")
            if args.force:
                command.append("--force")
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"bucket {bucket} validation failed: "
                    f"{completed.stderr.strip()}"
                )
            if args.preflight_only:
                row = json.loads(completed.stdout)
            else:
                row = json.loads(report_path.read_text(encoding="utf-8"))
            rows.append(row)
        report = {
            "schema_version": 1,
            "validation": "split_vs_camppmodel_v1_all_buckets",
            "preflight_only": args.preflight_only,
            "buckets": list(buckets),
            "ready": all(row.get("ready", row.get("passed")) is True
                         for row in rows),
            "results": rows,
        }
        if not args.preflight_only:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 3
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"bucket model validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
