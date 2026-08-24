#!/usr/bin/env python3
"""Pack one fixed-bucket Final V3 model into a camppmodel-v1 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_model_package.format import ModelPackageError  # noqa: E402
from runtime_model_package.packager import (  # noqa: E402
    FINAL_98_SUITE,
    SUPPORTED_BUCKET_FRAMES,
    build_final_bucket_package,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket-frames", type=int, choices=SUPPORTED_BUCKET_FRAMES,
        default=98,
    )
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument("--source-manifest", type=_path)
    parser.add_argument("--output", type=_path)
    parser.add_argument("--report", type=_path)
    parser.add_argument("--model-name")
    parser.add_argument("--optimization-suite", default=FINAL_98_SUITE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        bucket = args.bucket_frames
        source_root = (
            ROOT / "runs/models/campplus/final_v3/weight_residency"
            / str(bucket)
        )
        output_root = ROOT / "models/runtime" / f"campp_sv_{bucket}"
        output = args.output or output_root / f"campp_sv_{bucket}.camppmodel"
        report_path = args.report or output.with_suffix(".json")
        plan = args.plan or source_root / f"plan_{bucket}.bin"
        weights = args.weights or source_root / f"weights_{bucket}.bin"
        source_manifest = (
            args.source_manifest
            or source_root / f"weight_plan_{bucket}.json"
        )
        model_name = args.model_name or f"campp_sv_{bucket}"
        if (output.exists() or report_path.exists()) and not args.force:
            raise ModelPackageError("output exists; use --force")
        package, report = build_final_bucket_package(
            bucket_frames=bucket,
            plan_path=plan,
            weights_path=weights,
            source_manifest_path=source_manifest,
            model_name=model_name,
            optimization_suite=args.optimization_suite,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(package)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"model:  {output}")
        print(f"report: {report_path}")
        print(f"sha256: {report['package_sha256']}")
        return 0
    except (ModelPackageError, OSError, ValueError) as exc:
        print(f"model package build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
