#!/usr/bin/env python3
"""Pack Final-98 plan, weights, and future pipeline contracts into one model."""

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
    build_final_98_package,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    bundle = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
    output = ROOT / "models/runtime/campp_sv_98/campp_sv_98.camppmodel"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=_path,
        default=bundle / "execution_plans/plan_98.bin",
    )
    parser.add_argument("--weights", type=_path, default=bundle / "weights.bin")
    parser.add_argument(
        "--source-manifest", type=_path, default=bundle / "manifest.json"
    )
    parser.add_argument("--output", type=_path, default=output)
    parser.add_argument(
        "--report", type=_path, default=output.with_suffix(".json")
    )
    parser.add_argument("--model-name", default="campp_sv_98")
    parser.add_argument("--optimization-suite", default=FINAL_98_SUITE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        if (args.output.exists() or args.report.exists()) and not args.force:
            raise ModelPackageError("output exists; use --force")
        package, report = build_final_98_package(
            plan_path=args.plan,
            weights_path=args.weights,
            source_manifest_path=args.source_manifest,
            model_name=args.model_name,
            optimization_suite=args.optimization_suite,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(package)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"model:  {args.output}")
        print(f"report: {args.report}")
        print(f"sha256: {report['package_sha256']}")
        return 0
    except (ModelPackageError, OSError, ValueError) as exc:
        print(f"model package build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
