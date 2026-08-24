#!/usr/bin/env python3
"""Build four camppmodel-v1 files and a small bucket-selection manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_model_package.format import ModelPackageError  # noqa: E402
from runtime_model_package.packager import (  # noqa: E402
    AUDIO_SECONDS_BY_BUCKET,
    FINAL_98_SUITE,
    SUPPORTED_BUCKET_FRAMES,
    build_final_bucket_package,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buckets", type=int, nargs="+",
        choices=SUPPORTED_BUCKET_FRAMES,
        default=list(SUPPORTED_BUCKET_FRAMES),
    )
    parser.add_argument(
        "--source-root", type=_path,
        default=(
            ROOT / "runs/models/campplus/final_v3/weight_residency"
        ),
    )
    parser.add_argument(
        "--output-root", type=_path,
        default=ROOT / "models/runtime/campp_sv_multibucket",
    )
    parser.add_argument("--optimization-suite", default=FINAL_98_SUITE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        buckets = tuple(dict.fromkeys(args.buckets))
        if len(buckets) != len(args.buckets):
            raise ModelPackageError("each bucket must be unique")
        manifest_path = args.output_root / "campp_sv_multibucket.json"
        legacy_package = (
            args.output_root / "campp_sv_multibucket.camppmodel"
        )
        targets: list[tuple[int, Path, Path, Path, Path, Path]] = []
        for bucket in buckets:
            source = args.source_root / str(bucket)
            model_path = args.output_root / f"campp_sv_{bucket}.camppmodel"
            report_path = args.output_root / f"campp_sv_{bucket}.json"
            targets.append((
                bucket,
                source / f"plan_{bucket}.bin",
                source / f"weights_{bucket}.bin",
                source / f"weight_plan_{bucket}.json",
                model_path,
                report_path,
            ))
        missing = [
            path for _, plan, weights, source_manifest, _, _ in targets
            for path in (plan, weights, source_manifest)
            if not path.is_file()
        ]
        if missing:
            raise ModelPackageError(
                "missing bucket source files:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        outputs = [manifest_path]
        outputs.extend(
            path for target in targets for path in target[-2:]
        )
        existing = [path for path in outputs if path.exists()]
        if existing and not args.force:
            raise ModelPackageError(
                "outputs exist; use --force:\n  "
                + "\n  ".join(str(path) for path in existing)
            )

        built: list[tuple[Path, bytes, Path, dict]] = []
        bucket_manifest: dict[str, dict[str, object]] = {}
        for (bucket, plan, weights, source_manifest,
             model_path, report_path) in targets:
            package, report = build_final_bucket_package(
                bucket_frames=bucket,
                plan_path=plan,
                weights_path=weights,
                source_manifest_path=source_manifest,
                model_name=f"campp_sv_{bucket}",
                optimization_suite=args.optimization_suite,
            )
            built.append((model_path, package, report_path, report))
            bucket_manifest[str(bucket)] = {
                "audio_seconds": AUDIO_SECONDS_BY_BUCKET[bucket],
                "input_shape": [1, bucket, 80],
                "output_shape": [1, 192],
                "model": model_path.name,
                "report": report_path.name,
                "size_bytes": len(package),
                "sha256": _sha256(package),
            }

        manifest = {
            "schema_version": 1,
            "format": "campp-fixed-bucket-model-set-v1",
            "package_format": "camppmodel-v1",
            "selection_key": "feature_frames",
            "supported_buckets": list(buckets),
            "runtime_external": True,
            "runtime_requirements": {
                "architecture": "aarch64",
                "required_isa": ["neon"],
                "optimization_suite": args.optimization_suite,
                "model_package_loader": "camppmodel-v1",
            },
            "buckets": bucket_manifest,
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        for model_path, package, report_path, report in built:
            model_path.write_bytes(package)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        if legacy_package.exists():
            print(
                "warning: legacy campp_sv_multibucket.camppmodel is not used "
                "by this manifest; keep it only until the four packages pass "
                "board validation",
                file=sys.stderr,
            )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except (ModelPackageError, OSError, ValueError) as exc:
        print(f"bucket model build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
