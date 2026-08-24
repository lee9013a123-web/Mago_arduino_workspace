#!/usr/bin/env python3
"""Verify one fixed-bucket .camppmodel and its deployment contracts."""

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

from runtime_model_package.format import (  # noqa: E402
    ModelPackageError,
    ModelPackageSectionType,
    verify_model_package,
)
from runtime_model_package.packager import (  # noqa: E402
    SUPPORTED_BUCKET_FRAMES,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=_path,
        default=ROOT / "models/runtime/campp_sv_98/campp_sv_98.camppmodel",
    )
    parser.add_argument("--expect-plan", type=_path)
    parser.add_argument("--expect-weights", type=_path)
    parser.add_argument(
        "--bucket-frames", type=int, choices=SUPPORTED_BUCKET_FRAMES,
    )
    args = parser.parse_args()
    try:
        package = verify_model_package(args.model)
        bucket = package.bucket_frames
        if bucket not in SUPPORTED_BUCKET_FRAMES:
            raise ModelPackageError(f"unsupported model bucket: {bucket}")
        if args.bucket_frames is not None and bucket != args.bucket_frames:
            raise ModelPackageError(
                f"model bucket mismatch: {bucket} != {args.bucket_frames}"
            )
        metadata = package.json_section(
            ModelPackageSectionType.MODEL_METADATA_JSON
        )
        frontend = package.json_section(
            ModelPackageSectionType.FRONTEND_CONTRACT_JSON
        )
        postprocess = package.json_section(
            ModelPackageSectionType.POSTPROCESS_CONTRACT_JSON
        )
        if metadata.get("bucket_frames") != bucket:
            raise ModelPackageError("metadata bucket differs from package header")
        if metadata.get("input", {}).get("shape") != [1, bucket, 80]:
            raise ModelPackageError("invalid model input contract")
        if metadata.get("output", {}).get("shape") != [1, 192]:
            raise ModelPackageError("invalid model output contract")
        if frontend.get("implemented_in_model") is not False:
            raise ModelPackageError("frontend implementation boundary is unclear")
        if frontend.get("model_input_frames") != bucket:
            raise ModelPackageError("frontend bucket differs from package header")
        if postprocess.get("decision_threshold") is not None:
            raise ModelPackageError("uncalibrated threshold must not be frozen")
        for path, section_type, name in (
            (args.expect_plan, ModelPackageSectionType.EXECUTION_PLAN, "plan"),
            (args.expect_weights, ModelPackageSectionType.PACKED_WEIGHTS, "weights"),
        ):
            if path is not None and path.read_bytes() != package.sections[section_type]:
                raise ModelPackageError(f"{name} bytes differ from source")
        result = {
            "valid": True,
            "model": str(args.model),
            "size_bytes": package.file_size,
            "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "bucket_frames": package.bucket_frames,
            "model_name": metadata.get("model_name"),
            "input": metadata.get("input"),
            "output": metadata.get("output"),
            "frontend_contract_present": True,
            "postprocess_contract_present": True,
            "threshold_calibrated": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ModelPackageError, OSError, ValueError) as exc:
        print(f"model package verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
