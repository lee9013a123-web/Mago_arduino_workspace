#!/usr/bin/env python3
"""Select a fixed-bucket model from feature size and run the Final V3 C runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_model_package.format import (  # noqa: E402
    ModelPackageError,
    ModelPackageSectionType,
    verify_model_package,
)


FEATURE_BINS = 80
FLOAT32_BYTES = 4


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def infer_feature_frames(path: Path) -> int:
    size = path.stat().st_size
    bytes_per_frame = FEATURE_BINS * FLOAT32_BYTES
    if size == 0 or size % bytes_per_frame != 0:
        raise ModelPackageError(
            "feature must be raw [1,T,80] little-endian float32: "
            f"{size} bytes"
        )
    return size // bytes_per_frame


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPackageError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ModelPackageError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1", "ORT_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ModelPackageError(
            f"runtime failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ModelPackageError("runtime stdout is not JSON") from exc
    if not isinstance(value, dict):
        raise ModelPackageError("runtime JSON root is not an object")
    return value


def select_bucket_model(
    *, manifest_path: Path, feature_path: Path,
    requested_bucket: int | None = None,
) -> tuple[int, Path, float, dict[str, Any]]:
    manifest = _load_object(manifest_path)
    if manifest.get("format") != "campp-fixed-bucket-model-set-v1":
        raise ModelPackageError("unsupported bucket-model manifest format")
    frames = infer_feature_frames(feature_path)
    if requested_bucket is not None and requested_bucket != frames:
        raise ModelPackageError(
            f"requested bucket {requested_bucket} differs from input {frames}"
        )
    buckets = manifest.get("buckets")
    row = buckets.get(str(frames)) if isinstance(buckets, dict) else None
    if not isinstance(row, dict):
        supported = manifest.get("supported_buckets", [])
        raise ModelPackageError(
            f"no exact model for {frames} frames; supported: {supported}"
        )
    model_name = row.get("model")
    audio_seconds = row.get("audio_seconds")
    expected_sha256 = row.get("sha256")
    if (not isinstance(model_name, str) or
            not isinstance(audio_seconds, (int, float)) or
            not isinstance(expected_sha256, str)):
        raise ModelPackageError(f"invalid manifest row for bucket {frames}")
    model_path = manifest_path.parent / model_name
    if not model_path.is_file():
        raise ModelPackageError(f"selected model is missing: {model_path}")
    if _sha256(model_path) != expected_sha256:
        raise ModelPackageError(f"selected model checksum mismatch: {model_path}")
    package = verify_model_package(model_path)
    metadata = package.json_section(ModelPackageSectionType.MODEL_METADATA_JSON)
    if package.bucket_frames != frames:
        raise ModelPackageError("package header differs from selected bucket")
    if metadata.get("input", {}).get("shape") != [1, frames, FEATURE_BINS]:
        raise ModelPackageError("package input contract differs from feature")
    return frames, model_path, float(audio_seconds), manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    model_root = ROOT / "models/runtime/campp_sv_multibucket"
    parser.add_argument(
        "--manifest", type=_path,
        default=model_root / "campp_sv_multibucket.json",
    )
    parser.add_argument(
        "--runtime", type=_path,
        default=(
            ROOT / "build/profill/final_v3_hybrid"
            / "campp_runtime_benchmark_final"
        ),
    )
    parser.add_argument("--input", type=_path, required=True)
    parser.add_argument("--embedding-output", type=_path, required=True)
    parser.add_argument("--bucket-frames", type=int)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        if not args.input.is_file():
            raise ModelPackageError(f"input is missing: {args.input}")
        if not args.runtime.is_file():
            raise ModelPackageError(f"runtime is missing: {args.runtime}")
        if args.warmup < 0 or args.repeat <= 0 or args.threads <= 0:
            raise ModelPackageError("warmup/repeat/threads values are invalid")
        bucket, model, audio_seconds, manifest = select_bucket_model(
            manifest_path=args.manifest,
            feature_path=args.input,
            requested_bucket=args.bucket_frames,
        )
        capabilities = _run_json([str(args.runtime), "--capabilities"])
        requirements = manifest.get("runtime_requirements", {})
        if capabilities.get("model_package_format") != "camppmodel-v1":
            raise ModelPackageError("runtime cannot load camppmodel-v1")
        if capabilities.get("optimization_suite_config") != requirements.get(
            "optimization_suite"
        ):
            raise ModelPackageError("runtime optimization suite differs from model set")
        selection = {
            "bucket_frames": bucket,
            "audio_seconds": audio_seconds,
            "model": str(model),
            "input": str(args.input),
            "embedding_output": str(args.embedding_output),
        }
        if args.preflight_only:
            print(json.dumps({
                "ready": True,
                "selection": selection,
                "runtime_capabilities": capabilities,
            }, ensure_ascii=False, indent=2))
            return 0
        args.embedding_output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.runtime),
            "--model", str(model),
            "--input", str(args.input),
            "--audio-seconds", str(audio_seconds),
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
            "--threads", str(args.threads),
            "--embedding-output", str(args.embedding_output),
        ]
        result = _run_json(command)
        if not args.embedding_output.is_file() or (
            args.embedding_output.stat().st_size != 192 * FLOAT32_BYTES
        ):
            raise ModelPackageError("runtime did not write a 192-float embedding")
        print(json.dumps({
            "ready": True,
            "selection": selection,
            "runtime_result": result,
        }, ensure_ascii=False, indent=2))
        return 0
    except (ModelPackageError, OSError, ValueError) as exc:
        print(f"bucket model execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
