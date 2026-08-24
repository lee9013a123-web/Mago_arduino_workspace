#!/usr/bin/env python3
"""Build a self-contained runtime/ directory for microphone inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parents[1]
OUTPUT_ROOT = PIPELINE_ROOT / "runtime"
ORT_OUTPUT_ROOT = PIPELINE_ROOT / "runtime_onnx"
BUCKETS = (98, 298, 498, 998)
AUDIO_SECONDS = {98: 1, 298: 3, 498: 5, 998: 10}


class PreparationError(RuntimeError):
    """Raised when deployable runtime assets cannot be assembled."""


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path, *, force: bool) -> str:
    if not source.is_file():
        raise PreparationError(f"source artifact is missing: {source}")
    source_hash = _sha256(source)
    if destination.is_file():
        if _sha256(destination) == source_hash:
            return source_hash
        if not force:
            raise PreparationError(
                f"destination differs: {destination}; use --force to replace it"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return source_hash


def _relative(path: Path) -> str:
    return path.relative_to(OUTPUT_ROOT).as_posix()


def _prepare_packages(source_manifest: Path, *, force: bool) -> dict[str, dict]:
    source = _load_json(source_manifest)
    if source.get("format") != "campp-fixed-bucket-model-set-v1":
        raise PreparationError("source package manifest has an unsupported format")
    rows = source.get("buckets")
    if not isinstance(rows, dict):
        raise PreparationError("source package manifest has no buckets")
    output: dict[str, dict] = {}
    for bucket in BUCKETS:
        row = rows.get(str(bucket))
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            raise PreparationError(f"package manifest has no bucket {bucket}")
        source_model = source_manifest.parent / row["model"]
        destination = OUTPUT_ROOT / "models" / f"campp_sv_{bucket}.camppmodel"
        checksum = _copy(source_model, destination, force=force)
        expected = row.get("sha256")
        if isinstance(expected, str) and checksum != expected:
            raise PreparationError(f"source package checksum mismatch: {source_model}")
        output[str(bucket)] = {
            "audio_seconds": AUDIO_SECONDS[bucket],
            "model": _relative(destination),
            "sha256": checksum,
        }
    return output


def _prepare_windowed(source_manifest: Path, *, force: bool) -> dict[str, dict]:
    source = _load_json(source_manifest)
    if source.get("format") != "campp-weight-streaming-sidecar-v1":
        raise PreparationError("source streaming manifest has an unsupported format")
    rows = source.get("buckets")
    if not isinstance(rows, dict):
        raise PreparationError("source streaming manifest has no buckets")
    output: dict[str, dict] = {}
    for bucket in BUCKETS:
        row = rows.get(str(bucket))
        if not isinstance(row, dict):
            raise PreparationError(f"streaming manifest has no bucket {bucket}")
        destination_root = OUTPUT_ROOT / "models" / str(bucket)
        output_row: dict[str, object] = {"audio_seconds": AUDIO_SECONDS[bucket]}
        for key in ("plan", "weights", "schedule"):
            value = row.get(key)
            if not isinstance(value, str):
                raise PreparationError(f"bucket {bucket} has no {key}")
            source_path = Path(value)
            if not source_path.is_absolute():
                source_path = REPO_ROOT / source_path
            destination = destination_root / source_path.name
            checksum = _copy(source_path, destination, force=force)
            output_row[key] = _relative(destination)
            output_row[f"{key}_sha256"] = checksum
        output[str(bucket)] = output_row
    return output


def _prepare_ort(
    *, model: Path, fbank: Path, fbank_license: Path, force: bool,
) -> int:
    model_output = ORT_OUTPUT_ROOT / "models/campplus_int8_static_qop.onnx"
    model_sha256 = _copy(model, model_output, force=force)
    frontend_output = ORT_OUTPUT_ROOT / "campp_fbank"
    frontend_sha256 = _copy(fbank, frontend_output, force=force)
    frontend_output.chmod(frontend_output.stat().st_mode | stat.S_IXUSR)
    license_output = ORT_OUTPUT_ROOT / "licenses/kaldi-native-fbank-LICENSE"
    _copy(fbank_license, license_output, force=force)
    manifest = {
        "schema_version": 1,
        "format": "campp-onnx-pipeline-assets-v1",
        "backend": "onnxruntime-cpu",
        "provider": "CPUExecutionProvider",
        "supported_buckets": list(BUCKETS),
        "model": {
            "path": model_output.relative_to(ORT_OUTPUT_ROOT).as_posix(),
            "sha256": model_sha256,
        },
        "frontend": {
            "binary": frontend_output.relative_to(ORT_OUTPUT_ROOT).as_posix(),
            "sha256": frontend_sha256,
            "backend": "kaldi-native-fbank",
            "version": "1.22.3",
            "license": license_output.relative_to(ORT_OUTPUT_ROOT).as_posix(),
        },
    }
    ORT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ORT_OUTPUT_ROOT / "assets.json"
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(json.dumps({
        "ready": True,
        "backend": "onnxruntime-cpu",
        "pipeline_root": str(PIPELINE_ROOT),
        "runtime_root": str(ORT_OUTPUT_ROOT),
        "model": str(model_output),
        "frontend": str(frontend_output),
        "asset_manifest": str(manifest_path),
        "buckets": list(BUCKETS),
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--c", action="store_true", help="prepare C Runtime assets")
    backend.add_argument("--ort", action="store_true", help="prepare ONNX Runtime assets")
    parser.add_argument("--mode", choices=("package", "windowed"), default="package")
    parser.add_argument("--runtime", type=_repo_path)
    parser.add_argument(
        "--fbank",
        type=_repo_path,
        default=PIPELINE_ROOT / "build/native_fbank/campp_fbank",
        help="native Kaldi-compatible frontend built by build_native_fbank.sh",
    )
    parser.add_argument(
        "--fbank-license",
        type=_repo_path,
        default=(
            PIPELINE_ROOT / ".deps/kaldi-native-fbank-v1.22.3/LICENSE"
        ),
    )
    parser.add_argument(
        "--speaker-verify",
        type=_repo_path,
        default=(
            PIPELINE_ROOT / "build/native_pipeline/campp_speaker_verify"
        ),
        help="Python-free verification binary built by build_speaker_verify.sh",
    )
    parser.add_argument(
        "--onnx-model",
        type=_repo_path,
        default=REPO_ROOT / "models/source/campplus_int8_static_qop.onnx",
    )
    parser.add_argument(
        "--package-manifest",
        type=_repo_path,
        default=(
            REPO_ROOT / "models/runtime/campp_sv_multibucket"
            / "campp_sv_multibucket.json"
        ),
    )
    parser.add_argument(
        "--streaming-manifest",
        type=_repo_path,
        default=(
            REPO_ROOT / "runs/models/campplus/final_v3/weight_streaming"
            / "weight_streaming_manifest.json"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        if args.ort:
            if args.mode != "package":
                raise PreparationError("ORT backend does not support --mode windowed")
            return _prepare_ort(
                model=args.onnx_model,
                fbank=args.fbank,
                fbank_license=args.fbank_license,
                force=args.force,
            )
        source_runtime = args.runtime
        if source_runtime is None:
            build = "final_v3_hybrid" if args.mode == "package" else "weight_streaming_98"
            source_runtime = (
                REPO_ROOT / "build/profill" / build
                / "campp_runtime_benchmark_final"
            )
        runtime_output = OUTPUT_ROOT / "campp_runtime"
        runtime_sha256 = _copy(source_runtime, runtime_output, force=args.force)
        runtime_output.chmod(runtime_output.stat().st_mode | stat.S_IXUSR)
        frontend_output = OUTPUT_ROOT / "campp_fbank"
        frontend_sha256 = _copy(args.fbank, frontend_output, force=args.force)
        frontend_output.chmod(frontend_output.stat().st_mode | stat.S_IXUSR)
        frontend_license = OUTPUT_ROOT / "licenses/kaldi-native-fbank-LICENSE"
        _copy(args.fbank_license, frontend_license, force=args.force)
        verifier_output = OUTPUT_ROOT / "campp_speaker_verify"
        verifier_sha256 = _copy(
            args.speaker_verify, verifier_output, force=args.force,
        )
        verifier_output.chmod(verifier_output.stat().st_mode | stat.S_IXUSR)
        buckets = (
            _prepare_packages(args.package_manifest, force=args.force)
            if args.mode == "package"
            else _prepare_windowed(args.streaming_manifest, force=args.force)
        )
        manifest = {
            "schema_version": 1,
            "format": "campp-runtime-pipeline-assets-v1",
            "mode": args.mode,
            "selection_key": "bucket_frames",
            "runtime": {
                "binary": "campp_runtime",
                "sha256": runtime_sha256,
            },
            "frontend": {
                "binary": "campp_fbank",
                "sha256": frontend_sha256,
                "backend": "kaldi-native-fbank",
                "version": "1.22.3",
                "torch_required": False,
                "license": "licenses/kaldi-native-fbank-LICENSE",
            },
            "application": {
                "binary": "campp_speaker_verify",
                "sha256": verifier_sha256,
                "backend": "native-cpp-orchestrator",
                "python_required": False,
                "numpy_required": False,
            },
            "buckets": buckets,
        }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = OUTPUT_ROOT / "assets.json"
        temporary = manifest_path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        print(json.dumps({
            "ready": True,
            "mode": args.mode,
            "pipeline_root": str(PIPELINE_ROOT),
            "runtime": str(runtime_output),
            "frontend": str(frontend_output),
            "speaker_verify": str(verifier_output),
            "asset_manifest": str(manifest_path),
            "buckets": list(BUCKETS),
        }, ensure_ascii=False, indent=2))
        return 0
    except (PreparationError, OSError, ValueError) as exc:
        print(f"runtime preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
