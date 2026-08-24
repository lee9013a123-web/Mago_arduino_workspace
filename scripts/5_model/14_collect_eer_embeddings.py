#!/usr/bin/env python3
"""Generate ORT/C Runtime embedding manifests for bucket-wise EER evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from speaker_verification_evaluation import read_trials  # noqa: E402
from speaker_verification_evaluation.manifests import (  # noqa: E402
    ManifestError,
    file_sha256,
    required_input_ids,
)


SUPPORTED_BUCKETS = (98, 298, 498, 998)
INPUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EMBEDDING_DIMENSION = 192
FEATURE_BINS = 80


class CollectionError(RuntimeError):
    """Embedding collection cannot satisfy the accuracy contract."""


@dataclass(frozen=True)
class FeatureRecord:
    input_id: str
    bucket_frames: int
    path: Path
    sha256: str | None


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path, base: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CollectionError(f"JSON root must be an object: {path}")
    return value


def load_feature_records(path: Path) -> list[FeatureRecord]:
    document = _load_object(path)
    raw_records = document.get("features")
    if not isinstance(raw_records, list) or not raw_records:
        raise CollectionError("feature manifest requires a non-empty features array")
    records: list[FeatureRecord] = []
    seen: set[tuple[int, str]] = set()
    for index, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            raise CollectionError(f"feature row {index} must be an object")
        input_id = raw.get("input_id")
        bucket = raw.get("bucket_frames")
        raw_path = raw.get("path")
        if not isinstance(input_id, str) or not INPUT_ID_PATTERN.fullmatch(input_id):
            raise CollectionError(f"feature row {index} has an unsafe input_id")
        if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket <= 0:
            raise CollectionError(f"feature row {index} has an invalid bucket")
        if not isinstance(raw_path, str) or not raw_path:
            raise CollectionError(f"feature row {index} has no path")
        key = (bucket, input_id)
        if key in seen:
            raise CollectionError(
                f"duplicate feature for bucket={bucket}, input_id={input_id}"
            )
        seen.add(key)
        feature_path = Path(raw_path)
        if not feature_path.is_absolute():
            # Runtime feature manifests use repository-relative portable paths.
            feature_path = ROOT / feature_path
        if not feature_path.is_file():
            raise CollectionError(f"feature does not exist: {feature_path}")
        expected_bytes = bucket * FEATURE_BINS * np.dtype("<f4").itemsize
        if feature_path.stat().st_size != expected_bytes:
            raise CollectionError(
                f"{input_id}/{bucket}: expected {expected_bytes} feature bytes, "
                f"got {feature_path.stat().st_size}"
            )
        checksum = raw.get("sha256")
        if checksum is not None and not isinstance(checksum, str):
            raise CollectionError(f"feature row {index} sha256 must be a string")
        if checksum and file_sha256(feature_path) != checksum.lower():
            raise CollectionError(f"feature checksum mismatch: {feature_path}")
        records.append(FeatureRecord(input_id, bucket, feature_path, checksum))
    return records


def select_accuracy_records(
    records: Sequence[FeatureRecord],
    required_ids: set[str],
    buckets: Sequence[int],
) -> list[FeatureRecord]:
    by_key = {(record.bucket_frames, record.input_id): record for record in records}
    missing = [
        (bucket, input_id)
        for bucket in buckets
        for input_id in sorted(required_ids)
        if (bucket, input_id) not in by_key
    ]
    if missing:
        preview = ", ".join(
            f"{bucket}:{input_id}" for bucket, input_id in missing[:3]
        )
        suffix = " ..." if len(missing) > 3 else ""
        raise CollectionError(
            f"feature manifest is missing {len(missing)} trial inputs: "
            f"{preview}{suffix}"
        )
    return [
        by_key[(bucket, input_id)]
        for bucket in buckets
        for input_id in sorted(required_ids)
    ]


def _run_json(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1", "ORT_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CollectionError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    stdout = completed.stdout.strip()
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        try:
            value = json.loads(stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise CollectionError("runtime stdout is not JSON") from exc
    if not isinstance(value, dict):
        raise CollectionError("runtime JSON root is not an object")
    return value


def _model_rows(
    manifest_path: Path, buckets: Sequence[int]
) -> dict[int, tuple[Path, float]]:
    document = _load_object(manifest_path)
    raw_buckets = document.get("buckets")
    if not isinstance(raw_buckets, dict):
        raise CollectionError("model manifest has no buckets object")
    selected: dict[int, tuple[Path, float]] = {}
    for bucket in buckets:
        raw = raw_buckets.get(str(bucket))
        if not isinstance(raw, dict):
            raise CollectionError(f"model manifest has no bucket {bucket}")
        model_name = raw.get("model")
        audio_seconds = raw.get("audio_seconds")
        expected_sha = raw.get("sha256")
        if not isinstance(model_name, str) or not isinstance(
            audio_seconds, (int, float)
        ):
            raise CollectionError(f"model manifest bucket {bucket} is invalid")
        model_path = manifest_path.parent / model_name
        if not model_path.is_file():
            raise CollectionError(f"model package does not exist: {model_path}")
        if isinstance(expected_sha, str) and file_sha256(model_path) != expected_sha:
            raise CollectionError(f"model checksum mismatch: {model_path}")
        selected[bucket] = (model_path, float(audio_seconds))
    return selected


def _save_embedding(path: Path, value: np.ndarray) -> None:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != EMBEDDING_DIMENSION:
        raise CollectionError(
            f"embedding dimension {vector.size} != {EMBEDDING_DIMENSION}"
        )
    if not np.isfinite(vector).all():
        raise CollectionError("embedding contains NaN or Inf")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vector, allow_pickle=False)


def _collect_ort(
    records: Sequence[FeatureRecord],
    static_model_root: Path,
    output_root: Path,
    *,
    force: bool,
) -> list[dict[str, object]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise CollectionError("onnxruntime is required for ORT collection") from exc

    sessions: dict[int, Any] = {}
    rows: list[dict[str, object]] = []
    for record in records:
        session = sessions.get(record.bucket_frames)
        if session is None:
            model = static_model_root / f"campp_static_{record.bucket_frames}.onnx"
            if not model.is_file():
                raise CollectionError(f"static ONNX model does not exist: {model}")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(model), options, providers=["CPUExecutionProvider"]
            )
            sessions[record.bucket_frames] = session
        output = (
            output_root / "embeddings/ort" / str(record.bucket_frames)
            / f"{record.input_id}.npy"
        )
        if output.exists() and not force:
            raise CollectionError(f"embedding exists: {output}; pass --force")
        feature = np.fromfile(record.path, dtype="<f4").reshape(
            1, record.bucket_frames, FEATURE_BINS
        )
        embedding = session.run(["embedding"], {"feature": feature})[0]
        _save_embedding(output, embedding)
        rows.append(
            {
                "input_id": record.input_id,
                "bucket_frames": record.bucket_frames,
                "path": _display_path(output, output_root),
                "sha256": file_sha256(output),
            }
        )
        print(f"ORT {record.bucket_frames}: {record.input_id}", flush=True)
    return rows


def _collect_crt(
    records: Sequence[FeatureRecord],
    runtime: Path,
    model_rows: dict[int, tuple[Path, float]],
    output_root: Path,
    *,
    force: bool,
) -> list[dict[str, object]]:
    capabilities = _run_json([str(runtime), "--capabilities"])
    if capabilities.get("model_package_format") != "camppmodel-v1":
        raise CollectionError("C Runtime does not support camppmodel-v1")
    rows: list[dict[str, object]] = []
    for record in records:
        model, audio_seconds = model_rows[record.bucket_frames]
        raw_output = (
            output_root / "raw/crt" / str(record.bucket_frames)
            / f"{record.input_id}.f32"
        )
        output = (
            output_root / "embeddings/crt" / str(record.bucket_frames)
            / f"{record.input_id}.npy"
        )
        if output.exists() and not force:
            raise CollectionError(f"embedding exists: {output}; pass --force")
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        _run_json(
            [
                str(runtime),
                "--model",
                str(model),
                "--input",
                str(record.path),
                "--audio-seconds",
                str(audio_seconds),
                "--warmup",
                "0",
                "--repeat",
                "1",
                "--threads",
                "1",
                "--embedding-output",
                str(raw_output),
            ]
        )
        embedding = np.fromfile(raw_output, dtype="<f4")
        _save_embedding(output, embedding)
        rows.append(
            {
                "input_id": record.input_id,
                "bucket_frames": record.bucket_frames,
                "path": _display_path(output, output_root),
                "sha256": file_sha256(output),
            }
        )
        print(f"CRT {record.bucket_frames}: {record.input_id}", flush=True)
    return rows


def _write_manifest(
    output_root: Path,
    backend: str,
    feature_manifest: Path,
    rows: Sequence[dict[str, object]],
) -> Path:
    target = output_root / f"{backend}_embeddings.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": backend,
                "feature_manifest": _display_path(feature_manifest),
                "feature_manifest_sha256": file_sha256(feature_manifest),
                "embedding_dimension": EMBEDDING_DIMENSION,
                "embeddings": list(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=_path,
        default=ROOT / "benchmarks/campplus/manifests/trials.tsv",
    )
    parser.add_argument("--feature-manifest", type=_path, required=True)
    parser.add_argument(
        "--backends", choices=("ort", "crt"), nargs="+", default=["ort", "crt"]
    )
    parser.add_argument(
        "--buckets", type=int, choices=SUPPORTED_BUCKETS, nargs="+",
        default=list(SUPPORTED_BUCKETS),
    )
    parser.add_argument(
        "--static-model-root", type=_path, default=ROOT / "results/static"
    )
    parser.add_argument(
        "--runtime",
        type=_path,
        default=(
            ROOT / "build/profill/final_v3_hybrid/campp_runtime_benchmark_final"
        ),
    )
    parser.add_argument(
        "--model-manifest",
        type=_path,
        default=(
            ROOT
            / "models/runtime/campp_sv_multibucket/campp_sv_multibucket.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=_path,
        default=ROOT / "runs/models/campplus/eer",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        buckets = list(dict.fromkeys(args.buckets))
        backends = list(dict.fromkeys(args.backends))
        trials = read_trials(args.trials)
        records = select_accuracy_records(
            load_feature_records(args.feature_manifest),
            required_input_ids(trials),
            buckets,
        )
        model_rows: dict[int, tuple[Path, float]] = {}
        if "ort" in backends:
            missing_models = [
                args.static_model_root / f"campp_static_{bucket}.onnx"
                for bucket in buckets
                if not (
                    args.static_model_root / f"campp_static_{bucket}.onnx"
                ).is_file()
            ]
            if missing_models:
                raise CollectionError(f"static ONNX model missing: {missing_models[0]}")
        if "crt" in backends:
            if not args.runtime.is_file():
                raise CollectionError(f"C Runtime does not exist: {args.runtime}")
            model_rows = _model_rows(args.model_manifest, buckets)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "backends": backends,
                        "buckets": buckets,
                        "trial_count": len(trials),
                        "input_count_per_bucket": len(required_input_ids(trials)),
                        "feature_count": len(records),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        args.output_root.mkdir(parents=True, exist_ok=True)
        manifests: dict[str, str] = {}
        if "ort" in backends:
            rows = _collect_ort(
                records,
                args.static_model_root,
                args.output_root,
                force=args.force,
            )
            manifest = _write_manifest(
                args.output_root, "ort", args.feature_manifest, rows
            )
            manifests["ort"] = _display_path(manifest)
        if "crt" in backends:
            rows = _collect_crt(
                records,
                args.runtime,
                model_rows,
                args.output_root,
                force=args.force,
            )
            manifest = _write_manifest(
                args.output_root, "crt", args.feature_manifest, rows
            )
            manifests["crt"] = _display_path(manifest)
        print(json.dumps({"complete": True, "manifests": manifests}, indent=2))
        return 0
    except (CollectionError, ManifestError, OSError, ValueError) as exc:
        print(f"embedding collection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
