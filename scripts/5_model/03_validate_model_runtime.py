#!/usr/bin/env python3
"""Prove one bucket package and its split plan/weights are bitwise identical."""

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
from runtime_model_package.packager import (  # noqa: E402
    AUDIO_SECONDS_BY_BUCKET,
    FINAL_98_SUITE,
    SUPPORTED_BUCKET_FRAMES,
)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
            f"command failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ModelPackageError("runtime did not return JSON") from exc
    if not isinstance(value, dict):
        raise ModelPackageError("runtime JSON root is not an object")
    return value


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise ModelPackageError(
            f"command failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket-frames", type=int, choices=SUPPORTED_BUCKET_FRAMES,
        default=98,
    )
    parser.add_argument(
        "--binary", type=_path,
        default=(
            ROOT / "build/profill/final_v3_hybrid"
            / "campp_runtime_benchmark_final"
        ),
    )
    parser.add_argument("--model", type=_path)
    parser.add_argument(
        "--dump-binary", type=_path,
        default=(
            ROOT / "build/profill/final_v3_hybrid"
            / "campp_reference_dump_final"
        ),
    )
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument("--input", type=_path)
    parser.add_argument("--runs-dir", type=_path)
    parser.add_argument("--output", type=_path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        bucket = args.bucket_frames
        static_root = (
            ROOT / "runs/models/campplus/final_v3/weight_residency"
            / str(bucket)
        )
        model = args.model or (
            ROOT / "models/runtime/campp_sv_multibucket"
            / f"campp_sv_{bucket}.camppmodel"
        )
        plan = args.plan or static_root / f"plan_{bucket}.bin"
        weights = args.weights or static_root / f"weights_{bucket}.bin"
        feature = args.input or (
            ROOT / "benchmarks/campplus/features"
            / f"multi__speaker_0000__{bucket}.f32"
        )
        runs_dir = args.runs_dir or (
            ROOT / "runs/model_packages"
            / f"campp_sv_{bucket}/validation"
        )
        output = args.output or (
            ROOT / "results/model_packages"
            / f"campp_sv_{bucket}_validation.json"
        )
        missing = [
            path for path in (
                args.binary, args.dump_binary, model, plan, weights, feature
            )
            if not path.is_file()
        ]
        if missing:
            raise ModelPackageError(
                "missing files:\n  " + "\n  ".join(str(path) for path in missing)
            )
        package = verify_model_package(model)
        if package.bucket_frames != bucket:
            raise ModelPackageError(
                f"package bucket mismatch: {package.bucket_frames} != {bucket}"
            )
        metadata = package.json_section(
            ModelPackageSectionType.MODEL_METADATA_JSON
        )
        if metadata.get("optimization_suite") != FINAL_98_SUITE:
            raise ModelPackageError("model package suite is not Final V3")
        capabilities = _run_json([str(args.binary), "--capabilities"])
        if capabilities.get("model_package_format") != "camppmodel-v1":
            raise ModelPackageError("runtime has no camppmodel-v1 loader")
        if capabilities.get("optimization_suite_config") != FINAL_98_SUITE:
            raise ModelPackageError("runtime suite differs from model package")
        if args.preflight_only:
            print(json.dumps({
                "ready": True,
                "binary": str(args.binary),
                "model": str(model),
                "bucket_frames": package.bucket_frames,
            }, ensure_ascii=False, indent=2))
            return 0
        if output.exists() and not args.force:
            raise ModelPackageError("output exists; use --force")
        runs_dir.mkdir(parents=True, exist_ok=True)
        split_embedding = runs_dir / "split_embedding.f32"
        package_embedding = runs_dir / "package_embedding.f32"
        common = [
            "--input", str(feature),
            "--audio-seconds", str(AUDIO_SECONDS_BY_BUCKET[bucket]),
            "--warmup", "0",
            "--repeat", "1",
            "--threads", "1",
        ]
        split_result = _run_json([
            str(args.binary),
            "--plan", str(plan),
            "--weights", str(weights),
            *common,
            "--embedding-output", str(split_embedding),
        ])
        package_result = _run_json([
            str(args.binary),
            "--model", str(model),
            *common,
            "--embedding-output", str(package_embedding),
        ])
        split_prefix = runs_dir / "split_retained"
        package_prefix = runs_dir / "package_retained"
        _run([
            str(args.dump_binary), str(plan), str(weights),
            str(feature), str(split_prefix),
        ])
        _run([
            str(args.dump_binary), "--model", str(model),
            str(feature), str(package_prefix),
        ])
        split_bytes = split_embedding.read_bytes()
        package_bytes = package_embedding.read_bytes()
        bitwise = split_bytes == package_bytes
        split_retained = split_prefix.with_suffix(".bin").read_bytes()
        package_retained = package_prefix.with_suffix(".bin").read_bytes()
        split_index = json.loads(
            split_prefix.with_suffix(".json").read_text(encoding="utf-8")
        )
        package_index = json.loads(
            package_prefix.with_suffix(".json").read_text(encoding="utf-8")
        )
        retained_bitwise = split_retained == package_retained
        retained_index_identical = split_index == package_index
        report = {
            "schema_version": 1,
            "backend": "cpu_aarch64_o4i4_final",
            "bucket_frames": bucket,
            "input": str(feature),
            "comparison": "split_plan_weights_vs_camppmodel_v1",
            "tolerance": {"kind": "bitwise", "atol": 0.0, "rtol": 0.0},
            "embedding_bytes": len(split_bytes),
            "split_embedding_sha256": _sha256(split_embedding),
            "package_embedding_sha256": _sha256(package_embedding),
            "embedding_bitwise_identical": bitwise,
            "retained_tensor_payload_bytes": len(split_retained),
            "retained_tensor_payload_bitwise_identical": retained_bitwise,
            "retained_tensor_index_identical": retained_index_identical,
            "split_runtime": split_result.get("model"),
            "package_runtime": package_result.get("model"),
            "passed": (
                bitwise and len(split_bytes) == 192 * 4
                and retained_bitwise and retained_index_identical
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"model runtime validation: {'PASS' if report['passed'] else 'FAIL'}")
        return 0 if report["passed"] else 3
    except (ModelPackageError, OSError, ValueError) as exc:
        print(f"model runtime validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
