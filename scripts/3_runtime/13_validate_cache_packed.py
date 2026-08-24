#!/usr/bin/env python3
"""Bitwise-validate the cache-packed 98-frame bundle on fixed real inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "src" / "python"
DEFAULT_FEATURES = (
    ROOT / "benchmarks/campplus/features/multi__speaker_0000__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0005__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0006__98.f32",
)


class CachePackedValidationError(RuntimeError):
    """Validation input, execution, or output comparison failed."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _executable(path: Path) -> Path:
    if path.is_file():
        return path
    windows = path.with_suffix(path.suffix + ".exe")
    return windows if windows.is_file() else path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_dump(
    executable: Path, bundle: Path, feature: Path, prefix: Path
) -> None:
    completed = subprocess.run(
        [
            str(executable),
            str(bundle / "execution_plans/plan_98.bin"),
            str(bundle / "weights.bin"),
            str(feature),
            str(prefix),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CachePackedValidationError(
            f"dump failed for {feature.name}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _payloads(prefix: Path) -> dict[int, bytes]:
    index = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    blob = prefix.with_suffix(".bin").read_bytes()
    result: dict[int, bytes] = {}
    for entry in index["tensors"]:
        start = int(entry["offset"])
        end = start + int(entry["byte_size"])
        result[int(entry["tensor_id"])] = blob[start:end]
    return result


def validate(args: argparse.Namespace) -> dict:
    if str(PYTHON_SOURCE) not in sys.path:
        sys.path.insert(0, str(PYTHON_SOURCE))
    from runtime_bundle_exporter.writer.bundle_manifest_writer import (
        verify_bundle_manifest,
    )
    from runtime_bundle_exporter.writer.execution_plan_writer import (
        read_execution_plan,
    )

    executable = _executable(args.runtime_binary.resolve())
    if not executable.is_file():
        raise CachePackedValidationError(f"runtime binary not found: {executable}")
    baseline = args.baseline_bundle.resolve()
    optimized = args.optimized_bundle.resolve()
    baseline_manifest = verify_bundle_manifest(baseline / "manifest.json")
    optimized_manifest = verify_bundle_manifest(optimized / "manifest.json")
    baseline_plan = read_execution_plan(
        baseline / "execution_plans/plan_98.bin"
    )
    optimized_plan = read_execution_plan(
        optimized / "execution_plans/plan_98.bin"
    )
    layout_report = json.loads(
        (optimized / "cache_layout_plans/cache_layout_98.json").read_text(
            encoding="utf-8"
        )
    )
    removed_operators = layout_report["activation_layout"][
        "removed_descriptor_copy_operator_ids"
    ]
    removed_tensor_ids = sorted(
        baseline_plan.operators[int(operator_id)].output_tensor_ids[0]
        for operator_id in removed_operators
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    input_results = []
    prefixes: list[Path] = []
    try:
        for feature in args.features:
            feature = feature.resolve()
            if not feature.is_file() or feature.stat().st_size != 98 * 80 * 4:
                raise CachePackedValidationError(
                    f"feature must be float32 [1,98,80]: {feature}"
                )
            stem = feature.stem
            baseline_prefix = args.work_dir / f"baseline__{stem}"
            optimized_prefix = args.work_dir / f"optimized__{stem}"
            prefixes.extend((baseline_prefix, optimized_prefix))
            _run_dump(executable, baseline, feature, baseline_prefix)
            _run_dump(executable, optimized, feature, optimized_prefix)
            baseline_payloads = _payloads(baseline_prefix)
            optimized_payloads = _payloads(optimized_prefix)
            missing = sorted(set(baseline_payloads) - set(optimized_payloads))
            extra = sorted(set(optimized_payloads) - set(baseline_payloads))
            mismatched = sorted(
                tensor_id
                for tensor_id in set(baseline_payloads) & set(optimized_payloads)
                if baseline_payloads[tensor_id] != optimized_payloads[tensor_id]
            )
            passed = missing == removed_tensor_ids and not extra and not mismatched
            input_results.append(
                {
                    "path": feature.relative_to(ROOT).as_posix(),
                    "byte_size": feature.stat().st_size,
                    "sha256": _sha256(feature),
                    "baseline_tensor_count": len(baseline_payloads),
                    "optimized_tensor_count": len(optimized_payloads),
                    "compared_tensor_count": len(optimized_payloads),
                    "expected_removed_tensor_ids": removed_tensor_ids,
                    "missing_tensor_ids": missing,
                    "extra_tensor_ids": extra,
                    "mismatched_tensor_ids": mismatched,
                    "bitwise_identical": passed,
                }
            )
            print(f"{feature.name}: {'PASS' if passed else 'FAIL'}")
    finally:
        if not args.keep_dumps:
            for prefix in prefixes:
                for suffix in (".bin", ".json"):
                    path = prefix.with_suffix(suffix)
                    if path.is_file():
                        path.unlink()

    result = {
        "schema_version": 1,
        "comparison": "dense_slab_vs_cache_layout_qconv_o4i4",
        "bucket_frames": 98,
        "evaluation_features": [item["path"] for item in input_results],
        "all_bitwise_identical": all(
            item["bitwise_identical"] for item in input_results
        ),
        "baseline": {
            "bundle": str(baseline),
            "manifest_sha256": _sha256(baseline / "manifest.json"),
            "operator_count": baseline_plan.header.operator_count,
            "arena_size_bytes": next(
                item["arena_size_bytes"]
                for item in baseline_manifest["plans"]
                if int(item["bucket_frames"]) == 98
            ),
        },
        "optimized": {
            "bundle": str(optimized),
            "manifest_sha256": _sha256(optimized / "manifest.json"),
            "operator_count": optimized_plan.header.operator_count,
            "arena_size_bytes": next(
                item["arena_size_bytes"]
                for item in optimized_manifest["plans"]
                if int(item["bucket_frames"]) == 98
            ),
            "activation_layout": layout_report["activation_layout"],
            "weight_packing": layout_report["weight_packing"],
        },
        "inputs": input_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-binary",
        type=_path,
        default=ROOT / "build/cache_packed_validation/campp_reference_dump",
    )
    parser.add_argument(
        "--baseline-bundle",
        type=_path,
        default=ROOT / "runs/runtime/kernel_optimization/dense_slab/bundle",
    )
    parser.add_argument(
        "--optimized-bundle",
        type=_path,
        default=ROOT / "runs/runtime/kernel_optimization/cache_packed/bundle",
    )
    parser.add_argument(
        "--features", type=_path, nargs="+", default=list(DEFAULT_FEATURES)
    )
    parser.add_argument(
        "--work-dir",
        type=_path,
        default=ROOT / "runs/runtime/kernel_optimization/cache_packed/validation",
    )
    parser.add_argument(
        "--output",
        type=_path,
        default=ROOT / "results/runtime/cache_layout/cache_packed_validation.json",
    )
    parser.add_argument("--keep-dumps", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate(args)
    except (CachePackedValidationError, OSError, ValueError, KeyError) as exc:
        print(f"Cache-packed validation failed: {exc}", file=sys.stderr)
        return 1
    return 0 if result["all_bitwise_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
