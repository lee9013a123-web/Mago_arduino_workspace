#!/usr/bin/env python3
"""Compare the existing Tensor Arena bundle with the Dense slab bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUCKETS = (98, 298, 498, 998)


class DenseSlabValidationError(RuntimeError):
    """The runtime failed or a dump/report is inconsistent."""


def _runtime_binary(path: Path) -> Path:
    if path.is_file():
        return path
    executable = path.with_suffix(".exe")
    if executable.is_file():
        return executable
    raise DenseSlabValidationError(f"Runtime 실행 파일이 없다: {path}")


def _run_dump(
    runtime: Path, bundle: Path, frames: int, feature: Path, prefix: Path
) -> None:
    plan = bundle / "execution_plans" / f"plan_{frames}.bin"
    weights = bundle / "weights.bin"
    for required in (plan, weights, feature):
        if not required.is_file():
            raise DenseSlabValidationError(f"필요한 파일이 없다: {required}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(runtime), str(plan), str(weights), str(feature), str(prefix)],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise DenseSlabValidationError(
            f"Runtime 실행 실패({completed.returncode}):\n"
            f"{completed.stdout}{completed.stderr}"
        )


def _load_dump(prefix: Path) -> tuple[dict, bytes, dict[int, dict]]:
    index_path = prefix.with_suffix(".json")
    payload_path = prefix.with_suffix(".bin")
    if not index_path.is_file() or not payload_path.is_file():
        raise DenseSlabValidationError(f"Runtime dump가 없다: {prefix}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    payload = payload_path.read_bytes()
    entries: dict[int, dict] = {}
    for entry in index.get("tensors", []):
        tensor_id = int(entry["tensor_id"])
        start = int(entry["offset"])
        byte_size = int(entry["byte_size"])
        end = start + byte_size
        if tensor_id in entries or start < 0 or byte_size < 0 or end > len(payload):
            raise DenseSlabValidationError(
                f"손상된 dump index: tensor_id={tensor_id}, prefix={prefix}"
            )
        entries[tensor_id] = entry
    return index, payload, entries


def _tensor_bytes(entry: dict, payload: bytes) -> bytes:
    start = int(entry["offset"])
    return payload[start : start + int(entry["byte_size"])]


def _expected_removed_prefix_ids(report: dict) -> set[int]:
    return {
        int(tensor_id)
        for block in report.get("blocks", [])
        for tensor_id in block.get("prefix_tensor_ids", [])
    }


def _feature_evidence(feature: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    all_zero = True
    with feature.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            all_zero = all_zero and not any(chunk)
    return {
        "path": str(feature.resolve()),
        "byte_size": size,
        "sha256": digest.hexdigest(),
        "all_zero": all_zero,
    }


def compare_dumps(
    reference_prefix: Path, dense_prefix: Path, planner_report: Path
) -> dict:
    reference_index, reference_payload, reference_entries = _load_dump(
        reference_prefix
    )
    dense_index, dense_payload, dense_entries = _load_dump(dense_prefix)
    report = json.loads(planner_report.read_text(encoding="utf-8"))
    removed_prefix_ids = _expected_removed_prefix_ids(report)
    reference_ids = set(reference_entries)
    dense_ids = set(dense_entries)
    expected_dense_ids = reference_ids - removed_prefix_ids

    result = {
        "bucket_frames": int(reference_index["bucket_frames"]),
        "reference_operator_count": int(reference_index["operator_count"]),
        "dense_operator_count": int(dense_index["operator_count"]),
        "removed_dense_concat_count": int(
            report["removed_dense_concat_count"]
        ),
        "reference_dense_concat_copied_bytes": int(
            report["dense_concat_copied_bytes_before"]
        ),
        "dense_concat_copied_bytes": 0,
        "dense_concat_saved_copy_bytes": int(
            report["dense_concat_copied_bytes_before"]
        ),
        "reference_activation_bytes": int(reference_index["activation_bytes"]),
        "dense_activation_bytes": int(dense_index["activation_bytes"]),
        "compared_tensor_count": len(dense_ids),
        "removed_prefix_tensor_count": len(removed_prefix_ids),
        "bitwise_identical": False,
        "first_mismatch": None,
    }
    if reference_index.get("memory_layout") != "tensor_arena" or dense_index.get(
        "memory_layout"
    ) != "tensor_arena":
        raise DenseSlabValidationError("두 dump 모두 Tensor Arena여야 한다")
    if reference_index["bucket_frames"] != dense_index["bucket_frames"]:
        raise DenseSlabValidationError("두 dump의 bucket frame 수가 다르다")
    if dense_ids != expected_dense_ids:
        result["first_mismatch"] = {
            "kind": "tensor_id_set",
            "missing_unexpected": sorted(expected_dense_ids - dense_ids),
            "extra": sorted(dense_ids - expected_dense_ids),
        }
        return result

    metadata_keys = ("dtype", "rank", "shape", "byte_size")
    for tensor_id in sorted(dense_ids):
        reference_entry = reference_entries[tensor_id]
        dense_entry = dense_entries[tensor_id]
        for key in metadata_keys:
            if reference_entry.get(key) != dense_entry.get(key):
                result["first_mismatch"] = {
                    "kind": "metadata",
                    "tensor_id": tensor_id,
                    "field": key,
                    "reference": reference_entry.get(key),
                    "dense": dense_entry.get(key),
                }
                return result
        reference_bytes = _tensor_bytes(reference_entry, reference_payload)
        dense_bytes = _tensor_bytes(dense_entry, dense_payload)
        if reference_bytes != dense_bytes:
            first_byte = next(
                offset
                for offset, (left, right) in enumerate(
                    zip(reference_bytes, dense_bytes, strict=True)
                )
                if left != right
            )
            result["first_mismatch"] = {
                "kind": "payload",
                "tensor_id": tensor_id,
                "byte_offset_in_tensor": first_byte,
                "reference_byte": reference_bytes[first_byte],
                "dense_byte": dense_bytes[first_byte],
            }
            return result

    result["bitwise_identical"] = True
    result["activation_saved_bytes"] = (
        result["reference_activation_bytes"] - result["dense_activation_bytes"]
    )
    return result


def _delete_dump(prefix: Path) -> None:
    prefix.with_suffix(".bin").unlink(missing_ok=True)
    prefix.with_suffix(".json").unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-binary",
        type=Path,
        default=ROOT / "build" / "campp_reference_dump",
    )
    parser.add_argument(
        "--reference-bundle",
        type=Path,
        default=ROOT / "runs" / "runtime" / "tensor_arena" / "bundle",
    )
    parser.add_argument(
        "--dense-bundle",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "runtime"
            / "kernel_optimization"
            / "dense_slab"
            / "bundle"
        ),
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "runs" / "runtime" / "ort_reference",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=(
            ROOT
            / "runs"
            / "runtime"
            / "kernel_optimization"
            / "dense_slab"
            / "validation"
        ),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "results" / "runtime" / "dense" / "dense_slab_validation.json",
    )
    parser.add_argument("--buckets", type=int, nargs="*", default=DEFAULT_BUCKETS)
    parser.add_argument("--keep-dumps", action="store_true")
    args = parser.parse_args(argv)

    try:
        runtime = _runtime_binary(args.runtime_binary)
        bucket_results = []
        all_passed = True
        for frames in args.buckets:
            feature = args.feature_dir / f"feature_{frames}.f32"
            reference_prefix = args.work_dir / "reference" / f"c_{frames}"
            dense_prefix = args.work_dir / "dense_slab" / f"c_{frames}"
            report = (
                args.dense_bundle
                / "dense_slab_plans"
                / f"dense_slab_{frames}.json"
            )
            if not report.is_file():
                raise DenseSlabValidationError(f"planner report가 없다: {report}")
            print(f"[{frames} frames] 기존 Tensor Arena 실행", flush=True)
            _run_dump(
                runtime, args.reference_bundle, frames, feature, reference_prefix
            )
            print(f"[{frames} frames] Dense slab 실행", flush=True)
            _run_dump(runtime, args.dense_bundle, frames, feature, dense_prefix)
            comparison = compare_dumps(reference_prefix, dense_prefix, report)
            comparison["input_feature"] = _feature_evidence(feature)
            bucket_results.append(comparison)
            passed = bool(comparison["bitwise_identical"])
            all_passed = all_passed and passed
            print(
                f"  {'PASS' if passed else 'FAIL'}: "
                f"operators={comparison['reference_operator_count']}→"
                f"{comparison['dense_operator_count']}, "
                f"compared={comparison['compared_tensor_count']}",
                flush=True,
            )
            if not passed:
                print(f"  first mismatch: {comparison['first_mismatch']}")
                break
            if not args.keep_dumps:
                _delete_dump(reference_prefix)
                _delete_dump(dense_prefix)

        document = {
            "comparison": "tensor_arena_vs_dense_slab",
            "all_bitwise_identical": all_passed,
            "dense_concat_calls": 0 if all_passed else None,
            "dense_concat_copied_bytes": 0 if all_passed else None,
            "bucket_results": bucket_results,
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"결과: {args.result}")
        return 0 if all_passed else 1
    except (DenseSlabValidationError, OSError, ValueError, KeyError) as exc:
        print(f"Dense slab 검증 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
