#!/usr/bin/env python3
"""같은 입력으로 Reference buffer와 Tensor Arena 실행 결과를 bit-exact 비교한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUCKETS = (98, 298, 498, 998)


class ArenaValidationError(RuntimeError):
    """실행 실패, 손상된 dump 또는 Tensor 불일치를 나타낸다."""


def _runtime_binary(path: Path) -> Path:
    if path.is_file():
        return path
    exe = path.with_suffix(".exe")
    if exe.is_file():
        return exe
    raise ArenaValidationError(f"Runtime 실행 파일이 없다: {path}")


def _frames(feature: Path) -> int:
    try:
        return int(feature.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ArenaValidationError(
            f"feature 파일명에서 frame 수를 읽을 수 없다: {feature.name}"
        ) from exc


def _run_dump(
    runtime: Path,
    bundle: Path,
    feature: Path,
    prefix: Path,
) -> None:
    plan = bundle / "execution_plans" / f"plan_{_frames(feature)}.bin"
    weights = bundle / "weights.bin"
    for required in (plan, weights, feature):
        if not required.is_file():
            raise ArenaValidationError(f"필요한 파일이 없다: {required}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(runtime), str(plan), str(weights), str(feature), str(prefix)],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ArenaValidationError(
            f"Runtime 실행 실패({completed.returncode}):\n"
            f"{completed.stdout}{completed.stderr}"
        )
    if not prefix.with_suffix(".json").is_file() or not prefix.with_suffix(
        ".bin"
    ).is_file():
        raise ArenaValidationError(f"Runtime dump가 생성되지 않았다: {prefix}")


def _load_dump(prefix: Path) -> tuple[dict, bytes, dict[int, dict]]:
    index = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    payload = prefix.with_suffix(".bin").read_bytes()
    entries: dict[int, dict] = {}
    for entry in index.get("tensors", []):
        tensor_id = int(entry["tensor_id"])
        start = int(entry["offset"])
        byte_size = int(entry["byte_size"])
        end = start + byte_size
        if tensor_id in entries or start < 0 or byte_size < 0 or end > len(payload):
            raise ArenaValidationError(
                f"손상된 Tensor index: {prefix}, tensor_id={tensor_id}"
            )
        entries[tensor_id] = entry
    return index, payload, entries


def compare_dumps(reference_prefix: Path, arena_prefix: Path) -> dict:
    reference_index, reference_payload, reference_entries = _load_dump(
        reference_prefix
    )
    arena_index, arena_payload, arena_entries = _load_dump(arena_prefix)

    reference_ids = list(reference_entries)
    arena_ids = list(arena_entries)
    result = {
        "bucket_frames": int(reference_index["bucket_frames"]),
        "reference_activation_bytes": int(reference_index["activation_bytes"]),
        "arena_activation_bytes": int(arena_index["activation_bytes"]),
        "reference_tensor_count": len(reference_ids),
        "arena_tensor_count": len(arena_ids),
        "bitwise_identical": False,
        "first_mismatch": None,
    }
    if reference_index.get("memory_layout") != "reference":
        raise ArenaValidationError("기준 dump가 Reference memory layout이 아니다")
    if arena_index.get("memory_layout") != "tensor_arena":
        raise ArenaValidationError("비교 dump가 Tensor Arena memory layout이 아니다")
    if reference_index["bucket_frames"] != arena_index["bucket_frames"]:
        raise ArenaValidationError("두 dump의 bucket frame 수가 다르다")
    if reference_ids != arena_ids:
        result["first_mismatch"] = {
            "kind": "tensor_id_order",
            "reference_ids": reference_ids,
            "arena_ids": arena_ids,
        }
        return result

    metadata_keys = ("operator_id", "dtype", "rank", "shape", "byte_size")
    for tensor_id in reference_ids:
        reference_entry = reference_entries[tensor_id]
        arena_entry = arena_entries[tensor_id]
        for key in metadata_keys:
            if reference_entry.get(key) != arena_entry.get(key):
                result["first_mismatch"] = {
                    "kind": "metadata",
                    "tensor_id": tensor_id,
                    "field": key,
                    "reference": reference_entry.get(key),
                    "arena": arena_entry.get(key),
                }
                return result

        reference_start = int(reference_entry["offset"])
        arena_start = int(arena_entry["offset"])
        byte_size = int(reference_entry["byte_size"])
        reference_bytes = reference_payload[
            reference_start : reference_start + byte_size
        ]
        arena_bytes = arena_payload[arena_start : arena_start + byte_size]
        if reference_bytes != arena_bytes:
            first_byte = next(
                index
                for index, (left, right) in enumerate(
                    zip(reference_bytes, arena_bytes, strict=True)
                )
                if left != right
            )
            result["first_mismatch"] = {
                "kind": "payload",
                "operator_id": int(reference_entry["operator_id"]),
                "tensor_id": tensor_id,
                "byte_offset_in_tensor": first_byte,
                "reference_byte": reference_bytes[first_byte],
                "arena_byte": arena_bytes[first_byte],
            }
            return result

    result["bitwise_identical"] = True
    result["saved_activation_bytes"] = (
        result["reference_activation_bytes"] - result["arena_activation_bytes"]
    )
    return result


def _delete_dump(prefix: Path) -> None:
    for path in (prefix.with_suffix(".bin"), prefix.with_suffix(".json")):
        path.unlink(missing_ok=True)


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
        default=ROOT / "models" / "compiled" / "reference",
    )
    parser.add_argument(
        "--arena-bundle",
        type=Path,
        default=ROOT / "runs" / "runtime" / "tensor_arena" / "bundle",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "runs" / "runtime" / "ort_reference",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "runs" / "runtime" / "tensor_arena" / "validation",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "results" / "runtime" / "tensor_arena_validation.json",
    )
    parser.add_argument("--buckets", type=int, nargs="*", default=DEFAULT_BUCKETS)
    parser.add_argument(
        "--keep-dumps",
        action="store_true",
        help="통과한 bucket의 큰 중간 dump도 삭제하지 않는다",
    )
    args = parser.parse_args(argv)

    try:
        runtime = _runtime_binary(args.runtime_binary)
        results = []
        all_passed = True
        for frames in args.buckets:
            feature = args.feature_dir / f"feature_{frames}.f32"
            reference_prefix = args.work_dir / "reference" / f"c_{frames}"
            arena_prefix = args.work_dir / "arena" / f"c_{frames}"
            print(f"[{frames} frames] Reference buffer 실행", flush=True)
            _run_dump(runtime, args.reference_bundle, feature, reference_prefix)
            print(f"[{frames} frames] Tensor Arena 실행", flush=True)
            _run_dump(runtime, args.arena_bundle, feature, arena_prefix)
            comparison = compare_dumps(reference_prefix, arena_prefix)
            results.append(comparison)
            passed = bool(comparison["bitwise_identical"])
            all_passed = all_passed and passed
            print(
                f"  {'PASS' if passed else 'FAIL'}: "
                f"reference={comparison['reference_activation_bytes']:,} B, "
                f"arena={comparison['arena_activation_bytes']:,} B",
                flush=True,
            )
            if not passed:
                print(f"  first mismatch: {comparison['first_mismatch']}")
                print("  첫 실패에서 중단하고 해당 dump를 보존", flush=True)
                break
            elif not args.keep_dumps:
                _delete_dump(reference_prefix)
                _delete_dump(arena_prefix)
                print("  통과한 두 dump 삭제", flush=True)

        document = {
            "phase": 20,
            "comparison": "reference_buffers_vs_tensor_arena",
            "all_bitwise_identical": all_passed,
            "bucket_results": results,
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"결과: {args.result}")
        return 0 if all_passed else 1
    except (ArenaValidationError, OSError, ValueError, KeyError) as exc:
        print(f"Tensor Arena 검증 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
