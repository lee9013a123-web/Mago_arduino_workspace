#!/usr/bin/env python3
"""ORT 기준 출력과 C Runtime dump를 Operator 출력 Tensor 단위로 비교한다.

전체 graph dump와 독립 Operator replay dump가 같은 ``c_{frames}.bin/json``
index 형식을 사용하므로 이 도구 하나로 두 실행 방식을 모두 판정할 수 있다.
비교 순서는 Tensor ID, dtype, shape, 원소 수, 수치 순서다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "src" / "python"

BUCKETS: tuple[tuple[int, str], ...] = (
    (98, "1s"),
    (298, "3s"),
    (498, "5s"),
    (998, "10s"),
)

DTYPE_BY_CODE = {
    1: np.dtype(np.float32),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int8),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.bool_),
    7: np.dtype(np.float16),
}
INTEGER_CODES = {2, 3, 4, 5, 6}
FLOAT_ATOL = 1e-4
FLOAT_RTOL = 1e-3
EMBEDDING_COSINE_MIN = 0.999999


class RuntimeComparisonError(RuntimeError):
    """Dump 파일 자체가 손상됐거나 index와 payload가 다를 때 발생한다."""


def load_runtime_graph(
    frames: int,
    tag: str,
    static_dir: Path | None = None,
    graph_dir: Path | None = None,
):
    if str(PYTHON_SOURCE) not in sys.path:
        sys.path.insert(0, str(PYTHON_SOURCE))
    from runtime_bundle_exporter.reader.graph_ir_reader import read_graph_ir
    from runtime_bundle_exporter.reader.static_model_reader import (
        build_runtime_graph,
        read_static_model,
    )

    static_root = static_dir or ROOT / "results" / "static"
    graph_root = graph_dir or ROOT / "results" / "graph"
    static = read_static_model(static_root / f"campp_static_{frames}.onnx")
    graph_ir = read_graph_ir(graph_root / f"ir_{tag}.json")
    return build_runtime_graph(
        static, graph_ir, validate=True, name=f"campp_{frames}f"
    )


def load_c_dump(prefix: Path) -> tuple[dict, dict[int, np.ndarray]]:
    index_path = prefix.with_suffix(".json")
    payload_path = prefix.with_suffix(".bin")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    payload = np.fromfile(payload_path, dtype=np.uint8)
    tensors: dict[int, np.ndarray] = {}

    for entry in index.get("tensors", []):
        tensor_id = int(entry["tensor_id"])
        dtype = DTYPE_BY_CODE.get(int(entry["dtype"]))
        if dtype is None:
            raise RuntimeComparisonError(
                f"Tensor {tensor_id} has unsupported dtype code {entry['dtype']}"
            )
        if tensor_id in tensors:
            raise RuntimeComparisonError(f"duplicate Tensor ID {tensor_id} in C dump")
        shape = tuple(int(dimension) for dimension in entry["shape"])
        element_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        expected_bytes = element_count * dtype.itemsize
        start = int(entry["offset"])
        byte_size = int(entry["byte_size"])
        stop = start + byte_size
        if start < 0 or byte_size != expected_bytes or stop > payload.size:
            raise RuntimeComparisonError(
                f"Tensor {tensor_id} index is outside C payload or has wrong size"
            )
        raw = payload[start:stop]
        tensors[tensor_id] = raw.view(dtype).reshape(shape)
    return index, tensors


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm > 0.0 and right_norm > 0.0:
        return float(np.dot(left, right) / (left_norm * right_norm))
    return 1.0 if left_norm == right_norm else 0.0


def compare_arrays(
    reference: np.ndarray,
    actual: np.ndarray,
    dtype_code: int,
    *,
    float_atol: float = FLOAT_ATOL,
    float_rtol: float = FLOAT_RTOL,
) -> dict:
    """Tensor 구조를 확인한 뒤 요청된 모든 수치 지표를 계산한다."""

    expected_dtype = DTYPE_BY_CODE.get(dtype_code)
    result: dict[str, object] = {
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
        "reference_element_count": int(reference.size),
        "actual_element_count": int(actual.size),
    }
    if expected_dtype is None:
        result["status"] = "unsupported_dtype"
        return result
    if reference.dtype != expected_dtype or actual.dtype != expected_dtype:
        result["expected_dtype"] = str(expected_dtype)
        result["status"] = "dtype_mismatch"
        return result
    if reference.shape != actual.shape:
        result["status"] = "shape_mismatch"
        return result
    if reference.size != actual.size:
        result["status"] = "element_count_mismatch"
        return result

    left = reference.astype(np.float64).ravel()
    right = actual.astype(np.float64).ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.all():
        result["status"] = "non_finite"
        result["non_finite_elements"] = int((~finite).sum())
        return result

    absolute = np.abs(left - right)
    relative = absolute / np.maximum(np.abs(left), 1e-12)
    result["max_abs_error"] = float(absolute.max()) if absolute.size else 0.0
    result["mean_abs_error"] = float(absolute.mean()) if absolute.size else 0.0
    result["max_rel_error"] = float(relative.max()) if relative.size else 0.0
    result["cosine_similarity"] = _cosine_similarity(left, right)
    result["element_count"] = int(left.size)

    if dtype_code in INTEGER_CODES:
        mismatched = int(np.count_nonzero(reference != actual))
        result["mismatched_elements"] = mismatched
        result["bitwise_identical"] = mismatched == 0
        result["status"] = "exact" if mismatched == 0 else "integer_mismatch"
        return result

    reference_bytes = np.ascontiguousarray(reference).view(np.uint8).reshape(
        reference.size, reference.dtype.itemsize
    )
    actual_bytes = np.ascontiguousarray(actual).view(np.uint8).reshape(
        actual.size, actual.dtype.itemsize
    )
    differing = int(
        np.count_nonzero(np.any(reference_bytes != actual_bytes, axis=1))
    )
    result["bitwise_differing_elements"] = differing
    result["bitwise_identical"] = differing == 0
    tolerated = np.allclose(
        left, right, atol=float_atol, rtol=float_rtol, equal_nan=False
    )
    result["status"] = "within_tolerance" if tolerated else "float_mismatch"
    return result


def _missing_result(operator, tensor_id: int, name: str | None, status: str) -> dict:
    return {
        "operator_id": operator.operator_id,
        "operator_name": operator.name,
        "opcode": operator.opcode.name,
        "tensor_id": tensor_id,
        "tensor_name": name,
        "status": status,
    }


def compare_bucket(
    frames: int,
    tag: str,
    ort_dir: Path,
    c_dir: Path,
    *,
    static_dir: Path | None = None,
    graph_dir: Path | None = None,
    float_atol: float = FLOAT_ATOL,
    float_rtol: float = FLOAT_RTOL,
    embedding_cosine_min: float = EMBEDDING_COSINE_MIN,
) -> tuple[dict, list[dict]]:
    graph = load_runtime_graph(frames, tag, static_dir, graph_dir)
    name_by_id = {tensor.tensor_id: tensor.name for tensor in graph.tensors}
    c_index, c_tensors = load_c_dump(c_dir / f"c_{frames}")
    c_entries = {int(entry["tensor_id"]): entry for entry in c_index["tensors"]}

    comparisons: list[dict] = []
    with np.load(ort_dir / f"ort_{frames}.npz") as ort_tensors:
        for operator in graph.operators:
            for tensor_id in operator.output_tensor_ids:
                name = name_by_id.get(tensor_id)
                if name is None:
                    comparisons.append(
                        _missing_result(
                            operator, tensor_id, None, "missing_tensor_mapping"
                        )
                    )
                    continue
                if name not in ort_tensors.files:
                    comparisons.append(
                        _missing_result(operator, tensor_id, name, "missing_in_ort")
                    )
                    continue
                if tensor_id not in c_tensors or tensor_id not in c_entries:
                    comparisons.append(
                        _missing_result(operator, tensor_id, name, "missing_in_c")
                    )
                    continue

                metrics = compare_arrays(
                    np.asarray(ort_tensors[name]),
                    c_tensors[tensor_id],
                    int(c_entries[tensor_id]["dtype"]),
                    float_atol=float_atol,
                    float_rtol=float_rtol,
                )
                metrics.update(
                    {
                        "operator_id": operator.operator_id,
                        "operator_name": operator.name,
                        "opcode": operator.opcode.name,
                        "tensor_id": tensor_id,
                        "tensor_name": name,
                    }
                )
                comparisons.append(metrics)

    comparisons.sort(key=lambda item: (item["operator_id"], item["tensor_id"]))
    good = {"exact", "within_tolerance"}
    failures = [item for item in comparisons if item["status"] not in good]
    embedding = next(
        (item for item in comparisons if item["tensor_name"] == "embedding"),
        None,
    )
    embedding_cosine = (
        float(embedding.get("cosine_similarity", float("nan")))
        if embedding is not None
        else float("nan")
    )
    embedding_cosine_passed = (
        embedding is not None and embedding_cosine >= embedding_cosine_min
    )
    first_failure = failures[0] if failures else None
    if first_failure is None and not embedding_cosine_passed:
        first_failure = dict(embedding) if embedding is not None else {}
        first_failure["status"] = "embedding_cosine_below_threshold"
    summary = {
        "mode": c_index.get("mode", "full_graph"),
        "bucket_frames": frames,
        "compared_tensors": len(comparisons),
        "failed_tensors": len(failures),
        "first_failure": first_failure,
        "embedding": embedding,
        "embedding_cosine_min": embedding_cosine_min,
        "embedding_cosine_passed": embedding_cosine_passed,
        "all_match": not failures and embedding_cosine_passed,
        "float_atol": float_atol,
        "float_rtol": float_rtol,
    }
    return summary, comparisons


def print_bucket_report(summary: dict) -> None:
    frames = summary["bucket_frames"]
    total = summary["compared_tensors"]
    failed = summary["failed_tensors"]
    mark = "PASS" if summary["all_match"] else "FAIL"
    print(
        f"\n[{frames} frames] {mark} mode={summary['mode']} "
        f"compared={total} failed={failed}"
    )

    embedding = summary["embedding"]
    if embedding is not None:
        print(
            f"  embedding: status={embedding['status']} "
            f"max_abs={embedding.get('max_abs_error', float('nan')):.6e} "
            f"mean_abs={embedding.get('mean_abs_error', float('nan')):.6e} "
            f"max_rel={embedding.get('max_rel_error', float('nan')):.6e} "
            f"cos={embedding.get('cosine_similarity', float('nan')):.9f}"
        )

    first = summary["first_failure"]
    if first is not None:
        print(
            f"  first failure: operator #{first['operator_id']} "
            f"{first['opcode']} {first['operator_name']}"
        )
        print(
            f"    tensor '{first['tensor_name']}' (id={first['tensor_id']}) "
            f"status={first['status']}"
        )
        for key in (
            "reference_dtype",
            "actual_dtype",
            "reference_shape",
            "actual_shape",
            "max_abs_error",
            "mean_abs_error",
            "max_rel_error",
            "cosine_similarity",
            "mismatched_elements",
            "element_count",
        ):
            if key in first:
                print(f"    {key}: {first[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ort-dir",
        type=Path,
        default=ROOT / "runs" / "runtime" / "ort_reference",
    )
    parser.add_argument(
        "--c-dir",
        type=Path,
        default=ROOT / "runs" / "runtime" / "c_reference",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results" / "runtime"
    )
    parser.add_argument(
        "--static-dir", type=Path, default=ROOT / "results" / "static"
    )
    parser.add_argument(
        "--graph-dir", type=Path, default=ROOT / "results" / "graph"
    )
    parser.add_argument(
        "--buckets", type=int, nargs="*", default=[frames for frames, _ in BUCKETS]
    )
    parser.add_argument("--float-atol", type=float, default=FLOAT_ATOL)
    parser.add_argument("--float-rtol", type=float, default=FLOAT_RTOL)
    parser.add_argument(
        "--embedding-cosine-min", type=float, default=EMBEDDING_COSINE_MIN
    )
    parser.add_argument(
        "--max-report", type=int, default=20, help="출력할 실패 Operator 개수"
    )
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    all_pass = True
    for frames, tag in BUCKETS:
        if frames not in args.buckets:
            continue
        required = (
            args.c_dir / f"c_{frames}.json",
            args.c_dir / f"c_{frames}.bin",
            args.ort_dir / f"ort_{frames}.npz",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            print(
                "comparison inputs are missing:\n  "
                + "\n  ".join(str(path) for path in missing),
                file=sys.stderr,
            )
            return 1
        try:
            summary, comparisons = compare_bucket(
                frames,
                tag,
                args.ort_dir,
                args.c_dir,
                static_dir=args.static_dir,
                graph_dir=args.graph_dir,
                float_atol=args.float_atol,
                float_rtol=args.float_rtol,
                embedding_cosine_min=args.embedding_cosine_min,
            )
        except (OSError, ValueError, RuntimeComparisonError) as exc:
            print(f"[{frames} frames] comparison failed: {exc}", file=sys.stderr)
            return 1
        print_bucket_report(summary)
        summaries.append(summary)
        all_pass = all_pass and summary["all_match"]

        failures = [
            item
            for item in comparisons
            if item["status"] not in {"exact", "within_tolerance"}
        ]
        if failures:
            print(f"  failures (first {min(len(failures), args.max_report)}):")
            for item in failures[: args.max_report]:
                print(
                    f"    #{item['operator_id']:>5} {item['opcode']:<20} "
                    f"{item['status']:<22} {item['tensor_name']}"
                )
        (args.results_dir / f"compare_{frames}.json").write_text(
            json.dumps(comparisons, indent=2), encoding="utf-8"
        )

    (args.results_dir / "compare_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"\noverall: {'PASS' if all_pass else 'FAIL'}")
    print(f"results: {args.results_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
