#!/usr/bin/env python3
"""Phase 3의 다섯 번째 실행 스크립트다.

ORT 기준 출력과 C Reference Runtime 덤프를 Tensor 단위로 대조한다.
plan에는 이름이 없으므로 exporter의 RuntimeGraph를 다시 만들어
tensor_id와 ONNX Tensor 이름을 잇는다.

dtype과 shape를 먼저 확인하고, 그다음 max absolute error, relative error,
cosine similarity를 계산한다. topological order상 처음 허용 오차를 넘은
operator를 보고한다. INT8 정수 출력과 FP32 출력은 서로 다른 허용 오차를 쓴다.
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

# C 덤프의 dtype 코드는 CamppTensorDType와 같은 값이다.
DTYPE_BY_CODE = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.int32,
    5: np.int64,
    6: np.bool_,
    7: np.float16,
}

# 정수 Tensor는 한 눈금이라도 어긋나면 양자화 경로가 갈린 것이므로 0을 요구한다.
INTEGER_CODES = {2, 3, 4, 5, 6}
FLOAT_ATOL = 1e-4
FLOAT_RTOL = 1e-3


def load_runtime_graph(frames: int, tag: str):
    if str(PYTHON_SOURCE) not in sys.path:
        sys.path.insert(0, str(PYTHON_SOURCE))
    from runtime_bundle_exporter.graph_ir_reader import read_graph_ir
    from runtime_bundle_exporter.static_model_reader import (
        build_runtime_graph,
        read_static_model,
    )

    static = read_static_model(ROOT / "results" / "static" / f"campp_static_{frames}.onnx")
    graph_ir = read_graph_ir(ROOT / "results" / "graph" / f"ir_{tag}.json")
    return build_runtime_graph(static, graph_ir, validate=True, name=f"campp_{frames}f")


def load_c_dump(prefix: Path) -> tuple[dict, dict[int, np.ndarray]]:
    index = json.loads((prefix.with_suffix(".json")).read_text(encoding="utf-8"))
    payload = np.fromfile(prefix.with_suffix(".bin"), dtype=np.uint8)
    tensors: dict[int, np.ndarray] = {}
    for entry in index["tensors"]:
        dtype = DTYPE_BY_CODE.get(entry["dtype"])
        if dtype is None:
            continue
        start = entry["offset"]
        stop = start + entry["byte_size"]
        raw = payload[start:stop]
        array = raw.view(dtype)
        shape = tuple(entry["shape"]) if entry["shape"] else ()
        tensors[entry["tensor_id"]] = array.reshape(shape)
    return index, tensors


def compare_arrays(reference: np.ndarray, actual: np.ndarray, dtype_code: int) -> dict:
    """두 배열의 오차 지표를 계산한다. shape가 다르면 비교하지 않는다."""

    result: dict[str, object] = {
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
    }
    if reference.shape != actual.shape:
        result["status"] = "shape_mismatch"
        return result

    if dtype_code in INTEGER_CODES:
        difference = reference.astype(np.int64) - actual.astype(np.int64)
        mismatched = int(np.count_nonzero(difference))
        result["max_abs_error"] = float(np.abs(difference).max()) if difference.size else 0.0
        result["mismatched_elements"] = mismatched
        result["element_count"] = int(reference.size)
        result["status"] = "exact" if mismatched == 0 else "integer_mismatch"
        return result

    left = reference.astype(np.float64).ravel()
    right = actual.astype(np.float64).ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.all():
        result["status"] = "non_finite"
        result["non_finite_elements"] = int((~finite).sum())
        return result

    absolute = np.abs(left - right)
    max_abs = float(absolute.max()) if absolute.size else 0.0
    denominator = np.maximum(np.abs(left), 1e-12)
    max_rel = float((absolute / denominator).max()) if absolute.size else 0.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm > 0.0 and right_norm > 0.0:
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
    else:
        cosine = 1.0 if left_norm == right_norm else 0.0

    result["max_abs_error"] = max_abs
    result["max_rel_error"] = max_rel
    result["cosine_similarity"] = cosine
    result["element_count"] = int(left.size)
    # 허용 오차만 보면 quantize 경계에서 .5 tie를 뒤집는 미세 차이가 가려진다.
    # 비트 단위로 몇 개가 다른지 따로 센다.
    differing = int(np.count_nonzero(reference.astype(np.float32) != actual.astype(np.float32)))
    result["bitwise_differing_elements"] = differing
    result["bitwise_identical"] = differing == 0
    tolerated = np.allclose(left, right, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)
    result["status"] = "within_tolerance" if tolerated else "float_mismatch"
    return result


def compare_bucket(frames: int, tag: str, ort_dir: Path, c_dir: Path) -> dict:
    graph = load_runtime_graph(frames, tag)
    name_by_id = {tensor.tensor_id: tensor.name for tensor in graph.tensors}
    producer_by_id = {
        tensor.tensor_id: tensor.producer for tensor in graph.tensors
    }
    operator_by_id = {op.operator_id: op for op in graph.operators}

    ort_tensors = np.load(ort_dir / f"ort_{frames}.npz")
    _index, c_tensors = load_c_dump(c_dir / f"c_{frames}")
    c_dtype_by_id = {
        entry["tensor_id"]: entry["dtype"]
        for entry in json.loads((c_dir / f"c_{frames}.json").read_text(encoding="utf-8"))["tensors"]
    }

    comparisons: list[dict] = []
    for operator in graph.operators:
        for tensor_id in operator.output_tensor_ids:
            name = name_by_id.get(tensor_id)
            if name is None or name not in ort_tensors:
                continue
            if tensor_id not in c_tensors:
                comparisons.append(
                    {
                        "operator_id": operator.operator_id,
                        "operator_name": operator.name,
                        "opcode": operator.opcode.name,
                        "tensor_id": tensor_id,
                        "tensor_name": name,
                        "status": "missing_in_c",
                    }
                )
                continue
            metrics = compare_arrays(
                np.asarray(ort_tensors[name]),
                c_tensors[tensor_id],
                c_dtype_by_id[tensor_id],
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

    embedding_name = graph.tensors[-1].name
    embedding = next(
        (item for item in comparisons if item["tensor_name"] == "embedding"), None
    )

    summary = {
        "bucket_frames": frames,
        "compared_tensors": len(comparisons),
        "failed_tensors": len(failures),
        "first_failure": failures[0] if failures else None,
        "embedding": embedding,
        "all_match": not failures,
    }
    del embedding_name, producer_by_id, operator_by_id
    return summary, comparisons


def print_bucket_report(summary: dict) -> None:
    frames = summary["bucket_frames"]
    total = summary["compared_tensors"]
    failed = summary["failed_tensors"]
    mark = "PASS" if summary["all_match"] else "FAIL"
    print(f"\n[{frames} frames] {mark}  비교 {total}개 중 불일치 {failed}개")

    embedding = summary["embedding"]
    if embedding is not None:
        print(
            f"  embedding: status={embedding['status']} "
            f"max_abs={embedding.get('max_abs_error', float('nan')):.6e} "
            f"max_rel={embedding.get('max_rel_error', float('nan')):.6e} "
            f"cos={embedding.get('cosine_similarity', float('nan')):.9f}"
        )

    first = summary["first_failure"]
    if first is not None:
        print(
            f"  첫 불일치 operator #{first['operator_id']} "
            f"{first['opcode']} {first['operator_name']}"
        )
        print(
            f"    tensor '{first['tensor_name']}' (id={first['tensor_id']}) "
            f"status={first['status']}"
        )
        for key in (
            "reference_shape",
            "actual_shape",
            "max_abs_error",
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
        "--ort-dir", type=Path, default=ROOT / "runs" / "runtime" / "ort_reference"
    )
    parser.add_argument(
        "--c-dir", type=Path, default=ROOT / "runs" / "runtime" / "c_reference"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results" / "runtime"
    )
    parser.add_argument(
        "--buckets", type=int, nargs="*", default=[frames for frames, _ in BUCKETS]
    )
    parser.add_argument(
        "--max-report", type=int, default=20, help="보고할 불일치 operator 개수"
    )
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_pass = True
    for frames, tag in BUCKETS:
        if frames not in args.buckets:
            continue
        if not (args.c_dir / f"c_{frames}.json").is_file():
            print(f"C 덤프가 없다: {args.c_dir / f'c_{frames}.json'}", file=sys.stderr)
            return 1
        summary, comparisons = compare_bucket(frames, tag, args.ort_dir, args.c_dir)
        print_bucket_report(summary)
        summaries.append(summary)
        all_pass = all_pass and summary["all_match"]

        good = {"exact", "within_tolerance"}
        failures = [item for item in comparisons if item["status"] not in good]
        if failures:
            print(f"  불일치 상위 {min(len(failures), args.max_report)}개:")
            for item in failures[: args.max_report]:
                print(
                    f"    #{item['operator_id']:>5} {item['opcode']:<20} "
                    f"{item['status']:<18} {item['tensor_name']}"
                )
        (args.results_dir / f"compare_{frames}.json").write_text(
            json.dumps(comparisons, indent=2), encoding="utf-8"
        )

    (args.results_dir / "compare_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"\n전체 결과: {'PASS' if all_pass else 'FAIL'}")
    print(f"상세 결과: {args.results_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
