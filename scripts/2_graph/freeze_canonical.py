#!/usr/bin/env python3
"""Freeze the canonical CAM++ ONNX and validate all static length buckets.

This script implements the graph-freeze parts that do not require PyTorch:

1. fingerprint the canonical INT8 QOperator ONNX;
2. compare it with every bucket-specific static ONNX on several deterministic
   feature tensors (and optional real fbank ``.npy`` files);
3. verify that quantization/BN parameters survived unchanged;
4. write a machine-readable graph manifest and a sha256sum-compatible file.

The expected input contract is ``feature[1, frames, 80]`` and the expected
output is ``embedding[1, 192]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_BUCKETS = ((1.0, 98), (3.0, 298), (5.0, 498), (10.0, 998))
PARAMETER_OPS = {
    "QLinearConv": 1,
    "QuantizeLinear": 1,
    "DequantizeLinear": 1,
    "BatchNormalization": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_context(reference: Path) -> dict[str, Any]:
    root_text = run_git("rev-parse", "--show-toplevel")
    root = Path(root_text).resolve() if root_text else None
    label = str(reference.resolve())
    tracked: bool | None = None
    if root is not None:
        try:
            label = reference.resolve().relative_to(root).as_posix()
            tracked = (
                run_git("-C", str(root), "ls-files", "--error-unmatch", label)
                is not None
            )
        except ValueError:
            tracked = False
    return {
        "git_root": str(root) if root else None,
        "git_commit": run_git("rev-parse", "HEAD"),
        "git_branch": run_git("branch", "--show-current"),
        "reference_path_label": label,
        "reference_git_tracked": tracked,
    }


def import_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "numpy, onnx and onnxruntime are required; use the workspace venv"
        ) from exc
    return np, onnx, ort


def tensor_shape(value_info: Any) -> list[int | str | None]:
    shape: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


def graph_topology_hash(model: Any) -> str:
    """Hash node structure and attributes, intentionally excluding tensors."""
    digest = hashlib.sha256()
    for node in model.graph.node:
        digest.update(node.SerializeToString())
        digest.update(b"\x00")
    return digest.hexdigest()


def initializer_map(model: Any) -> dict[str, Any]:
    return {tensor.name: tensor for tensor in model.graph.initializer}


def arithmetic_parameter_names(model: Any) -> set[str]:
    """Collect initializer inputs that control quantized arithmetic or BN."""
    initializers = initializer_map(model)
    names: set[str] = set()
    for node in model.graph.node:
        start = PARAMETER_OPS.get(node.op_type)
        if start is None:
            continue
        for name in node.input[start:]:
            if name in initializers:
                names.add(name)
    return names


def parameter_fingerprints(model: Any, names: Iterable[str]) -> dict[str, str]:
    initializers = initializer_map(model)
    return {
        name: sha256_bytes(initializers[name].SerializeToString())
        for name in sorted(names)
        if name in initializers
    }


def compare_parameters(reference: Any, candidate: Any) -> dict[str, Any]:
    names = arithmetic_parameter_names(reference)
    expected = parameter_fingerprints(reference, names)
    actual = parameter_fingerprints(candidate, names)
    missing = sorted(set(expected) - set(actual))
    mismatched = sorted(
        name for name in set(expected) & set(actual) if expected[name] != actual[name]
    )
    return {
        "reference_parameter_count": len(expected),
        "candidate_parameter_count": len(actual),
        "missing": missing,
        "mismatched": mismatched,
        "passed": not missing and not mismatched,
    }


def model_metadata(onnx: Any, path: Path, model: Any) -> dict[str, Any]:
    operator_counts = Counter(node.op_type for node in model.graph.node)
    initializer_names = [tensor.name.lower() for tensor in model.graph.initializer]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "ir_version": int(model.ir_version),
        "opsets": {
            item.domain or "ai.onnx": int(item.version)
            for item in model.opset_import
        },
        "producer_name": model.producer_name or None,
        "producer_version": model.producer_version or None,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operator_counts": dict(sorted(operator_counts.items())),
        "inputs": [
            {
                "name": value.name,
                "shape": tensor_shape(value),
                "dtype": onnx.TensorProto.DataType.Name(
                    value.type.tensor_type.elem_type
                ),
            }
            for value in model.graph.input
        ],
        "outputs": [
            {
                "name": value.name,
                "shape": tensor_shape(value),
                "dtype": onnx.TensorProto.DataType.Name(
                    value.type.tensor_type.elem_type
                ),
            }
            for value in model.graph.output
        ],
        "topology_sha256": graph_topology_hash(model),
        "named_scale_initializers": sum("scale" in name for name in initializer_names),
        "named_zero_point_initializers": sum(
            "zero_point" in name for name in initializer_names
        ),
    }


def make_session(ort: Any, model_path: Path, threads: int) -> Any:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )


def session_io(session: Any) -> tuple[str, str]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError(f"expected one input, found {len(inputs)}")
    if not outputs:
        raise ValueError("model has no output")
    output_name = next(
        (output.name for output in outputs if output.name == "embedding"),
        outputs[0].name,
    )
    return inputs[0].name, output_name


def normalize_feature(np: Any, feature: Any) -> Any:
    array = np.asarray(feature)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != 80:
        raise ValueError(
            f"feature must be [frames,80] or [1,frames,80], got {array.shape}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def synthetic_cases(np: Any, frames: int, seeds: Sequence[int]) -> list[tuple[str, Any]]:
    shape = (1, frames, 80)
    cases: list[tuple[str, Any]] = [
        ("zeros", np.zeros(shape, dtype=np.float32)),
        ("ones", np.ones(shape, dtype=np.float32)),
    ]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        feature = rng.standard_normal(shape).astype(np.float32) * 5.0
        cases.append((f"random_seed_{seed}", feature))
    return cases


def feature_sha256(array: Any) -> str:
    contiguous = array if array.flags.c_contiguous else array.copy(order="C")
    return sha256_bytes(memoryview(contiguous).cast("B"))


def output_metrics(np: Any, reference: Any, candidate: Any) -> dict[str, Any]:
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    if ref.shape != got.shape:
        return {
            "shape_match": False,
            "reference_shape": list(ref.shape),
            "candidate_shape": list(got.shape),
            "exact": False,
            "max_abs_diff": None,
            "cosine_similarity": None,
        }
    ref64 = ref.astype(np.float64, copy=False).reshape(-1)
    got64 = got.astype(np.float64, copy=False).reshape(-1)
    difference = np.abs(ref64 - got64)
    denominator = float(np.linalg.norm(ref64) * np.linalg.norm(got64))
    cosine = float(ref64 @ got64 / denominator) if denominator else math.nan
    return {
        "shape_match": True,
        "reference_shape": list(ref.shape),
        "candidate_shape": list(got.shape),
        "exact": bool(np.array_equal(ref, got)),
        "max_abs_diff": float(np.max(difference)) if difference.size else 0.0,
        "mean_abs_diff": float(np.mean(difference)) if difference.size else 0.0,
        "cosine_similarity": cosine,
        "reference_embedding_sha256": feature_sha256(ref),
        "candidate_embedding_sha256": feature_sha256(got),
    }


def load_real_features(np: Any, paths: Sequence[Path]) -> dict[int, list[tuple[str, Any]]]:
    grouped: dict[int, list[tuple[str, Any]]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"feature not found: {path}")
        feature = normalize_feature(np, np.load(path, allow_pickle=False))
        grouped.setdefault(int(feature.shape[1]), []).append(
            (f"real:{path.name}", feature)
        )
    return grouped


def validate_bucket(
    np: Any,
    reference_session: Any,
    candidate_session: Any,
    cases: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    ref_input, ref_output = session_io(reference_session)
    got_input, got_output = session_io(candidate_session)
    results: list[dict[str, Any]] = []
    for case_name, feature in cases:
        expected = reference_session.run([ref_output], {ref_input: feature})[0]
        actual = candidate_session.run([got_output], {got_input: feature})[0]
        results.append(
            {
                "name": case_name,
                "feature_sha256": feature_sha256(feature),
                **output_metrics(np, expected, actual),
            }
        )
    return {
        "case_count": len(results),
        "exact_case_count": sum(bool(result["exact"]) for result in results),
        "all_exact": all(bool(result["exact"]) for result in results),
        "maximum_abs_diff": max(
            (
                float(result["max_abs_diff"])
                for result in results
                if result["max_abs_diff"] is not None
            ),
            default=None,
        ),
        "cases": results,
    }


def parse_bucket(text: str) -> tuple[float, int]:
    try:
        seconds_text, frames_text = text.split(":", 1)
        seconds = float(seconds_text)
        frames = int(frames_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"bucket must be SECONDS:FRAMES, got {text!r}"
        ) from exc
    if seconds <= 0 or frames <= 0:
        raise argparse.ArgumentTypeError("bucket values must be positive")
    return seconds, frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, default=Path("results/static"))
    parser.add_argument(
        "--bucket",
        type=parse_bucket,
        action="append",
        help="SECONDS:FRAMES; defaults to 1:98, 3:298, 5:498, 10:998",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--feature",
        type=Path,
        action="append",
        default=[],
        help="optional real fbank .npy; assigned by frame count",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/graph/campp_graph_manifest.json"),
    )
    parser.add_argument(
        "--checksum-output",
        type=Path,
        default=Path("results/graph/canonical_onnx.sha256"),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if not args.reference.is_file():
        raise FileNotFoundError(f"canonical ONNX not found: {args.reference}")
    if args.threads < 1:
        raise ValueError("threads must be >= 1")
    if not args.static_dir.is_dir():
        raise FileNotFoundError(f"static model directory not found: {args.static_dir}")

    np, onnx, ort = import_runtime_dependencies()
    buckets = args.bucket or list(DEFAULT_BUCKETS)
    git = git_context(args.reference)
    reference_model = onnx.load(str(args.reference), load_external_data=True)
    reference_metadata = model_metadata(
        onnx, args.reference, reference_model
    )
    reference_session = make_session(ort, args.reference, args.threads)
    real_features = load_real_features(np, args.feature)

    bucket_results: list[dict[str, Any]] = []
    for seconds, frames in buckets:
        static_path = args.static_dir / f"campp_static_{frames}.onnx"
        if not static_path.is_file():
            raise FileNotFoundError(f"static ONNX not found: {static_path}")
        static_model = onnx.load(str(static_path), load_external_data=True)
        static_metadata = model_metadata(onnx, static_path, static_model)
        parameter_check = compare_parameters(reference_model, static_model)
        static_session = make_session(ort, static_path, args.threads)
        cases = synthetic_cases(np, frames, args.seeds)
        cases.extend(real_features.get(frames, []))
        validation = validate_bucket(
            np, reference_session, static_session, cases
        )
        bucket_results.append(
            {
                "seconds": seconds,
                "frames": frames,
                "static_model": static_metadata,
                "parameter_preservation": parameter_check,
                "validation": validation,
            }
        )
        print(
            f"{seconds:g}s/{frames} frames: "
            f"{validation['exact_case_count']}/{validation['case_count']} exact, "
            f"max|diff|={validation['maximum_abs_diff']}"
        )

    topology_hashes = {
        result["static_model"]["topology_sha256"] for result in bucket_results
    }
    all_topologies_equal = len(topology_hashes) == 1
    all_parameters_preserved = all(
        result["parameter_preservation"]["passed"] for result in bucket_results
    )
    all_outputs_exact = all(
        result["validation"]["all_exact"] for result in bucket_results
    )
    overall_passed = (
        all_topologies_equal and all_parameters_preserved and all_outputs_exact
    )

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "CAM++ canonical ONNX and static bucket freeze",
        "reference_policy": (
            "The INT8 QOperator ONNX is canonical; PyTorch parity is out of scope."
        ),
        "git": git,
        "runtime": {
            "python": sys.version.split()[0],
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "threads": args.threads,
            "provider": "CPUExecutionProvider",
        },
        "canonical_model": reference_metadata,
        "buckets": bucket_results,
        "freeze_gate": {
            "static_topology_sha256": next(iter(topology_hashes), None)
            if all_topologies_equal
            else None,
            "all_static_topologies_equal": all_topologies_equal,
            "all_arithmetic_parameters_preserved": all_parameters_preserved,
            "all_outputs_bit_exact": all_outputs_exact,
            "passed": overall_passed,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.checksum_output.parent.mkdir(parents=True, exist_ok=True)
    args.checksum_output.write_text(
        f"{reference_metadata['sha256']}  {git['reference_path_label']}\n",
        encoding="utf-8",
    )

    print(f"manifest: {args.output}")
    print(f"checksum: {args.checksum_output}")
    print(f"freeze gate: {'PASS' if overall_passed else 'FAIL'}")
    return 0 if overall_passed else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
