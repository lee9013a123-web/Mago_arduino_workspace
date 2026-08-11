#!/usr/bin/env python3
"""298-frame CAM++ Reference Runtime end-to-end validation.

This module runs the C runtime with the same feature tensor used by ORT, compares
every operator output, and then reports the graph boundaries that are useful when
diagnosing a three-second inference failure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
FRAMES = 298
TAG = "3s"
EXPECTED_INPUT_SHAPE = (1, 298, 80)
EXPECTED_INPUT_BYTES = 1 * 298 * 80 * 4
EXPECTED_OPERATOR_COUNT = 1438
GOOD_STATUSES = {"exact", "within_tolerance"}


@dataclass(frozen=True)
class Checkpoint:
    key: str
    description: str
    operator_id: int
    opcode: str
    tensor_id: int
    tensor_name: str


# These are graph boundaries, not arbitrary samples.  The IDs come from the
# canonical 298-frame RuntimeGraph produced by the exporter.
CHECKPOINTS = (
    Checkpoint(
        "input_transform",
        "input transpose output",
        0,
        "TRANSPOSE",
        2281,
        "/Transpose_output_0",
    ),
    Checkpoint(
        "head",
        "head output",
        51,
        "QUANTIZE_LINEAR",
        2332,
        "/head/Reshape_output_0_quantized",
    ),
    Checkpoint(
        "dense_block_1",
        "Dense block 1 final concatenation",
        366,
        "CONCAT",
        2647,
        "/xvector/block1/Concat_11_output_0",
    ),
    Checkpoint(
        "dense_block_2",
        "Dense block 2 final concatenation",
        995,
        "CONCAT",
        3276,
        "/xvector/block2/Concat_23_output_0",
    ),
    Checkpoint(
        "dense_block_3",
        "Dense block 3 final concatenation",
        1416,
        "CONCAT",
        3697,
        "/xvector/block3/Concat_15_output_0",
    ),
    Checkpoint(
        "statistics_pooling",
        "statistics pooling concatenated mean/std",
        1431,
        "CONCAT",
        3712,
        "/xvector/stats/Concat_output_0",
    ),
    Checkpoint(
        "embedding",
        "final speaker embedding",
        1437,
        "BATCH_NORMALIZATION",
        3718,
        "embedding",
    ),
)


class EndToEndError(RuntimeError):
    """The end-to-end test inputs or generated dump are inconsistent."""


def _load_comparator():
    path = ROOT / "scripts" / "3_runtime" / "05_compare_runtime_outputs.py"
    spec = importlib.util.spec_from_file_location("campp_runtime_comparator", path)
    if spec is None or spec.loader is None:
        raise EndToEndError(f"cannot import comparator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_dump_tool() -> Path:
    candidates = [ROOT / "build" / "campp_reference_dump"]
    if os.name == "nt":
        candidates = [
            ROOT / "build" / "campp_reference_dump.exe",
            ROOT / "build" / "runtime_phase15" / "campp_reference_dump.exe",
        ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _required_source_files(ort_dir: Path) -> tuple[Path, ...]:
    return (
        ROOT / "results" / "static" / "campp_static_298.onnx",
        ROOT / "results" / "graph" / "ir_3s.json",
        ort_dir / "ort_298.npz",
    )


def _run_c_runtime(
    dump_tool: Path,
    plan: Path,
    weights: Path,
    feature: Path,
    output_prefix: Path,
) -> None:
    required = (dump_tool, plan, weights, feature)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise EndToEndError("required runtime files are missing:\n  " + "\n  ".join(missing))
    if feature.stat().st_size != EXPECTED_INPUT_BYTES:
        raise EndToEndError(
            f"feature size is {feature.stat().st_size} bytes; "
            f"{EXPECTED_INPUT_SHAPE} FLOAT32 requires {EXPECTED_INPUT_BYTES} bytes"
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(dump_tool), str(plan), str(weights), str(feature), str(output_prefix)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise EndToEndError(
            "C Reference Runtime failed\n"
            f"command: {' '.join(completed.args)}\n"
            f"stdout: {completed.stdout.strip()}\n"
            f"stderr: {completed.stderr.strip()}"
        )


def select_checkpoints(comparisons: list[dict]) -> list[dict]:
    """Select and verify the seven fixed graph-boundary tensors."""

    by_key = {
        (int(item["operator_id"]), int(item["tensor_id"])): item
        for item in comparisons
    }
    selected: list[dict] = []
    for checkpoint in CHECKPOINTS:
        item = by_key.get((checkpoint.operator_id, checkpoint.tensor_id))
        if item is None:
            raise EndToEndError(
                f"checkpoint {checkpoint.key} is missing: operator "
                f"#{checkpoint.operator_id}, tensor {checkpoint.tensor_id}"
            )
        if item.get("opcode") != checkpoint.opcode:
            raise EndToEndError(
                f"checkpoint {checkpoint.key} opcode changed: "
                f"expected {checkpoint.opcode}, got {item.get('opcode')}"
            )
        if item.get("tensor_name") != checkpoint.tensor_name:
            raise EndToEndError(
                f"checkpoint {checkpoint.key} tensor changed: "
                f"expected {checkpoint.tensor_name}, got {item.get('tensor_name')}"
            )
        selected.append({"checkpoint": asdict(checkpoint), "comparison": item})
    return selected


def format_first_failure(failure: dict | None) -> str:
    if failure is None:
        return "none"
    metrics = []
    for key in (
        "max_abs_error",
        "mean_abs_error",
        "max_rel_error",
        "cosine_similarity",
        "mismatched_elements",
    ):
        if key in failure:
            metrics.append(f"{key}={failure[key]}")
    suffix = f" ({', '.join(metrics)})" if metrics else ""
    return (
        f"operator #{failure.get('operator_id')} {failure.get('opcode')} "
        f"tensor {failure.get('tensor_id')} '{failure.get('tensor_name')}' "
        f"status={failure.get('status')}{suffix}"
    )


def run_end_to_end(
    *,
    dump_tool: Path,
    plan: Path,
    weights: Path,
    feature: Path,
    ort_dir: Path,
    c_dir: Path,
    skip_runtime: bool = False,
    float_atol: float = 1e-4,
    float_rtol: float = 1e-3,
    embedding_cosine_min: float = 0.999999,
) -> dict:
    """Run 298-frame inference and return a serializable validation report."""

    missing_inputs = [
        str(path) for path in (plan, weights, feature) if not path.is_file()
    ]
    if missing_inputs:
        raise EndToEndError(
            "compiled model/input files are missing:\n  "
            + "\n  ".join(missing_inputs)
        )
    if feature.stat().st_size != EXPECTED_INPUT_BYTES:
        raise EndToEndError(
            f"feature size is {feature.stat().st_size} bytes; "
            f"{EXPECTED_INPUT_SHAPE} FLOAT32 requires {EXPECTED_INPUT_BYTES} bytes"
        )

    missing_sources = [
        str(path) for path in _required_source_files(ort_dir) if not path.is_file()
    ]
    if missing_sources:
        raise EndToEndError(
            "ORT/RuntimeGraph reference files are missing:\n  "
            + "\n  ".join(missing_sources)
        )

    output_prefix = c_dir / "c_298"
    if not skip_runtime:
        _run_c_runtime(dump_tool, plan, weights, feature, output_prefix)
    else:
        missing_dump = [
            str(path)
            for path in (output_prefix.with_suffix(".bin"), output_prefix.with_suffix(".json"))
            if not path.is_file()
        ]
        if missing_dump:
            raise EndToEndError(
                "--skip-runtime requires an existing C dump:\n  "
                + "\n  ".join(missing_dump)
            )

    comparator = _load_comparator()
    c_index, _ = comparator.load_c_dump(output_prefix)
    if int(c_index.get("bucket_frames", -1)) != FRAMES:
        raise EndToEndError(
            f"C dump bucket is {c_index.get('bucket_frames')}, expected {FRAMES}"
        )
    operator_count = int(c_index.get("operator_count", -1))
    executed = int(c_index.get("executed", -1))
    if operator_count != EXPECTED_OPERATOR_COUNT or executed != operator_count:
        raise EndToEndError(
            f"C runtime executed {executed}/{operator_count} operators; "
            f"the canonical graph contains {EXPECTED_OPERATOR_COUNT}"
        )

    summary, comparisons = comparator.compare_bucket(
        FRAMES,
        TAG,
        ort_dir,
        c_dir,
        float_atol=float_atol,
        float_rtol=float_rtol,
    )
    checkpoints = select_checkpoints(comparisons)
    embedding = summary.get("embedding")
    if embedding is None:
        raise EndToEndError("embedding tensor is absent from the comparison")

    embedding_shape_ok = embedding.get("actual_shape") == [1, 192]
    embedding_cosine = float(embedding.get("cosine_similarity", float("nan")))
    embedding_ok = (
        embedding.get("status") in GOOD_STATUSES
        and embedding_shape_ok
        and embedding_cosine >= embedding_cosine_min
    )
    all_checkpoints_match = all(
        entry["comparison"].get("status") in GOOD_STATUSES
        for entry in checkpoints
    )
    comparison_count_ok = int(summary["compared_tensors"]) == operator_count
    passed = bool(
        summary["all_match"]
        and comparison_count_ok
        and all_checkpoints_match
        and embedding_ok
    )

    failures = [
        item for item in comparisons if item.get("status") not in GOOD_STATUSES
    ]
    first_failure = min(
        failures,
        key=lambda item: (int(item["operator_id"]), int(item["tensor_id"])),
        default=None,
    )
    if first_failure is None and not comparison_count_ok:
        first_failure = {
            "operator_id": None,
            "opcode": None,
            "tensor_id": None,
            "tensor_name": None,
            "status": "comparison_count_mismatch",
            "expected_comparisons": operator_count,
            "actual_comparisons": int(summary["compared_tensors"]),
        }
    if first_failure is None and not embedding_ok:
        first_failure = dict(embedding)
        first_failure["status"] = (
            "embedding_shape_mismatch"
            if not embedding_shape_ok
            else "embedding_cosine_below_threshold"
        )

    return {
        "phase": 16,
        "bucket_frames": FRAMES,
        "input": {
            "path": str(feature),
            "dtype": "FLOAT32",
            "shape": list(EXPECTED_INPUT_SHAPE),
            "byte_size": feature.stat().st_size,
        },
        "plan": str(plan),
        "weights": str(weights),
        "operator_count": operator_count,
        "executed_operator_count": executed,
        "compared_tensors": int(summary["compared_tensors"]),
        "failed_tensors": int(summary["failed_tensors"]),
        "float_atol": float_atol,
        "float_rtol": float_rtol,
        "embedding_cosine_min": embedding_cosine_min,
        "embedding": embedding,
        "checkpoints": checkpoints,
        "first_failure": first_failure,
        "passed": passed,
    }


def _print_report(report: dict) -> None:
    print(
        f"[298 frames] {'PASS' if report['passed'] else 'FAIL'} "
        f"operators={report['executed_operator_count']}/{report['operator_count']} "
        f"compared={report['compared_tensors']} failed={report['failed_tensors']}"
    )
    for entry in report["checkpoints"]:
        checkpoint = entry["checkpoint"]
        comparison = entry["comparison"]
        print(
            f"  #{checkpoint['operator_id']:>4} {checkpoint['key']:<19} "
            f"{comparison['status']:<18} "
            f"max_abs={comparison.get('max_abs_error', 0.0):.6e} "
            f"cos={comparison.get('cosine_similarity', 1.0):.9f}"
        )
    print(f"  first failure: {format_first_failure(report['first_failure'])}")


class CheckpointDefinitionTests(unittest.TestCase):
    def test_select_checkpoints_requires_exact_graph_boundaries(self) -> None:
        comparisons = [
            {
                "operator_id": checkpoint.operator_id,
                "opcode": checkpoint.opcode,
                "tensor_id": checkpoint.tensor_id,
                "tensor_name": checkpoint.tensor_name,
                "status": "within_tolerance",
            }
            for checkpoint in CHECKPOINTS
        ]
        selected = select_checkpoints(comparisons)
        self.assertEqual([item["checkpoint"]["key"] for item in selected],
                         [checkpoint.key for checkpoint in CHECKPOINTS])


class ReferenceRuntimeEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dump_tool = _default_dump_tool()
        cls.plan = ROOT / "models" / "compiled" / "reference" / "execution_plans" / "plan_298.bin"
        cls.weights = ROOT / "models" / "compiled" / "reference" / "weights.bin"
        cls.feature = ROOT / "runs" / "runtime" / "ort_reference" / "feature_298.f32"
        cls.ort_dir = ROOT / "runs" / "runtime" / "ort_reference"
        required = (
            cls.dump_tool,
            cls.plan,
            cls.weights,
            cls.feature,
            *_required_source_files(cls.ort_dir),
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "integration artifacts are not present: "
                + ", ".join(str(path) for path in missing)
            )
        cls.temporary = tempfile.TemporaryDirectory(prefix="campp_e2e_")
        cls.report = run_end_to_end(
            dump_tool=cls.dump_tool,
            plan=cls.plan,
            weights=cls.weights,
            feature=cls.feature,
            ort_dir=cls.ort_dir,
            c_dir=Path(cls.temporary.name),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def test_three_second_graph_matches_ort(self) -> None:
        self.assertTrue(
            self.report["passed"],
            format_first_failure(self.report["first_failure"]),
        )

    def test_all_graph_boundaries_match(self) -> None:
        for entry in self.report["checkpoints"]:
            self.assertIn(
                entry["comparison"]["status"],
                GOOD_STATUSES,
                entry["checkpoint"]["key"],
            )

    def test_embedding_shape_and_cosine(self) -> None:
        embedding = self.report["embedding"]
        self.assertEqual(embedding["actual_shape"], [1, 192])
        self.assertGreaterEqual(embedding["cosine_similarity"], 0.999999)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bundle = ROOT / "models" / "compiled" / "reference"
    parser.add_argument("--dump-tool", type=Path, default=_default_dump_tool())
    parser.add_argument(
        "--plan", type=Path,
        default=bundle / "execution_plans" / "plan_298.bin",
    )
    parser.add_argument("--weights", type=Path, default=bundle / "weights.bin")
    parser.add_argument(
        "--feature", type=Path,
        default=ROOT / "runs" / "runtime" / "ort_reference" / "feature_298.f32",
    )
    parser.add_argument(
        "--ort-dir", type=Path,
        default=ROOT / "runs" / "runtime" / "ort_reference",
    )
    parser.add_argument(
        "--c-dir", type=Path,
        default=ROOT / "runs" / "runtime" / "c_reference",
    )
    parser.add_argument(
        "--result", type=Path,
        default=ROOT / "results" / "runtime" / "end_to_end_298.json",
    )
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--float-atol", type=float, default=1e-4)
    parser.add_argument("--float-rtol", type=float, default=1e-3)
    parser.add_argument("--embedding-cosine-min", type=float, default=0.999999)
    args = parser.parse_args(argv)

    try:
        report = run_end_to_end(
            dump_tool=args.dump_tool,
            plan=args.plan,
            weights=args.weights,
            feature=args.feature,
            ort_dir=args.ort_dir,
            c_dir=args.c_dir,
            skip_runtime=args.skip_runtime,
            float_atol=args.float_atol,
            float_rtol=args.float_rtol,
            embedding_cosine_min=args.embedding_cosine_min,
        )
    except (EndToEndError, OSError, ValueError, ImportError) as exc:
        print(f"end-to-end validation could not run: {exc}", file=sys.stderr)
        return 2

    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"  report: {args.result}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
