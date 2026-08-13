#!/usr/bin/env python3
"""Run the same CAM++ C Reference Runtime binary with all four bucket plans.

For each bucket this module executes the complete graph, compares all operator
outputs with ORT, reports fixed graph boundaries, and writes a per-bucket JSON
report.  It also verifies that a plan rejects a feature tensor from a different
bucket with ``BUCKET_MISMATCH``.
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
FEATURE_DIM = 80
FLOAT32_BYTES = 4
EXPECTED_OPERATOR_COUNT = 1438
GOOD_STATUSES = {"exact", "within_tolerance"}


@dataclass(frozen=True)
class BucketSpec:
    frames: int
    tag: str


BUCKETS = (
    BucketSpec(98, "1s"),
    BucketSpec(298, "3s"),
    BucketSpec(498, "5s"),
    BucketSpec(998, "10s"),
)
BUCKET_BY_FRAMES = {bucket.frames: bucket for bucket in BUCKETS}


@dataclass(frozen=True)
class Checkpoint:
    key: str
    description: str
    operator_id: int
    opcode: str
    tensor_id: int
    tensor_name: str


# Topology and Tensor IDs are identical in all four static buckets.  Only Tensor
# shapes and the offsets of bucket-local constants differ between plans.
CHECKPOINTS = (
    Checkpoint(
        "input_transform", "input transpose output", 0, "TRANSPOSE", 2281,
        "/Transpose_output_0",
    ),
    Checkpoint(
        "head", "head output", 51, "QUANTIZE_LINEAR", 2332,
        "/head/Reshape_output_0_quantized",
    ),
    Checkpoint(
        "dense_block_1", "Dense block 1 final concatenation", 366, "CONCAT",
        2647, "/xvector/block1/Concat_11_output_0",
    ),
    Checkpoint(
        "dense_block_2", "Dense block 2 final concatenation", 995, "CONCAT",
        3276, "/xvector/block2/Concat_23_output_0",
    ),
    Checkpoint(
        "dense_block_3", "Dense block 3 final concatenation", 1416, "CONCAT",
        3697, "/xvector/block3/Concat_15_output_0",
    ),
    Checkpoint(
        "statistics_pooling", "statistics pooling concatenated mean/std", 1431,
        "CONCAT", 3712, "/xvector/stats/Concat_output_0",
    ),
    Checkpoint(
        "embedding", "final speaker embedding", 1437, "BATCH_NORMALIZATION",
        3718, "embedding",
    ),
)


class EndToEndError(RuntimeError):
    """The multi-bucket test inputs or generated dumps are inconsistent."""


def bucket_spec(frames: int) -> BucketSpec:
    try:
        return BUCKET_BY_FRAMES[frames]
    except KeyError as exc:
        supported = ", ".join(str(bucket.frames) for bucket in BUCKETS)
        raise EndToEndError(
            f"unsupported bucket {frames}; supported buckets: {supported}"
        ) from exc


def input_shape(frames: int) -> tuple[int, int, int]:
    return (1, frames, FEATURE_DIM)


def expected_input_bytes(frames: int) -> int:
    return frames * FEATURE_DIM * FLOAT32_BYTES


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
            ROOT / "build" / "runtime_phase17" / "campp_reference_dump.exe",
            ROOT / "build" / "runtime_phase15" / "campp_reference_dump.exe",
        ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _plan_path(bundle_dir: Path, frames: int) -> Path:
    return bundle_dir / "execution_plans" / f"plan_{frames}.bin"


def _feature_path(ort_dir: Path, frames: int) -> Path:
    return ort_dir / f"feature_{frames}.f32"


def _required_source_files(bucket: BucketSpec, ort_dir: Path) -> tuple[Path, ...]:
    return (
        ROOT / "results" / "static" / f"campp_static_{bucket.frames}.onnx",
        ROOT / "results" / "graph" / f"ir_{bucket.tag}.json",
        ort_dir / f"ort_{bucket.frames}.npz",
    )


def _validate_feature(feature: Path, frames: int) -> None:
    if not feature.is_file():
        raise EndToEndError(f"feature input is missing: {feature}")
    actual = feature.stat().st_size
    expected = expected_input_bytes(frames)
    if actual != expected:
        raise EndToEndError(
            f"feature size is {actual} bytes; FLOAT32{input_shape(frames)} "
            f"requires {expected} bytes"
        )


def _run_c_runtime(
    bucket: BucketSpec,
    dump_tool: Path,
    plan: Path,
    weights: Path,
    feature: Path,
    output_prefix: Path,
) -> None:
    required = (dump_tool, plan, weights)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise EndToEndError(
            "required runtime files are missing:\n  " + "\n  ".join(missing)
        )
    _validate_feature(feature, bucket.frames)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{bucket.frames} frames] C Reference Runtime start", flush=True)
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
    print(f"[{bucket.frames} frames] C Reference Runtime complete", flush=True)


def verify_bucket_mismatch(
    *,
    plan_bucket: BucketSpec,
    feature_bucket: BucketSpec,
    dump_tool: Path,
    bundle_dir: Path,
    ort_dir: Path,
    output_prefix: Path,
) -> dict:
    """Run one wrong plan/input pair and require BUCKET_MISMATCH."""

    if plan_bucket.frames == feature_bucket.frames:
        raise EndToEndError("bucket mismatch check requires two different buckets")
    plan = _plan_path(bundle_dir, plan_bucket.frames)
    weights = bundle_dir / "weights.bin"
    feature = _feature_path(ort_dir, feature_bucket.frames)
    required = (dump_tool, plan, weights, feature)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise EndToEndError(
            "bucket mismatch inputs are missing:\n  " + "\n  ".join(missing)
        )
    _validate_feature(feature, feature_bucket.frames)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(dump_tool), str(plan), str(weights), str(feature), str(output_prefix)],
        check=False,
        capture_output=True,
        text=True,
    )
    message = "\n".join((completed.stdout, completed.stderr)).strip()
    rejected = completed.returncode != 0 and "BUCKET_MISMATCH" in message
    if not rejected:
        raise EndToEndError(
            f"plan_{plan_bucket.frames} accepted feature_{feature_bucket.frames} "
            "or returned the wrong error\n"
            f"return code: {completed.returncode}\noutput: {message}"
        )
    return {
        "plan_frames": plan_bucket.frames,
        "feature_frames": feature_bucket.frames,
        "status": "BUCKET_MISMATCH",
        "rejected": True,
    }


def select_checkpoints(comparisons: list[dict]) -> list[dict]:
    """Select and verify the seven stable graph-boundary tensors."""

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
    bucket: BucketSpec,
    dump_tool: Path,
    bundle_dir: Path,
    ort_dir: Path,
    c_dir: Path,
    skip_runtime: bool = False,
    float_atol: float = 1e-4,
    float_rtol: float = 1e-3,
    embedding_cosine_min: float = 0.999999,
) -> dict:
    """Run one bucket and return its serializable numerical report."""

    plan = _plan_path(bundle_dir, bucket.frames)
    weights = bundle_dir / "weights.bin"
    feature = _feature_path(ort_dir, bucket.frames)
    missing_inputs = [
        str(path) for path in (plan, weights, feature) if not path.is_file()
    ]
    if missing_inputs:
        raise EndToEndError(
            "compiled model/input files are missing:\n  "
            + "\n  ".join(missing_inputs)
        )
    _validate_feature(feature, bucket.frames)

    missing_sources = [
        str(path)
        for path in _required_source_files(bucket, ort_dir)
        if not path.is_file()
    ]
    if missing_sources:
        raise EndToEndError(
            "ORT/RuntimeGraph reference files are missing:\n  "
            + "\n  ".join(missing_sources)
        )

    output_prefix = c_dir / f"c_{bucket.frames}"
    if not skip_runtime:
        _run_c_runtime(bucket, dump_tool, plan, weights, feature, output_prefix)
    else:
        missing_dump = [
            str(path)
            for path in (
                output_prefix.with_suffix(".bin"),
                output_prefix.with_suffix(".json"),
            )
            if not path.is_file()
        ]
        if missing_dump:
            raise EndToEndError(
                "--skip-runtime requires an existing C dump:\n  "
                + "\n  ".join(missing_dump)
            )

    comparator = _load_comparator()
    c_index, _ = comparator.load_c_dump(output_prefix)
    if int(c_index.get("bucket_frames", -1)) != bucket.frames:
        raise EndToEndError(
            f"C dump bucket is {c_index.get('bucket_frames')}, "
            f"expected {bucket.frames}"
        )
    operator_count = int(c_index.get("operator_count", -1))
    executed = int(c_index.get("executed", -1))
    if operator_count != EXPECTED_OPERATOR_COUNT or executed != operator_count:
        raise EndToEndError(
            f"C runtime executed {executed}/{operator_count} operators; "
            f"the canonical graph contains {EXPECTED_OPERATOR_COUNT}"
        )

    summary, comparisons = comparator.compare_bucket(
        bucket.frames,
        bucket.tag,
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
    numerical_passed = bool(
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
        "phase": 17,
        "bucket_frames": bucket.frames,
        "bucket_tag": bucket.tag,
        "input": {
            "path": str(feature),
            "dtype": "FLOAT32",
            "shape": list(input_shape(bucket.frames)),
            "byte_size": feature.stat().st_size,
        },
        "plan": str(plan),
        "weights": str(weights),
        "runtime_binary": str(dump_tool),
        "operator_count": operator_count,
        "executed_operator_count": executed,
        "execution_passed": executed == operator_count,
        "compared_tensors": int(summary["compared_tensors"]),
        "failed_tensors": int(summary["failed_tensors"]),
        "float_atol": float_atol,
        "float_rtol": float_rtol,
        "embedding_cosine_min": embedding_cosine_min,
        "embedding": embedding,
        "checkpoints": checkpoints,
        "first_failure": first_failure,
        "numerical_passed": numerical_passed,
        # Backward-compatible name used by the former Phase 16 report.
        "passed": numerical_passed,
    }


def _print_report(report: dict) -> None:
    frames = report["bucket_frames"]
    print(
        f"[{frames} frames] "
        f"{'PASS' if report['numerical_passed'] else 'NUMERICAL FAIL'} "
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


def _cyclic_mismatch_pairs(
    buckets: tuple[BucketSpec, ...],
) -> tuple[tuple[BucketSpec, BucketSpec], ...]:
    if len(buckets) < 2:
        return ()
    return tuple(
        (bucket, buckets[(index + 1) % len(buckets)])
        for index, bucket in enumerate(buckets)
    )


class BucketDefinitionTests(unittest.TestCase):
    def test_all_supported_input_sizes_are_distinct(self) -> None:
        self.assertEqual(
            len({expected_input_bytes(bucket.frames) for bucket in BUCKETS}),
            len(BUCKETS),
        )

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
        self.assertEqual(
            [item["checkpoint"]["key"] for item in selected],
            [checkpoint.key for checkpoint in CHECKPOINTS],
        )

    def test_cyclic_mismatch_pairs_cover_every_plan_once(self) -> None:
        pairs = _cyclic_mismatch_pairs(BUCKETS)
        self.assertEqual(
            {plan.frames for plan, _feature in pairs},
            {bucket.frames for bucket in BUCKETS},
        )
        self.assertTrue(
            all(plan.frames != feature.frames for plan, feature in pairs)
        )


class ReferenceRuntimeEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dump_tool = _default_dump_tool()
        cls.bundle_dir = ROOT / "models" / "compiled" / "reference"
        cls.ort_dir = ROOT / "runs" / "runtime" / "ort_reference"
        required = [cls.dump_tool, cls.bundle_dir / "weights.bin"]
        for bucket in BUCKETS:
            required.extend(
                (
                    _plan_path(cls.bundle_dir, bucket.frames),
                    _feature_path(cls.ort_dir, bucket.frames),
                    *_required_source_files(bucket, cls.ort_dir),
                )
            )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "multi-bucket integration artifacts are not present: "
                + ", ".join(str(path) for path in missing)
            )

        cls.temporary = tempfile.TemporaryDirectory(prefix="campp_e2e_buckets_")
        cls.c_dir = Path(cls.temporary.name) / "c"
        cls.reports = [
            run_end_to_end(
                bucket=bucket,
                dump_tool=cls.dump_tool,
                bundle_dir=cls.bundle_dir,
                ort_dir=cls.ort_dir,
                c_dir=cls.c_dir,
            )
            for bucket in BUCKETS
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def test_same_binary_executes_all_four_plans(self) -> None:
        self.assertEqual(
            {report["bucket_frames"] for report in self.reports},
            {bucket.frames for bucket in BUCKETS},
        )
        self.assertEqual(
            {report["runtime_binary"] for report in self.reports},
            {str(self.dump_tool)},
        )
        for report in self.reports:
            self.assertTrue(report["execution_passed"], report["bucket_frames"])

    def test_every_bucket_has_a_complete_comparison_report(self) -> None:
        for report in self.reports:
            self.assertEqual(report["compared_tensors"], EXPECTED_OPERATOR_COUNT)
            self.assertEqual(len(report["checkpoints"]), len(CHECKPOINTS))
            self.assertEqual(report["embedding"]["actual_shape"], [1, 192])

    def test_wrong_bucket_inputs_are_rejected(self) -> None:
        for plan_bucket, feature_bucket in _cyclic_mismatch_pairs(BUCKETS):
            with self.subTest(
                plan=plan_bucket.frames, feature=feature_bucket.frames
            ):
                result = verify_bucket_mismatch(
                    plan_bucket=plan_bucket,
                    feature_bucket=feature_bucket,
                    dump_tool=self.dump_tool,
                    bundle_dir=self.bundle_dir,
                    ort_dir=self.ort_dir,
                    output_prefix=(
                        Path(self.temporary.name)
                        / f"wrong_{plan_bucket.frames}_{feature_bucket.frames}"
                    ),
                )
                self.assertTrue(result["rejected"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_bundle = ROOT / "models" / "compiled" / "reference"
    parser.add_argument("--dump-tool", type=Path, default=_default_dump_tool())
    parser.add_argument("--bundle-dir", type=Path, default=default_bundle)
    parser.add_argument(
        "--ort-dir", type=Path,
        default=ROOT / "runs" / "runtime" / "ort_reference",
    )
    parser.add_argument(
        "--c-dir", type=Path,
        default=ROOT / "runs" / "runtime" / "c_reference",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=ROOT / "results" / "runtime" / "c_runtime_compare",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        default=[bucket.frames for bucket in BUCKETS],
    )
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-cross-bucket-check", action="store_true")
    parser.add_argument("--float-atol", type=float, default=1e-4)
    parser.add_argument("--float-rtol", type=float, default=1e-3)
    parser.add_argument("--embedding-cosine-min", type=float, default=0.999999)
    args = parser.parse_args(argv)

    try:
        selected = tuple(bucket_spec(frames) for frames in args.buckets)
    except EndToEndError as exc:
        print(f"end-to-end validation could not run: {exc}", file=sys.stderr)
        return 2
    if len({bucket.frames for bucket in selected}) != len(selected):
        print("each --buckets value must be unique", file=sys.stderr)
        return 2

    args.results_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    errors: list[dict] = []
    for bucket in selected:
        try:
            report = run_end_to_end(
                bucket=bucket,
                dump_tool=args.dump_tool,
                bundle_dir=args.bundle_dir,
                ort_dir=args.ort_dir,
                c_dir=args.c_dir,
                skip_runtime=args.skip_runtime,
                float_atol=args.float_atol,
                float_rtol=args.float_rtol,
                embedding_cosine_min=args.embedding_cosine_min,
            )
        except (EndToEndError, OSError, ValueError, ImportError) as exc:
            print(f"[{bucket.frames} frames] ERROR: {exc}", file=sys.stderr)
            errors.append({"bucket_frames": bucket.frames, "error": str(exc)})
            continue
        reports.append(report)
        result_path = args.results_dir / f"end_to_end_{bucket.frames}.json"
        result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _print_report(report)
        print(f"  report: {result_path}")

    mismatch_results: list[dict] = []
    mismatch_errors: list[dict] = []
    if not args.skip_cross_bucket_check:
        for plan_bucket, feature_bucket in _cyclic_mismatch_pairs(selected):
            try:
                result = verify_bucket_mismatch(
                    plan_bucket=plan_bucket,
                    feature_bucket=feature_bucket,
                    dump_tool=args.dump_tool,
                    bundle_dir=args.bundle_dir,
                    ort_dir=args.ort_dir,
                    output_prefix=(
                        args.c_dir
                        / f"wrong_{plan_bucket.frames}_{feature_bucket.frames}"
                    ),
                )
            except (EndToEndError, OSError, ValueError) as exc:
                print(
                    f"[plan {plan_bucket.frames} / feature "
                    f"{feature_bucket.frames}] ERROR: {exc}",
                    file=sys.stderr,
                )
                mismatch_errors.append(
                    {
                        "plan_frames": plan_bucket.frames,
                        "feature_frames": feature_bucket.frames,
                        "error": str(exc),
                    }
                )
                continue
            mismatch_results.append(result)
            print(
                f"[plan {plan_bucket.frames} / feature {feature_bucket.frames}] "
                "BUCKET_MISMATCH"
            )

    selected_execution_passed = (
        len(reports) == len(selected)
        and not errors
        and all(report["execution_passed"] for report in reports)
    )
    cross_check_expected = len(_cyclic_mismatch_pairs(selected))
    cross_check_passed = (
        not args.skip_cross_bucket_check
        and len(mismatch_results) == cross_check_expected
        and not mismatch_errors
    )
    all_four_selected = {bucket.frames for bucket in selected} == {
        bucket.frames for bucket in BUCKETS
    }
    phase17_passed = bool(
        all_four_selected and selected_execution_passed and cross_check_passed
    )
    summary = {
        "phase": 17,
        "runtime_binary": str(args.dump_tool),
        "shared_weights": str(args.bundle_dir / "weights.bin"),
        "selected_buckets": [bucket.frames for bucket in selected],
        "bucket_results": [
            {
                "bucket_frames": report["bucket_frames"],
                "execution_passed": report["execution_passed"],
                "numerical_passed": report["numerical_passed"],
                "failed_tensors": report["failed_tensors"],
                "embedding_cosine_similarity": report["embedding"].get(
                    "cosine_similarity"
                ),
            }
            for report in reports
        ],
        "errors": errors,
        "bucket_mismatch_checks": mismatch_results,
        "bucket_mismatch_errors": mismatch_errors,
        "selected_execution_passed": selected_execution_passed,
        "cross_bucket_rejection_passed": cross_check_passed,
        "cross_bucket_rejection_skipped": args.skip_cross_bucket_check,
        "all_numerically_passed": bool(
            reports and all(report["numerical_passed"] for report in reports)
        ),
        "phase17_passed": phase17_passed,
    }
    summary_path = args.results_dir / "end_to_end_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(
        "Phase 17: "
        f"{'PASS' if phase17_passed else 'INCOMPLETE'} "
        f"executed={len(reports)}/{len(selected)} "
        f"cross_bucket={'PASS' if cross_check_passed else 'FAIL'} "
        f"numerical={'PASS' if summary['all_numerically_passed'] else 'CHECK'}"
    )
    print(f"summary: {summary_path}")
    return 0 if phase17_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
