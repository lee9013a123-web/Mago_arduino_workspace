#!/usr/bin/env python3
"""E7 Bucket 98의 kernel exclusive time을 측정하고 Top 80%를 산출한다."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src" / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_bundle_exporter.format.binary_format_schema import (  # noqa: E402
    OperatorCode,
    TensorDType,
    TensorFlags,
    TensorStorageType,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
    LoadedPlan,
    read_execution_plan,
)

from profile_statistics import rank_operator_samples, summarize_ns  # noqa: E402


EXPECTED_INPUT_IDS = (
    "multi__speaker_0000",
    "multi__speaker_0005",
    "multi__speaker_0006",
)
EXPECTED_BUCKET = 98
EXPECTED_WARMUP = 20
EXPECTED_REPEAT = 100
EXPECTED_THREADS = 1
TOP_THRESHOLD_PCT = 80.0


def _load_benchmark_module() -> Any:
    path = ROOT / "scripts" / "3_runtime" / "09_benchmark_runtime.py"
    spec = importlib.util.spec_from_file_location("runtime_benchmark_09", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


class ProfileError(RuntimeError):
    """Profiling protocol이나 raw 결과가 유효하지 않을 때 발생한다."""


def _display_path(path: Path) -> str:
    """저장소 내부 경로는 결과 파일에 portable POSIX 상대경로로 기록한다."""

    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise ProfileError(f"{name} not found: {path}")


def _official_protocol(config: Any) -> None:
    mismatches: list[str] = []
    if config.buckets != {EXPECTED_BUCKET: 1.0}:
        mismatches.append(f"buckets={config.buckets!r}")
    if tuple(config.input_ids) != EXPECTED_INPUT_IDS:
        mismatches.append(f"input_ids={tuple(config.input_ids)!r}")
    if config.threads != EXPECTED_THREADS:
        mismatches.append(f"threads={config.threads}")
    if config.warmup != EXPECTED_WARMUP:
        mismatches.append(f"warmup={config.warmup}")
    if config.repeat != EXPECTED_REPEAT:
        mismatches.append(f"repeat={config.repeat}")
    if tuple(config.environment.affinity) != (0,):
        mismatches.append(f"affinity={tuple(config.environment.affinity)!r}")
    if mismatches:
        raise ProfileError(
            "E7 official profiling protocol mismatch: " + ", ".join(mismatches)
        )


def _load_e7_evidence(path: Path) -> dict[str, Any]:
    _require_file(path, "E7 validation evidence")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("bucket_frames") != EXPECTED_BUCKET:
        raise ProfileError("E7 validation evidence is not for bucket 98")
    if document.get("all_bitwise_identical") is not True:
        raise ProfileError("E7 validation evidence is not bitwise-identical")
    return {
        "path": _display_path(path),
        "sha256": BENCHMARK.sha256_file(path),
        "bucket_frames": document["bucket_frames"],
        "all_bitwise_identical": True,
        "evaluation_features": document.get("evaluation_features", []),
    }


def _selected_features(config: Any) -> tuple[list[Any], dict[str, Any]]:
    selected = BENCHMARK.selected_dataset_inputs(config.paths.dataset_manifest)
    requested = set(config.input_ids)
    selected = {
        input_id: row
        for input_id, row in selected.items()
        if input_id in requested
    }
    if set(selected) != requested:
        missing = sorted(requested - set(selected))
        raise ProfileError(f"dataset manifest is missing selected inputs: {missing}")
    all_features, feature_document = BENCHMARK.load_feature_manifest(
        config.paths.feature_manifest
    )
    filtered = [
        feature
        for feature in all_features
        if feature.bucket_frames == EXPECTED_BUCKET
        and feature.input_id in requested
    ]
    features = BENCHMARK.validate_features(config, selected, filtered)
    if tuple(feature.input_id for feature in features) != EXPECTED_INPUT_IDS:
        raise ProfileError("validated feature order or membership is not fixed")
    source_wavs = (
        BENCHMARK.verify_source_wavs(config.paths.data_root, selected)
        if config.verify_source_wavs
        else {"skipped": True}
    )
    return features, {
        "feature_manifest": _display_path(config.paths.feature_manifest),
        "feature_manifest_sha256": BENCHMARK.sha256_file(
            config.paths.feature_manifest
        ),
        "dataset_manifest": _display_path(config.paths.dataset_manifest),
        "dataset_manifest_sha256": BENCHMARK.sha256_file(
            config.paths.dataset_manifest
        ),
        "preprocessing": feature_document.get("preprocessing"),
        "source_wavs": source_wavs,
    }


def _profile_command(
    binary: Path,
    *,
    plan: Path,
    weights: Path,
    feature: Any,
    warmup: int,
    repeat: int,
    threads: int,
) -> list[str]:
    return [
        str(binary),
        "--plan",
        str(plan),
        "--weights",
        str(weights),
        "--input",
        str(feature.path),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--threads",
        str(threads),
    ]


def _validate_profiler_payload(
    payload: dict[str, Any], plan: LoadedPlan, repeat: int
) -> None:
    if payload.get("measurement_scope") != "kernel_run_exclusive":
        raise ProfileError("profiler did not report kernel exclusive scope")
    if payload.get("clock") != "CLOCK_MONOTONIC_RAW":
        raise ProfileError("profiler clock is not CLOCK_MONOTONIC_RAW")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ProfileError("profiler configuration is missing")
    if configuration.get("effective_threads") != EXPECTED_THREADS:
        raise ProfileError("profiler effective_threads is not 1")
    if configuration.get("repeat") != repeat:
        raise ProfileError("profiler repeat does not match the experiment")
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("bucket_frames") != EXPECTED_BUCKET:
        raise ProfileError("profiler model bucket is not 98")
    if model.get("operator_count") != len(plan.operators):
        raise ProfileError("profiler operator count does not match the plan")
    end_to_end = payload.get("end_to_end_ns")
    if (
        not isinstance(end_to_end, list)
        or len(end_to_end) != repeat
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in end_to_end)
    ):
        raise ProfileError("invalid end-to-end samples")
    operators = payload.get("operators")
    if not isinstance(operators, list) or len(operators) != len(plan.operators):
        raise ProfileError("invalid profiler operator list")
    for expected, raw in zip(plan.operators, operators):
        if not isinstance(raw, dict):
            raise ProfileError("profiler operator entry is not an object")
        if raw.get("operator_id") != expected.operator_id:
            raise ProfileError("profiler operator IDs are not dense and ordered")
        if raw.get("kernel_id") != expected.kernel_id:
            raise ProfileError(
                f"kernel ID mismatch at operator {expected.operator_id}"
            )
        if not isinstance(raw.get("kernel_name"), str) or not raw["kernel_name"]:
            raise ProfileError("profiler kernel name is missing")
        samples = raw.get("samples_ns")
        if (
            raw.get("call_count") != repeat
            or not isinstance(samples, list)
            or len(samples) != repeat
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in samples)
        ):
            raise ProfileError(
                f"incomplete samples for operator {expected.operator_id}"
            )


def _enum_name(enum_type: Any, value: int) -> str:
    try:
        return enum_type(value).name
    except ValueError:
        return f"UNKNOWN_{value}"


def _tensor_record(descriptor: Any) -> dict[str, Any]:
    return {
        "tensor_id": descriptor.tensor_id,
        "dtype": _enum_name(TensorDType, int(descriptor.dtype)),
        "rank": descriptor.rank,
        "shape": list(descriptor.dimensions[: descriptor.rank]),
        "byte_strides": list(descriptor.byte_strides[: descriptor.rank]),
        "storage_type": _enum_name(
            TensorStorageType, int(descriptor.storage_type)
        ),
        "logical_byte_size": descriptor.logical_byte_size,
        "storage_span_bytes": descriptor.storage_span_bytes,
        "packed_qconv_o4i4": bool(
            int(descriptor.flags) & int(TensorFlags.PACKED_QCONV_O4I4)
        ),
    }


def _fusion_index(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    entries = document.get("fusions")
    if not isinstance(entries, list):
        raise ProfileError("fusion_98.json does not contain fusions")
    result: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(
            entry.get("fused_operator_id"), int
        ):
            raise ProfileError("invalid fusion entry")
        operator_id = int(entry["fused_operator_id"])
        if operator_id in result:
            raise ProfileError(f"duplicate fused operator ID: {operator_id}")
        result[operator_id] = entry
    return result


def _build_operator_profile(
    plan: LoadedPlan,
    fusion_document: dict[str, Any],
    raw_payloads: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples_by_operator: dict[int, list[int]] = {
        descriptor.operator_id: [] for descriptor in plan.operators
    }
    kernel_names: dict[int, str] = {}
    end_to_end_samples: list[int] = []
    for payload in raw_payloads:
        end_to_end_samples.extend(payload["end_to_end_ns"])
        for raw in payload["operators"]:
            operator_id = int(raw["operator_id"])
            samples_by_operator[operator_id].extend(raw["samples_ns"])
            previous_name = kernel_names.setdefault(
                operator_id, str(raw["kernel_name"])
            )
            if previous_name != raw["kernel_name"]:
                raise ProfileError(
                    f"kernel name changed for operator {operator_id}"
                )

    ranked, timing_summary = rank_operator_samples(
        samples_by_operator,
        end_to_end_samples,
        threshold_pct=TOP_THRESHOLD_PCT,
    )
    ranked_by_id = {int(item["operator_id"]): item for item in ranked}
    fusions = _fusion_index(fusion_document)
    operator_profiles: list[dict[str, Any]] = []
    for timing in ranked:
        operator_id = int(timing["operator_id"])
        descriptor = plan.operators[operator_id]
        input_ids = descriptor.input_tensor_ids[: descriptor.input_count]
        output_ids = descriptor.output_tensor_ids[: descriptor.output_count]
        input_tensors = [_tensor_record(plan.tensors[tensor_id]) for tensor_id in input_ids]
        output_tensors = [
            _tensor_record(plan.tensors[tensor_id]) for tensor_id in output_ids
        ]
        constant_inputs = [
            tensor
            for tensor in input_tensors
            if tensor["storage_type"] == TensorStorageType.CONSTANT.name
        ]
        fusion = fusions.get(operator_id)
        operator_profiles.append(
            {
                **timing,
                "kernel_id": descriptor.kernel_id,
                "kernel_name": kernel_names[operator_id],
                "operator_type": _enum_name(OperatorCode, int(descriptor.opcode)),
                "fusion_family": fusion.get("family") if fusion else None,
                "fusion_pattern": fusion.get("pattern") if fusion else None,
                "original_operator_ids": (
                    list(fusion.get("original_operator_ids", []))
                    if fusion
                    else [operator_id]
                ),
                "input_tensors": input_tensors,
                "output_tensors": output_tensors,
                "constant_inputs": constant_inputs,
                "weight_shapes": [item["shape"] for item in constant_inputs],
                "samples_ns": samples_by_operator[operator_id],
            }
        )
    if set(ranked_by_id) != set(range(len(plan.operators))):
        raise ProfileError("ranked result does not cover every operator")
    timing_summary["end_to_end"] = summarize_ns(end_to_end_samples)
    return operator_profiles, timing_summary


def _overhead_document(
    comparisons: Sequence[dict[str, Any]], threshold_pct: float = 1.0
) -> dict[str, Any]:
    baseline_total_ns = sum(int(item["baseline_total_ns"]) for item in comparisons)
    profiled_total_ns = sum(int(item["profiled_total_ns"]) for item in comparisons)
    if baseline_total_ns <= 0:
        raise ProfileError("baseline total time must be positive")
    overhead_pct = (profiled_total_ns / baseline_total_ns - 1.0) * 100.0
    return {
        "schema_version": 1,
        "definition": (
            "(profiled_end_to_end / baseline_end_to_end - 1) * 100"
        ),
        "threshold_pct": threshold_pct,
        "overhead_pct": overhead_pct,
        "passes": overhead_pct < threshold_pct,
        "baseline_total_ns": baseline_total_ns,
        "profiled_total_ns": profiled_total_ns,
        "per_input": list(comparisons),
    }


def _write_csv(path: Path, operators: Sequence[dict[str, Any]]) -> None:
    fields = (
        "rank",
        "operator_id",
        "operator_type",
        "kernel_id",
        "kernel_name",
        "fusion_family",
        "fusion_pattern",
        "input_shapes",
        "output_shapes",
        "weight_shapes",
        "call_count",
        "total_exclusive_ms",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "end_to_end_share_pct",
        "cumulative_end_to_end_share_pct",
        "operator_exclusive_share_pct",
        "cumulative_operator_exclusive_share_pct",
        "in_top_bottleneck_set",
    )
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for item in operators:
            writer.writerow(
                {
                    "rank": item["rank"],
                    "operator_id": item["operator_id"],
                    "operator_type": item["operator_type"],
                    "kernel_id": item["kernel_id"],
                    "kernel_name": item["kernel_name"],
                    "fusion_family": item["fusion_family"],
                    "fusion_pattern": item["fusion_pattern"],
                    "input_shapes": json.dumps(
                        [tensor["shape"] for tensor in item["input_tensors"]],
                        separators=(",", ":"),
                    ),
                    "output_shapes": json.dumps(
                        [tensor["shape"] for tensor in item["output_tensors"]],
                        separators=(",", ":"),
                    ),
                    "weight_shapes": json.dumps(
                        item["weight_shapes"], separators=(",", ":")
                    ),
                    "call_count": item["call_count"],
                    "total_exclusive_ms": item["total_exclusive_ms"],
                    "mean_ms": item["mean_ms"],
                    "p50_ms": item["p50_ms"],
                    "p95_ms": item["p95_ms"],
                    "end_to_end_share_pct": item["end_to_end_share_pct"],
                    "cumulative_end_to_end_share_pct": item[
                        "cumulative_end_to_end_share_pct"
                    ],
                    "operator_exclusive_share_pct": item[
                        "operator_exclusive_share_pct"
                    ],
                    "cumulative_operator_exclusive_share_pct": item[
                        "cumulative_operator_exclusive_share_pct"
                    ],
                    "in_top_bottleneck_set": item[
                        "in_top_bottleneck_set"
                    ],
                }
            )


def _ensure_targets_available(paths: Sequence[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = ", ".join(str(path) for path in existing)
        raise ProfileError(f"output already exists; use --force: {rendered}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "benchmark" / "runtime_e7_98.json",
    )
    parser.add_argument(
        "--baseline-binary",
        type=Path,
        default=ROOT / "build" / "profill" / "campp_runtime_benchmark",
    )
    parser.add_argument(
        "--profiler-binary",
        type=Path,
        default=ROOT / "build" / "profill" / "campp_e7_profiler",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "runs" / "profiling" / "e7_98" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "profiling" / "e7_98",
    )
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = BENCHMARK.load_config(args.config, repository_root=ROOT)
        _official_protocol(config)
        baseline_binary = BENCHMARK.executable_path(args.baseline_binary)
        profiler_binary = BENCHMARK.executable_path(args.profiler_binary)
        _require_file(baseline_binary, "baseline binary")
        _require_file(profiler_binary, "profiler binary")
        plan_path = (
            config.paths.arena_bundle
            / "execution_plans"
            / f"plan_{EXPECTED_BUCKET}.bin"
        )
        weights_path = config.paths.arena_bundle / "weights.bin"
        fusion_path = (
            config.paths.arena_bundle
            / "fusion_plans"
            / f"fusion_{EXPECTED_BUCKET}.json"
        )
        for path, name in (
            (plan_path, "E7 plan"),
            (weights_path, "E7 weights"),
            (fusion_path, "E7 fusion plan"),
        ):
            _require_file(path, name)
        plan = read_execution_plan(plan_path)
        if plan.header.bucket_frames != EXPECTED_BUCKET:
            raise ProfileError("execution plan bucket is not 98")
        fusion_document = json.loads(fusion_path.read_text(encoding="utf-8"))
        if fusion_document.get("bucket_frames") != EXPECTED_BUCKET:
            raise ProfileError("fusion plan bucket is not 98")
        evidence = _load_e7_evidence(config.paths.verification_result)
        features, dataset = _selected_features(config)
        environment_before, environment_mismatches = (
            BENCHMARK.apply_and_validate_environment(
                config.environment,
                allow_mismatch=args.allow_environment_mismatch,
            )
        )
        capabilities, _ = BENCHMARK.run_json_command(
            [str(profiler_binary), "--capabilities"],
            environment=os.environ.copy(),
        )
        if capabilities.get("effective_threads") != EXPECTED_THREADS:
            raise ProfileError("profiler capability does not report one thread")
        if capabilities.get("clock") != "CLOCK_MONOTONIC_RAW":
            raise ProfileError("profiler capability does not provide raw clock")
    except (
        BENCHMARK.BenchmarkError,
        ProfileError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"E7 profiling preflight failed: {exc}", file=sys.stderr)
        return 2

    if args.preflight_only:
        print("E7 profiling preflight: PASS")
        print(f"  operators: {len(plan.operators)}")
        print(f"  inputs: {len(features)}")
        print(f"  environment mismatches: {environment_mismatches or 'none'}")
        return 0

    raw_targets = [args.raw_dir / f"{feature.input_id}.json" for feature in features]
    overhead_path = args.raw_dir / "overhead_comparison.json"
    result_targets = [
        args.output_dir / "operator_profile.json",
        args.output_dir / "operator_profile.csv",
        args.output_dir / "summary.json",
    ]
    try:
        _ensure_targets_available(
            [*raw_targets, overhead_path, *result_targets], force=args.force
        )
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        base_environment = os.environ.copy()
        base_environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "ORT_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        raw_payloads: list[dict[str, Any]] = []
        overhead_comparisons: list[dict[str, Any]] = []
        per_input: list[dict[str, Any]] = []
        for index, feature in enumerate(features):
            print(f"[{index + 1}/{len(features)}] {feature.input_id}", flush=True)
            order = (
                ("baseline", "profile")
                if index % 2 == 0
                else ("profile", "baseline")
            )
            baseline_payload: dict[str, Any] | None = None
            profile_payload: dict[str, Any] | None = None
            for mode in order:
                BENCHMARK.wait_for_thermal_start(config.environment)
                print(f"  {mode}", flush=True)
                if mode == "baseline":
                    command = BENCHMARK.c_command(
                        baseline_binary,
                        plan=plan_path,
                        weights=weights_path,
                        feature=feature,
                        threads=config.threads,
                        warmup=config.warmup,
                        repeat=config.repeat,
                    )
                    baseline_payload, _ = BENCHMARK.run_json_command(
                        command, environment=base_environment
                    )
                else:
                    command = _profile_command(
                        profiler_binary,
                        plan=plan_path,
                        weights=weights_path,
                        feature=feature,
                        warmup=config.warmup,
                        repeat=config.repeat,
                        threads=config.threads,
                    )
                    profile_payload, _ = BENCHMARK.run_json_command(
                        command, environment=base_environment
                    )
            if baseline_payload is None or profile_payload is None:
                raise ProfileError("baseline/profile pair is incomplete")
            _validate_profiler_payload(profile_payload, plan, config.repeat)
            baseline_timings_ms = baseline_payload.get("warm", {}).get(
                "timings_ms"
            )
            if (
                not isinstance(baseline_timings_ms, list)
                or len(baseline_timings_ms) != config.repeat
                or any(not isinstance(value, (int, float)) or value <= 0
                       for value in baseline_timings_ms)
            ):
                raise ProfileError("baseline returned invalid warm timings")
            baseline_total_ns = round(sum(baseline_timings_ms) * 1_000_000.0)
            profiled_total_ns = sum(profile_payload["end_to_end_ns"])
            input_overhead_pct = (
                profiled_total_ns / baseline_total_ns - 1.0
            ) * 100.0
            raw_document = {
                **profile_payload,
                "input_id": feature.input_id,
                "feature_sha256": feature.sha256,
            }
            raw_path = args.raw_dir / f"{feature.input_id}.json"
            raw_path.write_text(
                json.dumps(raw_document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raw_payloads.append(raw_document)
            overhead_comparisons.append(
                {
                    "input_id": feature.input_id,
                    "baseline_total_ns": baseline_total_ns,
                    "profiled_total_ns": profiled_total_ns,
                    "overhead_pct": input_overhead_pct,
                }
            )
            per_input.append(
                {
                    "input_id": feature.input_id,
                    "feature_path": _display_path(feature.path),
                    "feature_sha256": feature.sha256,
                    "raw_profile": _display_path(raw_path),
                    "end_to_end": summarize_ns(profile_payload["end_to_end_ns"]),
                }
            )

        overhead = _overhead_document(overhead_comparisons)
        overhead_path.write_text(
            json.dumps(overhead, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        operators, timing_summary = _build_operator_profile(
            plan, fusion_document, raw_payloads
        )
        expected_call_count = len(features) * config.repeat
        all_call_counts_valid = all(
            item["call_count"] == expected_call_count for item in operators
        )
        generated_at = datetime.now(timezone.utc).isoformat()
        artifacts = {
            "plan": {
                "path": _display_path(plan_path),
                "sha256": BENCHMARK.sha256_file(plan_path),
            },
            "weights": {
                "path": _display_path(weights_path),
                "sha256": BENCHMARK.sha256_file(weights_path),
            },
            "fusion_plan": {
                "path": _display_path(fusion_path),
                "sha256": BENCHMARK.sha256_file(fusion_path),
            },
            "baseline_binary": {
                "path": _display_path(baseline_binary),
                "sha256": BENCHMARK.sha256_file(baseline_binary),
            },
            "profiler_binary": {
                "path": _display_path(profiler_binary),
                "sha256": BENCHMARK.sha256_file(profiler_binary),
            },
        }
        configuration = {
            "config": _display_path(config.source),
            "bucket_frames": EXPECTED_BUCKET,
            "threads": config.threads,
            "cpu_affinity": list(config.environment.affinity),
            "warmup": config.warmup,
            "repeat": config.repeat,
            "input_ids": list(EXPECTED_INPUT_IDS),
            "expected_call_count_per_operator": expected_call_count,
            "measurement_scope": "kernel_run_exclusive",
        }
        operator_profile = {
            "schema_version": 1,
            "generated_at_utc": generated_at,
            "configuration": configuration,
            "artifacts": artifacts,
            "validation_evidence": evidence,
            "dataset": dataset,
            "environment_before": environment_before,
            "environment_mismatches": environment_mismatches,
            "per_input": per_input,
            "timing": timing_summary,
            "operators": operators,
        }
        summary = {
            "schema_version": 1,
            "generated_at_utc": generated_at,
            "configuration": configuration,
            "artifacts": artifacts,
            "overhead": overhead,
            "timing": timing_summary,
            "top_bottleneck_set": {
                **{
                    key: timing_summary[key]
                    for key in (
                        "threshold_pct",
                        "complete",
                        "operator_ids",
                        "operator_count",
                    )
                },
                "operators": [
                    {
                        "rank": item["rank"],
                        "operator_id": item["operator_id"],
                        "operator_type": item["operator_type"],
                        "kernel_id": item["kernel_id"],
                        "kernel_name": item["kernel_name"],
                        "total_exclusive_ms": item["total_exclusive_ms"],
                        "end_to_end_share_pct": item["end_to_end_share_pct"],
                        "cumulative_end_to_end_share_pct": item[
                            "cumulative_end_to_end_share_pct"
                        ],
                    }
                    for item in operators
                    if item["in_top_bottleneck_set"]
                ],
            },
            "validity": {
                "e7_bitwise_validated": evidence["all_bitwise_identical"],
                "operator_count": len(operators),
                "all_call_counts_valid": all_call_counts_valid,
                "profiling_overhead_below_1pct": overhead["passes"],
                "environment_matched": not environment_mismatches,
                "profile_valid": (
                    evidence["all_bitwise_identical"]
                    and all_call_counts_valid
                    and overhead["passes"]
                    and not environment_mismatches
                ),
            },
        }
        if "reason" in timing_summary:
            summary["top_bottleneck_set"]["reason"] = timing_summary["reason"]

        result_targets[0].write_text(
            json.dumps(operator_profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_csv(result_targets[1], operators)
        result_targets[2].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (
        BENCHMARK.BenchmarkError,
        ProfileError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"E7 profiling failed: {exc}", file=sys.stderr)
        return 1

    print(f"operator profile: {result_targets[0]}")
    print(f"operator CSV:     {result_targets[1]}")
    print(f"summary:          {result_targets[2]}")
    print(f"overhead:         {overhead['overhead_pct']:.6f}%")
    print(
        "top bottleneck set: "
        f"{timing_summary['operator_count']} operators, "
        f"complete={timing_summary['complete']}"
    )
    return 0 if summary["validity"]["profile_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
