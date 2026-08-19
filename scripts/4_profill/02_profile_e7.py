#!/usr/bin/env python3
"""E7 Bucket 98의 kernel exclusive time을 측정하고 Top 80%를 산출한다."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
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
EXPECTED_THREADS = 1
QUICK_WARMUP = 5
QUICK_REPEAT = 20
QUICK_BASELINE_REPEAT = 5
OFFICIAL_WARMUP = 20
OFFICIAL_REPEAT = 100
TOP_THRESHOLD_PCT = 80.0
HISTORICAL_E7_INFERENCE_SECONDS = 8.4


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


def _resolve_protocol(config: Any, mode: str) -> dict[str, Any]:
    mismatches: list[str] = []
    if config.buckets != {EXPECTED_BUCKET: 1.0}:
        mismatches.append(f"buckets={config.buckets!r}")
    if tuple(config.input_ids) != EXPECTED_INPUT_IDS:
        mismatches.append(f"input_ids={tuple(config.input_ids)!r}")
    if config.threads != EXPECTED_THREADS:
        mismatches.append(f"threads={config.threads}")
    if tuple(config.environment.affinity) != (0,):
        mismatches.append(f"affinity={tuple(config.environment.affinity)!r}")
    if mode == "official" and config.warmup != OFFICIAL_WARMUP:
        mismatches.append(f"warmup={config.warmup}")
    if mode == "official" and config.repeat != OFFICIAL_REPEAT:
        mismatches.append(f"repeat={config.repeat}")
    if mismatches:
        raise ProfileError(
            f"E7 {mode} profiling protocol mismatch: " + ", ".join(mismatches)
        )
    if mode == "quick":
        return {
            "mode": "quick",
            "warmup": QUICK_WARMUP,
            "repeat": QUICK_REPEAT,
            "baseline_repeat": QUICK_BASELINE_REPEAT,
            "official": False,
        }
    return {
        "mode": "official",
        "warmup": OFFICIAL_WARMUP,
        "repeat": OFFICIAL_REPEAT,
        "baseline_repeat": OFFICIAL_REPEAT,
        "official": True,
    }


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
    payload: dict[str, Any], plan: LoadedPlan, repeat: int,
    expected_suite: str,
) -> None:
    if payload.get("optimization_suite") != expected_suite:
        raise ProfileError(
            "profiler optimization suite does not match the experiment"
        )
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
    comparisons: Sequence[dict[str, Any]], *, preliminary: bool,
    threshold_pct: float = 1.0
) -> dict[str, Any]:
    baseline_total_ns = sum(int(item["baseline_total_ns"]) for item in comparisons)
    profiled_total_ns = sum(int(item["profiled_total_ns"]) for item in comparisons)
    baseline_count = sum(int(item["baseline_sample_count"]) for item in comparisons)
    profiled_count = sum(int(item["profiled_sample_count"]) for item in comparisons)
    if baseline_total_ns <= 0 or baseline_count <= 0 or profiled_count <= 0:
        raise ProfileError("overhead comparison samples must be positive")
    baseline_mean_ns = baseline_total_ns / baseline_count
    profiled_mean_ns = profiled_total_ns / profiled_count
    overhead_pct = (profiled_mean_ns / baseline_mean_ns - 1.0) * 100.0
    return {
        "schema_version": 1,
        "definition": (
            "(profiled_mean_end_to_end / baseline_mean_end_to_end - 1) * 100"
        ),
        "preliminary": preliminary,
        "threshold_pct": threshold_pct,
        "overhead_pct": overhead_pct,
        "passes": overhead_pct < threshold_pct,
        "baseline_total_ns": baseline_total_ns,
        "baseline_sample_count": baseline_count,
        "baseline_mean_ns": baseline_mean_ns,
        "profiled_total_ns": profiled_total_ns,
        "profiled_sample_count": profiled_count,
        "profiled_mean_ns": profiled_mean_ns,
        "per_input": list(comparisons),
    }


def _estimated_minutes(protocol: dict[str, Any], input_count: int) -> float:
    """기존 QRB2210 E7 p50을 이용해 전체 Quick/Official 시간을 예측한다."""

    inference_count = input_count * (
        protocol["warmup"] + protocol["repeat"]
        + protocol["warmup"] + protocol["baseline_repeat"]
    )
    return inference_count * HISTORICAL_E7_INFERENCE_SECONDS / 60.0


def _aggregate_operator_field(
    operators: Sequence[dict[str, Any]], field: str,
    total_end_to_end_ms: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for operator in operators:
        name = str(operator.get(field) or "NONE")
        item = grouped.setdefault(
            name,
            {
                "name": name,
                "operator_count": 0,
                "call_count": 0,
                "total_exclusive_ms": 0.0,
            },
        )
        item["operator_count"] += 1
        item["call_count"] += int(operator["call_count"])
        item["total_exclusive_ms"] += float(operator["total_exclusive_ms"])
    result = sorted(
        grouped.values(),
        key=lambda item: (-float(item["total_exclusive_ms"]), item["name"]),
    )
    for item in result:
        item["end_to_end_share_pct"] = (
            float(item["total_exclusive_ms"]) / total_end_to_end_ms * 100.0
            if total_end_to_end_ms > 0.0
            else 0.0
        )
    return result


def _analysis_document(
    operators: Sequence[dict[str, Any]], timing: dict[str, Any],
    overhead: dict[str, Any],
) -> dict[str, Any]:
    """결과를 kernel/type/fusion 축으로 집계하고 측정으로 확인된 사실만 요약한다."""

    total_ms = float(timing["total_end_to_end_ms"])
    kernels = _aggregate_operator_field(operators, "kernel_name", total_ms)
    operator_types = _aggregate_operator_field(operators, "operator_type", total_ms)
    fusion_families = _aggregate_operator_field(operators, "fusion_family", total_ms)
    top_operator = operators[0]
    observations = [
        (
            f"{operator_types[0]['name']} operators account for "
            f"{operator_types[0]['end_to_end_share_pct']:.3f}% of end-to-end latency."
        ),
        (
            f"The largest kernel family is {kernels[0]['name']} at "
            f"{kernels[0]['end_to_end_share_pct']:.3f}%."
        ),
        (
            f"Operator {top_operator['operator_id']} is the largest single operator "
            f"at {top_operator['end_to_end_share_pct']:.3f}%."
        ),
        (
            f"Kernel-exclusive timing accounts for "
            f"{timing['kernel_accounted_end_to_end_pct']:.3f}% of end-to-end latency."
        ),
    ]
    if overhead.get("preliminary"):
        observations.append(
            "Profiling overhead is a preliminary Quick-mode estimate."
        )
    return {
        "primary_bottleneck": {
            "operator_type": operator_types[0],
            "kernel": kernels[0],
            "top_operator": {
                key: top_operator[key]
                for key in (
                    "rank",
                    "operator_id",
                    "operator_type",
                    "kernel_id",
                    "kernel_name",
                    "fusion_family",
                    "mean_ms",
                    "p50_ms",
                    "p95_ms",
                    "end_to_end_share_pct",
                )
            },
        },
        "kernel_breakdown": kernels,
        "operator_type_breakdown": operator_types,
        "fusion_family_breakdown": fusion_families,
        "top_10_cumulative_end_to_end_share_pct": operators[
            min(9, len(operators) - 1)
        ]["cumulative_end_to_end_share_pct"],
        "observations": observations,
        "limitation": (
            "Exclusive latency alone does not distinguish compute, memory-bandwidth, "
            "or cache bottlenecks; hardware counters are required for that conclusion."
        ),
    }


def _print_final_report(
    *,
    elapsed_seconds: float,
    result_targets: Sequence[Path],
    summary: dict[str, Any],
) -> None:
    timing = summary["timing"]
    overhead = summary["overhead"]
    top_set = summary["top_bottleneck_set"]
    analysis = summary["analysis"]
    end_to_end = timing["end_to_end"]

    print("측정 완료")
    print(f"실제 소요 시간: {elapsed_seconds / 60.0:.1f}분")
    print(
        "End-to-end latency: "
        f"mean {end_to_end['mean_ms']:.3f} ms, "
        f"p50 {end_to_end['p50_ms']:.3f} ms, "
        f"p95 {end_to_end['p95_ms']:.3f} ms"
    )
    print(
        "Kernel 설명 비중: "
        f"{timing['kernel_accounted_end_to_end_pct']:.3f}% "
        f"(미설명 {timing['unattributed_runtime_ms']:.3f} ms)"
    )
    qualifier = "예비 " if overhead["preliminary"] else ""
    print(
        f"Profiling overhead: {overhead['overhead_pct']:.3f}% "
        f"({qualifier}{'PASS' if overhead['passes'] else 'FAIL'})"
    )
    print(
        f"Top 80% bottleneck set: {top_set['operator_count']}개, "
        f"complete={top_set['complete']}"
    )
    print("Kernel별 비중:")
    for item in analysis["kernel_breakdown"][:5]:
        print(
            f"  {item['name']}: {item['end_to_end_share_pct']:.3f}% "
            f"({item['operator_count']} operators)"
        )
    print("상위 Operator:")
    for item in top_set["operators"][:10]:
        print(
            f"  #{item['rank']} op={item['operator_id']} "
            f"{item['kernel_name']} {item['end_to_end_share_pct']:.3f}% "
            f"(누적 {item['cumulative_end_to_end_share_pct']:.3f}%)"
        )
    print("결과 파일:")
    for path in result_targets:
        print(f"  {path}")


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
        "--expected-suite",
        choices=("stock", "final"),
        default="stock",
        help="두 binary가 보고해야 하는 runtime optimization suite",
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
    parser.add_argument(
        "--mode",
        choices=("quick", "official"),
        default="quick",
        help="quick: 5 warm-up/20 profile/5 baseline; official: 20/100",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = BENCHMARK.load_config(args.config, repository_root=ROOT)
        protocol = _resolve_protocol(config, args.mode)
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
        baseline_capabilities, _ = BENCHMARK.run_json_command(
            [str(baseline_binary), "--capabilities"],
            environment=os.environ.copy(),
        )
        capabilities, _ = BENCHMARK.run_json_command(
            [str(profiler_binary), "--capabilities"],
            environment=os.environ.copy(),
        )
        if baseline_capabilities.get("optimization_suite") != args.expected_suite:
            raise ProfileError(
                "baseline binary optimization suite does not match "
                f"--expected-suite {args.expected_suite}"
            )
        if capabilities.get("optimization_suite") != args.expected_suite:
            raise ProfileError(
                "profiler binary optimization suite does not match "
                f"--expected-suite {args.expected_suite}"
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
        print(
            f"  mode: {protocol['mode']} "
            f"(warmup={protocol['warmup']}, repeat={protocol['repeat']}, "
            f"baseline_repeat={protocol['baseline_repeat']})"
        )
        print(f"  optimization suite: {args.expected_suite}")
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
        estimated_minutes = _estimated_minutes(protocol, len(features))
        print(f"예상 시간: 약 {estimated_minutes:.0f}분")
        print("진행 중", flush=True)
        measurement_started = time.perf_counter()

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
            order = (
                ("baseline", "profile")
                if index % 2 == 0
                else ("profile", "baseline")
            )
            baseline_payload: dict[str, Any] | None = None
            profile_payload: dict[str, Any] | None = None
            for mode in order:
                print("진행 중", flush=True)
                with redirect_stdout(io.StringIO()):
                    BENCHMARK.wait_for_thermal_start(config.environment)
                if mode == "baseline":
                    command = BENCHMARK.c_command(
                        baseline_binary,
                        plan=plan_path,
                        weights=weights_path,
                        feature=feature,
                        threads=config.threads,
                        warmup=protocol["warmup"],
                        repeat=protocol["baseline_repeat"],
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
                        warmup=protocol["warmup"],
                        repeat=protocol["repeat"],
                        threads=config.threads,
                    )
                    profile_payload, _ = BENCHMARK.run_json_command(
                        command, environment=base_environment
                    )
            if baseline_payload is None or profile_payload is None:
                raise ProfileError("baseline/profile pair is incomplete")
            _validate_profiler_payload(
                profile_payload, plan, protocol["repeat"],
                args.expected_suite,
            )
            if baseline_payload.get("optimization_suite") != args.expected_suite:
                raise ProfileError(
                    "baseline payload optimization suite changed during the run"
                )
            baseline_timings_ms = baseline_payload.get("warm", {}).get(
                "timings_ms"
            )
            if (
                not isinstance(baseline_timings_ms, list)
                or len(baseline_timings_ms) != protocol["baseline_repeat"]
                or any(not isinstance(value, (int, float)) or value <= 0
                       for value in baseline_timings_ms)
            ):
                raise ProfileError("baseline returned invalid warm timings")
            baseline_total_ns = round(sum(baseline_timings_ms) * 1_000_000.0)
            profiled_total_ns = sum(profile_payload["end_to_end_ns"])
            baseline_mean_ns = baseline_total_ns / len(baseline_timings_ms)
            profiled_mean_ns = profiled_total_ns / len(
                profile_payload["end_to_end_ns"]
            )
            input_overhead_pct = (
                profiled_mean_ns / baseline_mean_ns - 1.0
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
                    "baseline_sample_count": len(baseline_timings_ms),
                    "profiled_total_ns": profiled_total_ns,
                    "profiled_sample_count": len(
                        profile_payload["end_to_end_ns"]
                    ),
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

        overhead = _overhead_document(
            overhead_comparisons, preliminary=not protocol["official"]
        )
        overhead_path.write_text(
            json.dumps(overhead, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        operators, timing_summary = _build_operator_profile(
            plan, fusion_document, raw_payloads
        )
        analysis = _analysis_document(operators, timing_summary, overhead)
        measurement_elapsed_seconds = time.perf_counter() - measurement_started
        expected_call_count = len(features) * protocol["repeat"]
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
            "protocol_mode": protocol["mode"],
            "official_protocol": protocol["official"],
            "bucket_frames": EXPECTED_BUCKET,
            "threads": config.threads,
            "cpu_affinity": list(config.environment.affinity),
            "warmup": protocol["warmup"],
            "repeat": protocol["repeat"],
            "overhead_baseline_repeat": protocol["baseline_repeat"],
            "input_ids": list(EXPECTED_INPUT_IDS),
            "expected_call_count_per_operator": expected_call_count,
            "measurement_scope": "kernel_run_exclusive",
            "optimization_suite": args.expected_suite,
            "estimated_minutes": estimated_minutes,
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
            "analysis": analysis,
            "operators": operators,
        }
        summary = {
            "schema_version": 1,
            "generated_at_utc": generated_at,
            "configuration": configuration,
            "artifacts": artifacts,
            "overhead": overhead,
            "timing": timing_summary,
            "analysis": analysis,
            "measurement_elapsed_seconds": measurement_elapsed_seconds,
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
                "official_protocol": protocol["official"],
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

    _print_final_report(
        elapsed_seconds=measurement_elapsed_seconds,
        result_targets=result_targets,
        summary=summary,
    )
    return 0 if summary["validity"]["profile_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
