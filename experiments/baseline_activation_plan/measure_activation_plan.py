#!/usr/bin/env python3
"""Build a four-bucket native ORT RAM/RTF baseline for CAM++.

The measured process is ``build/campp_ort_benchmark``.  Python only generates
small null models, launches fresh native child processes, and analyzes JSON.
Consequently Python, NumPy, and the analyzer are excluded from every RSS/PSS
value in the result.

This experiment deliberately keeps three different memory concepts separate:

* resident delta: matched-phase CAM++ RSS minus a same-shape null-model RSS;
* allocator transient upper bound: ORT arena MaxInUse minus pre-run InUse;
* offline tensor arena: the exact arena planned for the custom C runtime.

None of these is silently relabelled as an exact ORT activation-tensor size.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments" / "baseline_activation_plan"
DEFAULT_RESULT_DIR = EXPERIMENT_DIR / "result"
DEFAULT_NULL_MODEL_DIR = EXPERIMENT_DIR / "generated"
DEFAULT_BINARY = ROOT / "build" / "campp_ort_benchmark"
DEFAULT_ARENA_MANIFEST = (
    ROOT / "runs" / "runtime" / "tensor_arena" / "bundle" / "manifest.json"
)
BUCKET_SECONDS = {98: 1.0, 298: 3.0, 498: 5.0, 998: 10.0}
DEFAULT_SPEAKERS = ("0000", "0005", "0006")
FEATURE_DIM = 80
FLOAT_BYTES = 4
MEMORY_METRICS = (
    "current_rss_bytes",
    "pss_bytes",
    "rss_anon_bytes",
    "rss_file_bytes",
    "private_clean_bytes",
    "private_dirty_bytes",
    "shared_clean_bytes",
    "shared_dirty_bytes",
    "anonymous_bytes",
)


class ExperimentError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ExperimentError("cannot summarize an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_numbers(values: Iterable[float]) -> dict[str, float | int]:
    sample = [float(value) for value in values]
    if not sample:
        raise ExperimentError("cannot summarize an empty sample")
    return {
        "samples": len(sample),
        "min": min(sample),
        "mean": statistics.fmean(sample),
        "p50": percentile(sample, 0.50),
        "p95": percentile(sample, 0.95),
        "max": max(sample),
    }


def median_optional(values: Iterable[int | float | None]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return int(statistics.median(present)) if present else None


def model_path(frames: int) -> Path:
    return ROOT / "results" / "static" / f"campp_static_{frames}.onnx"


def feature_path(speaker: str, frames: int) -> Path:
    return (
        ROOT
        / "benchmarks"
        / "campplus"
        / "features"
        / f"multi__speaker_{speaker}__{frames}.f32"
    )


def null_model_path(directory: Path, frames: int) -> Path:
    return directory / f"null_reduce_mean_{frames}.onnx"


def ensure_null_models(directory: Path, buckets: list[int]) -> dict[int, Path]:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise ExperimentError(
            "the onnx Python package is required once to generate null models"
        ) from exc

    directory.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}
    for frames in buckets:
        destination = null_model_path(directory, frames)
        input_info = helper.make_tensor_value_info(
            "feature", TensorProto.FLOAT, [1, frames, FEATURE_DIM]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 1, 1]
        )
        node = helper.make_node(
            "ReduceMean", ["feature"], ["output"], axes=[1, 2], keepdims=1
        )
        graph = helper.make_graph(
            [node], f"campp_null_floor_{frames}", [input_info], [output_info]
        )
        model = helper.make_model(
            graph,
            producer_name="campp-baseline-activation-plan",
            opset_imports=[helper.make_opsetid("", 11)],
        )
        model.ir_version = 6
        onnx.checker.check_model(model)
        onnx.save_model(model, destination)
        result[frames] = destination
    return result


def onnx_model_metadata(path: Path) -> dict[str, Any]:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise ExperimentError("onnx is required to inspect initializer bytes") from exc

    model = onnx.load(path, load_external_data=True)
    initializer_bytes = sum(
        int(numpy_helper.to_array(initializer).nbytes)
        for initializer in model.graph.initializer
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
        "initializer_count": len(model.graph.initializer),
        "serialized_initializer_payload_bytes": initializer_bytes,
        "graph_node_count": len(model.graph.node),
    }


def load_reference_arenas(manifest_path: Path) -> dict[int, int]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {
        int(plan["bucket_frames"]): int(plan["arena_size_bytes"])
        for plan in document.get("plans", [])
    }
    return result


def run_native(
    binary: Path,
    model: Path,
    feature: Path,
    audio_seconds: float,
    threads: int,
    memory_pattern: str,
    runs: int,
) -> dict[str, Any]:
    command = [
        str(binary),
        "--model", str(model),
        "--input", str(feature),
        "--audio-seconds", str(audio_seconds),
        "--warmup", "0",
        "--repeat", str(runs),
        "--threads", str(threads),
        "--graph-opt", "all",
        "--memory-pattern", memory_pattern,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ExperimentError(
            f"native ORT failed ({memory_pattern}, {model.name}, {feature.name}, "
            f"exit={completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExperimentError(
            f"native ORT did not return JSON ({memory_pattern}, {feature.name}): {exc}"
        ) from exc
    observations = result.get("inferences")
    if not isinstance(observations, list) or len(observations) != runs:
        raise ExperimentError(
            "native benchmark lacks per-run observations; rebuild it with "
            "experiments/baseline_ort_c/build.sh"
        )
    return result


def memory_value(memory: Any, key: str = "current_rss_bytes") -> int | None:
    if not isinstance(memory, dict):
        return None
    value = memory.get(key)
    return int(value) if value is not None else None


def allocator_delta(run: dict[str, Any]) -> dict[str, int] | None:
    value = run.get("allocator_run_delta")
    if not isinstance(value, dict):
        return None
    return {key: int(number) for key, number in value.items()}


def transient_peak_upper_bound(run: dict[str, Any]) -> dict[str, Any]:
    before = run.get("allocator_before")
    after = run.get("allocator_after_run")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {"bytes": None, "observed_new_max": None}
    before_max = int(before["max_in_use_bytes"])
    after_max = int(after["max_in_use_bytes"])
    before_in_use = int(before["in_use_bytes"])
    observed = after_max > before_max
    return {
        "bytes": max(0, after_max - before_in_use) if observed else None,
        "observed_new_max": observed,
        "allocator_in_use_before_bytes": before_in_use,
        "allocator_max_in_use_before_bytes": before_max,
        "allocator_max_in_use_after_bytes": after_max,
        "note": (
            "Upper bound includes output and allocator-managed scratch. It is only "
            "reported when this run raises the allocator MaxInUse high-water mark."
        ),
    }


def summarize_run(run: dict[str, Any], audio_seconds: float) -> dict[str, Any]:
    delta = allocator_delta(run)
    return {
        "session_run": int(run["session_run"]),
        "latency_ms": float(run["latency_ms"]),
        "rtf": float(run["latency_ms"]) / (audio_seconds * 1000.0),
        "output_hash_fnv1a64": run.get("output_hash_fnv1a64"),
        "output_bytes": int(run.get("output_bytes", 0)),
        "memory_before": run.get("memory_before"),
        "memory_after_run": run.get("memory_after_run"),
        "memory_after_output_release": run.get("memory_after_output_release"),
        "allocator_run_delta": delta,
        "allocation_events": (
            delta.get("num_allocs", 0) + delta.get("num_reserves", 0)
            if delta is not None
            else None
        ),
        "arena_backing_growth_bytes": (
            delta.get("total_allocated_bytes") if delta is not None else None
        ),
        "arena_extensions": (
            delta.get("num_arena_extensions") if delta is not None else None
        ),
        "allocator_transient_peak_upper_bound": transient_peak_upper_bound(run),
    }


def analyze_pattern_pair(
    pattern_on: dict[str, Any],
    pattern_off: dict[str, Any],
    audio_seconds: float,
) -> dict[str, Any]:
    on_runs = [summarize_run(run, audio_seconds) for run in pattern_on["inferences"]]
    off_runs = [summarize_run(run, audio_seconds) for run in pattern_off["inferences"]]
    hashes = {
        run["output_hash_fnv1a64"]
        for run in (*on_runs, *off_runs)
        if run["output_hash_fnv1a64"] is not None
    }
    outputs_match = len(hashes) == 1
    on_events = [run["allocation_events"] for run in on_runs]
    off_events = [run["allocation_events"] for run in off_runs]
    stats_available = all(
        value is not None for value in (*on_events[:2], *off_events[:2])
    )
    if stats_available:
        learned_after_first = on_events[1] < on_events[0]
        lower_than_off = on_events[1] < off_events[1]
        steady = len(on_events) < 3 or on_events[2] == on_events[1]
        running: bool | None = bool(
            outputs_match and learned_after_first and lower_than_off and steady
        )
    else:
        learned_after_first = lower_than_off = steady = None
        running = None

    first_latency = on_runs[0]["latency_ms"]
    warm_latency = [run["latency_ms"] for run in on_runs[1:]]
    first_rtf = on_runs[0]["rtf"]
    warm_rtf = [run["rtf"] for run in on_runs[1:]]
    return {
        "memory_pattern_status": (
            "confirmed" if running is True
            else "not_observed" if running is False
            else "inconclusive"
        ),
        "memory_pattern_running": running,
        "evidence": {
            "allocator_stats_available": stats_available,
            "all_outputs_match": outputs_match,
            "pattern_learned_after_run1": learned_after_first,
            "pattern_on_run2_lower_than_off": lower_than_off,
            "pattern_on_run2_run3_stable": steady,
            "pattern_on_allocation_events": on_events,
            "pattern_off_allocation_events": off_events,
        },
        "latency": {
            "first_inference_ms": first_latency,
            "first_inference_rtf": first_rtf,
            "warm_latency_ms": summarize_numbers(warm_latency),
            "warm_rtf": summarize_numbers(warm_rtf),
        },
        "pattern_on_runs": on_runs,
        "pattern_off_runs": off_runs,
    }


def raw_memory_phase(raw: dict[str, Any], phase: str) -> dict[str, Any] | None:
    if phase == "start":
        return raw.get("memory", {}).get("start")
    if phase == "after_session":
        return raw.get("memory", {}).get("after_context_create")
    if phase == "steady":
        return raw["inferences"][-1].get("memory_after_output_release")
    raise ValueError(f"unknown memory phase: {phase}")


def aggregate_memory_phase(
    raws: list[dict[str, Any]], phase: str
) -> dict[str, int | None]:
    return {
        metric: median_optional(
            memory_value(raw_memory_phase(raw, phase), metric) for raw in raws
        )
        for metric in MEMORY_METRICS
    }


def peak_rss_for_raw(raw: dict[str, Any]) -> int | None:
    values: list[int] = []
    for phase in ("start", "after_model_load", "after_context_create",
                  "after_warmup", "after_measurement"):
        value = memory_value(raw.get("memory", {}).get(phase), "peak_rss_bytes")
        if value is not None:
            values.append(value)
    for run in raw.get("inferences", []):
        for key in ("memory_before", "memory_after_run", "memory_after_output_release"):
            value = memory_value(run.get(key), "peak_rss_bytes")
            if value is not None:
                values.append(value)
    return max(values) if values else None


def decompose_metric(
    null_start: int | None,
    null_session: int | None,
    null_steady: int | None,
    cam_session: int | None,
    cam_steady: int | None,
) -> dict[str, int | None]:
    values = (null_start, null_session, null_steady, cam_session, cam_steady)
    if any(value is None for value in values):
        return {
            "process_and_loader_floor": None,
            "ort_env_session_input_increment": None,
            "null_inference_increment": None,
            "model_load_resident_delta": None,
            "campp_inference_resident_delta": None,
            "model_attributed_steady_delta": None,
            "model_independent_steady_floor": None,
            "campp_steady_total": None,
            "rounding_residual": None,
        }
    process_floor = int(null_start)
    session_increment = int(null_session) - process_floor
    null_inference = int(null_steady) - int(null_session)
    model_load = int(cam_session) - int(null_session)
    cam_inference = (int(cam_steady) - int(cam_session)) - null_inference
    model_attributed = int(cam_steady) - int(null_steady)
    return {
        "process_and_loader_floor": process_floor,
        "ort_env_session_input_increment": session_increment,
        "null_inference_increment": null_inference,
        "model_load_resident_delta": model_load,
        "campp_inference_resident_delta": cam_inference,
        "model_attributed_steady_delta": model_attributed,
        "model_independent_steady_floor": int(null_steady),
        "campp_steady_total": int(cam_steady),
        "rounding_residual": model_attributed - model_load - cam_inference,
    }


def summarize_bucket(
    frames: int,
    audio_seconds: float,
    model_metadata: dict[str, Any],
    reference_arena_bytes: int,
    null_raws: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    cam_raws = [item["raw"]["memory_pattern_on"] for item in measurements]
    null_phases = {
        phase: aggregate_memory_phase(null_raws, phase)
        for phase in ("start", "after_session", "steady")
    }
    cam_phases = {
        phase: aggregate_memory_phase(cam_raws, phase)
        for phase in ("start", "after_session", "steady")
    }
    decompositions = {
        metric: decompose_metric(
            null_phases["start"][metric],
            null_phases["after_session"][metric],
            null_phases["steady"][metric],
            cam_phases["after_session"][metric],
            cam_phases["steady"][metric],
        )
        for metric in MEMORY_METRICS
    }

    first_latencies = [
        item["analysis"]["pattern_on_runs"][0]["latency_ms"]
        for item in measurements
    ]
    warm_latencies = [
        run["latency_ms"]
        for item in measurements
        for run in item["analysis"]["pattern_on_runs"][1:]
    ]
    first_rtfs = [value / (audio_seconds * 1000.0) for value in first_latencies]
    warm_rtfs = [value / (audio_seconds * 1000.0) for value in warm_latencies]
    peak_values = [value for value in (peak_rss_for_raw(raw) for raw in cam_raws)
                   if value is not None]

    run_phase_rss = {}
    for index in range(3):
        run_phase_rss[f"run_{index + 1}"] = {
            "after_run_current_rss_bytes_p50": median_optional(
                memory_value(raw["inferences"][index].get("memory_after_run"))
                for raw in cam_raws
            ),
            "after_output_release_current_rss_bytes_p50": median_optional(
                memory_value(
                    raw["inferences"][index].get("memory_after_output_release")
                )
                for raw in cam_raws
            ),
            "peak_rss_bytes_p50": median_optional(
                memory_value(
                    raw["inferences"][index].get("memory_after_output_release"),
                    "peak_rss_bytes",
                )
                for raw in cam_raws
            ),
        }

    cam_transient = median_optional(
        item["analysis"]["pattern_on_runs"][0]
        ["allocator_transient_peak_upper_bound"]["bytes"]
        for item in measurements
    )
    null_transient = median_optional(
        transient_peak_upper_bound(raw["inferences"][0])["bytes"]
        for raw in null_raws
    )
    cam_specific_transient = (
        max(0, cam_transient - null_transient)
        if cam_transient is not None and null_transient is not None
        else None
    )
    rss_decomposition = decompositions["current_rss_bytes"]
    initializer_bytes = int(model_metadata["serialized_initializer_payload_bytes"])
    model_load_delta = rss_decomposition["model_load_resident_delta"]
    statuses = [item["analysis"]["memory_pattern_status"] for item in measurements]
    overall_status = (
        "confirmed" if statuses and all(value == "confirmed" for value in statuses)
        else "partial" if any(value == "confirmed" for value in statuses)
        else "inconclusive" if any(value == "inconclusive" for value in statuses)
        else "not_observed"
    )

    return {
        "bucket_frames": frames,
        "audio_seconds": audio_seconds,
        "memory_pattern_status": overall_status,
        "latency": {
            "first_inference_ms": summarize_numbers(first_latencies),
            "first_inference_rtf": summarize_numbers(first_rtfs),
            "warm_run2_run3_ms": summarize_numbers(warm_latencies),
            "warm_run2_run3_rtf": summarize_numbers(warm_rtfs),
        },
        "process_memory": {
            "peak_rss_bytes": {
                "samples": len(peak_values),
                "p50": median_optional(peak_values),
                "max": max(peak_values) if peak_values else None,
            },
            "matched_phase_medians": {
                "null_model": null_phases,
                "campp": cam_phases,
            },
            "run1_run2_run3": run_phase_rss,
            "decomposition_by_metric": decompositions,
        },
        "model_and_weight_memory": {
            **model_metadata,
            "model_load_resident_delta_bytes": model_load_delta,
            "resident_delta_beyond_serialized_initializers_bytes": (
                model_load_delta - initializer_bytes
                if model_load_delta is not None else None
            ),
            "note": (
                "The load delta contains weights, prepacked copies, graph/session "
                "metadata, and CAM++-specific code pages; RSS cannot split them exactly."
            ),
        },
        "activation_and_transient_memory": {
            "exact_ort_activation_tensor_bytes": None,
            "offline_custom_c_tensor_arena_bytes": reference_arena_bytes,
            "ort_allocator_run1_transient_peak_upper_bound_bytes": cam_transient,
            "null_allocator_run1_transient_peak_upper_bound_bytes": null_transient,
            "campp_specific_allocator_transient_upper_bound_bytes": cam_specific_transient,
            "campp_inference_resident_delta_bytes": (
                rss_decomposition["campp_inference_resident_delta"]
            ),
            "run1_arena_backing_growth_bytes_p50": median_optional(
                item["analysis"]["pattern_on_runs"][0]
                ["arena_backing_growth_bytes"]
                for item in measurements
            ),
            "run2_arena_backing_growth_bytes_p50": median_optional(
                item["analysis"]["pattern_on_runs"][1]
                ["arena_backing_growth_bytes"]
                for item in measurements
            ),
            "note": (
                "Resident delta includes activation, scratch, and lazy page faults. "
                "Arena backing growth is capacity growth, not activation size. The "
                "allocator transient value is an upper bound, not tensor-only memory."
            ),
        },
        "per_input": measurements,
        "null_floor_raw": null_raws,
    }


def environment_snapshot(cpu: int) -> dict[str, Any]:
    def command(arguments: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                arguments, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "kernel": os.uname().release if hasattr(os, "uname") else None,
        "machine": os.uname().machine if hasattr(os, "uname") else None,
        "logical_cpu_count": os.cpu_count(),
        "process_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None
        ),
        "requested_pinned_cpu": cpu if cpu >= 0 else None,
        "git_commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_status_porcelain": command(
            ["git", "-C", str(ROOT), "status", "--porcelain"]
        ),
    }


def mb(value: int | float | None) -> str:
    return "—" if value is None else f"{float(value) / 1_000_000.0:.2f}"


def render_markdown(document: dict[str, Any]) -> str:
    lines = [
        "# Native ORT four-bucket RAM/RTF baseline",
        "",
        "| frames | audio | first p50 | warm p50 | warm RTF | peak RSS max | "
        "model-attributed RSS | inference resident delta | allocator transient upper | "
        "offline arena | pattern |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for bucket in document["buckets"]:
        rss = bucket["process_memory"]["decomposition_by_metric"]["current_rss_bytes"]
        transient = bucket["activation_and_transient_memory"]
        lines.append(
            f"| {bucket['bucket_frames']} | {bucket['audio_seconds']:.1f}s | "
            f"{bucket['latency']['first_inference_ms']['p50']:.2f} ms | "
            f"{bucket['latency']['warm_run2_run3_ms']['p50']:.2f} ms | "
            f"{bucket['latency']['warm_run2_run3_rtf']['p50']:.3f} | "
            f"{mb(bucket['process_memory']['peak_rss_bytes']['max'])} MB | "
            f"{mb(rss['model_attributed_steady_delta'])} MB | "
            f"{mb(rss['campp_inference_resident_delta'])} MB | "
            f"{mb(transient['campp_specific_allocator_transient_upper_bound_bytes'])} MB | "
            f"{mb(transient['offline_custom_c_tensor_arena_bytes'])} MB | "
            f"{bucket['memory_pattern_status']} |"
        )
    lines.extend(
        [
            "",
            "`inference resident delta` is a same-shape null-model differential. "
            "It includes activation, scratch, and lazy page faults.",
            "",
            "`allocator transient upper` includes allocator-managed scratch/output and is "
            "reported only when run 1 establishes a new MaxInUse high-water mark.",
            "",
            "The authoritative result, including every native observation, is the sibling JSON file.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--buckets", type=int, nargs="+", default=list(BUCKET_SECONDS)
    )
    parser.add_argument(
        "--speakers", nargs="+", default=list(DEFAULT_SPEAKERS)
    )
    parser.add_argument("--arena-manifest", type=Path, default=DEFAULT_ARENA_MANIFEST)
    parser.add_argument("--null-model-dir", type=Path, default=DEFAULT_NULL_MODEL_DIR)
    parser.add_argument("--null-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--cpu", type=int, default=0,
        help="single CPU inherited by native children; use -1 to disable affinity",
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-as-baseline", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.threads != 1:
            raise ExperimentError(
                "the reproducible baseline requires --threads 1; a multi-core "
                "experiment needs a CPU set instead of pinning all threads to one CPU"
            )
        if args.null_repeats < 1:
            raise ExperimentError("--null-repeats must be at least 1")
        invalid_buckets = [value for value in args.buckets if value not in BUCKET_SECONDS]
        if invalid_buckets:
            raise ExperimentError(f"unsupported bucket(s): {invalid_buckets}")
        if not args.binary.is_file() or not args.arena_manifest.is_file():
            raise ExperimentError(
                f"missing native binary or arena manifest: {args.binary}, "
                f"{args.arena_manifest}"
            )
        arenas = load_reference_arenas(args.arena_manifest)
        null_models = ensure_null_models(args.null_model_dir, args.buckets)
        required: list[Path] = []
        for frames in args.buckets:
            required.append(model_path(frames))
            required.extend(feature_path(speaker, frames) for speaker in args.speakers)
            if frames not in arenas:
                raise ExperimentError(f"arena manifest has no {frames}-frame plan")
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise ExperimentError("missing required file(s): " + ", ".join(map(str, missing)))
        for frames in args.buckets:
            expected_bytes = frames * FEATURE_DIM * FLOAT_BYTES
            for speaker in args.speakers:
                feature = feature_path(speaker, frames)
                if feature.stat().st_size != expected_bytes:
                    raise ExperimentError(
                        f"{feature} is {feature.stat().st_size} bytes; expected "
                        f"{expected_bytes} for [1,{frames},{FEATURE_DIM}] float32"
                    )
        if args.cpu >= 0:
            try:
                os.sched_setaffinity(0, {args.cpu})
            except (AttributeError, OSError) as exc:
                raise ExperimentError(f"could not pin CPU {args.cpu}: {exc}") from exc

        print(
            f"native ORT baseline: buckets={args.buckets}, speakers={args.speakers}, "
            f"threads=1, cpu={args.cpu}, ON runs=3, OFF runs=2",
            flush=True,
        )
        bucket_results: list[dict[str, Any]] = []
        for frames in args.buckets:
            seconds = BUCKET_SECONDS[frames]
            model = model_path(frames)
            floor_feature = feature_path(args.speakers[0], frames)
            print(f"\n[{frames} frames / {seconds:.1f}s] null floor", flush=True)
            null_raws = [
                run_native(
                    args.binary, null_models[frames], floor_feature,
                    seconds, 1, "on", 3,
                )
                for _ in range(args.null_repeats)
            ]

            measurements: list[dict[str, Any]] = []
            for speaker in args.speakers:
                feature = feature_path(speaker, frames)
                pattern_on = run_native(
                    args.binary, model, feature, seconds, 1, "on", 3
                )
                pattern_off = run_native(
                    args.binary, model, feature, seconds, 1, "off", 2
                )
                analysis = analyze_pattern_pair(pattern_on, pattern_off, seconds)
                measurements.append(
                    {
                        "speaker": speaker,
                        "input": str(feature),
                        "input_sha256": sha256_file(feature),
                        "analysis": analysis,
                        "raw": {
                            "memory_pattern_on": pattern_on,
                            "memory_pattern_off": pattern_off,
                        },
                    }
                )
                print(
                    f"  speaker {speaker}: pattern={analysis['memory_pattern_status']}, "
                    f"first={analysis['latency']['first_inference_ms']:.2f} ms, "
                    f"warm p50={analysis['latency']['warm_latency_ms']['p50']:.2f} ms",
                    flush=True,
                )

            result = summarize_bucket(
                frames,
                seconds,
                onnx_model_metadata(model),
                arenas[frames],
                null_raws,
                measurements,
            )
            bucket_results.append(result)
            rss = result["process_memory"]["decomposition_by_metric"]["current_rss_bytes"]
            print(
                f"  bucket: RTF={result['latency']['warm_run2_run3_rtf']['p50']:.3f}, "
                f"peak={mb(result['process_memory']['peak_rss_bytes']['max'])} MB, "
                f"model-attributed={mb(rss['model_attributed_steady_delta'])} MB, "
                f"inference-resident={mb(rss['campp_inference_resident_delta'])} MB",
                flush=True,
            )

        document = {
            "schema_version": 2,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_scope": {
                "process": "native ONNX Runtime C API child only; Python excluded",
                "input": "precomputed FBank float32 [1,frames,80]",
                "output": "embedding [1,192]",
                "excluded": ["WAV read", "FBank extraction", "Python orchestrator"],
                "memory": (
                    "fresh-process VmRSS/VmHWM plus smaps_rollup; matched-phase "
                    "same-shape ReduceMean null-model differential"
                ),
                "latency": "OrtRun only; run1 first inference, run2/run3 warm",
            },
            "configuration": {
                "buckets": args.buckets,
                "speakers": args.speakers,
                "threads": 1,
                "pinned_cpu": args.cpu if args.cpu >= 0 else None,
                "memory_pattern_on_runs": 3,
                "memory_pattern_off_runs": 2,
                "null_fresh_process_repeats": args.null_repeats,
                "graph_optimization_level": "all",
                "cpu_memory_arena": True,
                "binary": str(args.binary),
                "binary_sha256": sha256_file(args.binary),
                "arena_manifest": str(args.arena_manifest),
                "arena_manifest_sha256": sha256_file(args.arena_manifest),
            },
            "interpretation_contract": {
                "exact_ort_activation_tensor_bytes": "unavailable via public RSS/allocator counters",
                "inference_resident_delta": (
                    "CAM++ run-time resident increase minus null-model run-time increase"
                ),
                "allocator_transient_upper_bound": (
                    "MaxInUse after run minus InUse before run, only when run raises MaxInUse"
                ),
                "arena_backing_growth": (
                    "TotalAllocated counter delta; capacity growth, never activation size"
                ),
                "serialized_initializer_payload": (
                    "exact ONNX initializer payload, not necessarily resident weight RSS"
                ),
            },
            "environment": environment_snapshot(args.cpu),
            "buckets": bucket_results,
        }

        destination = args.output
        if destination is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            destination = args.result_dir / f"{stamp}__native_ort_4bucket_baseline.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        destination.write_text(payload, encoding="utf-8")
        markdown_path = destination.with_suffix(".md")
        markdown = render_markdown(document)
        markdown_path.write_text(markdown, encoding="utf-8")
        if args.save_as_baseline:
            (args.result_dir / "baseline.json").write_text(payload, encoding="utf-8")
            (args.result_dir / "baseline.md").write_text(markdown, encoding="utf-8")
        print(f"\nresult: {destination}")
        print(f"summary: {markdown_path}")
        return 0
    except (ExperimentError, OSError, ValueError, KeyError) as exc:
        print(f"native ORT baseline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
