#!/usr/bin/env python3
"""Measure whether ORT reuses a memory pattern and estimate its RSS share.

The inference process is the existing native C benchmark in
``experiments/baseline_ort_c``.  This Python process only launches fresh child
processes and analyzes their JSON; it is never included in the reported RSS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments" / "baseline_activation_plan"
DEFAULT_RESULT_DIR = EXPERIMENT_DIR / "result"
DEFAULT_MODEL = ROOT / "results" / "static" / "campp_static_98.onnx"
DEFAULT_BINARY = ROOT / "build" / "campp_ort_benchmark"
DEFAULT_ARENA_MANIFEST = (
    ROOT / "runs" / "runtime" / "tensor_arena" / "bundle" / "manifest.json"
)
DEFAULT_INPUTS = (
    ROOT / "benchmarks" / "campplus" / "features" / "multi__speaker_0000__98.f32",
    ROOT / "benchmarks" / "campplus" / "features" / "multi__speaker_0005__98.f32",
    ROOT / "benchmarks" / "campplus" / "features" / "multi__speaker_0006__98.f32",
)
FRAMES = 98
FEATURE_DIM = 80
FLOAT_BYTES = 4


class ExperimentError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_arena_bytes(manifest_path: Path, frames: int = FRAMES) -> int:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    for plan in document.get("plans", []):
        if int(plan["bucket_frames"]) == frames:
            return int(plan["arena_size_bytes"])
    raise ExperimentError(f"manifest has no {frames}-frame arena: {manifest_path}")


def run_native(
    binary: Path,
    model: Path,
    feature: Path,
    threads: int,
    memory_pattern: str,
) -> dict[str, Any]:
    command = [
        str(binary),
        "--model", str(model),
        "--input", str(feature),
        "--audio-seconds", "1",
        "--warmup", "0",
        "--repeat", "2",
        "--threads", str(threads),
        "--graph-opt", "all",
        "--memory-pattern", memory_pattern,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ExperimentError(
            f"native ORT failed ({memory_pattern}, {feature.name}, "
            f"exit={completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExperimentError(
            f"native ORT did not return JSON ({memory_pattern}, {feature.name}): {exc}"
        ) from exc
    runs = result.get("inferences")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ExperimentError(
            "campp_ort_benchmark is missing the two per-inference observations; "
            "rebuild it with experiments/baseline_ort_c/build.sh"
        )
    return result


def _allocator_delta(run: dict[str, Any]) -> dict[str, int] | None:
    value = run.get("allocator_run_delta")
    if not isinstance(value, dict):
        return None
    required = ("total_allocated_bytes", "num_allocs", "num_reserves")
    if not all(key in value for key in required):
        return None
    return {key: int(number) for key, number in value.items()}


def _rss(run: dict[str, Any], point: str) -> int | None:
    memory = run.get(point)
    if not isinstance(memory, dict):
        return None
    value = memory.get("current_rss_bytes")
    return int(value) if value is not None else None


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    delta = _allocator_delta(run)
    return {
        "session_run": int(run["session_run"]),
        "latency_ms": float(run["latency_ms"]),
        "output_hash_fnv1a64": run.get("output_hash_fnv1a64"),
        "output_bytes": int(run.get("output_bytes", 0)),
        "rss_before_bytes": _rss(run, "memory_before"),
        "rss_after_run_bytes": _rss(run, "memory_after_run"),
        "rss_after_output_release_bytes": _rss(
            run, "memory_after_output_release"
        ),
        "allocator_run_delta": delta,
        "allocation_events": (
            delta["num_allocs"] + delta["num_reserves"] if delta else None
        ),
    }


def analyze_pair(
    pattern_on: dict[str, Any],
    pattern_off: dict[str, Any],
    reference_arena_bytes: int,
) -> dict[str, Any]:
    on_runs = [summarize_run(run) for run in pattern_on["inferences"]]
    off_runs = [summarize_run(run) for run in pattern_off["inferences"]]
    hashes = {
        run["output_hash_fnv1a64"]
        for run in (*on_runs, *off_runs)
        if run["output_hash_fnv1a64"] is not None
    }
    outputs_match = len(hashes) == 1

    on1_events = on_runs[0]["allocation_events"]
    on2_events = on_runs[1]["allocation_events"]
    off2_events = off_runs[1]["allocation_events"]
    allocator_stats_available = all(
        value is not None for value in (on1_events, on2_events, off2_events)
    )
    if allocator_stats_available:
        second_run_collapsed = on2_events < on1_events
        lower_than_pattern_off = on2_events < off2_events
        plan_running: bool | None = bool(
            outputs_match and second_run_collapsed and lower_than_pattern_off
        )
    else:
        second_run_collapsed = None
        lower_than_pattern_off = None
        plan_running = None

    activation_estimate = None
    estimate_method = None
    if plan_running:
        on2_delta = on_runs[1]["allocator_run_delta"]
        output_bytes = on_runs[1]["output_bytes"]
        # ORT's second same-shape run requests one memory-pattern block plus
        # non-pattern outputs.  Subtracting the output gives a reproducible
        # activation-block estimate without counting Python or model weights.
        activation_estimate = max(
            0, int(on2_delta["total_allocated_bytes"]) - output_bytes
        )
        estimate_method = "pattern_on_run2_allocator_delta_minus_output"

    native_rss = on_runs[1]["rss_after_run_bytes"]
    if native_rss is not None and activation_estimate is not None:
        non_activation_rss = max(0, native_rss - activation_estimate)
        activation_share = activation_estimate / native_rss if native_rss else None
        reference_share = reference_arena_bytes / native_rss if native_rss else None
    else:
        non_activation_rss = None
        activation_share = None
        reference_share = None

    run1_rss = on_runs[0]["rss_after_run_bytes"]
    run2_rss = on_runs[1]["rss_after_run_bytes"]
    return {
        "activation_plan_running": plan_running,
        "evidence": {
            "allocator_stats_available": allocator_stats_available,
            "all_outputs_match": outputs_match,
            "pattern_on_run2_has_fewer_events_than_run1": second_run_collapsed,
            "pattern_on_run2_has_fewer_events_than_pattern_off": lower_than_pattern_off,
            "pattern_on_allocation_events": [on1_events, on2_events],
            "pattern_off_allocation_events": [
                off_runs[0]["allocation_events"], off2_events
            ],
        },
        "memory_decomposition": {
            "native_rss_at_pattern_on_run2_bytes": native_rss,
            "activation_plan_estimate_bytes": activation_estimate,
            "non_activation_rss_estimate_bytes": non_activation_rss,
            "activation_share_of_native_rss": activation_share,
            "reference_tensor_arena_bytes": reference_arena_bytes,
            "reference_tensor_arena_share_of_native_rss": reference_share,
            "activation_estimate_method": estimate_method,
        },
        "same_session_rss": {
            "pattern_on_run1_bytes": run1_rss,
            "pattern_on_run2_bytes": run2_rss,
            "run2_minus_run1_bytes": (
                run2_rss - run1_rss
                if run1_rss is not None and run2_rss is not None
                else None
            ),
            "note": (
                "RSS may stay flat because the CPU arena retains pages; plan operation "
                "is decided from allocator-event deltas, not VmHWM."
            ),
        },
        "pattern_on_runs": on_runs,
        "pattern_off_runs": off_runs,
    }


def median_int(values: list[int]) -> int | None:
    return int(statistics.median(values)) if values else None


def aggregate(per_input: list[dict[str, Any]]) -> dict[str, Any]:
    states = [item["analysis"]["activation_plan_running"] for item in per_input]
    if states and all(value is True for value in states):
        status = "confirmed"
    elif any(value is True for value in states):
        status = "partial"
    elif any(value is None for value in states):
        status = "inconclusive"
    else:
        status = "not_observed"

    decompositions = [item["analysis"]["memory_decomposition"] for item in per_input]
    activation_values = [
        int(item["activation_plan_estimate_bytes"])
        for item in decompositions
        if item["activation_plan_estimate_bytes"] is not None
    ]
    rss_values = [
        int(item["native_rss_at_pattern_on_run2_bytes"])
        for item in decompositions
        if item["native_rss_at_pattern_on_run2_bytes"] is not None
    ]
    activation = median_int(activation_values)
    rss = median_int(rss_values)
    return {
        "activation_plan_status": status,
        "confirmed_inputs": sum(value is True for value in states),
        "total_inputs": len(states),
        "median_native_rss_at_run2_bytes": rss,
        "median_activation_plan_estimate_bytes": activation,
        "median_non_activation_rss_estimate_bytes": (
            max(0, rss - activation)
            if rss is not None and activation is not None
            else None
        ),
        "activation_share_of_native_rss": (
            activation / rss
            if rss is not None and activation is not None and rss > 0
            else None
        ),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--inputs", type=Path, nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument("--arena-manifest", type=Path, default=DEFAULT_ARENA_MANIFEST)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--cpu", type=int, default=0,
        help="CPU affinity inherited by native children; use -1 to disable",
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.threads < 1:
            raise ExperimentError("--threads must be at least 1")
        required = [args.binary, args.model, args.arena_manifest, *args.inputs]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise ExperimentError("missing required file(s): " + ", ".join(map(str, missing)))
        expected_input_bytes = FRAMES * FEATURE_DIM * FLOAT_BYTES
        for feature in args.inputs:
            if feature.stat().st_size != expected_input_bytes:
                raise ExperimentError(
                    f"{feature} is {feature.stat().st_size} bytes; "
                    f"expected {expected_input_bytes} for [1,{FRAMES},{FEATURE_DIM}] float32"
                )
        if args.cpu >= 0:
            try:
                os.sched_setaffinity(0, {args.cpu})
            except (AttributeError, OSError) as exc:
                raise ExperimentError(f"could not pin CPU {args.cpu}: {exc}") from exc

        reference_arena_bytes = load_reference_arena_bytes(args.arena_manifest)
        print(
            f"CAM++ {FRAMES} frames, threads={args.threads}, inputs={len(args.inputs)}, "
            "same-session runs=2, memory-pattern=ON/OFF",
            flush=True,
        )
        per_input = []
        for feature in args.inputs:
            pattern_on = run_native(args.binary, args.model, feature, args.threads, "on")
            pattern_off = run_native(args.binary, args.model, feature, args.threads, "off")
            analysis = analyze_pair(pattern_on, pattern_off, reference_arena_bytes)
            per_input.append(
                {
                    "input": str(feature),
                    "input_sha256": sha256_file(feature),
                    "analysis": analysis,
                    "raw": {"memory_pattern_on": pattern_on, "memory_pattern_off": pattern_off},
                }
            )
            memory = analysis["memory_decomposition"]
            state = analysis["activation_plan_running"]
            share = memory["activation_share_of_native_rss"]
            print(
                f"  {feature.name}: plan={state}, "
                f"activation={memory['activation_plan_estimate_bytes']} B, "
                f"RSS share={share:.2%}" if share is not None else
                f"  {feature.name}: plan={state}, activation=unavailable",
                flush=True,
            )

        document = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "Native ORT C API RSS plus same-session run-1/run-2 allocator "
                "deltas; the Python orchestrator is excluded from RSS."
            ),
            "configuration": {
                "model": str(args.model),
                "model_sha256": sha256_file(args.model),
                "frames": FRAMES,
                "feature_dim": FEATURE_DIM,
                "threads": args.threads,
                "pinned_cpu": args.cpu if args.cpu >= 0 else None,
                "runs_per_session": 2,
                "cpu_memory_arena": True,
                "graph_optimization_level": "all",
                "memory_pattern_variants": ["on", "off"],
                "binary": str(args.binary),
                "reference_arena_manifest": str(args.arena_manifest),
                "reference_arena_bytes": reference_arena_bytes,
            },
            "summary": aggregate(per_input),
            "per_input": per_input,
        }
        destination = args.output
        if destination is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            destination = args.result_dir / f"{stamp}__activation_plan.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary = document["summary"]
        print(
            f"status={summary['activation_plan_status']}, "
            f"activation={summary['median_activation_plan_estimate_bytes']} B, "
            f"non-activation={summary['median_non_activation_rss_estimate_bytes']} B"
        )
        print(f"result: {destination}")
        return 0
    except (ExperimentError, OSError, ValueError, KeyError) as exc:
        print(f"activation-plan experiment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
