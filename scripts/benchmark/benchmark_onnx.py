#!/usr/bin/env python3
"""Reproducible cold/warm benchmark for a static CAM++ ONNX model.

The ONNX models in this repository consume precomputed fbank features with
shape ``[1, frames, 80]``.  This runner therefore accepts a feature ``.npy``
file, not a WAV file.  If no feature is supplied, it creates deterministic
synthetic features for performance-only measurements.

Cold measurements launch a fresh Python process for every sample.  Warm
measurements create one ONNX Runtime session, discard warm-up runs, and then
record every measured inference.  The final JSON contains raw timings and the
derived p50/p95/p99, RTF, CV, RSS, planned static arena, and optional cosine
similarity against a reference embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROCESS_START_NS = time.perf_counter_ns()

DEFAULTS: dict[str, Any] = {
    "warmup": 20,
    "repeat": 100,
    "cold_runs": 10,
    "threads": 1,
    "seed": 0,
    "cv_threshold_pct": 3.0,
    "cosine_threshold": 0.999,
}

COMMON_BUCKET_SECONDS = {98: 1.0, 298: 3.0, 498: 5.0, 998: 10.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: Any) -> str:
    contiguous = array if array.flags.c_contiguous else array.copy(order="C")
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def percentile(values: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("cannot compute a percentile from an empty sequence")
    if not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("at least one timing is required")
    samples = [float(value) for value in values]
    mean = statistics.fmean(samples)
    stddev = statistics.pstdev(samples)
    cv = stddev / mean * 100.0 if mean > 0.0 else math.inf
    return {
        "count": len(samples),
        "mean_ms": mean,
        "stddev_ms": stddev,
        "cv_pct": cv,
        "min_ms": min(samples),
        "p50_ms": percentile(samples, 50.0),
        "p95_ms": percentile(samples, 95.0),
        "p99_ms": percentile(samples, 99.0),
        "max_ms": max(samples),
    }


def cosine_metrics(reference: Any, actual: Any) -> dict[str, float | list[int]]:
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    got = np.asarray(actual, dtype=np.float64).reshape(-1)
    if ref.shape != got.shape:
        raise ValueError(
            f"reference and output sizes differ: {ref.shape} != {got.shape}"
        )
    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    if ref_norm == 0.0 or got_norm == 0.0:
        cosine = math.nan
    else:
        cosine = float(ref @ got / (ref_norm * got_norm))
    return {
        "shape": list(actual.shape),
        "cosine_similarity": cosine,
        "max_abs_diff": float(np.max(np.abs(ref - got))),
        "mean_abs_diff": float(np.mean(np.abs(ref - got))),
    }


def read_linux_memory() -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "current_rss_bytes": None,
        "peak_rss_bytes": None,
    }
    status = Path("/proc/self/status")
    if not status.exists():
        return result
    wanted = {"VmRSS:": "current_rss_bytes", "VmHWM:": "peak_rss_bytes"}
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if fields and fields[0] in wanted and len(fields) >= 2:
            result[wanted[fields[0]]] = int(fields[1]) * 1024
    return result


def read_text_if_present(path: str) -> str | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        return candidate.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return None


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def system_metadata(ort_version: str, numpy_version: str) -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_version": platform.python_version(),
        "onnxruntime_version": ort_version,
        "numpy_version": numpy_version,
        "device_tree_model": read_text_if_present("/proc/device-tree/model"),
        "cpu_governor": read_text_if_present(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "cpu_affinity": affinity,
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "ORT_NUM_THREADS",
            )
        },
    }


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a JSON object")
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    return config


def setting(args: argparse.Namespace, config: dict[str, Any], name: str) -> Any:
    cli_value = getattr(args, name)
    return cli_value if cli_value is not None else config.get(name, DEFAULTS[name])


def load_ir(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"IR not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    arena = data.get("arena") or {}
    return {
        "path": str(path),
        "frames": data.get("frames"),
        "audio_seconds": data.get("seconds"),
        "planned_static_arena_bytes": arena.get("arena_bytes"),
        "planned_peak_live_bytes": arena.get("peak_live_bytes"),
        "weight_bytes": arena.get("weight_bytes"),
    }


def create_session(model: Path, threads: int) -> tuple[Any, str, str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required; activate the Arduino venv that contains it"
        ) from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError(f"expected one model input, found {len(inputs)}")
    if not outputs:
        raise ValueError("model has no outputs")
    output_name = next(
        (output.name for output in outputs if output.name == "embedding"),
        outputs[0].name,
    )
    return session, inputs[0].name, output_name


def static_frames_from_session(session: Any, input_name: str) -> int | None:
    input_meta = next(item for item in session.get_inputs() if item.name == input_name)
    shape = input_meta.shape
    if len(shape) != 3:
        raise ValueError(f"expected model input rank 3 [1,T,80], got {shape}")
    frame_value = shape[1]
    return frame_value if isinstance(frame_value, int) else None


def load_feature(
    path: Path | None,
    frames: int | None,
    seed: int,
    session: Any,
    input_name: str,
) -> tuple[Any, str]:
    import numpy as np

    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"feature file not found: {path}")
        feature = np.load(path, allow_pickle=False)
        source = "npy"
    else:
        chosen_frames = frames or static_frames_from_session(session, input_name)
        if chosen_frames is None:
            raise ValueError("dynamic model requires --frames when --input-npy is omitted")
        rng = np.random.default_rng(seed)
        feature = (
            rng.standard_normal((1, chosen_frames, 80)).astype(np.float32) * 5.0
        )
        source = "synthetic"

    feature = np.asarray(feature)
    if feature.ndim == 2:
        feature = feature[None, :, :]
    if feature.ndim != 3 or feature.shape[0] != 1 or feature.shape[2] != 80:
        raise ValueError(
            f"feature must have shape [T,80] or [1,T,80], got {feature.shape}"
        )
    feature = np.ascontiguousarray(feature, dtype=np.float32)

    expected = session.get_inputs()[0].shape
    for index, (expected_dim, actual_dim) in enumerate(zip(expected, feature.shape)):
        if isinstance(expected_dim, int) and expected_dim != actual_dim:
            raise ValueError(
                f"input dimension {index} mismatch: model={expected_dim}, "
                f"feature={actual_dim}"
            )
    return feature, source


def infer_audio_seconds(frames: int, ir: dict[str, Any] | None) -> float:
    if ir and ir.get("audio_seconds") is not None:
        return float(ir["audio_seconds"])
    if frames in COMMON_BUCKET_SECONDS:
        return COMMON_BUCKET_SECONDS[frames]
    return (frames - 1) * 0.010 + 0.025


def run_cold_child(args: argparse.Namespace) -> int:
    import numpy as np

    model = args.model.resolve()
    create_start = time.perf_counter_ns()
    session, input_name, output_name = create_session(model, args.threads)
    create_end = time.perf_counter_ns()
    feature, source = load_feature(
        args.input_npy, args.frames, args.seed, session, input_name
    )
    inference_start = time.perf_counter_ns()
    embedding = session.run([output_name], {input_name: feature})[0]
    inference_end = time.perf_counter_ns()
    memory = read_linux_memory()
    payload = {
        "session_create_ms": (create_end - create_start) / 1e6,
        "first_inference_ms": (inference_end - inference_start) / 1e6,
        "child_total_ms": (inference_end - PROCESS_START_NS) / 1e6,
        "peak_rss_bytes": memory["peak_rss_bytes"],
        "feature_source": source,
        "feature_shape": list(feature.shape),
        "embedding_shape": list(np.asarray(embedding).shape),
        "embedding_sha256": sha256_array(np.asarray(embedding)),
    }
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


def child_command(args: argparse.Namespace, threads: int, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-cold",
        "--model",
        str(args.model.resolve()),
        "--threads",
        str(threads),
        "--seed",
        str(seed),
    ]
    if args.input_npy is not None:
        command += ["--input-npy", str(args.input_npy.resolve())]
    if args.frames is not None:
        command += ["--frames", str(args.frames)]
    return command


def run_cold_processes(
    args: argparse.Namespace, cold_runs: int, threads: int, seed: int
) -> dict[str, Any] | None:
    if cold_runs == 0 or args.skip_cold:
        return None
    samples: list[dict[str, Any]] = []
    for index in range(cold_runs):
        started = time.perf_counter_ns()
        completed = subprocess.run(
            child_command(args, threads, seed),
            check=False,
            capture_output=True,
            text=True,
        )
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        if completed.returncode != 0:
            raise RuntimeError(
                f"cold child {index + 1} failed ({completed.returncode}):\n"
                f"{completed.stderr.strip()}"
            )
        try:
            sample = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cold child {index + 1} returned invalid JSON: "
                f"{completed.stdout!r}"
            ) from exc
        sample["run"] = index + 1
        sample["process_wall_ms"] = wall_ms
        samples.append(sample)

    return {
        "definition": "fresh process; OS file page cache is not dropped",
        "samples": samples,
        "process_wall_summary": summarize_ms(
            [sample["process_wall_ms"] for sample in samples]
        ),
        "session_create_summary": summarize_ms(
            [sample["session_create_ms"] for sample in samples]
        ),
        "first_inference_summary": summarize_ms(
            [sample["first_inference_ms"] for sample in samples]
        ),
        "max_peak_rss_bytes": max(
            (
                sample["peak_rss_bytes"]
                for sample in samples
                if sample["peak_rss_bytes"] is not None
            ),
            default=None,
        ),
    }


def run_warm(
    args: argparse.Namespace,
    warmup: int,
    repeat: int,
    threads: int,
    seed: int,
    cv_threshold_pct: float,
    cosine_threshold: float,
    ir: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any], str, str]:
    import numpy as np
    import onnxruntime as ort

    if args.skip_warm:
        return None, {}, ort.__version__, np.__version__

    create_start = time.perf_counter_ns()
    session, input_name, output_name = create_session(args.model.resolve(), threads)
    session_create_ms = (time.perf_counter_ns() - create_start) / 1e6
    feature, feature_source = load_feature(
        args.input_npy, args.frames, seed, session, input_name
    )
    frames = int(feature.shape[1])
    audio_seconds = (
        float(args.audio_seconds)
        if args.audio_seconds is not None
        else infer_audio_seconds(frames, ir)
    )
    if audio_seconds <= 0:
        raise ValueError("audio seconds must be positive")

    embedding = None
    for _ in range(warmup):
        embedding = session.run([output_name], {input_name: feature})[0]

    timings_ms: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter_ns()
        embedding = session.run([output_name], {input_name: feature})[0]
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)

    assert embedding is not None
    summary = summarize_ms(timings_ms)
    rtf = {
        name.replace("_ms", ""): float(summary[name]) / 1000.0 / audio_seconds
        for name in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
    }
    accuracy: dict[str, Any] = {
        "reference_path": str(args.reference_npy.resolve())
        if args.reference_npy
        else None,
        "cosine_threshold": cosine_threshold,
        "evaluated": args.reference_npy is not None,
        "passed": None,
    }
    if args.reference_npy is not None:
        if not args.reference_npy.is_file():
            raise FileNotFoundError(
                f"reference embedding not found: {args.reference_npy}"
            )
        reference = np.load(args.reference_npy, allow_pickle=False)
        accuracy.update(cosine_metrics(reference, embedding))
        cosine = float(accuracy["cosine_similarity"])
        accuracy["passed"] = math.isfinite(cosine) and cosine >= cosine_threshold

    memory = read_linux_memory()
    warm_result = {
        "session_create_ms": session_create_ms,
        "warmup_count": warmup,
        "measured_count": repeat,
        "timings_ms": timings_ms,
        "latency": summary,
        "rtf": rtf,
        "stability_gate": {
            "threshold_cv_pct": cv_threshold_pct,
            "passed": float(summary["cv_pct"]) <= cv_threshold_pct,
        },
        "memory": {
            **memory,
            "planned_static_arena_bytes": ir.get("planned_static_arena_bytes")
            if ir
            else None,
            "planned_peak_live_bytes": ir.get("planned_peak_live_bytes")
            if ir
            else None,
            "weight_bytes": ir.get("weight_bytes") if ir else None,
            "note": "planned arena is from offline IR, not ONNX Runtime RSS",
        },
        "accuracy": accuracy,
        "embedding": {
            "shape": list(np.asarray(embedding).shape),
            "dtype": str(np.asarray(embedding).dtype),
            "sha256": sha256_array(np.asarray(embedding)),
        },
    }
    input_result = {
        "source": feature_source,
        "path": str(args.input_npy.resolve()) if args.input_npy else None,
        "sha256": sha256_file(args.input_npy) if args.input_npy else None,
        "feature_sha256": sha256_array(feature),
        "shape": list(feature.shape),
        "dtype": str(feature.dtype),
        "frames": frames,
        "audio_seconds": audio_seconds,
    }
    return warm_result, input_result, ort.__version__, np.__version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument("--reference-npy", type=Path)
    parser.add_argument("--ir", type=Path, help="matching results/graph/ir_*.json")
    parser.add_argument("--config", type=Path, help="JSON defaults for QRB2210")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--cold-runs", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--audio-seconds", type=float)
    parser.add_argument("--cv-threshold-pct", type=float)
    parser.add_argument("--cosine-threshold", type=float)
    parser.add_argument("--skip-cold", action="store_true")
    parser.add_argument("--skip-warm", action="store_true")
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="return exit code 2 when CV or evaluated cosine gate fails",
    )
    parser.add_argument("--child-cold", action="store_true", help=argparse.SUPPRESS)
    return parser


def validate_positive_settings(
    warmup: int, repeat: int, cold_runs: int, threads: int
) -> None:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if cold_runs < 0:
        raise ValueError("cold-runs must be >= 0")
    if threads < 1:
        raise ValueError("threads must be >= 1")


def run(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        raise FileNotFoundError(f"model not found: {args.model}")

    if args.child_cold:
        if args.threads is None:
            args.threads = DEFAULTS["threads"]
        if args.seed is None:
            args.seed = DEFAULTS["seed"]
        return run_cold_child(args)

    if args.output is None:
        raise ValueError("--output is required")
    if args.skip_cold and args.skip_warm:
        raise ValueError("cannot use --skip-cold and --skip-warm together")

    config = load_config(args.config)
    warmup = int(setting(args, config, "warmup"))
    repeat = int(setting(args, config, "repeat"))
    cold_runs = int(setting(args, config, "cold_runs"))
    threads = int(setting(args, config, "threads"))
    seed = int(setting(args, config, "seed"))
    cv_threshold_pct = float(setting(args, config, "cv_threshold_pct"))
    cosine_threshold = float(setting(args, config, "cosine_threshold"))
    validate_positive_settings(warmup, repeat, cold_runs, threads)

    # Child commands need the resolved values, not the optional CLI defaults.
    args.threads = threads
    args.seed = seed

    ir = load_ir(args.ir)
    cold = run_cold_processes(args, cold_runs, threads, seed)
    warm, input_result, ort_version, numpy_version = run_warm(
        args,
        warmup,
        repeat,
        threads,
        seed,
        cv_threshold_pct,
        cosine_threshold,
        ir,
    )

    result = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "runtime": "onnxruntime-cpu",
        "model": {
            "path": str(args.model.resolve()),
            "sha256": sha256_file(args.model),
        },
        "configuration": {
            "threads": threads,
            "warmup": warmup,
            "repeat": repeat,
            "cold_runs": cold_runs,
            "seed": seed,
            "cv_threshold_pct": cv_threshold_pct,
            "cosine_threshold": cosine_threshold,
            "execution_provider": "CPUExecutionProvider",
            "execution_mode": "ORT_SEQUENTIAL",
        },
        "input": input_result,
        "ir": ir,
        "environment": system_metadata(ort_version, numpy_version),
        "cold": cold,
        "warm": warm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {args.output}")
    if warm:
        latency = warm["latency"]
        print(
            "warm: "
            f"p50={latency['p50_ms']:.3f} ms, "
            f"p95={latency['p95_ms']:.3f} ms, "
            f"p99={latency['p99_ms']:.3f} ms, "
            f"CV={latency['cv_pct']:.2f}% "
            f"({'PASS' if warm['stability_gate']['passed'] else 'FAIL'})"
        )
        accuracy = warm["accuracy"]
        if accuracy["evaluated"]:
            print(
                f"cosine={accuracy['cosine_similarity']:.9f} "
                f"({'PASS' if accuracy['passed'] else 'FAIL'})"
            )

    if args.enforce_gates and warm:
        stability_failed = not warm["stability_gate"]["passed"]
        accuracy = warm["accuracy"]
        accuracy_failed = accuracy["evaluated"] and not accuracy["passed"]
        if stability_failed or accuracy_failed:
            return 2
    return 0


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
