#!/usr/bin/env python3
"""Isolated ONNX Runtime process for one fixed-bucket embedding request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


FEATURE_BINS = 80
EMBEDDING_DIMENSION = 192


def read_linux_memory() -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "current_rss_bytes": None,
        "peak_rss_bytes": None,
    }
    status = Path("/proc/self/status")
    if not status.is_file():
        return result
    for line in status.read_text(
        encoding="utf-8", errors="replace",
    ).splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        if fields[0] == "VmRSS:":
            result["current_rss_bytes"] = int(fields[1]) * 1024
        elif fields[0] == "VmHWM:":
            result["peak_rss_bytes"] = int(fields[1]) * 1024
    return result


def _capabilities() -> int:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(f"ORT dependency check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "backend": "onnxruntime-cpu",
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "available_providers": ort.get_available_providers(),
        "required_provider": "CPUExecutionProvider",
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--embedding-output", type=Path)
    parser.add_argument("--audio-seconds", type=float)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.capabilities:
        return _capabilities()
    if any(value is None for value in (
        args.model, args.input, args.frames,
        args.embedding_output, args.audio_seconds,
    )):
        parser.error("model, input, frames, embedding-output and audio-seconds are required")
    if args.frames <= 0 or args.audio_seconds <= 0:
        parser.error("frames and audio-seconds must be positive")
    if args.warmup < 0 or args.repeat <= 0 or args.threads <= 0:
        parser.error("warmup/repeat/threads values are invalid")
    if not args.model.is_file() or not args.input.is_file():
        parser.error("model or input file is missing")

    memory_start = read_linux_memory()
    process_started = time.perf_counter()
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(f"ORT dependency import failed: {exc}", file=sys.stderr)
        return 1

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    load_started = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model), options, providers=["CPUExecutionProvider"],
    )
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    memory_after_model = read_linux_memory()

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "feature":
        raise RuntimeError("ONNX model must expose one input named 'feature'")
    if not outputs or not any(item.name == "embedding" for item in outputs):
        raise RuntimeError("ONNX model has no 'embedding' output")
    expected_bytes = args.frames * FEATURE_BINS * 4
    if args.input.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"feature must contain [1,{args.frames},{FEATURE_BINS}] float32"
        )
    feature = np.fromfile(args.input, dtype="<f4").reshape(
        1, args.frames, FEATURE_BINS,
    )
    feed = {"feature": feature}

    first_started = time.perf_counter()
    embedding = session.run(["embedding"], feed)[0]
    first_inference_ms = (time.perf_counter() - first_started) * 1000.0
    for _ in range(args.warmup):
        session.run(["embedding"], feed)
    timings: list[float] = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        embedding = session.run(["embedding"], feed)[0]
        timings.append((time.perf_counter() - started) * 1000.0)

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != EMBEDDING_DIMENSION or not np.isfinite(vector).all():
        raise RuntimeError("ORT embedding is not a finite 192-float vector")
    args.embedding_output.parent.mkdir(parents=True, exist_ok=True)
    vector.astype("<f4").tofile(args.embedding_output)
    memory_after_measurement = read_linux_memory()
    latency_mean_ms = statistics.fmean(timings)
    print(json.dumps({
        "schema_version": 1,
        "backend": "onnxruntime-cpu",
        "provider": "CPUExecutionProvider",
        "frames": args.frames,
        "audio_seconds": args.audio_seconds,
        "threads": args.threads,
        "model": {
            "path": str(args.model),
            "file_bytes": args.model.stat().st_size,
        },
        "lifecycle": {
            "model_load_ms": model_load_ms,
            "first_inference_ms": first_inference_ms,
            "process_elapsed_ms": (time.perf_counter() - process_started) * 1000.0,
        },
        "warm": {
            "timings_ms": timings,
            "mean_ms": latency_mean_ms,
        },
        "memory": {
            "start": memory_start,
            "after_model_load": memory_after_model,
            "after_measurement": memory_after_measurement,
            "activation_bytes": None,
        },
        "embedding": {
            "shape": list(np.asarray(embedding).shape),
            "dimension": EMBEDDING_DIMENSION,
            "output_path": str(args.embedding_output),
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
