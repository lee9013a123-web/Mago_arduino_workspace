#!/usr/bin/env python3
"""ORT 단독 실행기: bucket 하나를 warmup/repeat 하고 timing과 peak RSS를 낸다.

peak RSS를 backend별로 격리해 재려면 별도 프로세스여야 한다.  그래서 비교
스크립트가 이 파일을 subprocess로 띄운다.  C runtime의 유효 thread가 1개이므로
기본값도 intra/inter 모두 1로 맞춘다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    load_started = time.perf_counter()
    sess = ort.InferenceSession(args.model, options,
                                providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - load_started) * 1000.0

    raw = np.fromfile(args.input, dtype=np.float32).reshape(1, args.frames, 80)
    feed = {"feature": raw}

    first_started = time.perf_counter()
    out = sess.run(["embedding"], feed)[0]
    first_ms = (time.perf_counter() - first_started) * 1000.0

    for _ in range(args.warmup):
        sess.run(["embedding"], feed)

    timings = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        sess.run(["embedding"], feed)
        timings.append((time.perf_counter() - started) * 1000.0)

    # ru_maxrss는 리눅스에서 KiB 단위다.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    json.dump({
        "backend": "onnxruntime",
        "frames": args.frames,
        "threads": args.threads,
        "model_load_ms": load_ms,
        "first_inference_ms": first_ms,
        "timings_ms": timings,
        "peak_rss_bytes": peak_rss,
        "embedding_shape": list(out.shape),
    }, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
