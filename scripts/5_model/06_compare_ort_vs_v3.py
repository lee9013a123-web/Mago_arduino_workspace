#!/usr/bin/env python3
"""ONNX Runtime과 Final V3 C Runtime을 RTF / RAM 으로 비교한다.

V3가 이제 baseline이므로 비교 대상은 V2가 아니라 ORT다.

    RTF = 추론 latency(초) / 오디오 길이(초)
    RAM = 프로세스 peak RSS (model load + context + 추론 전체)

두 backend를 bucket마다 번갈아 실행한다.  CPU governor가 schedutil이면 절대값이
흔들리지만 번갈아 재므로 비율은 유지된다.  governor를 performance로 고정하면
절대값 재현성까지 확보된다(root 필요).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
FEATURES = ROOT / "benchmarks/campplus/features"
MODEL = ROOT / "models/source/campplus_int8_static_qop.onnx"
V3 = ROOT / "build/profill/final_v3_hybrid/campp_runtime_benchmark_final"
ORT_RUNNER = ROOT / "scripts/5_model/06_ort_runner.py"
PYTHON = os.environ.get("PY", "/home/arduino/workspace/venv/bin/python3")
# 25ms/10ms 기준 frame 수 -> 오디오 길이.  RTF 분모다.
AUDIO_SECONDS = {98: 1.0, 298: 3.0, 498: 5.0, 998: 10.0}
SPEAKER = "0000"


def cpu_frequency_mhz() -> float | None:
    path = pathlib.Path(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    try:
        return int(path.read_text().strip()) / 1000.0
    except OSError:
        return None


def governor() -> str:
    path = pathlib.Path(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text().strip()
    except OSError:
        return "unknown"


def run_ort(bucket: int, feature: pathlib.Path,
            warmup: int, repeat: int, threads: int) -> dict:
    proc = subprocess.run(
        ["taskset", "-c", "0", PYTHON, str(ORT_RUNNER),
         "--model", str(MODEL), "--input", str(feature),
         "--frames", str(bucket), "--warmup", str(warmup),
         "--repeat", str(repeat), "--threads", str(threads)],
        check=True, capture_output=True, text=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_v3(bucket: int, feature: pathlib.Path,
           warmup: int, repeat: int) -> dict:
    proc = subprocess.run(
        ["taskset", "-c", "0", str(V3),
         "--plan", str(BUNDLE / f"execution_plans/plan_{bucket}.bin"),
         "--weights", str(BUNDLE / "weights.bin"),
         "--input", str(feature),
         "--audio-seconds", str(int(AUDIO_SECONDS[bucket])),
         "--warmup", str(warmup), "--repeat", str(repeat), "--threads", "1"],
        check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "backend": "campp_c_runtime_v3",
        "model_load_ms": payload["lifecycle"]["model_load_ms"],
        "first_inference_ms": payload["lifecycle"]["first_inference_ms"],
        "timings_ms": payload["warm"]["timings_ms"],
        "peak_rss_bytes": payload["memory"]["after_measurement"][
            "peak_rss_bytes"],
        "arena_bytes": payload["memory"].get("arena_bytes"),
    }


def summarise(raw: dict, audio_seconds: float) -> dict:
    t = sorted(raw["timings_ms"])
    mean_ms = statistics.mean(t)
    return {
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(t),
        "p95_ms": t[max(0, int(len(t) * 0.95) - 1)],
        "rtf": mean_ms / 1000.0 / audio_seconds,
        "peak_rss_mb": raw["peak_rss_bytes"] / (1024 * 1024),
        "model_load_ms": raw["model_load_ms"],
        "first_inference_ms": raw["first_inference_ms"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, nargs="+",
                        default=[98, 298, 498, 998])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--ort-threads", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    if not V3.exists():
        print(f"V3 바이너리가 없다: {V3}", file=sys.stderr)
        return 1

    gov = governor()
    if gov != "performance":
        print(f"주의: CPU governor가 '{gov}'다.  절대값 재현성이 떨어진다.")
        print("      performance 고정: for c in 0 1 2 3; do echo performance |"
              " sudo tee /sys/devices/system/cpu/cpu$c/cpufreq/"
              "scaling_governor; done\n")

    rows = []
    for bucket in args.buckets:
        feature = FEATURES / f"multi__speaker_{SPEAKER}__{bucket}.f32"
        if not feature.exists():
            print(f"bucket {bucket}: feature 없음", file=sys.stderr)
            continue
        seconds = AUDIO_SECONDS[bucket]
        entry = {"bucket": bucket, "audio_seconds": seconds,
                 "cpu_mhz_at_start": cpu_frequency_mhz()}
        # 번갈아 실행해 governor 변동이 한쪽에만 몰리지 않게 한다.
        entry["v3"] = summarise(
            run_v3(bucket, feature, args.warmup, args.repeat), seconds)
        for threads in args.ort_threads:
            entry[f"ort_t{threads}"] = summarise(
                run_ort(bucket, feature, args.warmup, args.repeat, threads),
                seconds)
        rows.append(entry)
        print(f"  bucket {bucket} 완료", flush=True)

    print(f"\n=== RTF (낮을수록 좋다) | governor={gov} ===")
    head = f"{'bucket':>7}{'audio s':>9}{'V3 RTF':>10}"
    for t in args.ort_threads:
        head += f"{'ORT t' + str(t):>11}"
    head += f"{'V3 우위':>10}"
    print(head)
    for r in rows:
        line = f"{r['bucket']:>7}{r['audio_seconds']:>9.0f}{r['v3']['rtf']:>10.4f}"
        for t in args.ort_threads:
            line += f"{r[f'ort_t{t}']['rtf']:>11.4f}"
        base = r[f"ort_t{args.ort_threads[0]}"]["rtf"]
        line += f"{base / r['v3']['rtf']:>9.2f}x"
        print(line)

    print(f"\n=== Peak RSS (MB) ===")
    head = f"{'bucket':>7}{'V3':>10}"
    for t in args.ort_threads:
        head += f"{'ORT t' + str(t):>11}"
    head += f"{'V3 절감':>10}"
    print(head)
    for r in rows:
        line = f"{r['bucket']:>7}{r['v3']['peak_rss_mb']:>10.1f}"
        for t in args.ort_threads:
            line += f"{r[f'ort_t{t}']['peak_rss_mb']:>11.1f}"
        base = r[f"ort_t{args.ort_threads[0]}"]["peak_rss_mb"]
        line += f"{(base - r['v3']['peak_rss_mb']) / base * 100:>9.1f}%"
        print(line)

    print(f"\n=== latency mean / p50 / p95 (ms) ===")
    print(f"{'bucket':>7}{'backend':>12}{'mean':>10}{'p50':>10}{'p95':>10}"
          f"{'load':>9}{'first':>10}")
    for r in rows:
        for key in ["v3"] + [f"ort_t{t}" for t in args.ort_threads]:
            s = r[key]
            print(f"{r['bucket']:>7}{key:>12}{s['mean_ms']:>10.2f}"
                  f"{s['p50_ms']:>10.2f}{s['p95_ms']:>10.2f}"
                  f"{s['model_load_ms']:>9.1f}{s['first_inference_ms']:>10.1f}")

    out = ROOT / "results/model_evaluation/final_v3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ort_vs_v3_rtf_ram.json"
    path.write_text(json.dumps(
        {"schema_version": 1, "governor": gov,
         "warmup": args.warmup, "repeat": args.repeat,
         "speaker": SPEAKER, "buckets": rows},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n결과: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
