#!/usr/bin/env python3
"""bucket별로 V3 full-resident와 weight-windowed의 RAM/latency를 잰다.

peak RSS(VmHWM)는 OOM과 직결되므로 gate 지표로 그대로 쓴다.  windowing이 정상
상태에서만 이득을 내는지 구분하려고 current RSS도 함께 남긴다.

전제: weights 매핑에 MADV_NOHUGEPAGE가 걸려 있어야 한다.  THP가 [always]인 커널에서는
파일 매핑이 2 MiB PMD로 backing되고, 그러면 부분 MADV_DONTNEED가 huge page를 쪼개지
못해 windowing이 통째로 무력화된다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
STREAM = ROOT / "runs/models/campplus/final_v3/weight_streaming"
FEATURES = ROOT / "benchmarks/campplus/features"
V3 = ROOT / "build/profill/final_v3_hybrid/campp_runtime_benchmark_final"
WS = (ROOT / "build/profill/weight_streaming_98"
           / "campp_runtime_benchmark_final")
AUDIO_SECONDS = {98: 1.0, 298: 3.0, 498: 5.0, 998: 10.0}
SPEAKER = "0000"


def run(binary: pathlib.Path, plan: pathlib.Path, weights: pathlib.Path,
        feature: pathlib.Path, seconds: float, warmup: int, repeat: int,
        schedule: pathlib.Path | None) -> dict:
    command = [
        "taskset", "-c", "0", str(binary),
        "--plan", str(plan), "--weights", str(weights),
        "--input", str(feature), "--audio-seconds", str(int(seconds)),
        "--warmup", str(warmup), "--repeat", str(repeat), "--threads", "1",
    ]
    if schedule is not None:
        command += ["--weight-mode", "windowed",
                    "--weight-schedule", str(schedule)]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    timings = sorted(payload["warm"]["timings_ms"])
    memory = payload["memory"]
    return {
        "mean_ms": statistics.mean(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": timings[max(0, int(len(timings) * 0.95) - 1)],
        "peak_rss_bytes": memory["after_measurement"]["peak_rss_bytes"],
        "current_rss_bytes": memory["after_measurement"]["current_rss_bytes"],
        "arena_bytes": memory.get("arena_bytes"),
        "rtf": statistics.mean(timings) / 1000.0 / seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, nargs="+",
                        default=[98, 298, 498, 998])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--output", type=pathlib.Path,
                        default=ROOT / "results/models/campplus/final_v3"
                                       "/weight_streaming/ram_by_bucket.json")
    args = parser.parse_args()

    rows = []
    for bucket in args.buckets:
        seconds = AUDIO_SECONDS[bucket]
        feature = FEATURES / f"multi__speaker_{SPEAKER}__{bucket}.f32"
        stream_dir = STREAM / str(bucket)
        needed = [feature,
                  BUNDLE / f"execution_plans/plan_{bucket}.bin",
                  stream_dir / f"plan_{bucket}.bin",
                  stream_dir / f"weights_{bucket}.bin",
                  stream_dir / f"weight_schedule_{bucket}.bin"]
        missing = [p for p in needed if not p.exists()]
        if missing:
            print(f"bucket {bucket}: 누락 {missing[0].name}", file=sys.stderr)
            continue

        full = run(V3, BUNDLE / f"execution_plans/plan_{bucket}.bin",
                   BUNDLE / "weights.bin", feature, seconds,
                   args.warmup, args.repeat, None)
        win = run(WS, stream_dir / f"plan_{bucket}.bin",
                  stream_dir / f"weights_{bucket}.bin", feature, seconds,
                  args.warmup, args.repeat,
                  stream_dir / f"weight_schedule_{bucket}.bin")
        plan = json.loads((ROOT / "results/models/campplus/final_v3"
                                  f"/weight_streaming/{bucket}/plan.json")
                          .read_text())
        rows.append({
            "bucket": bucket, "audio_seconds": seconds,
            "block_count": plan["block_count"],
            "logical_weight_bytes": plan["logical_weight_bytes"],
            "v3_full": full, "windowed": win,
            "peak_reduction_bytes": full["peak_rss_bytes"]
                                    - win["peak_rss_bytes"],
            "current_reduction_bytes": full["current_rss_bytes"]
                                       - win["current_rss_bytes"],
            "p50_delta_pct": (win["p50_ms"] - full["p50_ms"])
                             / full["p50_ms"] * 100.0,
        })
        print(f"  bucket {bucket} 완료", flush=True)

    print(f"\n=== Peak RSS (VmHWM) ===")
    print(f"{'bucket':>7}{'V3 full':>12}{'windowed':>12}{'절감':>12}{'절감율':>9}")
    for r in rows:
        f, w = r["v3_full"]["peak_rss_bytes"], r["windowed"]["peak_rss_bytes"]
        print(f"{r['bucket']:>7}{f:>12,}{w:>12,}{f - w:>12,}"
              f"{(f - w) / f * 100:>8.1f}%")

    print(f"\n=== Current RSS (정상 상태) ===")
    print(f"{'bucket':>7}{'V3 full':>12}{'windowed':>12}{'절감':>12}{'절감율':>9}")
    for r in rows:
        f = r["v3_full"]["current_rss_bytes"]
        w = r["windowed"]["current_rss_bytes"]
        print(f"{r['bucket']:>7}{f:>12,}{w:>12,}{f - w:>12,}"
              f"{(f - w) / f * 100:>8.1f}%")

    print(f"\n=== latency / RTF ===")
    print(f"{'bucket':>7}{'full p50':>11}{'win p50':>11}{'Δ%':>8}"
          f"{'full RTF':>10}{'win RTF':>10}")
    for r in rows:
        print(f"{r['bucket']:>7}{r['v3_full']['p50_ms']:>11.2f}"
              f"{r['windowed']['p50_ms']:>11.2f}{r['p50_delta_pct']:>7.2f}%"
              f"{r['v3_full']['rtf']:>10.4f}{r['windowed']['rtf']:>10.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"schema_version": 1, "speaker": SPEAKER,
         "warmup": args.warmup, "repeat": args.repeat, "buckets": rows},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n결과: {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
