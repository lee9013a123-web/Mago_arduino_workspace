#!/usr/bin/env python3
"""windowed 실행 중 weight mapping RSS를 고빈도로 추적해 peak 발생 지점을 찾는다.

runtime을 건드리지 않고 밖에서 `/proc/<pid>/smaps`를 샘플링한다.  weights 파일
mapping의 Rss를 따로 뽑아 전체 RSS/VmHWM과 함께 기록하므로, peak가 weight 때문인지
다른 영역(arena, scratch, 바이너리) 때문인지 분리할 수 있다.

VmHWM은 high-water mark라 되돌아가지 않는다.  따라서 "언제 처음 그 값에 도달했는지"가
곧 peak 발생 시점이다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def read_status(pid: int) -> dict[str, int]:
    out = {}
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            for key in ("VmRSS:", "VmHWM:"):
                if line.startswith(key):
                    out[key.rstrip(":")] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return out


def read_smaps_rss(pid: int, needles: tuple[str, ...]) -> dict[str, int]:
    """파일 mapping별 Rss 합계.  needle이 경로에 포함된 mapping만 센다."""
    totals = {n: 0 for n in needles}
    current = None
    try:
        with open(f"/proc/{pid}/smaps", "r") as handle:
            for line in handle:
                if "-" in line.split()[0] and not line.startswith(" "):
                    path = line.rstrip("\n").split(" ", 5)[-1].strip()
                    current = next((n for n in needles if n in path), None)
                elif current and line.startswith("Rss:"):
                    totals[current] += int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path,
                        default=ROOT / "build/profill/weight_streaming_98"
                                       "/campp_runtime_benchmark_final")
    parser.add_argument("--plan", type=pathlib.Path,
                        default=ROOT / "runs/models/campplus/final_v3"
                                       "/weight_streaming/98/plan_98.bin")
    parser.add_argument("--weights", type=pathlib.Path,
                        default=ROOT / "runs/models/campplus/final_v3"
                                       "/weight_streaming/98/weights_98.bin")
    parser.add_argument("--schedule", type=pathlib.Path,
                        default=ROOT / "runs/models/campplus/final_v3"
                                       "/weight_streaming/98/"
                                       "weight_schedule_98.bin")
    parser.add_argument("--input", type=pathlib.Path,
                        default=ROOT / "benchmarks/campplus/features"
                                       "/multi__speaker_0000__98.f32")
    parser.add_argument("--weight-mode", default="windowed")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--interval-ms", type=float, default=2.0)
    parser.add_argument("--output", type=pathlib.Path,
                        default=ROOT / "results/models/campplus/final_v3"
                                       "/weight_streaming/98/rss_trace.json")
    args = parser.parse_args()

    command = [
        "taskset", "-c", "0", str(args.binary),
        "--plan", str(args.plan), "--weights", str(args.weights),
        "--input", str(args.input), "--audio-seconds", "1",
        "--warmup", str(args.warmup), "--repeat", str(args.repeat),
        "--threads", "1",
    ]
    if args.weight_mode:
        command += ["--weight-mode", args.weight_mode]
    if args.weight_mode == "windowed":
        command += ["--weight-schedule", str(args.schedule)]

    needles = (args.weights.name, args.plan.name, args.binary.name)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    samples = []
    started = time.perf_counter()
    while proc.poll() is None:
        status = read_status(proc.pid)
        if status:
            smaps = read_smaps_rss(proc.pid, needles)
            samples.append({
                "t_ms": (time.perf_counter() - started) * 1000.0,
                "rss": status.get("VmRSS", 0),
                "hwm": status.get("VmHWM", 0),
                "weights_rss": smaps.get(args.weights.name, 0),
                "plan_rss": smaps.get(args.plan.name, 0),
                "binary_rss": smaps.get(args.binary.name, 0),
            })
        time.sleep(args.interval_ms / 1000.0)
    stdout, stderr = proc.communicate()

    if not samples:
        print("샘플을 얻지 못했다 (프로세스가 너무 빨리 끝났을 수 있다)",
              file=sys.stderr)
        print(stderr[-500:], file=sys.stderr)
        return 1

    final_hwm = max(s["hwm"] for s in samples)
    # VmHWM이 처음 최대값에 도달한 샘플이 peak 발생 시점이다.
    peak_index = next(i for i, s in enumerate(samples) if s["hwm"] >= final_hwm)
    peak = samples[peak_index]
    max_weights = max(samples, key=lambda s: s["weights_rss"])

    print(f"샘플 {len(samples)}개, 간격 목표 {args.interval_ms} ms, "
          f"실행 {samples[-1]['t_ms']:.0f} ms")
    print(f"\nVmHWM 최종           : {final_hwm:,} B")
    print(f"VmHWM 최초 도달       : t={peak['t_ms']:.1f} ms "
          f"({peak_index}/{len(samples)} 샘플, 실행의 "
          f"{peak['t_ms'] / samples[-1]['t_ms'] * 100:.0f}% 지점)")
    print(f"  그 시점 RSS         : {peak['rss']:,} B")
    print(f"  그 시점 weights RSS : {peak['weights_rss']:,} B")
    print(f"  그 시점 binary RSS  : {peak['binary_rss']:,} B")
    print(f"\nweights RSS 최대     : {max_weights['weights_rss']:,} B "
          f"(t={max_weights['t_ms']:.1f} ms)")
    print(f"weights RSS 최종     : {samples[-1]['weights_rss']:,} B")
    print(f"RSS 최종             : {samples[-1]['rss']:,} B")

    print(f"\n{'t_ms':>9}{'RSS':>12}{'HWM':>12}{'weights':>11}{'binary':>10}")
    step = max(1, len(samples) // 25)
    for s in samples[::step]:
        print(f"{s['t_ms']:>9.1f}{s['rss']:>12,}{s['hwm']:>12,}"
              f"{s['weights_rss']:>11,}{s['binary_rss']:>10,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": 1,
        "weight_mode": args.weight_mode,
        "command": command,
        "sample_count": len(samples),
        "final_hwm_bytes": final_hwm,
        "peak_sample": peak,
        "max_weights_sample": max_weights,
        "final_sample": samples[-1],
        "samples": samples,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n결과: {args.output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
