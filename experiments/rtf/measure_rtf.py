#!/usr/bin/env python3
"""짧은 음성 하나에 대한 C Runtime RTF를 반복 측정한다.

추론 시간을 줄여나가는 최적화 루프용이다. 정확도는 보지 않는다 —
그 판정은 `scripts/3_runtime/05_compare_runtime_outputs.py`의 몫이다.

측정 대상은 **추론 루프뿐**이다. WAV 읽기와 FBank 생성은 payload를 미리
만들어 제외하고, model load와 context create도 `lifecycle`에 따로 기록만 하고
RTF에는 넣지 않는다.

    python3 experiments/rtf/measure_rtf.py --label baseline
    python3 experiments/rtf/measure_rtf.py --label neon-qconv

각 실행은 `experiments/rtf/result/<UTC>__<label>.json`으로 남고, baseline이
있으면 표에 delta를 함께 출력한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT / "experiments" / "rtf" / "result"
BASELINE_NAME = "baseline.json"


class RtfError(RuntimeError):
    """측정을 신뢰할 수 없는 상태를 나타낸다."""


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise RtfError("빈 표본에서 백분위를 구할 수 없다")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def read_first_line(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def environment_snapshot(cpu: int) -> dict:
    governors = sorted(
        {
            text
            for text in (
                read_first_line(path)
                for path in Path("/sys/devices/system/cpu").glob(
                    "cpu[0-9]*/cpufreq/scaling_governor"
                )
            )
            if text
        }
    )
    temperatures = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        raw = read_first_line(path)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        temperatures.append(value / 1000.0 if abs(value) > 1000.0 else value)

    def command(argv: list[str]) -> str | None:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, check=False)
        except OSError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    build_info = {}
    build_note = ROOT / "build" / "campp_runtime_benchmark.build.txt"
    if build_note.is_file():
        for line in build_note.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key:
                build_info[key] = value

    return {
        "device_tree_model": (
            read_first_line(Path("/sys/firmware/devicetree/base/model")) or ""
        ).replace("\x00", ""),
        "kernel": os.uname().release,
        "machine": os.uname().machine,
        "logical_cpu_count": os.cpu_count(),
        "pinned_cpu": cpu,
        "cpu_governors": governors,
        "max_thermal_zone_c": max(temperatures) if temperatures else None,
        "git_commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_dirty": bool(
            command(["git", "-C", str(ROOT), "status", "--porcelain"]) or ""
        ),
        "compiler": command(["gcc", "-dumpfullversion"]),
        "build": build_info,
    }


def load_inputs(
    manifest_path: Path, bucket: int, count: int
) -> list[tuple[str, Path, float]]:
    if not manifest_path.is_file():
        raise RtfError(f"feature manifest가 없다: {manifest_path}")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [
        item
        for item in document.get("features", [])
        if int(item["bucket_frames"]) == bucket
    ]
    entries.sort(key=lambda item: item["input_id"])
    if len(entries) < count:
        raise RtfError(
            f"{bucket} frame payload가 {len(entries)}개뿐이다 (요구 {count}개)"
        )
    chosen = []
    for item in entries[:count]:
        path = ROOT / item["path"]
        if not path.is_file():
            raise RtfError(f"payload가 없다: {path}")
        chosen.append((item["input_id"], path, float(item["audio_seconds"])))
    return chosen


def run_one(
    binary: Path,
    bundle: Path,
    bucket: int,
    feature: Path,
    audio_seconds: float,
    warmup: int,
    repeat: int,
    threads: int,
) -> dict:
    plan = bundle / "execution_plans" / f"plan_{bucket}.bin"
    weights = bundle / "weights.bin"
    for required in (binary, plan, weights, feature):
        if not required.is_file():
            raise RtfError(f"필요한 파일이 없다: {required}")
    argv = [
        str(binary),
        "--plan", str(plan),
        "--weights", str(weights),
        "--input", str(feature),
        "--audio-seconds", str(audio_seconds),
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--threads", str(threads),
    ]
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RtfError(f"실행 실패({done.returncode}): {done.stderr.strip()}")
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise RtfError(f"JSON 출력을 읽을 수 없다: {exc}") from exc


def summarize(timings_ms: list[float], audio_seconds: float) -> dict:
    if not timings_ms:
        raise RtfError("측정 표본이 없다")
    rtfs = [ms / (audio_seconds * 1000.0) for ms in timings_ms]
    spread = (max(timings_ms) - min(timings_ms)) / min(timings_ms) * 100.0
    return {
        "samples": len(timings_ms),
        "latency_ms": {
            "min": min(timings_ms),
            "mean": statistics.fmean(timings_ms),
            "p50": _percentile(timings_ms, 0.5),
            "max": max(timings_ms),
        },
        "rtf": {
            "min": min(rtfs),
            "mean": statistics.fmean(rtfs),
            "p50": _percentile(rtfs, 0.5),
            "max": max(rtfs),
        },
        "spread_pct": spread,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="unlabeled", help="이 실행을 구분할 이름")
    parser.add_argument("--bucket", type=int, default=98, help="frame 수 (98=1초)")
    parser.add_argument("--inputs", type=int, default=3, help="측정할 음성 개수")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=0, help="고정할 CPU 번호")
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build" / "campp_runtime_benchmark"
    )
    parser.add_argument(
        "--bundle", type=Path, default=ROOT / "runs" / "runtime" / "tensor_arena" / "bundle"
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=ROOT / "benchmarks" / "campplus" / "manifests" / "runtime_features.json",
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=f"비교할 이전 결과 (기본: result/{BASELINE_NAME})",
    )
    parser.add_argument(
        "--save-as-baseline",
        action="store_true",
        help=f"이 결과를 result/{BASELINE_NAME}으로도 저장한다",
    )
    args = parser.parse_args(argv)

    try:
        try:
            os.sched_setaffinity(0, {args.cpu})
        except (AttributeError, OSError) as exc:
            raise RtfError(f"CPU {args.cpu} 고정에 실패했다: {exc}") from exc

        inputs = load_inputs(args.feature_manifest, args.bucket, args.inputs)
        environment = environment_snapshot(args.cpu)
        print(
            f"bucket={args.bucket} ({inputs[0][2]}s)  inputs={len(inputs)}  "
            f"warmup={args.warmup} repeat={args.repeat} threads={args.threads} "
            f"cpu={args.cpu}",
            flush=True,
        )

        measurements = []
        for input_id, feature, audio_seconds in inputs:
            raw = run_one(
                args.binary, args.bundle, args.bucket, feature, audio_seconds,
                args.warmup, args.repeat, args.threads,
            )
            timings = [float(v) for v in raw["warm"]["timings_ms"]]
            summary = summarize(timings, audio_seconds)
            memory = raw.get("memory", {})
            measurements.append(
                {
                    "input_id": input_id,
                    "audio_seconds": audio_seconds,
                    "timings_ms": timings,
                    **summary,
                    "lifecycle_ms": raw.get("lifecycle", {}),
                    "peak_rss_bytes": max(
                        (
                            value["peak_rss_bytes"]
                            for value in memory.values()
                            if isinstance(value, dict) and "peak_rss_bytes" in value
                        ),
                        default=None,
                    ),
                    "arena_bytes": memory.get("arena_bytes"),
                }
            )
            print(
                f"  {input_id}: RTF p50={summary['rtf']['p50']:.3f}  "
                f"{summary['latency_ms']['p50'] / 1000.0:.2f} s  "
                f"spread={summary['spread_pct']:.2f}%",
                flush=True,
            )

        pooled = [ms for item in measurements for ms in item["timings_ms"]]
        overall = summarize(pooled, inputs[0][2])
        document = {
            "schema_version": 1,
            "label": args.label,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_scope": (
                "추론 루프만. WAV 읽기·FBank·model load·context create 제외."
            ),
            "configuration": {
                "bucket_frames": args.bucket,
                "audio_seconds": inputs[0][2],
                "inputs": len(inputs),
                "warmup": args.warmup,
                "repeat": args.repeat,
                "threads": args.threads,
                "pinned_cpu": args.cpu,
                "bundle": str(args.bundle),
                "feature_manifest": str(args.feature_manifest),
            },
            "environment": environment,
            "overall": overall,
            "per_input": measurements,
        }

        args.result_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_label = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in args.label
        )
        destination = args.result_dir / f"{stamp}__{safe_label}.json"
        payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        destination.write_text(payload, encoding="utf-8")
        if args.save_as_baseline:
            (args.result_dir / BASELINE_NAME).write_text(payload, encoding="utf-8")

        print()
        print(f"RTF p50 {overall['rtf']['p50']:.3f}   "
              f"mean {overall['rtf']['mean']:.3f}   "
              f"min {overall['rtf']['min']:.3f}   "
              f"({overall['samples']} samples, spread {overall['spread_pct']:.2f}%)")

        baseline_path = args.baseline or (args.result_dir / BASELINE_NAME)
        if baseline_path.is_file() and baseline_path.resolve() != destination.resolve():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            before = baseline["overall"]["rtf"]["p50"]
            after = overall["rtf"]["p50"]
            change = (after - before) / before * 100.0
            print(
                f"baseline '{baseline['label']}' RTF p50 {before:.3f} → {after:.3f}  "
                f"({change:+.2f}%, {before / after:.2f}x)"
            )
        print(f"결과: {destination}")
        return 0
    except (RtfError, OSError, ValueError, KeyError) as exc:
        print(f"RTF 측정 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
