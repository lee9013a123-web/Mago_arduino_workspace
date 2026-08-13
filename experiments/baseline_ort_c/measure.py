#!/usr/bin/env python3
"""정적 ONNX 모델의 ORT baseline을 native(C API)와 Python 양쪽으로 측정한다.

두 backend는 같은 모델·같은 payload·같은 스레드 설정으로 같은 추론을 한다.
차이는 프로세스에 Python 인터프리터와 numpy가 있느냐뿐이다. 그래서 latency는
거의 같고 RSS만 갈라진다 — 이 스크립트의 목적이 그 차이를 재는 것이다.

    python3 experiments/baseline_ort_c/measure.py --label baseline
    python3 experiments/baseline_ort_c/measure.py --buckets 98 --native-only

결과는 `experiments/baseline_ort_c/result/<UTC>__<label>.json`에 저장된다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT / "experiments" / "baseline_ort_c" / "result"
DEFAULT_BUCKETS = (98, 298, 498, 998)


class MeasureError(RuntimeError):
    """측정을 신뢰할 수 없는 상태를 나타낸다."""


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise MeasureError("빈 표본")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (1.0 - (position - low)) + ordered[high] * (position - low)


def read_first_line(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def environment_snapshot(cpu: int) -> dict:
    def command(argv: list[str]) -> str | None:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, check=False)
        except OSError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

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

    build_info = {}
    note = ROOT / "build" / "campp_ort_benchmark.build.txt"
    if note.is_file():
        for line in note.read_text(encoding="utf-8").splitlines():
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
        "git_dirty": bool(command(["git", "-C", str(ROOT), "status", "--porcelain"]) or ""),
        "compiler": command(["gcc", "-dumpfullversion"]),
        "build": build_info,
    }


def load_inputs(manifest: Path, bucket: int, count: int) -> list[tuple[str, Path, float]]:
    if not manifest.is_file():
        raise MeasureError(f"feature manifest가 없다: {manifest}")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    entries = [
        item for item in document.get("features", [])
        if int(item["bucket_frames"]) == bucket
    ]
    entries.sort(key=lambda item: item["input_id"])
    if len(entries) < count:
        raise MeasureError(f"{bucket} frame payload가 {len(entries)}개뿐이다")
    chosen = []
    for item in entries[:count]:
        path = ROOT / item["path"]
        if not path.is_file():
            raise MeasureError(f"payload가 없다: {path}")
        chosen.append((item["input_id"], path, float(item["audio_seconds"])))
    return chosen


def run_native(
    binary: Path, model: Path, feature: Path, audio_seconds: float,
    warmup: int, repeat: int, threads: int, graph_opt: str,
) -> tuple[list[float], int, float]:
    argv = [
        str(binary), "--model", str(model), "--input", str(feature),
        "--audio-seconds", str(audio_seconds), "--warmup", str(warmup),
        "--repeat", str(repeat), "--threads", str(threads),
        "--graph-opt", graph_opt,
    ]
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise MeasureError(f"native 실행 실패({done.returncode}): {done.stderr.strip()}")
    raw = json.loads(done.stdout)
    timings = [float(v) for v in raw["warm"]["timings_ms"]]
    peak = max(
        value["peak_rss_bytes"]
        for value in raw["memory"].values()
        if isinstance(value, dict) and value.get("peak_rss_bytes") is not None
    )
    return timings, int(peak), float(raw["lifecycle"]["model_load_ms"])


def run_python(
    script: Path, model: Path, feature: Path, frames: int, audio_seconds: float,
    warmup: int, repeat: int, threads: int,
) -> tuple[list[float], int, float]:
    with tempfile.TemporaryDirectory() as workspace:
        output = Path(workspace) / "result.json"
        argv = [
            sys.executable, str(script), "--model", str(model),
            "--input-f32", str(feature), "--frames", str(frames),
            "--audio-seconds", str(audio_seconds), "--warmup", str(warmup),
            "--repeat", str(repeat), "--threads", str(threads),
            "--skip-cold", "--output", str(output),
        ]
        done = subprocess.run(argv, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise MeasureError(f"python 실행 실패({done.returncode}): {done.stderr.strip()[-400:]}")
        raw = json.loads(output.read_text(encoding="utf-8"))
    warm = raw["warm"]
    timings = [float(v) for v in warm["timings_ms"]]
    return timings, int(warm["memory"]["peak_rss_bytes"]), float(warm["session_create_ms"])


def summarize(timings: list[float], audio_seconds: float) -> dict:
    rtfs = [ms / (audio_seconds * 1000.0) for ms in timings]
    return {
        "samples": len(timings),
        "latency_ms": {
            "min": min(timings),
            "mean": statistics.fmean(timings),
            "p50": _percentile(timings, 0.5),
            "p95": _percentile(timings, 0.95),
            "max": max(timings),
        },
        "rtf": {"p50": _percentile(rtfs, 0.5), "mean": statistics.fmean(rtfs)},
        "spread_pct": (max(timings) - min(timings)) / min(timings) * 100.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--buckets", type=int, nargs="*", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--inputs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--graph-opt", default="all",
                        choices=("disable", "basic", "extended", "all"))
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "results" / "static")
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "build" / "campp_ort_benchmark")
    parser.add_argument("--python-script", type=Path,
                        default=ROOT / "scripts" / "1_benchmark" / "benchmark_onnx.py")
    parser.add_argument("--feature-manifest", type=Path,
                        default=ROOT / "benchmarks" / "campplus" / "manifests"
                        / "runtime_features.json")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args(argv)

    try:
        try:
            os.sched_setaffinity(0, {args.cpu})
        except (AttributeError, OSError) as exc:
            raise MeasureError(f"CPU {args.cpu} 고정 실패: {exc}") from exc
        if not args.binary.is_file():
            raise MeasureError(
                f"native 바이너리가 없다: {args.binary}\n"
                "  bash experiments/baseline_ort_c/build.sh"
            )

        environment = environment_snapshot(args.cpu)
        buckets_out = []
        print(f"warmup={args.warmup} repeat={args.repeat} threads={args.threads} "
              f"cpu={args.cpu} graph_opt={args.graph_opt}", flush=True)

        for bucket in args.buckets:
            model = args.model_dir / f"campp_static_{bucket}.onnx"
            if not model.is_file():
                raise MeasureError(f"모델이 없다: {model}")
            inputs = load_inputs(args.feature_manifest, bucket, args.inputs)
            audio_seconds = inputs[0][2]
            print(f"\n[{bucket} frame / {audio_seconds}s] {model.name}", flush=True)

            native_timings: list[float] = []
            native_peaks: list[int] = []
            native_loads: list[float] = []
            python_timings: list[float] = []
            python_peaks: list[int] = []
            python_loads: list[float] = []

            for input_id, feature, seconds in inputs:
                timings, peak, load = run_native(
                    args.binary, model, feature, seconds,
                    args.warmup, args.repeat, args.threads, args.graph_opt,
                )
                native_timings += timings
                native_peaks.append(peak)
                native_loads.append(load)
                line = (f"  {input_id}  native p50={_percentile(timings, 0.5):8.2f} ms  "
                        f"RSS={peak / 1e6:6.2f} MB")
                if not args.native_only:
                    timings, peak, load = run_python(
                        args.python_script, model, feature, bucket, seconds,
                        args.warmup, args.repeat, args.threads,
                    )
                    python_timings += timings
                    python_peaks.append(peak)
                    python_loads.append(load)
                    line += (f"   |  python p50={_percentile(timings, 0.5):8.2f} ms  "
                             f"RSS={peak / 1e6:6.2f} MB")
                print(line, flush=True)

            entry = {
                "bucket_frames": bucket,
                "audio_seconds": audio_seconds,
                "model": str(model.relative_to(ROOT)),
                "inputs": [item[0] for item in inputs],
                "native": {
                    **summarize(native_timings, audio_seconds),
                    "peak_rss_bytes": max(native_peaks),
                    "session_create_ms": statistics.fmean(native_loads),
                },
            }
            if not args.native_only:
                entry["python"] = {
                    **summarize(python_timings, audio_seconds),
                    "peak_rss_bytes": max(python_peaks),
                    "session_create_ms": statistics.fmean(python_loads),
                }
                entry["python_overhead"] = {
                    "rss_bytes": entry["python"]["peak_rss_bytes"]
                    - entry["native"]["peak_rss_bytes"],
                    "latency_p50_ms": entry["python"]["latency_ms"]["p50"]
                    - entry["native"]["latency_ms"]["p50"],
                }
            buckets_out.append(entry)

        document = {
            "schema_version": 1,
            "label": args.label,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_scope": (
                "warm 추론 루프만. WAV·FBank·session create 제외 "
                "(session create는 별도 기록)."
            ),
            "configuration": {
                "buckets": args.buckets,
                "inputs": args.inputs,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "threads": args.threads,
                "pinned_cpu": args.cpu,
                "graph_optimization_level": args.graph_opt,
                "model_dir": str(args.model_dir.relative_to(ROOT)),
                "feature_manifest": str(args.feature_manifest.relative_to(ROOT)),
            },
            "environment": environment,
            "buckets": buckets_out,
        }
        args.result_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in args.label)
        destination = args.result_dir / f"{stamp}__{safe}.json"
        destination.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        print("\n" + "=" * 78)
        header = f"{'bucket':>7} {'sec':>5} {'native p50':>12} {'native RSS':>11}"
        if not args.native_only:
            header += f" {'python p50':>12} {'python RSS':>11} {'RSS 절감':>10}"
        print(header)
        for entry in buckets_out:
            row = (f"{entry['bucket_frames']:>7} {entry['audio_seconds']:>5.1f} "
                   f"{entry['native']['latency_ms']['p50']:>9.2f} ms "
                   f"{entry['native']['peak_rss_bytes'] / 1e6:>8.2f} MB")
            if not args.native_only:
                row += (f" {entry['python']['latency_ms']['p50']:>9.2f} ms "
                        f"{entry['python']['peak_rss_bytes'] / 1e6:>8.2f} MB "
                        f"{entry['python_overhead']['rss_bytes'] / 1e6:>7.2f} MB")
            print(row)
        print(f"\n결과: {destination}")
        return 0
    except (MeasureError, OSError, ValueError, KeyError) as exc:
        print(f"측정 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
