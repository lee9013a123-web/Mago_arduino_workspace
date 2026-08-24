#!/usr/bin/env python3
"""Run the weight-streaming gate independently for all selected buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SINGLE = ROOT / "scripts/5_model/09_benchmark_weight_streaming_98.py"
DEFAULT_RESULTS = ROOT / "results/models/campplus/final_v3/weight_streaming"
SUPPORTED_BUCKETS = (98, 298, 498, 998)


class MultibucketBenchmarkError(RuntimeError):
    pass


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _compact(document: dict[str, Any]) -> dict[str, Any]:
    baseline = document.get("variants", {}).get("v3_full", {})
    candidate = document.get("variants", {}).get("stream_windowed", {})
    return {
        "bucket_frames": document.get("bucket_frames"),
        "audio_seconds": document.get("audio_seconds"),
        "ready": document.get("ready"),
        "retained_bitwise": document.get(
            "retained_tensor_validation", {}
        ).get("passed"),
        "embedding_bitwise": document.get(
            "all_embeddings_bitwise_identical"
        ),
        "baseline_p50_ms": baseline.get("p50_ms"),
        "windowed_p50_ms": candidate.get("p50_ms"),
        "p50_delta_pct": document.get("p50_delta_pct"),
        "p95_delta_pct": document.get("p95_delta_pct"),
        "baseline_peak_rss_bytes": baseline.get("peak_rss_bytes"),
        "windowed_peak_rss_bytes": candidate.get("peak_rss_bytes"),
        "windowed_current_rss_bytes": candidate.get("current_rss_bytes"),
        "peak_rss_reduction_bytes": document.get(
            "peak_rss_reduction_bytes"
        ),
        "measurement_major_faults": document.get(
            "windowed_measurement_major_faults"
        ),
        "weight_event_trace": document.get("weight_event_trace"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket-frames", type=int, nargs="+",
        choices=SUPPORTED_BUCKETS, default=list(SUPPORTED_BUCKETS),
    )
    parser.add_argument("--mode", choices=("quick", "official"), default="quick")
    parser.add_argument("--results", type=_path, default=DEFAULT_RESULTS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--trace-weight-events", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        buckets = tuple(dict.fromkeys(args.bucket_frames))
        rows: list[dict[str, Any]] = []
        command_results: list[dict[str, Any]] = []
        for bucket in buckets:
            command = [
                sys.executable,
                str(SINGLE),
                "--bucket-frames", str(bucket),
                "--mode", args.mode,
                "--results", str(args.results),
            ]
            if args.preflight_only:
                command.append("--preflight-only")
            if args.trace_weight_events:
                command.append("--trace-weight-events")
            if args.force:
                command.append("--force")
            completed = subprocess.run(
                command, cwd=ROOT, text=True,
                capture_output=True, check=False,
            )
            command_results.append({
                "bucket_frames": bucket,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            })
            if args.preflight_only:
                if completed.returncode != 0:
                    raise MultibucketBenchmarkError(
                        f"bucket {bucket} preflight failed: "
                        f"{completed.stderr.strip()}"
                    )
                continue
            if completed.returncode not in (0, 3):
                raise MultibucketBenchmarkError(
                    f"bucket {bucket} benchmark failed: "
                    f"{completed.stderr.strip()}"
                )
            result_path = (
                args.results / str(bucket) / f"benchmark_{args.mode}.json"
            )
            if not result_path.is_file():
                raise MultibucketBenchmarkError(
                    f"bucket {bucket} result is missing: {result_path}"
                )
            document = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(_compact(document))
            print(
                f"bucket {bucket}: {'PASS' if document.get('ready') else 'FAIL'}",
                flush=True,
            )

        report = {
            "schema_version": 1,
            "ready": (
                True if args.preflight_only
                else all(row.get("ready") is True for row in rows)
            ),
            "mode": args.mode,
            "preflight_only": args.preflight_only,
            "buckets": list(buckets),
            "results": rows,
            "commands": command_results if args.preflight_only else None,
        }
        args.results.mkdir(parents=True, exist_ok=True)
        suffix = "preflight" if args.preflight_only else args.mode
        output = args.results / f"benchmark_multibucket_{suffix}.json"
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 3
    except (MultibucketBenchmarkError, OSError, ValueError) as exc:
        print(f"multibucket weight benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
