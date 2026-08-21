#!/usr/bin/env python3
"""Run and merge ONNX Runtime vs final C Runtime evaluation for four buckets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
BUCKETS = (98, 298, 498, 998)
BUCKET_TAGS = {98: "1s", 298: "3s", 498: "5s", 998: "10s"}
FINAL_V3_SUITE = (
    "qconv_layer_hybrid_v3+fused_layer_hybrid_v3+bn_v2_spatial2+"
    "dequant_neon_combined+fused_dqrq_neon+remaining_optimized"
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class EvaluationError(RuntimeError):
    pass


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _run(command: Sequence[str], *, environment: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        list(command), cwd=ROOT, env=environment, check=False
    )
    if completed.returncode != 0:
        raise EvaluationError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


def _run_json(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command), cwd=ROOT, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise EvaluationError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise EvaluationError("runtime capabilities are not valid JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationError("runtime capabilities JSON must be an object")
    return value


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read JSON {path}: {exc}") from exc


def _require_files(paths: Sequence[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise EvaluationError(
            "required files are missing:\n  " + "\n  ".join(str(path) for path in missing)
        )


def require_compiled_v3_buckets(
    capabilities: dict[str, Any], requested_buckets: Sequence[int]
) -> set[int]:
    plans = capabilities.get("optimization_bucket_plans")
    if not isinstance(plans, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in plans
    ):
        raise EvaluationError("C Runtime V3 bucket plan list is invalid")
    compiled = set(plans)
    missing = sorted(set(requested_buckets) - compiled)
    if missing:
        raise EvaluationError(
            "C Runtime is not a full multibucket V3 build; missing plans: "
            f"{missing}. Run 16_select_multibucket_v3.py --mode official "
            "--force --build-final first."
        )
    return compiled


def _verify_runtime_capabilities(
    binary: Path, requested_buckets: Sequence[int]
) -> tuple[dict[str, Any], set[int]]:
    capabilities = _run_json([str(binary), "--capabilities"])
    if capabilities.get("optimization_suite_config") != FINAL_V3_SUITE:
        raise EvaluationError("C Runtime binary is not the Final V3 suite")
    if capabilities.get("optimization_bucket_policy_source") != (
        "compiled_bucket_plan"
    ):
        raise EvaluationError(
            "C Runtime must be rebuilt: compiled bucket policy is unavailable"
        )
    compiled = require_compiled_v3_buckets(capabilities, requested_buckets)
    return capabilities, compiled


def _metric(document: dict[str, Any], *keys: str) -> Any:
    value: Any = document
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def build_frame_matrix(
    accuracy: Sequence[dict[str, Any]],
    performance: dict[str, Any],
    buckets: Sequence[int],
    *,
    expected_policies: dict[int, str],
    embedding_cosine_min: float,
    baseline_98_rtf: float,
    max_regression_pct: float,
) -> dict[str, Any]:
    accuracy_by_bucket = {
        int(item["bucket_frames"]): item for item in accuracy
    }
    ort_by_bucket = performance["aggregates"]["onnxruntime-cpu"]
    c_by_bucket = performance["aggregates"]["campp-c-runtime"]
    comparisons = {
        int(item["bucket_frames"]): item
        for item in performance.get("comparisons", [])
    }
    direct_accuracy = performance.get("accuracy", {}).get("buckets", {})
    rows: list[dict[str, Any]] = []
    for frames in buckets:
        accuracy_item = accuracy_by_bucket.get(frames)
        ort = ort_by_bucket.get(str(frames))
        crt = c_by_bucket.get(str(frames))
        comparison = comparisons.get(frames)
        direct_item = direct_accuracy.get(str(frames))
        if (
            accuracy_item is None
            or ort is None
            or crt is None
            or comparison is None
            or direct_item is None
        ):
            raise EvaluationError(f"incomplete evaluation result for {frames} frames")
        embedding = accuracy_item.get("embedding") or {}
        actual_policy = crt.get("optimization_bucket_policy")
        expected_policy = expected_policies[frames]
        crt_rtf = _metric(crt, "rtf", "p50_rtf")
        regression_pct = None
        baseline_passed = True
        if frames == 98 and crt_rtf is not None:
            regression_pct = (float(crt_rtf) / baseline_98_rtf - 1.0) * 100.0
            baseline_passed = regression_pct <= max_regression_pct
        stability_passed = bool(
            _metric(ort, "stability_gate", "all_inputs_passed")
            and _metric(ort, "stability_gate", "aggregate_passed")
            and _metric(crt, "stability_gate", "all_inputs_passed")
            and _metric(crt, "stability_gate", "aggregate_passed")
        )
        row = {
            "bucket_frames": frames,
            "audio_seconds": {98: 1.0, 298: 3.0, 498: 5.0, 998: 10.0}[frames],
            "expected_c_policy": expected_policy,
            "actual_c_policy": actual_policy,
            "policy_passed": actual_policy == expected_policy,
            "embedding_cosine": direct_item.get("minimum_cosine_similarity"),
            "embedding_finite": direct_item.get("all_finite"),
            "accuracy_passed": bool(
                direct_item.get("all_passed") and accuracy_item.get("gate_passed")
            ),
            "retained_reference_cosine": embedding.get("cosine_similarity"),
            "retained_reference_gate_passed": accuracy_item.get("gate_passed"),
            "ort_first_p50_ms": _metric(ort, "first_inference", "p50_ms"),
            "crt_first_p50_ms": _metric(crt, "first_inference", "p50_ms"),
            "ort_warm_p50_ms": _metric(ort, "latency", "p50_ms"),
            "crt_warm_p50_ms": _metric(crt, "latency", "p50_ms"),
            "ort_warm_p95_ms": _metric(ort, "latency", "p95_ms"),
            "crt_warm_p95_ms": _metric(crt, "latency", "p95_ms"),
            "ort_warm_p99_ms": _metric(ort, "latency", "p99_ms"),
            "crt_warm_p99_ms": _metric(crt, "latency", "p99_ms"),
            "ort_rtf_p50": _metric(ort, "rtf", "p50_rtf"),
            "crt_rtf_p50": crt_rtf,
            "crt_speedup_over_ort_p50": comparison.get("c_speedup_over_ort_p50"),
            "ort_peak_rss_bytes": ort.get("peak_rss_bytes"),
            "crt_peak_rss_bytes": crt.get("peak_rss_bytes"),
            "ort_incremental_peak_rss_bytes": ort.get(
                "incremental_peak_rss_bytes"
            ),
            "crt_incremental_peak_rss_bytes": crt.get(
                "incremental_peak_rss_bytes"
            ),
            "crt_peak_rss_reduction_ratio": comparison.get(
                "peak_rss_reduction_ratio"
            ),
            "crt_incremental_peak_rss_reduction_ratio": comparison.get(
                "incremental_peak_rss_reduction_ratio"
            ),
            "stability_passed": stability_passed,
            "baseline_98_rtf": baseline_98_rtf if frames == 98 else None,
            "baseline_98_regression_pct": regression_pct,
            "baseline_98_gate_passed": baseline_passed,
        }
        row["passed"] = bool(
            row["accuracy_passed"]
            and row["embedding_finite"]
            and row["policy_passed"]
            and row["stability_passed"]
            and row["baseline_98_gate_passed"]
        )
        rows.append(row)
    return {
        "schema_version": 1,
        "comparison": "canonical_onnxruntime_vs_multibucket_v3_c_runtime",
        "required_c_policy": "layer_hybrid_v3",
        "accuracy_gate": {
            "kind": "embedding_cosine",
            "minimum": embedding_cosine_min,
            "bitwise_required": False,
            "finite_required": True,
        },
        "baseline_98_gate": {
            "rtf": baseline_98_rtf,
            "maximum_regression_pct": max_regression_pct,
        },
        "rows": rows,
        "all_passed": all(row["passed"] for row in rows),
    }


def _format_number(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def write_matrix(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frame_matrix.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = report["rows"]
    if rows:
        with (output_dir / "frame_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# ONNX Runtime vs Final C Runtime",
        "",
        "| frame | C policy | cosine | ORT RTF | CRT RTF | speedup | "
        "ORT peak MiB | CRT peak MiB | gate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        ort_mib = (
            float(row["ort_peak_rss_bytes"]) / (1024.0 * 1024.0)
            if row["ort_peak_rss_bytes"] is not None
            else None
        )
        crt_mib = (
            float(row["crt_peak_rss_bytes"]) / (1024.0 * 1024.0)
            if row["crt_peak_rss_bytes"] is not None
            else None
        )
        lines.append(
            f"| {row['bucket_frames']} | {row['actual_c_policy']} | "
            f"{_format_number(row['embedding_cosine'], 6)} | "
            f"{_format_number(row['ort_rtf_p50'])} | "
            f"{_format_number(row['crt_rtf_p50'])} | "
            f"{_format_number(row['crt_speedup_over_ort_p50'], 2)}x | "
            f"{_format_number(ort_mib, 2)} | {_format_number(crt_mib, 2)} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    (output_dir / "frame_matrix.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def default_run_id(mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"final-v3-onnx-crt-{mode}-{stamp}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "official"), default="quick")
    parser.add_argument("--buckets", type=int, nargs="+", default=list(BUCKETS))
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument("--embedding-cosine-min", type=float, default=0.99)
    parser.add_argument("--baseline-98-rtf", type=float, default=0.522322)
    parser.add_argument("--max-baseline-regression-pct", type=float, default=3.0)
    parser.add_argument(
        "--binary",
        type=_path,
        default=ROOT / "build/profill/final_v3_hybrid/campp_runtime_benchmark_final",
    )
    parser.add_argument(
        "--dump-binary",
        type=_path,
        default=ROOT / "build/profill/final_v3_hybrid/campp_reference_dump_final",
    )
    parser.add_argument(
        "--bundle",
        type=_path,
        default=ROOT / "runs/runtime/kernel_optimization/e7/bundle",
    )
    parser.add_argument(
        "--ort-dir",
        type=_path,
        default=ROOT / "runs/runtime/ort_reference",
    )
    args = parser.parse_args(argv)

    if args.skip_accuracy and args.skip_performance:
        parser.error("cannot skip both accuracy and performance")
    if not 0.0 <= args.embedding_cosine_min <= 1.0:
        parser.error("--embedding-cosine-min must be in [0, 1]")
    unknown = sorted(set(args.buckets) - set(BUCKETS))
    if unknown:
        parser.error(f"unsupported buckets: {unknown}")
    buckets = tuple(frames for frames in BUCKETS if frames in set(args.buckets))
    run_id = args.run_id or default_run_id(args.mode)
    if not RUN_ID_PATTERN.fullmatch(run_id):
        parser.error("invalid --run-id")

    config = ROOT / "configs/benchmark" / (
        "runtime_final_v3_multibucket_quick.json"
        if args.mode == "quick"
        else "runtime_final_v3_multibucket_official.json"
    )
    result_dir = (
        ROOT / "results/models/campplus/final_v3/onnx_crt_evaluation" / run_id
    )
    c_dump_dir = (
        ROOT / "runs/models/campplus/final_v3/onnx_crt_evaluation/accuracy"
        / run_id
    )
    accuracy_dir = result_dir / "accuracy"
    config_document = _load(config)
    configured_binary = _path(config_document["paths"]["c_benchmark"])
    configured_bundle = _path(config_document["paths"]["arena_bundle"])
    configured_cosine = float(config_document.get("embedding_cosine_min", 0.99))
    if args.binary.resolve() != configured_binary.resolve():
        parser.error("--binary must match the selected benchmark config")
    if args.bundle.resolve() != configured_bundle.resolve():
        parser.error("--bundle must match the selected benchmark config")
    if args.embedding_cosine_min != configured_cosine:
        parser.error(
            "--embedding-cosine-min must match the selected benchmark config"
        )
    benchmark_root = _path(config_document["paths"]["workspace_root"])
    benchmark_summary = benchmark_root / run_id / "benchmark_summary.json"

    required = [
        args.binary,
        args.dump_binary,
        args.bundle / "weights.bin",
        config,
        ROOT / "models/source/campplus_int8_static_qop.onnx",
        ROOT / "scripts/3_runtime/04_run_reference_runtime.sh",
        ROOT / "scripts/3_runtime/05_compare_runtime_outputs.py",
        ROOT / "scripts/3_runtime/09_benchmark_runtime.py",
    ]
    for frames in buckets:
        required.extend(
            [
                args.bundle / "execution_plans" / f"plan_{frames}.bin",
                args.ort_dir / f"feature_{frames}.f32",
                args.ort_dir / f"ort_{frames}.npz",
                args.ort_dir / f"ort_{frames}.json",
                ROOT / "results/static" / f"campp_static_{frames}.onnx",
                ROOT / "results/graph" / f"ir_{BUCKET_TAGS[frames]}.json",
            ]
        )
    try:
        _require_files(required)
        capabilities, compiled_v3_buckets = _verify_runtime_capabilities(
            args.binary, buckets
        )
        expected_policies = {frames: "layer_hybrid_v3" for frames in buckets}
        preflight_command = [
            sys.executable,
            str(ROOT / "scripts/3_runtime/09_benchmark_runtime.py"),
            "--config",
            str(config),
            "--buckets",
            *(str(value) for value in buckets),
            "--preflight-only",
        ]
        if args.allow_environment_mismatch:
            preflight_command.append("--allow-environment-mismatch")
        _run(preflight_command)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "mode": args.mode,
                        "buckets": list(buckets),
                        "runtime_suite": capabilities.get(
                            "optimization_suite_config"
                        ),
                        "compiled_v3_buckets": sorted(compiled_v3_buckets),
                        "expected_bucket_policies": expected_policies,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if not args.skip_accuracy:
            accuracy_environment = os.environ.copy()
            accuracy_environment.update(
                {
                    "BUILD_DIR": str(args.binary.parent),
                    "BUNDLE_DIR": str(args.bundle),
                    "DUMP_TOOL": str(args.dump_binary),
                    "ORT_DIR": str(args.ort_dir),
                    "OUT_DIR": str(c_dump_dir),
                    "BUCKETS": " ".join(str(value) for value in buckets),
                }
            )
            _run(
                ["bash", str(ROOT / "scripts/3_runtime/04_run_reference_runtime.sh")],
                environment=accuracy_environment,
            )
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts/3_runtime/05_compare_runtime_outputs.py"),
                    "--ort-dir",
                    str(args.ort_dir),
                    "--c-dir",
                    str(c_dump_dir),
                    "--results-dir",
                    str(accuracy_dir),
                    "--buckets",
                    *(str(value) for value in buckets),
                    "--gate",
                    "embedding-cosine",
                    "--embedding-cosine-min",
                    str(args.embedding_cosine_min),
                ]
            )

        if not args.skip_performance:
            performance_command = [
                sys.executable,
                str(ROOT / "scripts/3_runtime/09_benchmark_runtime.py"),
                "--config",
                str(config),
                "--buckets",
                *(str(value) for value in buckets),
                "--run-id",
                run_id,
            ]
            if args.allow_environment_mismatch:
                performance_command.append("--allow-environment-mismatch")
            _run(performance_command)

        if args.skip_accuracy or args.skip_performance:
            print("partial evaluation complete; matrix merge was skipped")
            return 0
        accuracy = _load(accuracy_dir / "compare_summary.json")
        performance = _load(benchmark_summary)
        report = build_frame_matrix(
            accuracy,
            performance,
            buckets,
            expected_policies=expected_policies,
            embedding_cosine_min=args.embedding_cosine_min,
            baseline_98_rtf=args.baseline_98_rtf,
            max_regression_pct=args.max_baseline_regression_pct,
        )
        report.update(
            {
                "run_id": run_id,
                "mode": args.mode,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "accuracy_summary": str(accuracy_dir / "compare_summary.json"),
                "performance_summary": str(benchmark_summary),
            }
        )
        write_matrix(report, result_dir)
        print(f"matrix: {result_dir / 'frame_matrix.json'}")
        print(f"overall: {'PASS' if report['all_passed'] else 'FAIL'}")
        return 0 if report["all_passed"] else 1
    except (EvaluationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
