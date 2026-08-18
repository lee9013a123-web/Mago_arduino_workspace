#!/usr/bin/env python3
"""E7 fused BN/ReLU/Quantize의 target-only cycle을 세부 구간으로 분류한다."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DIAGNOSIS_SCRIPT = SCRIPT_DIR / "02_diagnose_top4.py"
COMMON_SCRIPT = SCRIPT_DIR / "_perf_hotspot_common.py"
BN_SOURCE = (
    ROOT
    / "src/c/runtime/backends/cpu_aarch64/fused_kernels/bn_relu_quant_conv.c"
)
TARGET_CASE_NAME = "fused_bn_relu_quant"
TARGET_SYMBOL = "campp_fused_bn_relu_quant"
CORE_CATEGORIES = (
    "bn_index_address",
    "bn_parameter_load",
    "bn_sqrt_affine",
    "bn_relu_quant",
    "bn_output_write",
)
CATEGORIES = (*CORE_CATEGORIES, "setup_other", "unclassified")


class BnHotspotError(RuntimeError):
    """BN perf 입력, 실행 또는 분류 결과가 유효하지 않다."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BnHotspotError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSIS = _load_module("e7_optimization_diagnosis_bn", DIAGNOSIS_SCRIPT)
COMMON = _load_module("e7_perf_hotspot_common", COMMON_SCRIPT)


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_source_map(source: Path = BN_SOURCE) -> dict[str, Any]:
    lines = source.read_text(encoding="utf-8").splitlines()
    main_span = COMMON.function_span(
        lines, "CamppStatus campp_fused_bn_relu_quant("
    )
    loop_begin = COMMON.find_line(
        lines, "for (index = 0u; index < count; ++index)", start=main_span[0]
    )
    index_begin = COMMON.find_line(
        lines, "campp_reference_unravel_index(", start=loop_begin
    )
    input_read = COMMON.find_line(
        lines, "status = campp_reference_read_f32(&inputs[0]", start=index_begin
    )
    parameter_begin = COMMON.find_line(
        lines, "status = campp_reference_read_f32(&inputs[1]", start=input_read
    )
    parameter_end = COMMON.find_line(
        lines, "status = campp_reference_read_f32(&inputs[4]", start=parameter_begin
    ) + 1
    affine_begin = COMMON.find_line(lines, "inv_std =", start=parameter_end)
    affine_end = COMMON.find_line(lines, "value = x *", start=affine_begin)
    relu_begin = COMMON.find_line(lines, "if (value < 0.0f)", start=affine_end)
    output_begin = COMMON.find_line(
        lines, "status = campp_reference_write_quantized(", start=relu_begin
    )
    output_end = COMMON.find_line(
        lines, "if (status != CAMPP_STATUS_OK)", start=output_begin
    )
    loop_end = COMMON.find_line(
        lines, "CAMPP_OPT_STAGE_BN_ELEMENTWISE", start=output_end
    )
    return {
        "source": source,
        "source_name": source.name,
        "line_count": len(lines),
        "main_span": main_span,
        "loop_begin": loop_begin,
        "index_span": (index_begin, parameter_begin - 1),
        "input_read": input_read,
        "parameter_span": (parameter_begin, parameter_end),
        "affine_span": (affine_begin, affine_end),
        "relu_quant_span": (relu_begin, output_begin - 1),
        "output_span": (output_begin, output_end),
        "loop_end": loop_end,
    }


def classify_sample(sample: dict[str, Any], source_map: dict[str, Any]) -> str:
    symbol = str(sample["symbol"]).lower()
    source_name = Path(str(sample["source"])).name.lower()
    line = int(sample["line"])

    if "sqrt" in symbol:
        return "bn_sqrt_affine"
    if "nearbyint" in symbol:
        return "bn_relu_quant"
    if "write_quantized" in symbol:
        return "bn_output_write"
    if any(
        name in symbol
        for name in (
            "unravel_index",
            "offset_for_linear",
            "tensor_view_byte_offset",
            "tensor_view_element_count",
        )
    ):
        return "bn_index_address"
    if "reference_read_f32" in symbol:
        return "bn_index_address"
    if source_name == "tensor_view.h":
        return "bn_index_address"
    if source_name != source_map["source_name"].lower():
        return "unclassified"
    if COMMON.inside(line, source_map["index_span"]):
        return "bn_index_address"
    if COMMON.inside(line, source_map["parameter_span"]):
        return "bn_parameter_load"
    if COMMON.inside(line, source_map["affine_span"]):
        return "bn_sqrt_affine"
    if COMMON.inside(line, source_map["relu_quant_span"]):
        return "bn_relu_quant"
    if COMMON.inside(line, source_map["output_span"]):
        return "bn_output_write"
    if source_map["loop_begin"] <= line <= source_map["loop_end"]:
        return "bn_index_address" if line == source_map["input_read"] else "setup_other"
    if COMMON.inside(line, source_map["main_span"]):
        return "setup_other"
    return "unclassified"


def classify_perf_script(text: str, source_map: dict[str, Any]) -> dict[str, Any]:
    periods = {category: 0 for category in CATEGORIES}
    counts = {category: 0 for category in CATEGORIES}
    line_periods: dict[tuple[str, int, str, str], int] = {}
    samples, malformed = COMMON.iter_perf_samples(text)
    for sample in samples:
        category = classify_sample(sample, source_map)
        period = int(sample["period"])
        periods[category] += period
        counts[category] += 1
        key = (
            Path(str(sample["source"])).name,
            int(sample["line"]),
            str(sample["symbol"]),
            category,
        )
        line_periods[key] = line_periods.get(key, 0) + period
    total = sum(periods.values())
    if not samples or total == 0:
        raise BnHotspotError("perf script contains no attributable cycle samples")
    classified = sum(periods[name] for name in CORE_CATEGORIES)
    ranked = sorted(
        CORE_CATEGORIES, key=lambda name: periods[name], reverse=True
    )
    cumulative = 0
    top_set = []
    for name in ranked:
        if periods[name] == 0:
            continue
        cumulative += periods[name]
        top_set.append(name)
        if classified and cumulative / classified >= 0.8:
            break
    top_lines = sorted(line_periods.items(), key=lambda item: item[1], reverse=True)
    return {
        "parsed_sample_count": len(samples),
        "malformed_line_count": malformed,
        "total_sample_period": total,
        "classified_core_period": classified,
        "category_sample_counts": counts,
        "category_periods": periods,
        "category_share_pct": {
            name: value / total * 100.0 for name, value in periods.items()
        },
        "classified_core_share_pct": classified / total * 100.0,
        "top_bottleneck_set": top_set,
        "top_lines": [
            {
                "source": source,
                "line": line,
                "symbol": symbol,
                "category": category,
                "sample_period": period,
                "total_share_pct": period / total * 100.0,
            }
            for (source, line, symbol, category), period in top_lines[:20]
        ],
    }


def _microbench_command(
    binary: Path, *, plan: Path, weights: Path, feature: Path,
    operator_id: int, warmup: int, repeat: int,
) -> list[str]:
    command = DIAGNOSIS._command(
        binary,
        plan=plan,
        weights=weights,
        feature=feature,
        operator_id=operator_id,
        warmup=warmup,
        repeat=repeat,
    )
    command.append("--perf-window")
    return command


def _run_one(
    *, perf: str, binary: Path, plan: Path, weights: Path, feature: Path,
    case: dict[str, Any], warmup: int, repeat: int, sample_period: int,
    output_dir: Path, source_map: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_data = output_dir / "perf.data"
    payload_path = output_dir / "microbench.json"
    script_path = output_dir / "perf_script.txt"
    annotate_path = output_dir / "perf_annotate.txt"
    report_path = output_dir / "perf_report.txt"
    command_path = output_dir / "command.json"
    stderr_path = output_dir / "perf_record.stderr.txt"
    microbench = _microbench_command(
        binary, plan=plan, weights=weights, feature=feature,
        operator_id=int(case["operator_id"]), warmup=warmup, repeat=repeat,
    )
    record_command = [
        perf, "record", "--quiet", "--no-buildid", "--output", str(perf_data),
        "--event", "cycles:u", "--count", str(sample_period), "--", *microbench,
    ]
    command_path.write_text(
        json.dumps(
            {"record": record_command, "cpu_affinity": [0],
             "measurement_scope": "target kernel only"},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8", newline="\n",
    )
    try:
        completed = COMMON.run_command(record_command, root=ROOT)
    except RuntimeError as exc:
        raise BnHotspotError(str(exc)) from exc
    stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BnHotspotError("hotspot binary did not return JSON") from exc
    operator = payload.get("operator", {})
    window = payload.get("perf_window", {})
    if (
        operator.get("operator_id") != case["operator_id"]
        or operator.get("kernel_name") != case["kernel_name"]
        or payload.get("bn_candidate") != "baseline"
        or payload.get("output_hash_matches") is not True
        or window.get("requested") is not True
        or window.get("supported") is not True
        or window.get("enabled_at_exit") is not False
    ):
        raise BnHotspotError("invalid BN hotspot payload")
    payload["input"] = {"path": _display_path(feature)}
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    if not perf_data.is_file() or perf_data.stat().st_size == 0:
        raise BnHotspotError("perf record did not create data")
    try:
        script = COMMON.run_command(
            [perf, "script", "--input", str(perf_data), "--fields",
             "period,ip,sym,srcline"], root=ROOT,
        )
        annotate = COMMON.run_command(
            [perf, "annotate", "--stdio", "--input", str(perf_data),
             "--symbol", TARGET_SYMBOL], root=ROOT, check=False,
        )
        report = COMMON.run_command(
            [perf, "report", "--stdio", "--input", str(perf_data),
             "--sort", "symbol,srcline", "--percent-limit", "0"],
            root=ROOT, check=False,
        )
    except RuntimeError as exc:
        raise BnHotspotError(str(exc)) from exc
    script_path.write_text(script.stdout, encoding="utf-8", newline="\n")
    annotate_path.write_text(
        annotate.stdout + annotate.stderr, encoding="utf-8", newline="\n"
    )
    report_path.write_text(
        report.stdout + report.stderr, encoding="utf-8", newline="\n"
    )
    result = classify_perf_script(script.stdout, source_map)
    result.update(
        {
            "input": _display_path(feature),
            "output_hash": payload["output_hash"],
            "artifacts": {
                "perf_data": _display_path(perf_data),
                "perf_record_stderr": _display_path(stderr_path),
                "perf_script": _display_path(script_path),
                "perf_annotate": _display_path(annotate_path),
                "perf_report": _display_path(report_path),
                "microbench": _display_path(payload_path),
            },
        }
    )
    return result


def _aggregate_case(
    case: dict[str, Any], inputs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    periods = {category: 0 for category in CATEGORIES}
    for item in inputs:
        for category in CATEGORIES:
            periods[category] += int(item["category_periods"][category])
    total = sum(periods.values())
    classified = sum(periods[name] for name in CORE_CATEGORIES)
    ranked = sorted(CORE_CATEGORIES, key=lambda name: periods[name], reverse=True)
    cumulative = 0
    top_set = []
    for name in ranked:
        if periods[name] == 0:
            continue
        cumulative += periods[name]
        top_set.append(name)
        if classified and cumulative / classified >= 0.8:
            break
    input_winners = [
        max(CORE_CATEGORIES, key=lambda name: item["category_periods"][name])
        for item in inputs
    ]
    max_unclassified = max(
        float(item["category_share_pct"]["unclassified"]) for item in inputs
    )
    stable = len(set(input_winners)) == 1 and max_unclassified <= 20.0
    return {
        **case,
        "inputs": list(inputs),
        "aggregate": {
            "total_sample_period": total,
            "classified_core_period": classified,
            "category_periods": periods,
            "category_share_pct": {
                name: value / total * 100.0 if total else 0.0
                for name, value in periods.items()
            },
            "classified_core_share_pct": (
                classified / total * 100.0 if total else 0.0
            ),
            "top_bottleneck_set": top_set,
        },
        "decision": {
            "ready": stable,
            "winner": input_winners[0] if len(set(input_winners)) == 1 else "mixed",
            "input_winners": input_winners,
            "maximum_unclassified_share_pct": max_unclassified,
            "rule": "same top category on 3 inputs and unclassified <=20%; select cumulative 80% of classified BN core",
        },
    }


def build_summary(
    case: dict[str, Any], *, warmup: int, repeat: int, sample_period: int,
    elapsed_seconds: float, source_map: dict[str, Any],
) -> dict[str, Any]:
    ready = bool(case["decision"]["ready"])
    top_set = case["aggregate"]["top_bottleneck_set"]
    candidate_map = {
        "bn_index_address": "address",
        "bn_parameter_load": "affine",
        "bn_sqrt_affine": "affine",
        "bn_relu_quant": "quant",
        "bn_output_write": "address/quant",
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "locate internal bottlenecks in E7 fused BN/ReLU/Quantize",
        "configuration": {
            "bucket_frames": 98,
            "threads": 1,
            "cpu_affinity": [0],
            "warmup": warmup,
            "repeat_per_input": repeat,
            "input_count": 3,
            "perf_event": "cycles:u",
            "sample_period": sample_period,
            "stage_probe_compiled": False,
            "elapsed_seconds": elapsed_seconds,
        },
        "source_map": {
            key: value for key, value in source_map.items()
            if key not in ("source", "source_name")
        } | {"source": _display_path(source_map["source"])},
        "cases": [case],
        "decision": {
            "ready": ready,
            "overall": case["decision"]["winner"] if ready else "inconclusive",
            "top_bottleneck_set": top_set,
            "next_candidates": sorted({candidate_map[name] for name in top_set}),
            "next_step": (
                "benchmark baseline/address/affine/quant/combined"
                if ready else "inspect perf_annotate and reduce unclassified samples"
            ),
        },
    }


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "case_name", "operator_id", "input", *(
            f"{name}_pct" for name in CATEGORIES
        ), "classified_core_share_pct", "top_bottleneck_set", "winner",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        case = summary["cases"][0]
        for item in case["inputs"]:
            shares = item["category_share_pct"]
            winner = max(CORE_CATEGORIES, key=lambda name: item["category_periods"][name])
            row = {
                "case_name": case["case_name"],
                "operator_id": case["operator_id"],
                "input": item["input"],
                "classified_core_share_pct": item["classified_core_share_pct"],
                "top_bottleneck_set": ";".join(item["top_bottleneck_set"]),
                "winner": winner,
            }
            row.update({f"{name}_pct": shares[name] for name in CATEGORIES})
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=_path,
        default=ROOT / "results/profiling/e7_98/operator_profile.json",
    )
    parser.add_argument(
        "--binary", type=_path,
        default=ROOT / "build/profill/optimization/campp_operator_hotspot",
    )
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument(
        "--features", type=_path, nargs="+", default=list(DIAGNOSIS.EXPECTED_INPUTS)
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sample-period", type=int, default=100_000)
    parser.add_argument(
        "--runs-dir", type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/bn_hotspot",
    )
    parser.add_argument(
        "--results-dir", type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/bn_hotspot",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.warmup < 0 or args.repeat <= 0 or args.sample_period <= 0:
            raise BnHotspotError("warmup, repeat or sample period is invalid")
        if os.name != "posix" and not args.preflight_only:
            raise BnHotspotError("actual perf sampling requires Linux/QRB2210")
        perf = shutil.which("perf")
        if perf is None:
            raise BnHotspotError("Linux perf executable not found")
        if not args.profile.is_file():
            raise BnHotspotError(f"profile not found: {args.profile}")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        selected = DIAGNOSIS.select_representative_cases(profile.get("operators", []))
        case = next(
            (item for item in selected if item["case_name"] == TARGET_CASE_NAME),
            None,
        )
        if case is None:
            raise BnHotspotError("fused BN case was not selected")
        plan = args.plan or DIAGNOSIS._resolve_profile_artifact(profile, "plan")
        weights = args.weights or DIAGNOSIS._resolve_profile_artifact(profile, "weights")
        required = [args.binary, plan, weights, BN_SOURCE, *args.features]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise BnHotspotError(
                "required files are missing:\n  " + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3 or any(
            path.stat().st_size != 98 * 80 * 4 for path in args.features
        ):
            raise BnHotspotError("exactly three float32 [1,98,80] inputs are required")
        summary_path = args.results_dir / "bn_hotspot.json"
        csv_path = args.results_dir / "bn_hotspot.csv"
        if summary_path.exists() and not args.force and not args.preflight_only:
            raise BnHotspotError(f"output exists: {summary_path} (use --force)")
        capabilities = DIAGNOSIS._run_payload([str(args.binary), "--capabilities"])
        if capabilities.get("perf_sample_window") is not True:
            raise BnHotspotError("binary does not support target-only perf windows")
        if capabilities.get("stage_probe") is not False:
            raise BnHotspotError("hotspot binary must be built without stage probes")
        source_map = build_source_map()
        if args.preflight_only:
            print(json.dumps(
                {
                    "ready": True, "perf": perf,
                    "binary": _display_path(args.binary),
                    "plan": _display_path(plan), "weights": _display_path(weights),
                    "features": [_display_path(path) for path in args.features],
                    "case": case,
                    "source_map": {
                        key: value for key, value in source_map.items()
                        if key not in ("source", "source_name")
                    },
                }, ensure_ascii=False, indent=2,
            ))
            return 0

        execution_seconds = (
            float(case["profile_mean_ms"]) / 1000.0
            * (args.warmup + args.repeat + 1) * len(args.features)
        )
        estimated_seconds = 45.0 + 75.0 * len(args.features) + execution_seconds
        print(f"예상 시간: 약 {max(1, math.ceil(estimated_seconds / 60.0))}분")
        started = time.monotonic()
        inputs = []
        for feature in args.features:
            print("진행 중", flush=True)
            output_dir = args.runs_dir / TARGET_CASE_NAME / feature.stem
            if output_dir.exists() and args.force:
                for generated in output_dir.iterdir():
                    if generated.is_file():
                        generated.unlink()
            inputs.append(_run_one(
                perf=perf, binary=args.binary, plan=plan, weights=weights,
                feature=feature, case=case, warmup=args.warmup,
                repeat=args.repeat, sample_period=args.sample_period,
                output_dir=output_dir, source_map=source_map,
            ))
        aggregated = _aggregate_case(case, inputs)
        summary = build_summary(
            aggregated, warmup=args.warmup, repeat=args.repeat,
            sample_period=args.sample_period,
            elapsed_seconds=time.monotonic() - started, source_map=source_map,
        )
        summary["artifacts"] = {
            "profile": _display_path(args.profile), "plan": _display_path(plan),
            "weights": _display_path(weights), "raw_dir": _display_path(args.runs_dir),
        }
        args.results_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        _write_csv(csv_path, summary)
        print(f"BN hotspot 완료: {_display_path(summary_path)}")
        aggregate = summary["cases"][0]["aggregate"]
        print(f"  실측 시간: {summary['configuration']['elapsed_seconds']:.1f}초")
        for category in CATEGORIES:
            print(
                f"  {category}: "
                f"{aggregate['category_share_pct'][category]:.2f}%"
            )
        print(
            "  Top bottleneck set: "
            + ", ".join(summary["decision"]["top_bottleneck_set"])
        )
        print(
            f"  gate.ready={str(summary['decision']['ready']).lower()} "
            f"winner={summary['decision']['overall']}"
        )
        print(f"  next: {summary['decision']['next_step']}")
        return 0 if summary["decision"]["ready"] else 3
    except (BnHotspotError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"BN hotspot failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
