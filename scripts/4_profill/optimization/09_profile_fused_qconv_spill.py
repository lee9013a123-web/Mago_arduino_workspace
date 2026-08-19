#!/usr/bin/env python3
"""Profile register spill in representative fused Quant-QConv shapes."""

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
DIAGNOSIS_SCRIPT = Path(__file__).with_name("02_diagnose_top4.py")
HOTSPOT_SCRIPT = Path(__file__).with_name("04_profile_qconv_hotspot.py")
DEFAULT_FEATURES = (
    ROOT / "benchmarks/campplus/features/multi__speaker_0000__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0005__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0006__98.f32",
)
FUSED_CANDIDATES = (
    "mac", "combined", "mac_fixed", "quant_neon", "combined_fixed",
    "combined_v4", "combined_hybrid",
)


class FusedSpillError(RuntimeError):
    """Fused spill profiling inputs or measurements are invalid."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FusedSpillError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSIS = _load_module("e7_fused_spill_diagnosis", DIAGNOSIS_SCRIPT)
HOTSPOT = _load_module("e7_fused_spill_hotspot", HOTSPOT_SCRIPT)


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def select_shape_representatives(
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick the highest-profile fused operator from every weight shape."""

    by_shape: dict[tuple[int, ...], dict[str, Any]] = {}
    for case in cases:
        shape = tuple(int(value) for value in case.get("weight_shape", []))
        current = by_shape.get(shape)
        if current is None or (
            float(case["profile_mean_ms"]), -int(case["operator_id"])
        ) > (
            float(current["profile_mean_ms"]), -int(current["operator_id"])
        ):
            by_shape[shape] = case
    return sorted(by_shape.values(), key=lambda item: int(item["operator_id"]))


def select_cases(
    profile: dict[str, Any], operator_ids: Sequence[int] | None,
    all_operators: bool,
) -> list[dict[str, Any]]:
    family = DIAGNOSIS.select_fused_qconv_family_cases(
        profile.get("operators", [])
    )
    if all_operators:
        return family
    if operator_ids:
        by_id = {int(case["operator_id"]): case for case in family}
        missing = [operator_id for operator_id in operator_ids if operator_id not in by_id]
        if missing:
            raise FusedSpillError(
                f"requested operators are not fused Quant-QConv: {missing}"
            )
        return [by_id[operator_id] for operator_id in dict.fromkeys(operator_ids)]
    return select_shape_representatives(family)


def _spill_summary(value: Any) -> tuple[bool, float]:
    if not isinstance(value, dict):
        return False, 0.0
    available = float(value.get("annotated_percent_total", 0.0)) > 0.0
    return available, float(
        value.get("stack_spill_share_of_annotated_pct", 0.0)
    )


def aggregate_case(
    case: dict[str, Any], inputs: Sequence[dict[str, Any]],
    *, spill_limit_pct: float, quantize_expected: bool,
) -> dict[str, Any]:
    paths = [str(item["execution_path"]) for item in inputs]
    symbols = [str(item["mac_annotate_symbol"]) for item in inputs]
    mac_values = [_spill_summary(item.get("spill")) for item in inputs]
    quant_values = [
        _spill_summary(item.get("quantize_spill")) for item in inputs
    ]
    mac_available = all(value[0] for value in mac_values)
    quant_available = all(value[0] for value in quant_values)
    mac_spills = [value[1] for value in mac_values]
    quant_spills = [value[1] for value in quant_values]
    mac_passed = mac_available and max(mac_spills, default=0.0) <= spill_limit_pct
    quant_passed = (
        not quantize_expected
        or quant_available
        and max(quant_spills, default=0.0) <= spill_limit_pct
    )
    stable_path = len(set(paths)) == 1 and len(set(symbols)) == 1
    ready = stable_path and mac_passed and quant_passed
    if not stable_path:
        next_action = "inspect inconsistent fixed/fallback dispatch across inputs"
    elif not mac_available:
        next_action = "increase perf samples for the selected MAC symbol"
    elif max(mac_spills, default=0.0) > spill_limit_pct:
        next_action = "inspect MAC annotate and reduce stack-resident vectors"
    elif quantize_expected and not quant_available:
        next_action = "lower sample period to capture input quantize"
    elif quantize_expected and max(quant_spills, default=0.0) > spill_limit_pct:
        next_action = "inspect input quantize register pressure"
    else:
        next_action = "spill gate passed"
    return {
        **case,
        "input_count": len(inputs),
        "execution_path": paths[0] if stable_path else "mixed",
        "mac_annotate_symbol": symbols[0] if stable_path else "mixed",
        "fixed_microkernel_sample_count": sum(
            int(item.get("fixed_microkernel_sample_count", 0))
            for item in inputs
        ),
        "mac_spill": {
            "available_all_inputs": mac_available,
            "mean_pct": sum(mac_spills) / len(mac_spills) if mac_spills else 0.0,
            "max_pct": max(mac_spills, default=0.0),
        },
        "input_quantize_spill": {
            "expected": quantize_expected,
            "available_all_inputs": quant_available,
            "mean_pct": (
                sum(quant_spills) / len(quant_spills) if quant_spills else 0.0
            ),
            "max_pct": max(quant_spills, default=0.0),
        },
        "output_hashes": {
            str(item["input"]): str(item["output_hash"]) for item in inputs
        },
        "inputs": list(inputs),
        "spill_gate": {
            "ready": ready,
            "limit_pct": spill_limit_pct,
            "stable_execution_path": stable_path,
            "mac_passed": mac_passed,
            "input_quantize_passed": quant_passed,
            "next_action": next_action,
        },
    }


def build_summary(
    cases: Sequence[dict[str, Any]], *, candidate: str, warmup: int,
    repeat: int, sample_period: int, spill_limit_pct: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    ready = all(case["spill_gate"]["ready"] for case in cases)
    fixed_count = sum(
        str(case["execution_path"]).startswith("fixed_4x8") for case in cases
    )
    fallback_count = sum(
        case["execution_path"] == "fallback_v2" for case in cases
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "measure fused Quant-QConv MAC and quantize register spill",
        "configuration": {
            "bucket_frames": 98,
            "threads": 1,
            "cpu_affinity": [0],
            "warmup": warmup,
            "repeat": repeat,
            "sample_period": sample_period,
            "spill_limit_pct": spill_limit_pct,
            "fused_qconv_candidate": candidate,
            "elapsed_seconds": elapsed_seconds,
        },
        "case_count": len(cases),
        "shape_count": len(
            {tuple(case.get("weight_shape", [])) for case in cases}
        ),
        "fixed_path_case_count": fixed_count,
        "fallback_v2_case_count": fallback_count,
        "cases": list(cases),
        "spill_gate": {
            "ready": ready,
            "rule": (
                "all inputs use one stable path and annotated stack spill "
                f"does not exceed {spill_limit_pct:g}%"
            ),
            "failing_operator_ids": [
                case["operator_id"]
                for case in cases if not case["spill_gate"]["ready"]
            ],
        },
    }


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = (
        "operator_id", "weight_shape", "profile_mean_ms", "execution_path",
        "mac_annotate_symbol", "fixed_microkernel_sample_count",
        "mac_spill_mean_pct", "mac_spill_max_pct",
        "quantize_spill_mean_pct", "quantize_spill_max_pct",
        "spill_gate_ready", "next_action",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in summary["cases"]:
            writer.writerow(
                {
                    "operator_id": case["operator_id"],
                    "weight_shape": "x".join(
                        str(value) for value in case["weight_shape"]
                    ),
                    "profile_mean_ms": case["profile_mean_ms"],
                    "execution_path": case["execution_path"],
                    "mac_annotate_symbol": case["mac_annotate_symbol"],
                    "fixed_microkernel_sample_count": case[
                        "fixed_microkernel_sample_count"
                    ],
                    "mac_spill_mean_pct": case["mac_spill"]["mean_pct"],
                    "mac_spill_max_pct": case["mac_spill"]["max_pct"],
                    "quantize_spill_mean_pct": case[
                        "input_quantize_spill"
                    ]["mean_pct"],
                    "quantize_spill_max_pct": case[
                        "input_quantize_spill"
                    ]["max_pct"],
                    "spill_gate_ready": case["spill_gate"]["ready"],
                    "next_action": case["spill_gate"]["next_action"],
                }
            )


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
        "--features", type=_path, nargs="+", default=list(DEFAULT_FEATURES),
    )
    parser.add_argument(
        "--fused-qconv-candidate", choices=FUSED_CANDIDATES,
        default="combined_fixed",
    )
    parser.add_argument("--operator-ids", type=int, nargs="+")
    parser.add_argument("--all-operators", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sample-period", type=int, default=100_000)
    parser.add_argument("--spill-limit-pct", type=float, default=5.0)
    parser.add_argument(
        "--runs-dir", type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/fused_qconv_spill",
    )
    parser.add_argument(
        "--results-dir", type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/fused_qconv_spill",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        if (
            args.warmup < 0 or args.repeat <= 0 or args.sample_period <= 0
            or args.spill_limit_pct < 0.0
        ):
            raise FusedSpillError("invalid warmup, repeat, sample period or limit")
        if args.operator_ids and args.all_operators:
            raise FusedSpillError("--operator-ids and --all-operators are exclusive")
        if os.name != "posix" and not args.preflight_only:
            raise FusedSpillError("actual perf sampling requires Linux/QRB2210")
        perf = shutil.which("perf")
        if perf is None:
            raise FusedSpillError("Linux perf executable not found")
        if not args.profile.is_file():
            raise FusedSpillError(f"profile not found: {args.profile}")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        cases = select_cases(profile, args.operator_ids, args.all_operators)
        plan = args.plan or DIAGNOSIS._resolve_profile_artifact(profile, "plan")
        weights = args.weights or DIAGNOSIS._resolve_profile_artifact(
            profile, "weights"
        )
        required = [
            args.binary, plan, weights, HOTSPOT.V2_MAC_SOURCE,
            HOTSPOT.V2_CANDIDATE_SOURCE, HOTSPOT.FIXED_MAC_SOURCE,
            HOTSPOT.FIXED_DISPATCH_SOURCE, *args.features,
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FusedSpillError(
                "required files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3 or any(
            path.stat().st_size != 98 * 80 * 4 for path in args.features
        ):
            raise FusedSpillError(
                "exactly three float32 [1,98,80] inputs are required"
            )
        summary_path = args.results_dir / "fused_qconv_spill.json"
        csv_path = args.results_dir / "fused_qconv_spill.csv"
        if summary_path.exists() and not args.force and not args.preflight_only:
            raise FusedSpillError(f"output exists: {summary_path} (use --force)")
        capabilities = DIAGNOSIS._run_payload([str(args.binary), "--capabilities"])
        if capabilities.get("perf_sample_window") is not True or (
            capabilities.get("stage_probe") is not False
        ):
            raise FusedSpillError("hotspot binary lacks target-only perf support")
        if args.fused_qconv_candidate not in capabilities.get(
            "fused_qconv_candidates", []
        ):
            raise FusedSpillError("hotspot binary lacks requested fused candidate")
        source_map = HOTSPOT.build_source_map()
        v2_source_map = HOTSPOT.build_v2_source_map()
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "perf": perf,
                        "binary": _display_path(args.binary),
                        "plan": _display_path(plan),
                        "weights": _display_path(weights),
                        "features": [_display_path(path) for path in args.features],
                        "fused_qconv_candidate": args.fused_qconv_candidate,
                        "case_count": len(cases),
                        "shape_count": len(
                            {tuple(case["weight_shape"]) for case in cases}
                        ),
                        "cases": cases,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        estimated_seconds = (
            75.0 * len(cases) * len(args.features)
            + sum(
                float(case["profile_mean_ms"]) / 1000.0
                * (args.warmup + args.repeat + 1)
                * len(args.features)
                for case in cases
            )
        )
        print(f"예상 시간: 약 {max(1, math.ceil(estimated_seconds / 60.0))}분")
        started = time.monotonic()
        results = []
        quantize_expected = args.fused_qconv_candidate in (
            "quant_neon", "combined_fixed", "combined_v4",
            "combined_hybrid",
        )
        for case in cases:
            inputs = []
            for feature in args.features:
                print("진행 중", flush=True)
                output_dir = args.runs_dir / case["case_name"] / feature.stem
                if output_dir.exists() and args.force:
                    for generated in output_dir.iterdir():
                        if generated.is_file():
                            generated.unlink()
                inputs.append(
                    HOTSPOT._run_one(
                        perf=perf,
                        binary=args.binary,
                        plan=plan,
                        weights=weights,
                        feature=feature,
                        case=case,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        sample_period=args.sample_period,
                        output_dir=output_dir,
                        source_map=source_map,
                        fused_qconv_candidate=args.fused_qconv_candidate,
                        v2_source_map=v2_source_map,
                    )
                )
            results.append(
                aggregate_case(
                    case, inputs,
                    spill_limit_pct=args.spill_limit_pct,
                    quantize_expected=quantize_expected,
                )
            )
        summary = build_summary(
            results,
            candidate=args.fused_qconv_candidate,
            warmup=args.warmup,
            repeat=args.repeat,
            sample_period=args.sample_period,
            spill_limit_pct=args.spill_limit_pct,
            elapsed_seconds=time.monotonic() - started,
        )
        summary["artifacts"] = {
            "profile": _display_path(args.profile),
            "plan": _display_path(plan),
            "weights": _display_path(weights),
            "raw_dir": _display_path(args.runs_dir),
        }
        args.results_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        write_csv(csv_path, summary)
        print(f"fused QConv spill: {_display_path(summary_path)}")
        return 0 if summary["spill_gate"]["ready"] else 3
    except (FusedSpillError, HOTSPOT.HotspotError, OSError, ValueError, KeyError) as exc:
        print(f"fused spill profiling failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
