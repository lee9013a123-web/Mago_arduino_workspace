#!/usr/bin/env python3
"""E7 상위 4개 kernel family를 실제 중간 Tensor로 독립 진단한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
PROFILL_SCRIPT_DIR = ROOT / "scripts" / "4_profill"
if str(PROFILL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILL_SCRIPT_DIR))

from profile_statistics import summarize_ns  # noqa: E402


EXPECTED_INPUTS = (
    ROOT / "benchmarks/campplus/features/multi__speaker_0000__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0005__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0006__98.f32",
)
TARGET_KERNELS = (
    "qlinear_conv_o4i4_neon",
    "fused_quant_qlinear_conv_o4i4",
    "fused_bn_relu_quant",
    "dequantize_linear_stride",
)
RELEVANT_STAGES = {
    "qlinear_conv_o4i4_neon": (
        "qconv_setup",
        "qconv_mac_address",
        "qconv_requant_write",
    ),
    "fused_quant_qlinear_conv_o4i4": (
        "fused_input_quantize",
        "fused_qconv",
    ),
    "fused_bn_relu_quant": ("bn_setup", "bn_elementwise"),
    "dequantize_linear_stride": ("dequant_setup", "dequant_elementwise"),
}
SOURCE_HOTSPOTS = {
    "qlinear_conv_o4i4_neon": {
        "path": "src/c/runtime/backends/cpu_aarch64/int8_neon/qlinear_convolution_neon.c",
        "line_start": 288,
        "line_end": 390,
        "question": "좌표·offset 계산과 dot4 MAC 중 어느 쪽이 cycle을 지배하는가",
    },
    "fused_quant_qlinear_conv_o4i4": {
        "path": "src/c/runtime/backends/cpu_aarch64/fused_kernels/conv_relu_requant.c",
        "line_start": 164,
        "line_end": 183,
        "question": "full-tensor quantize scratch pass의 비중이 충분히 큰가",
    },
    "fused_bn_relu_quant": {
        "path": "src/c/runtime/backends/cpu_aarch64/fused_kernels/bn_relu_quant_conv.c",
        "line_start": 86,
        "line_end": 136,
        "question": "element마다 반복되는 unravel·parameter read·sqrtf가 지배적인가",
    },
    "dequantize_linear_stride": {
        "path": "src/c/runtime/backends/cpu_reference/quantization_operators.c",
        "line_start": 197,
        "line_end": 225,
        "question": "generic stride helper와 element별 memcpy가 산술보다 비싼가",
    },
}


class DiagnosisError(RuntimeError):
    """진단 입력, 실행 또는 결과가 유효하지 않을 때 발생한다."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _meaningful_weight_shape(operator: dict[str, Any]) -> tuple[int, ...]:
    shapes = operator.get("weight_shapes")
    if not isinstance(shapes, list):
        return ()
    candidates = [
        tuple(int(value) for value in shape)
        for shape in shapes
        if isinstance(shape, list) and len(shape) >= 3
    ]
    return max(candidates, key=lambda shape: math.prod(shape), default=())


def select_representative_cases(
    operators: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """상위 family와 QConv의 3x3/1x1 shape를 대표하는 case를 고른다."""

    by_kernel: dict[str, list[dict[str, Any]]] = {
        kernel: [] for kernel in TARGET_KERNELS
    }
    for operator in operators:
        kernel = operator.get("kernel_name")
        if kernel in by_kernel:
            by_kernel[kernel].append(operator)
    missing = [kernel for kernel, rows in by_kernel.items() if not rows]
    if missing:
        raise DiagnosisError(f"profile is missing target kernels: {missing}")
    for rows in by_kernel.values():
        rows.sort(key=lambda item: int(item["total_exclusive_ns"]), reverse=True)

    ordinary = by_kernel["qlinear_conv_o4i4_neon"]
    qconv_3x3 = next(
        (row for row in ordinary if _meaningful_weight_shape(row)[-2:] == (3, 3)),
        ordinary[0],
    )
    qconv_1x1 = next(
        (row for row in ordinary if _meaningful_weight_shape(row)[-1:] == (1,)),
        ordinary[0],
    )
    selected = [
        ("qconv_3x3", qconv_3x3),
        ("qconv_1x1", qconv_1x1),
        ("fused_quant_qconv", by_kernel["fused_quant_qlinear_conv_o4i4"][0]),
        ("fused_bn_relu_quant", by_kernel["fused_bn_relu_quant"][0]),
        ("dequantize_linear", by_kernel["dequantize_linear_stride"][0]),
    ]
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for case_name, operator in selected:
        identity = (case_name, int(operator["operator_id"]))
        if identity in seen:
            continue
        seen.add(identity)
        cases.append(
            {
                "case_name": case_name,
                "operator_id": int(operator["operator_id"]),
                "kernel_id": int(operator["kernel_id"]),
                "kernel_name": str(operator["kernel_name"]),
                "operator_type": str(operator["operator_type"]),
                "weight_shape": list(_meaningful_weight_shape(operator)),
                "profile_mean_ms": float(operator["mean_ms"]),
                "profile_share_pct": float(operator["end_to_end_share_pct"]),
            }
        )
    return cases


def _pin_cpu_zero() -> None:
    os.sched_setaffinity(0, {0})


def _command(
    binary: Path,
    *,
    plan: Path,
    weights: Path,
    feature: Path,
    operator_id: int,
    warmup: int,
    repeat: int,
    expected_hash: str | None = None,
    qconv_candidate: str = "baseline",
    fused_qconv_candidate: str = "baseline",
    bn_candidate: str = "baseline",
    dequant_candidate: str = "baseline",
) -> list[str]:
    command = [
        str(binary),
        "--plan",
        str(plan),
        "--weights",
        str(weights),
        "--input",
        str(feature),
        "--operator-id",
        str(operator_id),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--threads",
        "1",
        "--qconv-candidate",
        qconv_candidate,
        "--fused-qconv-candidate",
        fused_qconv_candidate,
        "--bn-candidate",
        bn_candidate,
        "--dequant-candidate",
        dequant_candidate,
    ]
    if expected_hash is not None:
        command.extend(("--expected-output-hash", expected_hash))
    return command


def _run_payload(command: Sequence[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        preexec_fn=_pin_cpu_zero if os.name == "posix" else None,
    )
    if completed.returncode != 0:
        raise DiagnosisError(
            f"microbench failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DiagnosisError("microbench did not return one JSON object") from exc
    return payload


def _validate_payload(
    payload: dict[str, Any], case: dict[str, Any], repeat: int,
    qconv_candidate: str | None = None,
    bn_candidate: str | None = None,
    fused_qconv_candidate: str | None = None,
    dequant_candidate: str | None = None,
) -> None:
    if payload.get("mode") != "operator_microbench":
        raise DiagnosisError("unexpected microbench mode")
    if payload.get("clock") != "CLOCK_MONOTONIC_RAW":
        raise DiagnosisError("microbench clock is not CLOCK_MONOTONIC_RAW")
    if (
        qconv_candidate is not None
        and payload.get("qconv_candidate") != qconv_candidate
    ):
        raise DiagnosisError("QConv candidate mode mismatch")
    if bn_candidate is not None and payload.get("bn_candidate") != bn_candidate:
        raise DiagnosisError("BN candidate mode mismatch")
    if (
        fused_qconv_candidate is not None
        and payload.get("fused_qconv_candidate") != fused_qconv_candidate
    ):
        raise DiagnosisError("fused QConv candidate mode mismatch")
    if (
        dequant_candidate is not None
        and payload.get("dequant_candidate") != dequant_candidate
    ):
        raise DiagnosisError("Dequant candidate mode mismatch")
    operator = payload.get("operator")
    if not isinstance(operator, dict) or (
        operator.get("operator_id") != case["operator_id"]
        or operator.get("kernel_id") != case["kernel_id"]
        or operator.get("kernel_name") != case["kernel_name"]
    ):
        raise DiagnosisError(f"operator identity mismatch for {case['case_name']}")
    samples = payload.get("samples_ns")
    if not isinstance(samples, list) or len(samples) != repeat or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in samples
    ):
        raise DiagnosisError(f"invalid wall samples for {case['case_name']}")
    if payload.get("output_hash_matches") is not True:
        raise DiagnosisError(f"output hash mismatch for {case['case_name']}")
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise DiagnosisError("stage samples are missing")
    for stage in stages:
        values = stage.get("samples_ns") if isinstance(stage, dict) else None
        if not isinstance(values, list) or len(values) != repeat:
            raise DiagnosisError("invalid stage samples")


def _aggregate_case(
    case: dict[str, Any], payloads: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    wall_samples = [
        int(value) for payload in payloads for value in payload["samples_ns"]
    ]
    stage_values: dict[str, list[int]] = {}
    for payload in payloads:
        for stage in payload["stages"]:
            stage_values.setdefault(stage["name"], []).extend(
                int(value) for value in stage["samples_ns"]
            )
    relevant = RELEVANT_STAGES[case["kernel_name"]]
    wall_total = sum(wall_samples)
    stages = []
    for name in relevant:
        values = stage_values.get(name, [])
        summary = summarize_ns(values)
        stages.append(
            {
                "name": name,
                **summary,
                "wall_share_pct": (
                    float(summary["total_exclusive_ns"]) / wall_total * 100.0
                    if wall_total
                    else 0.0
                ),
            }
        )
    stages.sort(key=lambda item: item["total_exclusive_ns"], reverse=True)

    pmu_available = all(payload["pmu"]["available"] for payload in payloads)
    pmu_totals: dict[str, int] = {}
    if pmu_available:
        for payload in payloads:
            for event in payload["pmu"]["events"]:
                pmu_totals[event["name"]] = pmu_totals.get(event["name"], 0) + sum(
                    int(value) for value in event["samples"]
                )
    cycles = pmu_totals.get("cpu_cycles", 0)
    instructions = pmu_totals.get("instructions", 0)
    cache_references = pmu_totals.get("cache_references", 0)
    cache_misses = pmu_totals.get("cache_misses", 0)
    branches = pmu_totals.get("branches", 0)
    branch_misses = pmu_totals.get("branch_misses", 0)
    dominant = stages[0] if stages else None
    return {
        **case,
        "input_count": len(payloads),
        "output_hashes": {
            str(payload.get("input", {}).get("path", f"input_{index}")):
                str(payload["output_hash"])
            for index, payload in enumerate(payloads)
        },
        "wall": summarize_ns(wall_samples),
        "stages": stages,
        "dominant_stage": dominant["name"] if dominant else None,
        "dominant_stage_wall_share_pct": (
            dominant["wall_share_pct"] if dominant else 0.0
        ),
        "pmu": {
            "available": pmu_available,
            "unavailable_errno": sorted(
                {
                    int(payload["pmu"].get("unavailable_errno", 0))
                    for payload in payloads
                    if not payload["pmu"]["available"]
                }
            ),
            "totals": pmu_totals,
            "instructions_per_cycle": instructions / cycles if cycles else None,
            "cache_miss_pct": (
                cache_misses / cache_references * 100.0
                if cache_references
                else None
            ),
            "branch_miss_pct": (
                branch_misses / branches * 100.0 if branches else None
            ),
        },
        "source_hotspot": SOURCE_HOTSPOTS[case["kernel_name"]],
        "diagnosis_scope": (
            "coarse stage attribution plus target-only PMU; "
            "qconv_mac_address requires perf annotate to split address from MAC"
        ),
    }


def build_diagnosis(
    cases: Sequence[dict[str, Any]],
    payloads_by_case: dict[str, list[dict[str, Any]]],
    *,
    warmup: int,
    repeat: int,
    elapsed_seconds: float,
    qconv_candidate: str = "baseline",
    fused_qconv_candidate: str = "baseline",
    bn_candidate: str = "baseline",
    dequant_candidate: str = "baseline",
) -> dict[str, Any]:
    results = [
        _aggregate_case(case, payloads_by_case[case["case_name"]])
        for case in cases
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "locate internal bottlenecks before implementing E7 optimizations",
        "configuration": {
            "bucket_frames": 98,
            "threads": 1,
            "cpu_affinity": [0],
            "warmup": warmup,
            "repeat_per_input": repeat,
            "input_count": 3,
            "elapsed_seconds": elapsed_seconds,
            "qconv_candidate": qconv_candidate,
            "fused_qconv_candidate": fused_qconv_candidate,
            "bn_candidate": bn_candidate,
            "dequant_candidate": dequant_candidate,
        },
        "target_kernel_families": list(TARGET_KERNELS),
        "case_count": len(results),
        "cases": results,
        "optimization_gate": {
            "ready": True,
            "rule": (
                "implement only after a stage dominates stably across all three inputs; "
                "retain bitwise output hashes and rerun full retained-tensor validation"
            ),
        },
    }


def _resolve_profile_artifact(profile: dict[str, Any], name: str) -> Path:
    artifacts = profile.get("artifacts")
    entry = artifacts.get(name) if isinstance(artifacts, dict) else None
    value = entry.get("path") if isinstance(entry, dict) else None
    if not isinstance(value, str) or not value:
        raise DiagnosisError(f"profile artifact is missing: {name}")
    return _path(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=_path,
        default=ROOT / "results/profiling/e7_98/operator_profile.json",
    )
    parser.add_argument(
        "--binary",
        type=_path,
        default=ROOT / "build/profill/optimization/campp_operator_microbench",
    )
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument("--features", type=_path, nargs="+", default=list(EXPECTED_INPUTS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--qconv-candidate",
        choices=(
            "baseline", "address", "mac", "combined", "mac_fixed", "mac_asm"
        ),
        default="baseline",
    )
    parser.add_argument("--qconv-only", action="store_true")
    parser.add_argument(
        "--fused-qconv-candidate",
        choices=("baseline", "mac", "combined"),
        default="baseline",
    )
    parser.add_argument("--fused-qconv-only", action="store_true")
    parser.add_argument(
        "--bn-candidate",
        choices=("baseline", "address", "affine", "quant", "combined"),
        default="baseline",
    )
    parser.add_argument("--bn-only", action="store_true")
    parser.add_argument(
        "--dequant-candidate",
        choices=(
            "baseline", "address", "parameter",
            "scalar_combined", "neon_combined",
        ),
        default="baseline",
    )
    parser.add_argument("--dequant-only", action="store_true")
    parser.add_argument(
        "--runs-dir",
        type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/diagnosis",
    )
    parser.add_argument(
        "--output",
        type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/diagnosis.json",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.warmup < 0 or args.repeat <= 0:
            raise DiagnosisError("warmup must be >= 0 and repeat must be > 0")
        selected_only_modes = sum(
            (
                args.qconv_only, args.fused_qconv_only,
                args.bn_only, args.dequant_only,
            )
        )
        if selected_only_modes > 1:
            raise DiagnosisError(
                "--qconv-only, --fused-qconv-only, --bn-only and "
                "--dequant-only "
                "are mutually exclusive"
            )
        if os.name != "posix" and not args.preflight_only:
            raise DiagnosisError("actual diagnostics require Linux/QRB2210")
        if not args.profile.is_file():
            raise DiagnosisError(f"profile not found: {args.profile}")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        cases = select_representative_cases(profile.get("operators", []))
        if args.qconv_only:
            cases = [
                case
                for case in cases
                if case["case_name"] in ("qconv_3x3", "qconv_1x1")
            ]
        elif args.fused_qconv_only:
            cases = [
                case for case in cases
                if case["case_name"] == "fused_quant_qconv"
            ]
        elif args.bn_only:
            cases = [
                case for case in cases
                if case["case_name"] == "fused_bn_relu_quant"
            ]
        elif args.dequant_only:
            cases = [
                case for case in cases
                if case["case_name"] == "dequantize_linear"
            ]
        plan = args.plan or _resolve_profile_artifact(profile, "plan")
        weights = args.weights or _resolve_profile_artifact(profile, "weights")
        required = [args.binary, plan, weights, *args.features]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise DiagnosisError(
                "required files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3 or any(path.stat().st_size != 98 * 80 * 4 for path in args.features):
            raise DiagnosisError("exactly three float32 [1,98,80] inputs are required")
        if args.output.exists() and not args.force and not args.preflight_only:
            raise DiagnosisError(f"output exists: {args.output} (use --force)")
        capabilities = _run_payload([str(args.binary), "--capabilities"])
        if capabilities.get("stage_probe") is not True:
            raise DiagnosisError("binary does not expose optimization stage probes")
        if args.qconv_candidate not in capabilities.get("qconv_candidates", []):
            raise DiagnosisError("binary does not expose requested QConv candidate")
        if args.fused_qconv_candidate not in capabilities.get(
            "fused_qconv_candidates", []
        ):
            raise DiagnosisError(
                "binary does not expose requested fused QConv candidate"
            )
        if args.bn_candidate not in capabilities.get("bn_candidates", []):
            raise DiagnosisError("binary does not expose requested BN candidate")
        if args.dequant_candidate not in capabilities.get(
            "dequant_candidates", []
        ):
            raise DiagnosisError(
                "binary does not expose requested Dequant candidate"
            )
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "plan": _display_path(plan),
                        "weights": _display_path(weights),
                        "features": [_display_path(path) for path in args.features],
                        "cases": cases,
                        "qconv_candidate": args.qconv_candidate,
                        "fused_qconv_candidate": args.fused_qconv_candidate,
                        "bn_candidate": args.bn_candidate,
                        "dequant_candidate": args.dequant_candidate,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        expected_seconds = 30.0 + sum(
            (float(case["profile_mean_ms"]) / 1000.0)
            * (args.warmup + args.repeat + 1)
            * len(args.features)
            for case in cases
        )
        print(f"예상 시간: 약 {max(1, math.ceil(expected_seconds / 60.0))}분")
        raw_dir = args.runs_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        payloads_by_case: dict[str, list[dict[str, Any]]] = {
            case["case_name"]: [] for case in cases
        }
        started = time.monotonic()
        for case in cases:
            for feature in args.features:
                print("진행 중", flush=True)
                payload = _run_payload(
                    _command(
                        args.binary,
                        plan=plan,
                        weights=weights,
                        feature=feature,
                        operator_id=case["operator_id"],
                        warmup=args.warmup,
                        repeat=args.repeat,
                        qconv_candidate=(
                            args.qconv_candidate
                            if case["case_name"]
                            in ("qconv_3x3", "qconv_1x1")
                            else "baseline"
                        ),
                        fused_qconv_candidate=(
                            args.fused_qconv_candidate
                            if case["case_name"] == "fused_quant_qconv"
                            else "baseline"
                        ),
                        bn_candidate=(
                            args.bn_candidate
                            if case["case_name"] == "fused_bn_relu_quant"
                            else "baseline"
                        ),
                        dequant_candidate=(
                            args.dequant_candidate
                            if case["case_name"] == "dequantize_linear"
                            else "baseline"
                        ),
                    )
                )
                expected_candidate = (
                    args.qconv_candidate
                    if case["case_name"] in ("qconv_3x3", "qconv_1x1")
                    else "baseline"
                )
                expected_bn_candidate = (
                    args.bn_candidate
                    if case["case_name"] == "fused_bn_relu_quant"
                    else "baseline"
                )
                expected_fused_qconv_candidate = (
                    args.fused_qconv_candidate
                    if case["case_name"] == "fused_quant_qconv"
                    else "baseline"
                )
                expected_dequant_candidate = (
                    args.dequant_candidate
                    if case["case_name"] == "dequantize_linear"
                    else "baseline"
                )
                _validate_payload(
                    payload, case, args.repeat, expected_candidate,
                    expected_bn_candidate, expected_fused_qconv_candidate,
                    expected_dequant_candidate,
                )
                payload["input"] = {
                    "path": _display_path(feature),
                    "byte_size": feature.stat().st_size,
                }
                payloads_by_case[case["case_name"]].append(payload)
                raw_path = raw_dir / f"{case['case_name']}__{feature.stem}.json"
                raw_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
        diagnosis = build_diagnosis(
            cases,
            payloads_by_case,
            warmup=args.warmup,
            repeat=args.repeat,
            elapsed_seconds=time.monotonic() - started,
            qconv_candidate=args.qconv_candidate,
            fused_qconv_candidate=args.fused_qconv_candidate,
            bn_candidate=args.bn_candidate,
            dequant_candidate=args.dequant_candidate,
        )
        diagnosis["artifacts"] = {
            "profile": _display_path(args.profile),
            "plan": _display_path(plan),
            "weights": _display_path(weights),
            "raw_dir": _display_path(raw_dir),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"진단 완료: {_display_path(args.output)}")
        return 0
    except (DiagnosisError, OSError, ValueError, KeyError) as exc:
        print(f"diagnosis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
