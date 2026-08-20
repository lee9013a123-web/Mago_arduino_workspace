#!/usr/bin/env python3
"""Measure every ordinary E7 QLinearConv operator for layer-hybrid selection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
DIAGNOSIS_PATH = Path(__file__).with_name("02_diagnose_top4.py")
COMPARE = Path(__file__).with_name("03_compare_candidate.py")
SUPPORTED_MODES = ("baseline", "mac_fixed", "v4", "v5")
DEFAULT_MODES = ("baseline", "mac_fixed", "v4", "v5")

SPEC = importlib.util.spec_from_file_location("diagnose_top4", DIAGNOSIS_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {DIAGNOSIS_PATH}")
DIAGNOSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DIAGNOSIS
SPEC.loader.exec_module(DIAGNOSIS)


class QconvFamilyError(RuntimeError):
    """QConv family benchmark input or execution failed."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def select_qconv_family_cases(profile: dict[str, Any]) -> list[dict[str, Any]]:
    operators = profile.get("operators")
    if not isinstance(operators, list):
        raise QconvFamilyError("profile operator list is missing")
    selected = [
        operator for operator in operators
        if operator.get("kernel_name") == "qlinear_conv_o4i4_neon"
    ]
    if not selected:
        raise QconvFamilyError("profile is missing ordinary QLinearConv ops")
    selected.sort(key=lambda item: int(item["operator_id"]))
    return [
        {
            "case_name": f"qconv_op_{int(operator['operator_id'])}",
            "operator_id": int(operator["operator_id"]),
            "kernel_id": int(operator["kernel_id"]),
            "kernel_name": str(operator["kernel_name"]),
            "operator_type": str(operator["operator_type"]),
            "weight_shape": list(DIAGNOSIS._meaningful_weight_shape(operator)),
            "profile_mean_ms": float(operator["mean_ms"]),
            "profile_share_pct": float(operator["end_to_end_share_pct"]),
        }
        for operator in selected
    ]


def _run(command: list[str], allowed_returncodes: tuple[int, ...] = (0,)) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in allowed_returncodes:
        raise QconvFamilyError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


def run_mode(
    *,
    mode: str,
    cases: Sequence[dict[str, Any]],
    binary: Path,
    plan: Path,
    weights: Path,
    features: Sequence[Path],
    warmup: int,
    repeat: int,
    runs_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    raw_dir = runs_dir / mode / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payloads_by_case: dict[str, list[dict[str, Any]]] = {
        case["case_name"]: [] for case in cases
    }
    started = time.monotonic()
    for case in cases:
        for feature in features:
            print(f"진행 중 qconv {mode} op {case['operator_id']}", flush=True)
            payload = DIAGNOSIS._run_payload(
                DIAGNOSIS._command(
                    binary,
                    plan=plan,
                    weights=weights,
                    feature=feature,
                    operator_id=int(case["operator_id"]),
                    warmup=warmup,
                    repeat=repeat,
                    qconv_candidate=mode,
                )
            )
            DIAGNOSIS._validate_payload(
                payload, case, repeat, mode, "baseline", "baseline",
                "baseline",
            )
            payload["input"] = {
                "path": _display_path(feature),
                "byte_size": feature.stat().st_size,
            }
            payloads_by_case[case["case_name"]].append(payload)
            (raw_dir / f"{case['case_name']}__{feature.stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
    document = DIAGNOSIS.build_diagnosis(
        cases,
        payloads_by_case,
        warmup=warmup,
        repeat=repeat,
        elapsed_seconds=time.monotonic() - started,
        qconv_candidate=mode,
    )
    document["objective"] = "measure every ordinary QLinearConv for hybrid selection"
    document["artifacts"] = {
        "plan": _display_path(plan),
        "weights": _display_path(weights),
        "raw_dir": _display_path(raw_dir),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{mode}.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return document


def build_summary(modes: Sequence[str], results_dir: Path) -> dict[str, Any]:
    comparisons = []
    for mode in modes:
        if mode == "baseline":
            continue
        path = results_dir / f"{mode}_comparison.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        comparisons.append(
            {
                "mode": mode,
                "gate_passed": document["candidate_microbench_gate_passed"],
                "all_output_hashes_bitwise_identical": document[
                    "all_output_hashes_bitwise_identical"
                ],
                "no_case_regressed_over_1pct": document[
                    "no_case_regressed_over_1pct"
                ],
                **document["family_operator_sum"],
                "comparison": _display_path(path),
            }
        )
    eligible = [item for item in comparisons if item["gate_passed"]]
    winner = min(
        eligible, key=lambda item: float(item["candidate_mean_ms"]),
        default=None,
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "all ordinary qlinear_conv_o4i4 operators",
        "measurement_strategy": "single-operator replay per input",
        "modes": list(modes),
        "comparisons": comparisons,
        "winner": winner["mode"] if winner is not None else None,
        "production_gate_ready": winner is not None,
        "next_gate": "build layer-hybrid plan, then retained-tensor and Quick E2E",
    }


def compare_modes(modes: Sequence[str], results_dir: Path, force: bool) -> None:
    for mode in modes:
        if mode == "baseline":
            continue
        output = results_dir / f"{mode}_comparison.json"
        command = [
            sys.executable,
            str(COMPARE),
            "--baseline",
            str(results_dir / "baseline.json"),
            "--candidate",
            str(results_dir / f"{mode}.json"),
            "--output",
            str(output),
        ]
        if force:
            command.append("--force")
        _run(command, allowed_returncodes=(0, 3))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=_path, default=ROOT / "results/profiling/e7_98/operator_profile.json")
    parser.add_argument("--binary", type=_path, default=ROOT / "build/profill/optimization/campp_operator_microbench")
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument("--features", type=_path, nargs="+", default=list(DIAGNOSIS.EXPECTED_INPUTS))
    parser.add_argument("--modes", nargs="+", choices=SUPPORTED_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--runs-dir", type=_path, default=ROOT / "runs/profiling/e7_98/optimization/qconv_family")
    parser.add_argument("--results-dir", type=_path, default=ROOT / "results/profiling/e7_98/optimization/qconv_family")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.warmup < 0 or args.repeat <= 0:
            raise QconvFamilyError("warmup must be >= 0 and repeat > 0")
        modes = list(dict.fromkeys(args.modes))
        if "baseline" not in modes:
            modes.insert(0, "baseline")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        cases = select_qconv_family_cases(profile)
        plan = args.plan or DIAGNOSIS._resolve_profile_artifact(profile, "plan")
        weights = args.weights or DIAGNOSIS._resolve_profile_artifact(profile, "weights")
        required = [args.binary, plan, weights, *args.features]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise QconvFamilyError(
                "required files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3 or any(path.stat().st_size != 98 * 80 * 4 for path in args.features):
            raise QconvFamilyError("exactly three float32 [1,98,80] inputs are required")
        capabilities = DIAGNOSIS._run_payload([str(args.binary), "--capabilities"])
        available = capabilities.get("qconv_candidates", [])
        if any(mode not in available for mode in modes):
            raise QconvFamilyError("binary does not expose requested QConv modes")
        if args.preflight_only:
            print(json.dumps({
                "ready": True,
                "operator_count": len(cases),
                "modes": modes,
                "plan": _display_path(plan),
                "weights": _display_path(weights),
                "features": [_display_path(path) for path in args.features],
            }, ensure_ascii=False, indent=2))
            return 0
        planned = [
            args.results_dir / "summary.json",
            *(args.results_dir / f"{mode}.json" for mode in modes),
            *(args.results_dir / f"{mode}_comparison.json" for mode in modes if mode != "baseline"),
        ]
        existing = [path for path in planned if path.exists()]
        if existing and not args.force:
            raise QconvFamilyError(f"output exists: {existing[0]} (use --force)")
        estimated = sum(float(case["profile_mean_ms"]) for case in cases) / 1000.0
        estimated *= (args.warmup + args.repeat + 1) * len(args.features) * len(modes)
        print(f"예상 시간: 약 {max(1, math.ceil((30.0 + estimated) / 60.0))}분")
        for mode in modes:
            run_mode(
                mode=mode,
                cases=cases,
                binary=args.binary,
                plan=plan,
                weights=weights,
                features=args.features,
                warmup=args.warmup,
                repeat=args.repeat,
                runs_dir=args.runs_dir,
                results_dir=args.results_dir,
            )
        compare_modes(modes, args.results_dir, args.force)
        summary = build_summary(modes, args.results_dir)
        (args.results_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(f"qconv family summary: {_display_path(args.results_dir / 'summary.json')}")
        return 0 if summary["production_gate_ready"] else 3
    except (QconvFamilyError, OSError, ValueError, KeyError) as exc:
        print(f"qconv family benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
