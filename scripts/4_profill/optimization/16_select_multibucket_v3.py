#!/usr/bin/env python3
"""Measure and generate layer-hybrid V3 plans for additional frame buckets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3]
OPTIMIZATION_DIR = ROOT / "scripts/4_profill/optimization"
QCONV_BENCHMARK = OPTIMIZATION_DIR / "12_benchmark_qconv_family.py"
FUSED_BENCHMARK = OPTIMIZATION_DIR / "08_benchmark_fused_qconv_family.py"
PLAN_BUILDER = OPTIMIZATION_DIR / "13_build_conv_hybrid_plan.py"
SOURCE_GENERATOR = OPTIMIZATION_DIR / "17_generate_multibucket_v3_source.py"
OPTIMIZATION_BUILD = OPTIMIZATION_DIR / "01_build_optimization.sh"
FINAL_V3_BUILD = ROOT / "scripts/4_profill/05_build_final_v3.sh"
DEFAULT_PROFILE = ROOT / "results/profiling/e7_98/operator_profile.json"
DEFAULT_BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
DEFAULT_FEATURE_DIR = ROOT / "benchmarks/campplus/features"
DEFAULT_QCONV_BINARY = (
    ROOT / "build/profill/optimization/campp_qconv_family_bench"
)
DEFAULT_FUSED_BINARY = (
    ROOT / "build/profill/optimization/campp_fused_qconv_family_bench"
)
DEFAULT_SPEAKERS = ("0000", "0005", "0006")
PROTOCOLS = {
    "quick": {"warmup": 2, "repeat": 10},
    "official": {"warmup": 5, "repeat": 20},
}


class MultibucketSelectionError(RuntimeError):
    """The multibucket selection pipeline could not complete safely."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _run(command: Sequence[str], allowed: tuple[int, ...] = (0,)) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(list(command), cwd=ROOT, check=False)
    if completed.returncode not in allowed:
        raise MultibucketSelectionError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


def _features(feature_dir: Path, bucket: int, speakers: Sequence[str]) -> list[Path]:
    return [
        feature_dir / f"multi__speaker_{speaker}__{bucket}.f32"
        for speaker in speakers
    ]


def _bucket_paths(bucket: int) -> dict[str, Path]:
    result_root = ROOT / f"results/profiling/e7_{bucket}/optimization"
    run_root = ROOT / f"runs/profiling/e7_{bucket}/optimization"
    return {
        "qconv_results": result_root / "qconv_family",
        "qconv_runs": run_root / "qconv_family",
        "fused_results": result_root / "fused_qconv_family",
        "fused_runs": run_root / "fused_qconv_family",
        "plan_json": result_root / "conv_hybrid_plan.json",
        "plan_csv": result_root / "conv_hybrid_plan.csv",
    }


def _benchmark_command(
    *, script: Path, bucket: int, profile: Path, binary: Path,
    plan: Path, weights: Path, features: Sequence[Path],
    results_dir: Path, runs_dir: Path, modes: Sequence[str],
    warmup: int, repeat: int, preflight_only: bool, force: bool,
) -> list[str]:
    command = [
        sys.executable, str(script),
        "--bucket-frames", str(bucket),
        "--profile", str(profile),
        "--binary", str(binary),
        "--plan", str(plan),
        "--weights", str(weights),
        "--features", *(str(path) for path in features),
        "--modes", *modes,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--runs-dir", str(runs_dir),
        "--results-dir", str(results_dir),
        "--baseline-check-only",
    ]
    if preflight_only:
        command.append("--preflight-only")
    if force and not preflight_only:
        command.append("--force")
    return command


def _ensure_measurement_binaries(
    qconv_binary: Path, fused_binary: Path, rebuild: bool,
    preflight_only: bool,
) -> bool:
    missing = [
        path for path in (qconv_binary, fused_binary) if not path.is_file()
    ]
    if not rebuild and not missing:
        print("build 재사용: family benchmark binary 2개", flush=True)
        return True
    if preflight_only:
        raise MultibucketSelectionError(
            "preflight does not build binaries; run "
            "bash scripts/4_profill/optimization/01_build_optimization.sh"
        )
    if os.name != "posix":
        raise MultibucketSelectionError(
            "benchmark binary build requires Linux/QRB2210"
        )
    reason = "--rebuild" if rebuild else f"missing: {missing[0]}"
    print(f"optimization build 실행 ({reason})", flush=True)
    _run(["bash", str(OPTIMIZATION_BUILD)])
    return False


def _required_artifacts(
    buckets: Sequence[int], bundle: Path, feature_dir: Path,
    speakers: Sequence[str], profile: Path,
) -> list[Path]:
    required = [profile, bundle / "weights.bin"]
    for bucket in buckets:
        required.append(bundle / "execution_plans" / f"plan_{bucket}.bin")
        required.extend(_features(feature_dir, bucket, speakers))
    return required


def _generate_source_command(
    buckets: Sequence[int], include_98_plan: Path, force: bool,
    check_only: bool = False,
) -> list[str]:
    command = [sys.executable, str(SOURCE_GENERATOR)]
    plan_specs = [(98, include_98_plan)] + [
        (bucket, _bucket_paths(bucket)["plan_json"])
        for bucket in buckets if bucket != 98
    ]
    for bucket, path in plan_specs:
        command.extend(("--plan", f"{bucket}={path}"))
    if check_only:
        command.append("--check-only")
    elif force:
        command.append("--force")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buckets", type=int, nargs="+", default=[298, 498, 998]
    )
    parser.add_argument("--mode", choices=PROTOCOLS, default="official")
    parser.add_argument("--profile", type=_path, default=DEFAULT_PROFILE)
    parser.add_argument("--bundle", type=_path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--feature-dir", type=_path, default=DEFAULT_FEATURE_DIR
    )
    parser.add_argument(
        "--speakers", nargs="+", default=list(DEFAULT_SPEAKERS)
    )
    parser.add_argument(
        "--qconv-binary", type=_path, default=DEFAULT_QCONV_BINARY
    )
    parser.add_argument(
        "--fused-binary", type=_path, default=DEFAULT_FUSED_BINARY
    )
    parser.add_argument(
        "--plan-98", type=_path,
        default=(
            ROOT / "results/profiling/e7_98/optimization/conv_hybrid_plan.json"
        ),
    )
    parser.add_argument("--min-margin-pct", type=float, default=1.0)
    parser.add_argument(
        "--plan-only", action="store_true",
        help="existing family results에서 plan/source만 다시 생성한다",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--build-final", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        buckets = list(dict.fromkeys(args.buckets))
        if not buckets or any(bucket <= 0 or bucket == 98 for bucket in buckets):
            raise MultibucketSelectionError(
                "--buckets must contain positive non-98 frame buckets"
            )
        if args.min_margin_pct < 0.0:
            raise MultibucketSelectionError("min-margin-pct must be non-negative")
        if not args.speakers:
            raise MultibucketSelectionError("at least one speaker is required")
        protocol = PROTOCOLS[args.mode]
        if args.plan_only:
            required = [args.plan_98]
            for bucket in buckets:
                paths = _bucket_paths(bucket)
                for result_dir, modes in (
                    (
                        paths["qconv_results"],
                        ("mac_fixed", "v4", "v5"),
                    ),
                    (
                        paths["fused_results"],
                        ("combined_fixed", "combined_hybrid", "combined_v5"),
                    ),
                ):
                    required.extend(
                        result_dir / f"{mode}_comparison.json"
                        for mode in modes
                    )
        else:
            required = _required_artifacts(
                buckets, args.bundle, args.feature_dir,
                args.speakers, args.profile
            )
            required.append(args.plan_98)
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise MultibucketSelectionError(
                "required artifacts are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        build_reused = True
        if not args.plan_only:
            build_reused = _ensure_measurement_binaries(
                args.qconv_binary, args.fused_binary, args.rebuild,
                args.preflight_only,
            )

        print(json.dumps({
            "buckets": buckets,
            "protocol": args.mode,
            "warmup": protocol["warmup"],
            "repeat": protocol["repeat"],
            "input_count_per_bucket": len(args.speakers),
            "baseline_check_only": True,
            "min_margin_pct": args.min_margin_pct,
            "build_reused": build_reused,
        }, ensure_ascii=False, indent=2), flush=True)

        if not args.plan_only:
            for bucket in buckets:
                paths = _bucket_paths(bucket)
                plan = (
                    args.bundle / "execution_plans" / f"plan_{bucket}.bin"
                )
                weights = args.bundle / "weights.bin"
                features = _features(
                    args.feature_dir, bucket, args.speakers
                )
                common = {
                    "bucket": bucket,
                    "profile": args.profile,
                    "plan": plan,
                    "weights": weights,
                    "features": features,
                    "warmup": protocol["warmup"],
                    "repeat": protocol["repeat"],
                    "preflight_only": args.preflight_only,
                    "force": args.force,
                }
                _run(_benchmark_command(
                    script=QCONV_BENCHMARK,
                    binary=args.qconv_binary,
                    results_dir=paths["qconv_results"],
                    runs_dir=paths["qconv_runs"],
                    modes=("baseline", "mac_fixed", "v4", "v5"),
                    **common,
                ))
                _run(_benchmark_command(
                    script=FUSED_BENCHMARK,
                    binary=args.fused_binary,
                    results_dir=paths["fused_results"],
                    runs_dir=paths["fused_runs"],
                    modes=(
                        "baseline", "combined_fixed",
                        "combined_hybrid", "combined_v5",
                    ),
                    **common,
                ))
        if args.preflight_only:
            print("multibucket V3 selection preflight: ready")
            return 0

        for bucket in buckets:
            paths = _bucket_paths(bucket)
            command = [
                sys.executable, str(PLAN_BUILDER),
                "--bucket-frames", str(bucket),
                "--qconv-dir", str(paths["qconv_results"]),
                "--fused-dir", str(paths["fused_results"]),
                "--min-margin-pct", str(args.min_margin_pct),
                "--output", str(paths["plan_json"]),
                "--csv", str(paths["plan_csv"]),
            ]
            if args.force:
                command.append("--force")
            _run(command)

        # The target is a generated source file. Reaching this point means all
        # measured family gates and per-bucket plan builds succeeded, so update
        # it even on the first non-force pipeline run.
        _run(_generate_source_command(
            buckets, args.plan_98, force=True
        ))
        if args.build_final:
            if os.name != "posix":
                raise MultibucketSelectionError(
                    "final V3 build requires Linux/QRB2210"
                )
            _run(["bash", str(FINAL_V3_BUILD)])
        print("multibucket V3 layer selection: complete")
        return 0
    except (MultibucketSelectionError, OSError, ValueError) as exc:
        print(f"multibucket V3 selection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
