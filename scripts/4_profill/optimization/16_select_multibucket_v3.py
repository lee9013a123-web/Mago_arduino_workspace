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
PROFILL_DIR = ROOT / "scripts/4_profill"
if str(PROFILL_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILL_DIR))

from bucket_profile_identity import (  # noqa: E402
    BucketProfileIdentityError,
    validate_profile_against_plan,
)

OPTIMIZATION_DIR = ROOT / "scripts/4_profill/optimization"
PROFILE_SCRIPT = PROFILL_DIR / "02_profile_e7.py"
QCONV_BENCHMARK = OPTIMIZATION_DIR / "12_benchmark_qconv_family.py"
FUSED_BENCHMARK = OPTIMIZATION_DIR / "08_benchmark_fused_qconv_family.py"
PLAN_BUILDER = OPTIMIZATION_DIR / "13_build_conv_hybrid_plan.py"
SOURCE_GENERATOR = OPTIMIZATION_DIR / "17_generate_multibucket_v3_source.py"
OPTIMIZATION_BUILD = OPTIMIZATION_DIR / "01_build_optimization.sh"
FINAL_V3_BUILD = ROOT / "scripts/4_profill/05_build_final_v3.sh"
DEFAULT_BUNDLE = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
DEFAULT_FEATURE_DIR = ROOT / "benchmarks/campplus/features"
DEFAULT_PROFILE_CONFIG = (
    ROOT / "configs/benchmark/runtime_final_v3_multibucket_official.json"
)
DEFAULT_PROFILE_BASELINE = (
    ROOT / "build/profill/final_v3_hybrid/campp_runtime_benchmark_final"
)
DEFAULT_PROFILE_BINARY = (
    ROOT / "build/profill/final_v3_hybrid/campp_e7_profiler_final"
)
DEFAULT_RUNS_ROOT = ROOT / "runs/models/campplus/final_v3/layer_selection"
DEFAULT_RESULTS_ROOT = ROOT / "results/models/campplus/final_v3/layer_selection"
DEFAULT_FINAL_SUITE_CONFIG = (
    "qconv_layer_hybrid_v3+fused_layer_hybrid_v3+bn_v2_spatial2+"
    "dequant_neon_combined+fused_dqrq_neon+remaining_optimized"
)
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


def _bucket_paths(
    bucket: int, runs_root: Path = DEFAULT_RUNS_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict[str, Path]:
    result_root = results_root / str(bucket)
    run_root = runs_root / str(bucket)
    return {
        "profile": result_root / "operator_profile/operator_profile.json",
        "profile_results": result_root / "operator_profile",
        "profile_runs": run_root / "operator_profile/raw",
        "qconv_results": result_root / "qconv_family",
        "qconv_runs": run_root / "qconv_family",
        "fused_results": result_root / "fused_qconv_family",
        "fused_runs": run_root / "fused_qconv_family",
        "plan_json": result_root / "conv_hybrid_plan.json",
        "plan_csv": result_root / "conv_hybrid_plan.csv",
    }


def _profile_command(
    *, bucket: int, config: Path, baseline_binary: Path,
    profiler_binary: Path, raw_dir: Path, output_dir: Path,
    suite_config: str, force: bool,
) -> list[str]:
    command = [
        sys.executable, str(PROFILE_SCRIPT),
        "--config", str(config),
        "--bucket-frames", str(bucket),
        "--baseline-binary", str(baseline_binary),
        "--profiler-binary", str(profiler_binary),
        "--expected-suite", "final",
        "--expected-suite-config", suite_config,
        "--mode", "quick",
        "--allow-unvalidated-bucket-profile",
        "--raw-dir", str(raw_dir),
        "--output-dir", str(output_dir),
    ]
    if force:
        command.append("--force")
    return command


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
    stale = []
    if not missing:
        for path in (qconv_binary, fused_binary):
            completed = subprocess.run(
                [str(path), "--capabilities"], cwd=ROOT,
                text=True, capture_output=True, check=False,
            )
            try:
                capabilities = json.loads(completed.stdout)
            except json.JSONDecodeError:
                capabilities = {}
            if completed.returncode != 0 or (
                capabilities.get("bucket_support") != "plan_header"
            ):
                stale.append(path)
    if not rebuild and not missing and not stale:
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
    if rebuild:
        reason = "--rebuild"
    elif missing:
        reason = f"missing: {missing[0]}"
    else:
        reason = f"stale bucket capability: {stale[0]}"
    print(f"optimization build 실행 ({reason})", flush=True)
    _run(["bash", str(OPTIMIZATION_BUILD)])
    return False


def _required_artifacts(
    buckets: Sequence[int], bundle: Path, feature_dir: Path,
    speakers: Sequence[str], profile_config: Path,
    profile_baseline: Path, profile_binary: Path,
    *, include_profile_tools: bool,
) -> list[Path]:
    required = [bundle / "weights.bin"]
    if include_profile_tools:
        required.extend((profile_config, profile_baseline, profile_binary))
    for bucket in buckets:
        required.append(bundle / "execution_plans" / f"plan_{bucket}.bin")
        required.extend(_features(feature_dir, bucket, speakers))
    return required


def _generate_source_command(
    buckets: Sequence[int], include_98_plan: Path, force: bool,
    *, runs_root: Path = DEFAULT_RUNS_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    check_only: bool = False,
) -> list[str]:
    command = [sys.executable, str(SOURCE_GENERATOR)]
    plan_specs = [(98, include_98_plan)] + [
        (
            bucket,
            _bucket_paths(bucket, runs_root, results_root)["plan_json"],
        )
        for bucket in buckets if bucket != 98
    ]
    for bucket, path in plan_specs:
        command.extend(("--plan", f"{bucket}={path}"))
    command.extend((
        "--manifest", str(results_root / "multibucket_manifest.json")
    ))
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
    parser.add_argument("--bundle", type=_path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--runs-root", type=_path, default=DEFAULT_RUNS_ROOT
    )
    parser.add_argument(
        "--results-root", type=_path, default=DEFAULT_RESULTS_ROOT
    )
    parser.add_argument(
        "--profile-config", type=_path, default=DEFAULT_PROFILE_CONFIG
    )
    parser.add_argument(
        "--profile-baseline-binary", type=_path,
        default=DEFAULT_PROFILE_BASELINE,
    )
    parser.add_argument(
        "--profile-binary", type=_path, default=DEFAULT_PROFILE_BINARY
    )
    parser.add_argument(
        "--expected-suite-config", default=DEFAULT_FINAL_SUITE_CONFIG
    )
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
    parser.add_argument(
        "--profiles-only", action="store_true",
        help="bucket별 operator profile 생성·identity 검증 후 중지한다",
    )
    parser.add_argument(
        "--reprofile", action="store_true",
        help="기존 bucket operator profile을 다시 측정한다",
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
        if args.plan_only and args.profiles_only:
            raise MultibucketSelectionError(
                "--plan-only and --profiles-only are mutually exclusive"
            )
        protocol = PROTOCOLS[args.mode]
        bucket_paths = {
            bucket: _bucket_paths(
                bucket, args.runs_root, args.results_root
            )
            for bucket in buckets
        }
        needs_profile_run = args.reprofile or any(
            not paths["profile"].is_file()
            for paths in bucket_paths.values()
        )
        required = [args.plan_98, args.bundle / "weights.bin"]
        for bucket in buckets:
            required.append(
                args.bundle / "execution_plans" / f"plan_{bucket}.bin"
            )
        if args.plan_only:
            for bucket in buckets:
                paths = _bucket_paths(
                    bucket, args.runs_root, args.results_root
                )
                required.append(paths["profile"])
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
            required.extend(_required_artifacts(
                buckets, args.bundle, args.feature_dir,
                args.speakers, args.profile_config,
                args.profile_baseline_binary, args.profile_binary,
                include_profile_tools=needs_profile_run,
            ))
        required = list(dict.fromkeys(required))
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise MultibucketSelectionError(
                "required artifacts are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        build_reused = True
        if not args.plan_only and not args.profiles_only:
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
            "profile_source": "bucket-specific exact execution plan",
            "runs_root": _display_path(args.runs_root),
            "results_root": _display_path(args.results_root),
        }, ensure_ascii=False, indent=2), flush=True)

        profile_identity: dict[int, dict[str, object]] = {}
        for bucket in buckets:
            paths = _bucket_paths(
                bucket, args.runs_root, args.results_root
            )
            plan = args.bundle / "execution_plans" / f"plan_{bucket}.bin"
            if args.reprofile or not paths["profile"].is_file():
                if args.preflight_only or args.plan_only:
                    raise MultibucketSelectionError(
                        f"bucket {bucket} profile is missing or requires "
                        "--reprofile; run --profiles-only first"
                    )
                _run(_profile_command(
                    bucket=bucket,
                    config=args.profile_config,
                    baseline_binary=args.profile_baseline_binary,
                    profiler_binary=args.profile_binary,
                    raw_dir=paths["profile_runs"],
                    output_dir=paths["profile_results"],
                    suite_config=args.expected_suite_config,
                    force=args.reprofile,
                ))
            profile_identity[bucket] = validate_profile_against_plan(
                paths["profile"], plan, bucket
            )
            print(
                f"profile identity PASS: bucket={bucket} "
                f"operators={profile_identity[bucket]['operator_count']}",
                flush=True,
            )

        if args.profiles_only:
            print("multibucket operator profiles: complete")
            return 0

        if not args.plan_only:
            for bucket in buckets:
                paths = _bucket_paths(
                    bucket, args.runs_root, args.results_root
                )
                plan = (
                    args.bundle / "execution_plans" / f"plan_{bucket}.bin"
                )
                weights = args.bundle / "weights.bin"
                features = _features(
                    args.feature_dir, bucket, args.speakers
                )
                common = {
                    "bucket": bucket,
                    "profile": paths["profile"],
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
            paths = _bucket_paths(
                bucket, args.runs_root, args.results_root
            )
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
            buckets, args.plan_98, force=True,
            runs_root=args.runs_root,
            results_root=args.results_root,
        ))
        if args.build_final:
            if os.name != "posix":
                raise MultibucketSelectionError(
                    "final V3 build requires Linux/QRB2210"
                )
            _run(["bash", str(FINAL_V3_BUILD)])
        print("multibucket V3 layer selection: complete")
        return 0
    except (
        MultibucketSelectionError, BucketProfileIdentityError,
        OSError, ValueError,
    ) as exc:
        print(f"multibucket V3 selection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
