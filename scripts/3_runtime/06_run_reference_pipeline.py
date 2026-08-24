#!/usr/bin/env python3
"""Run exporter, ORT reference, C build, inference, and comparison in staging.

The pipeline never writes to ``models/compiled/reference``.  Every invocation
uses a unique run directory below the configured workspace root, so an existing
verified bundle cannot be mixed with a partial or failed run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from runtime_pipeline_config import (
    RuntimePipelineConfig,
    RuntimePipelineConfigError,
    load_runtime_pipeline_config,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "3_runtime"
CANONICAL_BUNDLE_DIR = ROOT / "models" / "compiled" / "reference"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class PipelinePaths:
    run_dir: Path
    bundle_dir: Path
    ort_dir: Path
    build_dir: Path
    c_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class PipelineStep:
    number: int
    name: str
    command: tuple[str, ...]
    environment: dict[str, str]


def _default_run_id(profile: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{profile}-{stamp}"


def _pipeline_paths(config: RuntimePipelineConfig, run_id: str) -> PipelinePaths:
    run_dir = config.paths.workspace_root / run_id
    return PipelinePaths(
        run_dir=run_dir,
        bundle_dir=run_dir / "bundle",
        ort_dir=run_dir / "ort_reference",
        build_dir=run_dir / "build",
        c_dir=run_dir / "c_reference",
        results_dir=run_dir / "results",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_snapshot(directory: Path) -> dict[str, dict[str, object]]:
    if not directory.is_dir():
        return {}
    return {
        str(path.relative_to(directory)): {
            "byte_size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _require_pipeline_inputs(config: RuntimePipelineConfig) -> None:
    required = [config.paths.canonical_model]
    tags = {98: "1s", 298: "3s", 498: "5s", 998: "10s"}
    # Bundle export always creates all four plans, even when later steps select
    # a subset for development.
    for frames, tag in tags.items():
        required.extend(
            (
                config.paths.static_dir / f"campp_static_{frames}.onnx",
                config.paths.graph_dir / f"ir_{tag}.json",
            )
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimePipelineConfigError(
            "pipeline inputs are missing:\n  " + "\n  ".join(missing)
        )


def _build_steps(
    config: RuntimePipelineConfig,
    paths: PipelinePaths,
    *,
    python: str,
    bash: str,
) -> tuple[PipelineStep, ...]:
    bucket_args = tuple(str(frames) for frames in config.buckets)
    bucket_env = " ".join(bucket_args)
    weight_index_flag = (
        "--include-weight-index"
        if config.include_weight_index
        else "--no-weight-index"
    )
    return (
        PipelineStep(
            1,
            "export reference bundle",
            (
                python,
                str(SCRIPT_DIR / "01_export_reference_bundle.py"),
                "--static-dir",
                str(config.paths.static_dir),
                "--graph-dir",
                str(config.paths.graph_dir),
                "--canonical-model",
                str(config.paths.canonical_model),
                "--output-dir",
                str(paths.bundle_dir),
                weight_index_flag,
            ),
            {},
        ),
        PipelineStep(
            2,
            "generate ORT references",
            (
                python,
                str(SCRIPT_DIR / "02_dump_ort_references.py"),
                "--static-dir",
                str(config.paths.static_dir),
                "--output-dir",
                str(paths.ort_dir),
                "--seed",
                str(config.seed),
                "--buckets",
                *bucket_args,
            ),
            {},
        ),
        PipelineStep(
            3,
            "build C Reference Runtime",
            (bash, str(SCRIPT_DIR / "03_build_reference_runtime.sh")),
            {
                "BUILD_DIR": str(paths.build_dir),
                "CC": config.build.cc,
                "CFLAGS": config.build.cflags,
            },
        ),
        PipelineStep(
            4,
            "run C Reference Runtime",
            (bash, str(SCRIPT_DIR / "04_run_reference_runtime.sh")),
            {
                "BUILD_DIR": str(paths.build_dir),
                "BUNDLE_DIR": str(paths.bundle_dir),
                "ORT_DIR": str(paths.ort_dir),
                "OUT_DIR": str(paths.c_dir),
                "BUCKETS": bucket_env,
            },
        ),
        PipelineStep(
            5,
            "compare C outputs with ORT",
            (
                python,
                str(SCRIPT_DIR / "05_compare_runtime_outputs.py"),
                "--static-dir",
                str(config.paths.static_dir),
                "--graph-dir",
                str(config.paths.graph_dir),
                "--ort-dir",
                str(paths.ort_dir),
                "--c-dir",
                str(paths.c_dir),
                "--results-dir",
                str(paths.results_dir),
                "--buckets",
                *bucket_args,
                "--float-atol",
                str(config.tolerances.float_atol),
                "--float-rtol",
                str(config.tolerances.float_rtol),
                "--embedding-cosine-min",
                str(config.tolerances.embedding_cosine_min),
                "--max-report",
                str(config.diagnostics.max_failure_report),
            ),
            {},
        ),
    )


def _display_step(step: PipelineStep, *, dry_run: bool) -> None:
    label = "DRY RUN" if dry_run else "RUN"
    print(f"[{label} {step.number}/5] {step.name}")
    for key in sorted(step.environment):
        print(f"  env {key}={step.environment[key]}")
    print(f"  $ {shlex.join(step.command)}")


def _resolved_config_document(
    config: RuntimePipelineConfig,
    paths: PipelinePaths,
    run_id: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile": config.profile,
        "run_id": run_id,
        "source_config": str(config.source),
        "backend": config.backend,
        "buckets": list(config.buckets),
        "seed": config.seed,
        "diagnostics": {
            "enabled": config.diagnostics.enabled,
            "dump_tensor_ids": config.diagnostics.dump_tensor_ids,
            "checkpoint_tensor_ids": list(
                config.diagnostics.checkpoint_tensor_ids
            ),
        },
        "tolerances": {
            "float_atol": config.tolerances.float_atol,
            "float_rtol": config.tolerances.float_rtol,
            "embedding_cosine_min": config.tolerances.embedding_cosine_min,
        },
        "paths": {
            "run_dir": str(paths.run_dir),
            "bundle_dir": str(paths.bundle_dir),
            "ort_dir": str(paths.ort_dir),
            "build_dir": str(paths.build_dir),
            "c_dir": str(paths.c_dir),
            "results_dir": str(paths.results_dir),
            "protected_canonical_bundle": str(CANONICAL_BUNDLE_DIR),
        },
    }


def _write_verification_report(
    config: RuntimePipelineConfig,
    paths: PipelinePaths,
) -> Path | None:
    """Collect configured checkpoints from 05 comparison artifacts."""

    checkpoints = set(config.diagnostics.checkpoint_tensor_ids)
    buckets: list[dict[str, object]] = []
    for frames in config.buckets:
        comparison_path = paths.results_dir / f"compare_{frames}.json"
        if not comparison_path.is_file():
            continue
        try:
            comparisons = json.loads(comparison_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(comparisons, list):
            continue
        failures = [
            item
            for item in comparisons
            if isinstance(item, dict)
            and item.get("status") not in {"exact", "within_tolerance"}
        ]
        embedding = next(
            (
                item
                for item in comparisons
                if isinstance(item, dict) and item.get("tensor_name") == "embedding"
            ),
            None,
        )
        embedding_cosine = (
            float(embedding.get("cosine_similarity", float("nan")))
            if embedding is not None
            else float("nan")
        )
        embedding_cosine_passed = (
            embedding is not None
            and embedding_cosine >= config.tolerances.embedding_cosine_min
        )
        selected = [
            item
            for item in comparisons
            if isinstance(item, dict) and item.get("tensor_id") in checkpoints
        ]
        buckets.append(
            {
                "bucket_frames": frames,
                "compared_tensors": len(comparisons),
                "failed_tensors": len(failures),
                "first_failure": (
                    failures[0]
                    if failures
                    else (
                        None
                        if embedding_cosine_passed
                        else {
                            "status": "embedding_cosine_below_threshold",
                            "cosine_similarity": embedding_cosine,
                        }
                    )
                ),
                "embedding": embedding,
                "embedding_cosine_passed": embedding_cosine_passed,
                "checkpoints": selected,
                "numerical_passed": not failures and embedding_cosine_passed,
            }
        )
    if not buckets:
        return None

    report = {
        "profile": config.profile,
        "buckets": buckets,
        "checkpoint_tensor_ids": sorted(checkpoints),
        "complete": len(buckets) == len(config.buckets),
        "all_numerically_passed": (
            len(buckets) == len(config.buckets)
            and all(bool(bucket["numerical_passed"]) for bucket in buckets)
        ),
    }
    path = paths.results_dir / "verification_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "runtime" / "reference.json",
    )
    parser.add_argument(
        "--run-id",
        help="staging folder name; default is <profile>-<UTC timestamp>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print all commands without creating files or executing them",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="explicitly allow reuse of an existing run-id staging directory",
    )
    args = parser.parse_args(argv)

    try:
        config = load_runtime_pipeline_config(
            args.config, repository_root=ROOT
        )
        _require_pipeline_inputs(config)
    except RuntimePipelineConfigError as exc:
        print(f"pipeline config error: {exc}", file=sys.stderr)
        return 2

    run_id = args.run_id or _default_run_id(config.profile)
    if not RUN_ID_PATTERN.fullmatch(run_id):
        print(
            "run-id must contain only letters, numbers, '.', '_' and '-'",
            file=sys.stderr,
        )
        return 2
    paths = _pipeline_paths(config, run_id)
    if paths.run_dir.exists() and not args.resume and not args.dry_run:
        print(
            f"staging directory already exists: {paths.run_dir}\n"
            "use a new --run-id or explicitly pass --resume",
            file=sys.stderr,
        )
        return 2

    bash = shutil.which("bash") or "bash"
    steps = _build_steps(
        config,
        paths,
        python=sys.executable,
        bash=bash,
    )

    print("CAM++ Runtime verification pipeline")
    print(f"  profile: {config.profile}")
    print(f"  config: {config.source}")
    print(f"  buckets: {' '.join(str(item) for item in config.buckets)}")
    print(f"  run-id: {run_id}")
    print(f"  staging: {paths.run_dir}")
    print(f"  protected: {CANONICAL_BUNDLE_DIR}")
    print("  publish: disabled")
    print()

    if args.dry_run:
        for step in steps:
            _display_step(step, dry_run=True)
        print()
        print("dry-run complete: no files were created or modified")
        return 0

    paths.run_dir.mkdir(parents=True, exist_ok=args.resume)
    resolved = _resolved_config_document(config, paths, run_id)
    (paths.run_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )

    canonical_before = _bundle_snapshot(CANONICAL_BUNDLE_DIR)
    base_environment = os.environ.copy()
    python_source = str(ROOT / "src" / "python")
    previous_python_path = base_environment.get("PYTHONPATH")
    base_environment["PYTHONPATH"] = (
        python_source
        if not previous_python_path
        else os.pathsep.join((python_source, previous_python_path))
    )

    step_results: list[dict[str, object]] = []
    pipeline_started = time.perf_counter()
    failed_step: int | None = None
    for step in steps:
        _display_step(step, dry_run=False)
        environment = base_environment.copy()
        environment.update(step.environment)
        started = time.perf_counter()
        completed = subprocess.run(
            step.command,
            cwd=ROOT,
            env=environment,
            check=False,
        )
        elapsed = time.perf_counter() - started
        result = {
            "step": step.number,
            "name": step.name,
            "return_code": completed.returncode,
            "elapsed_seconds": elapsed,
            "passed": completed.returncode == 0,
        }
        step_results.append(result)
        print(
            f"[{'PASS' if completed.returncode == 0 else 'FAIL'} "
            f"{step.number}/5] {elapsed:.2f}s"
        )
        print()
        if completed.returncode != 0:
            failed_step = step.number
            break

    canonical_after = _bundle_snapshot(CANONICAL_BUNDLE_DIR)
    canonical_unchanged = canonical_before == canonical_after
    verification_report = _write_verification_report(config, paths)
    passed = failed_step is None and canonical_unchanged
    summary = {
        "profile": config.profile,
        "run_id": run_id,
        "passed": passed,
        "failed_step": failed_step,
        "elapsed_seconds": time.perf_counter() - pipeline_started,
        "steps": step_results,
        "staging_directory": str(paths.run_dir),
        "canonical_bundle_directory": str(CANONICAL_BUNDLE_DIR),
        "canonical_bundle_unchanged": canonical_unchanged,
        "verification_report": (
            str(verification_report) if verification_report is not None else None
        ),
    }
    summary_path = paths.run_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Pipeline result")
    print(f"  status: {'PASS' if passed else 'FAIL'}")
    print(f"  failed step: {failed_step if failed_step is not None else 'none'}")
    print(f"  canonical bundle unchanged: {canonical_unchanged}")
    print(f"  staging outputs: {paths.run_dir}")
    print(
        "  verification report: "
        f"{verification_report if verification_report is not None else 'not generated'}"
    )
    print(f"  summary: {summary_path}")
    if not canonical_unchanged:
        print(
            "safety check failed: models/compiled/reference changed",
            file=sys.stderr,
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
