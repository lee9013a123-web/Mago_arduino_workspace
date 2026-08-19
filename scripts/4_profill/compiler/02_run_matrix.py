#!/usr/bin/env python3
"""Build and Quick-benchmark staged compiler variants for the final E7 suite."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EXPECTED_BUCKET = 98


class MatrixError(RuntimeError):
    """The matrix configuration, build, validation, or benchmark is invalid."""


def _load_benchmark_module() -> Any:
    path = ROOT / "scripts" / "3_runtime" / "09_benchmark_runtime.py"
    spec = importlib.util.spec_from_file_location("runtime_benchmark_09_matrix", path)
    if spec is None or spec.loader is None:
        raise MatrixError(f"cannot load benchmark module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()


def _resolve_path(value: str, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _merge_flags(*values: str) -> str:
    tokens: list[str] = []
    for value in values:
        if value:
            tokens.extend(shlex.split(value))
    for token in tokens:
        if not token.startswith("-"):
            raise MatrixError(f"compiler option is not a flag: {token!r}")
    return " ".join(tokens)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MatrixError(f"{name} must be an object")
    return value


def _load_matrix_config(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise MatrixError("unsupported compiler matrix schema")
    if document.get("expected_suite") != "final":
        raise MatrixError("compiler matrix must target the final optimization suite")
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise MatrixError("compiler matrix has no stages")
    seen_stages: set[str] = set()
    for stage_index, raw_stage in enumerate(stages):
        stage = _require_mapping(raw_stage, f"stages[{stage_index}]")
        name = stage.get("name")
        objective = stage.get("objective", "latency")
        candidates = stage.get("candidates")
        if not isinstance(name, str) or not name or name in seen_stages:
            raise MatrixError(f"invalid or duplicate stage name: {name!r}")
        if objective not in ("latency", "size"):
            raise MatrixError(f"invalid objective for stage {name}: {objective}")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise MatrixError(f"stage {name} requires at least two candidates")
        seen_stages.add(name)
        seen_candidates: set[str] = set()
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate = _require_mapping(
                raw_candidate, f"stages[{stage_index}].candidates[{candidate_index}]"
            )
            candidate_name = candidate.get("name")
            if (
                not isinstance(candidate_name, str)
                or not candidate_name
                or candidate_name in seen_candidates
            ):
                raise MatrixError(
                    f"invalid or duplicate candidate in stage {name}: "
                    f"{candidate_name!r}"
                )
            seen_candidates.add(candidate_name)
            for field in ("cppflags", "cflags", "ldflags"):
                raw_flags = candidate.get(field, "")
                if not isinstance(raw_flags, str):
                    raise MatrixError(f"{name}.{candidate_name}.{field} must be text")
                _merge_flags(raw_flags)
            if not isinstance(candidate.get("strip", False), bool):
                raise MatrixError(f"{name}.{candidate_name}.strip must be boolean")
    selection = _require_mapping(document.get("selection"), "selection")
    for key in (
        "minimum_p50_improvement_pct",
        "maximum_p95_regression_pct",
        "maximum_cv_pct",
        "maximum_peak_rss_increase_bytes",
    ):
        if not isinstance(selection.get(key), (int, float)) or selection[key] < 0:
            raise MatrixError(f"selection.{key} must be non-negative")
    for field in ("base_cppflags", "base_cflags", "base_ldflags"):
        if not isinstance(document.get(field, ""), str):
            raise MatrixError(f"{field} must be text")
        _merge_flags(document.get(field, ""))
    for field in ("benchmark_config", "build_script"):
        value = document.get(field)
        if not isinstance(value, str) or not value:
            raise MatrixError(f"{field} is required")
        document[f"_{field}_path"] = _resolve_path(value, root=root)
    document["_source_path"] = path
    return document


def _candidate_metrics(result: dict[str, Any]) -> dict[str, float | int | None]:
    aggregate = result.get("aggregate", {})
    latency = aggregate.get("latency", {})
    return {
        "p50_ms": latency.get("p50_ms"),
        "p95_ms": latency.get("p95_ms"),
        "cv_pct": latency.get("cv_pct"),
        "peak_rss_bytes": aggregate.get("peak_rss_bytes"),
        "binary_size_bytes": result.get("binary", {}).get("size_bytes"),
    }


def _choose_stage_winner(
    results: Sequence[dict[str, Any]],
    selection: dict[str, Any],
    objective: str,
) -> dict[str, Any]:
    if not results:
        raise MatrixError("cannot choose a winner from an empty stage")
    control = results[0]
    control_metrics = _candidate_metrics(control)
    if control.get("status") != "passed":
        raise MatrixError("the stage control candidate did not pass")
    for key in ("p50_ms", "p95_ms", "cv_pct", "binary_size_bytes"):
        if control_metrics[key] is None:
            raise MatrixError(f"stage control is missing {key}")
    eligible: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    max_p95 = float(control_metrics["p95_ms"]) * (
        1.0 + float(selection["maximum_p95_regression_pct"]) / 100.0
    )
    control_rss = control_metrics["peak_rss_bytes"]
    max_rss = (
        int(control_rss) + int(selection["maximum_peak_rss_increase_bytes"])
        if control_rss is not None
        else None
    )
    for result in results:
        metrics = _candidate_metrics(result)
        reasons: list[str] = []
        if result.get("status") != "passed":
            reasons.append("run_failed")
        if result.get("all_output_hashes_match_reference") is not True:
            reasons.append("output_hash_mismatch")
        if result.get("all_retained_tensor_hashes_match_reference") is not True:
            reasons.append("retained_tensor_hash_mismatch")
        if metrics["cv_pct"] is None or float(metrics["cv_pct"]) > float(
            selection["maximum_cv_pct"]
        ):
            reasons.append("cv_gate")
        if metrics["p95_ms"] is None or float(metrics["p95_ms"]) > max_p95:
            reasons.append("p95_regression")
        if (
            max_rss is not None
            and metrics["peak_rss_bytes"] is not None
            and int(metrics["peak_rss_bytes"]) > max_rss
        ):
            reasons.append("rss_regression")
        passed = not reasons
        evaluations.append(
            {
                "variant": result.get("variant"),
                "eligible": passed,
                "reasons": reasons,
                **metrics,
            }
        )
        if passed:
            eligible.append(result)
    if not eligible:
        raise MatrixError("no candidate passed the stage gates")

    if objective == "size":
        winner = min(
            eligible,
            key=lambda item: (
                int(_candidate_metrics(item)["binary_size_bytes"]),
                float(_candidate_metrics(item)["p50_ms"]),
            ),
        )
        reason = "smallest binary among correctness/stability candidates"
    else:
        fastest = min(
            eligible,
            key=lambda item: float(_candidate_metrics(item)["p50_ms"]),
        )
        improvement_pct = (
            1.0
            - float(_candidate_metrics(fastest)["p50_ms"])
            / float(control_metrics["p50_ms"])
        ) * 100.0
        if (
            fastest is control
            or improvement_pct
            < float(selection["minimum_p50_improvement_pct"])
        ):
            winner = control
            reason = "no candidate exceeded the minimum p50 improvement"
        else:
            winner = fastest
            reason = f"best eligible p50; improvement={improvement_pct:.3f}%"
    return {
        "control": control["variant"],
        "winner": winner["variant"],
        "objective": objective,
        "reason": reason,
        "evaluations": evaluations,
    }


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as sink:
        completed = subprocess.run(
            list(command), cwd=cwd, env=environment,
            stdout=sink, stderr=subprocess.STDOUT, text=True, check=False,
        )
    if completed.returncode != 0:
        raise MatrixError(
            f"command failed ({completed.returncode}); see {_display_path(log_path)}"
        )


def _command_output(command: Sequence[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, *command[1:]], cwd=ROOT, capture_output=True,
        text=True, check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    return output if output else None


def _environment_probe(cc: str) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cc": cc,
        "cc_version": _command_output([cc, "--version"]),
        "ld_version": _command_output(["ld", "--version"]),
        "uname": _command_output(["uname", "-a"]),
        "lscpu": _command_output(["lscpu"]),
        "native_target": _command_output(
            [cc, "-Q", "--help=target", "-mcpu=native"]
        ),
    }


def _selected_features(config: Any) -> list[Any]:
    selected = BENCHMARK.selected_dataset_inputs(config.paths.dataset_manifest)
    requested = set(config.input_ids)
    selected = {key: value for key, value in selected.items() if key in requested}
    if set(selected) != requested:
        raise MatrixError(
            f"dataset manifest is missing inputs: {sorted(requested - set(selected))}"
        )
    all_features, _ = BENCHMARK.load_feature_manifest(config.paths.feature_manifest)
    filtered = [
        item for item in all_features
        if item.bucket_frames == EXPECTED_BUCKET and item.input_id in requested
    ]
    features = BENCHMARK.validate_features(config, selected, filtered)
    if config.verify_source_wavs:
        BENCHMARK.verify_source_wavs(config.paths.data_root, selected)
    return features


def _validate_matrix_artifacts(config: Any) -> dict[str, Any]:
    plan = config.paths.arena_bundle / "execution_plans" / "plan_98.bin"
    weights = config.paths.arena_bundle / "weights.bin"
    for path, name in (
        (plan, "E7 execution plan"),
        (weights, "E7 packed weights"),
        (config.paths.verification_result, "E7 validation evidence"),
    ):
        if not path.is_file():
            raise MatrixError(f"{name} not found: {path}")
    evidence = json.loads(
        config.paths.verification_result.read_text(encoding="utf-8")
    )
    if evidence.get("bucket_frames") != EXPECTED_BUCKET:
        raise MatrixError("E7 validation evidence is not for bucket 98")
    if evidence.get("all_bitwise_identical") is not True:
        raise MatrixError("existing E7 validation is not bitwise-identical")
    return {
        "path": _display_path(config.paths.verification_result),
        "sha256": BENCHMARK.sha256_file(config.paths.verification_result),
        "all_bitwise_identical": True,
    }


def _retained_tensor_hashes(
    *,
    dump_binary: Path,
    bundle: Path,
    features: Sequence[Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    plan = bundle / "execution_plans" / f"plan_{EXPECTED_BUCKET}.bin"
    weights = bundle / "weights.bin"
    records: list[dict[str, Any]] = []
    dump_dir = run_dir / "retained_tensors"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        prefix = dump_dir / feature.input_id
        payload_path = prefix.with_suffix(".bin")
        index_path = prefix.with_suffix(".json")
        _run_logged(
            [
                str(dump_binary), str(plan), str(weights),
                str(feature.path), str(prefix),
            ],
            cwd=ROOT,
            environment=os.environ.copy(),
            log_path=dump_dir / f"{feature.input_id}.log",
        )
        if not payload_path.is_file() or not index_path.is_file():
            raise MatrixError(f"retained Tensor dump is incomplete: {prefix}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        tensors = index.get("tensors")
        if not isinstance(tensors, list) or not tensors:
            raise MatrixError(f"retained Tensor index is empty: {index_path}")
        records.append(
            {
                "input_id": feature.input_id,
                "tensor_count": len(tensors),
                "payload_bytes": payload_path.stat().st_size,
                "payload_sha256": BENCHMARK.sha256_file(payload_path),
            }
        )
        payload_path.unlink()
        index_path.unlink()
    return records


def _measure_variant(
    *,
    matrix: dict[str, Any],
    benchmark_config: Any,
    features: Sequence[Any],
    variant: str,
    parent_variant: str | None,
    stage_name: str,
    candidate_name: str,
    cppflags: str,
    cflags: str,
    ldflags: str,
    strip_final: bool,
    build_dir: Path,
    run_dir: Path,
    cc: str,
    allow_environment_mismatch: bool,
) -> dict[str, Any]:
    build_started = time.perf_counter()
    build_environment = os.environ.copy()
    build_environment.update(
        {
            "BUILD_DIR": str(build_dir),
            "BUILD_VARIANT": variant,
            "BUILD_SCOPE": "compiler_matrix",
            "CC": cc,
            "CPPFLAGS": cppflags,
            "CFLAGS": cflags,
            "LDFLAGS": ldflags,
            "STRIP_FINAL": "1" if strip_final else "0",
        }
    )
    _run_logged(
        ["bash", str(matrix["_build_script_path"])], cwd=ROOT,
        environment=build_environment, log_path=run_dir / "build.log",
    )
    build_seconds = time.perf_counter() - build_started
    test_binary = BENCHMARK.executable_path(build_dir / "test_final_candidate_suite")
    runtime_binary = BENCHMARK.executable_path(
        build_dir / "campp_runtime_benchmark_final"
    )
    dump_binary = BENCHMARK.executable_path(
        build_dir / "campp_reference_dump_final"
    )
    for path, name in (
        (test_binary, "suite test"),
        (runtime_binary, "runtime"),
        (dump_binary, "retained Tensor dump"),
    ):
        if not path.is_file():
            raise MatrixError(f"{name} binary not found: {path}")
    _run_logged(
        [str(test_binary)], cwd=ROOT, environment=os.environ.copy(),
        log_path=run_dir / "suite_test.log",
    )
    capabilities, _ = BENCHMARK.run_json_command(
        [str(runtime_binary), "--capabilities"], environment=os.environ.copy()
    )
    if capabilities.get("optimization_suite") != matrix["expected_suite"]:
        raise MatrixError(
            f"{variant} reports optimization_suite="
            f"{capabilities.get('optimization_suite')!r}"
        )
    retained_tensors = _retained_tensor_hashes(
        dump_binary=dump_binary,
        bundle=benchmark_config.paths.arena_bundle,
        features=features,
        run_dir=run_dir,
    )

    variant_config = replace(
        benchmark_config,
        paths=replace(benchmark_config.paths, c_benchmark=runtime_binary),
    )
    BENCHMARK.apply_and_validate_environment(
        variant_config.environment,
        allow_mismatch=allow_environment_mismatch,
    )
    runtime_environment = os.environ.copy()
    runtime_environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "ORT_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    benchmark_started = time.perf_counter()
    per_input: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        print(f"    [{index}/{len(features)}] {feature.input_id}", flush=True)
        BENCHMARK.wait_for_thermal_start(variant_config.environment)
        output_dir = run_dir / "raw" / feature.input_id
        result = BENCHMARK.run_c_benchmark(
            variant_config, feature, output_dir, runtime_environment
        )
        raw_results.append(result)
        per_input.append(
            {
                "input_id": feature.input_id,
                "feature_sha256": feature.sha256,
                "embedding_sha256": result["embedding"]["sha256"],
                "result": _display_path(output_dir / "result.json"),
            }
        )
    aggregate = BENCHMARK.aggregate_bucket(
        "campp-c-runtime", EXPECTED_BUCKET, raw_results,
        variant_config.cv_threshold_pct,
    )
    benchmark_seconds = time.perf_counter() - benchmark_started
    result = {
        "schema_version": 1,
        "variant": variant,
        "parent_variant": parent_variant,
        "stage": stage_name,
        "candidate": candidate_name,
        "status": "passed",
        "flags": {
            "cppflags": cppflags,
            "cflags": cflags,
            "ldflags": ldflags,
            "strip_final": strip_final,
        },
        "build": {
            "directory": _display_path(build_dir),
            "seconds": build_seconds,
            "metadata": _display_path(build_dir / "build_metadata.txt"),
        },
        "binary": {
            "path": _display_path(runtime_binary),
            "sha256": BENCHMARK.sha256_file(runtime_binary),
            "size_bytes": runtime_binary.stat().st_size,
        },
        "capabilities": capabilities,
        "retained_tensors": retained_tensors,
        "benchmark_seconds": benchmark_seconds,
        "per_input": per_input,
        "aggregate": aggregate,
    }
    (run_dir / "variant_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": result.get("variant"),
        "parent_variant": result.get("parent_variant"),
        "stage": result.get("stage"),
        "candidate": result.get("candidate"),
        "status": result.get("status"),
        "flags": result.get("flags"),
        "build": result.get("build"),
        "binary": result.get("binary"),
        "benchmark_seconds": result.get("benchmark_seconds"),
        "per_input": result.get("per_input", []),
        "retained_tensors": result.get("retained_tensors", []),
        "aggregate": result.get("aggregate", {}),
        "all_output_hashes_match_reference": result.get(
            "all_output_hashes_match_reference"
        ),
        "all_retained_tensor_hashes_match_reference": result.get(
            "all_retained_tensor_hashes_match_reference"
        ),
        "error": result.get("error"),
    }


def _write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    fields = [
        "stage", "variant", "parent_variant", "candidate", "status",
        "p50_ms", "p95_ms", "cv_pct", "peak_rss_bytes",
        "binary_size_bytes", "hashes_match", "retained_hashes_match",
        "cflags", "ldflags",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = _candidate_metrics(result)
            writer.writerow(
                {
                    "stage": result.get("stage"),
                    "variant": result.get("variant"),
                    "parent_variant": result.get("parent_variant"),
                    "candidate": result.get("candidate"),
                    "status": result.get("status"),
                    "p50_ms": metrics["p50_ms"],
                    "p95_ms": metrics["p95_ms"],
                    "cv_pct": metrics["cv_pct"],
                    "peak_rss_bytes": metrics["peak_rss_bytes"],
                    "binary_size_bytes": metrics["binary_size_bytes"],
                    "hashes_match": result.get(
                        "all_output_hashes_match_reference"
                    ),
                    "retained_hashes_match": result.get(
                        "all_retained_tensor_hashes_match_reference"
                    ),
                    "cflags": result.get("flags", {}).get("cflags"),
                    "ldflags": result.get("flags", {}).get("ldflags"),
                }
            )


def _load_base_decision(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    winner = document.get("winner")
    if not isinstance(winner, dict) or not isinstance(winner.get("flags"), dict):
        raise MatrixError("base decision has no winner flags")
    return winner


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/runtime/compiler_qrb2210_strict.json",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--base-decision", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    args = parser.parse_args(argv)

    try:
        matrix = _load_matrix_config(args.config)
        for path, name in (
            (matrix["_benchmark_config_path"], "benchmark config"),
            (matrix["_build_script_path"], "build script"),
        ):
            if not path.is_file():
                raise MatrixError(f"{name} not found: {path}")
        benchmark_config = BENCHMARK.load_config(
            matrix["_benchmark_config_path"], repository_root=ROOT
        )
        if benchmark_config.buckets != {EXPECTED_BUCKET: 1.0}:
            raise MatrixError("compiler Quick matrix must use bucket 98 only")
        validation_evidence = _validate_matrix_artifacts(benchmark_config)
        features = _selected_features(benchmark_config)
        BENCHMARK.apply_and_validate_environment(
            benchmark_config.environment,
            allow_mismatch=args.allow_environment_mismatch,
        )
        base_winner = None
        if matrix.get("requires_base_decision"):
            if args.base_decision is None:
                raise MatrixError("this matrix requires --base-decision")
            base_winner = _load_base_decision(args.base_decision)
    except (MatrixError, BENCHMARK.BenchmarkError, OSError, ValueError) as exc:
        print(f"compiler matrix configuration failed: {exc}", file=sys.stderr)
        return 2

    if args.preflight_only:
        print("Compiler matrix preflight: PASS")
        print(f"  matrix: {matrix['name']}")
        print(f"  stages: {len(matrix['stages'])}")
        print(f"  inputs: {len(features)}")
        print(f"  suite: {matrix['expected_suite']}")
        print("  retained Tensor gate: enabled")
        return 0

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        print("invalid run id", file=sys.stderr)
        return 2
    build_root = ROOT / "build/profill/compiler_matrix" / run_id
    run_root = ROOT / "runs/profiling/e7_98/compiler_matrix" / run_id
    result_root = ROOT / "results/profiling/e7_98/compiler_matrix" / run_id
    if result_root.exists() and not (args.resume or args.force or args.dry_run):
        print(
            f"result already exists; use --resume or --force: {result_root}",
            file=sys.stderr,
        )
        return 2
    if args.force:
        for path in (build_root, run_root, result_root):
            if path.exists():
                shutil.rmtree(path)
    result_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "environment.json").write_text(
        json.dumps(_environment_probe(args.cc), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    base_cppflags = matrix.get("base_cppflags", "")
    base_cflags = matrix.get("base_cflags", "")
    base_ldflags = matrix.get("base_ldflags", "")
    inherited_strip = False
    parent_variant: str | None = None
    if base_winner is not None:
        base_cppflags = _merge_flags(
            base_cppflags, base_winner["flags"].get("cppflags", "")
        )
        base_cflags = _merge_flags(
            base_cflags, base_winner["flags"].get("cflags", "")
        )
        base_ldflags = _merge_flags(
            base_ldflags, base_winner["flags"].get("ldflags", "")
        )
        inherited_strip = bool(base_winner["flags"].get("strip_final", False))
        parent_variant = base_winner.get("variant")

    all_results: list[dict[str, Any]] = []
    stage_decisions: list[dict[str, Any]] = []
    reference_hashes: dict[str, str] | None = None
    reference_retained_hashes: dict[str, str] | None = None
    baseline_variant: dict[str, Any] | None = None
    inherited = {
        "cppflags": base_cppflags,
        "cflags": base_cflags,
        "ldflags": base_ldflags,
        "strip_final": inherited_strip,
    }

    print(f"Compiler matrix: {matrix['name']}")
    print(f"  run: {run_id}")
    print(f"  suite: {matrix['expected_suite']}")
    for stage_index, stage in enumerate(matrix["stages"]):
        stage_name = stage["name"]
        print(f"[{stage_index + 1}/{len(matrix['stages'])}] {stage_name}")
        stage_results: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(stage["candidates"]):
            variant = (
                f"{stage_index + 1:02d}_{stage_name}__{candidate['name']}"
            )
            candidate_parent = (
                parent_variant
                if stage.get("inherit_winner") or base_winner is not None
                else None
            )
            cppflags = _merge_flags(
                inherited["cppflags"] if stage.get("inherit_winner") else base_cppflags,
                candidate.get("cppflags", ""),
            )
            cflags = _merge_flags(
                inherited["cflags"] if stage.get("inherit_winner") else base_cflags,
                candidate.get("cflags", ""),
            )
            ldflags = _merge_flags(
                inherited["ldflags"] if stage.get("inherit_winner") else base_ldflags,
                candidate.get("ldflags", ""),
            )
            strip_final = bool(
                candidate.get(
                    "strip",
                    inherited["strip_final"] if stage.get("inherit_winner") else False,
                )
            )
            print(f"  - {variant}")
            print(f"    CFLAGS={cflags}")
            print(f"    LDFLAGS={ldflags or '(none)'}")
            if args.dry_run:
                continue
            variant_run_dir = run_root / variant
            cached_path = variant_run_dir / "variant_result.json"
            try:
                if args.resume and cached_path.is_file():
                    result = json.loads(cached_path.read_text(encoding="utf-8"))
                else:
                    result = _measure_variant(
                        matrix=matrix,
                        benchmark_config=benchmark_config,
                        features=features,
                        variant=variant,
                        parent_variant=candidate_parent,
                        stage_name=stage_name,
                        candidate_name=candidate["name"],
                        cppflags=cppflags,
                        cflags=cflags,
                        ldflags=ldflags,
                        strip_final=strip_final,
                        build_dir=build_root / variant,
                        run_dir=variant_run_dir,
                        cc=args.cc,
                        allow_environment_mismatch=args.allow_environment_mismatch,
                    )
            except (MatrixError, BENCHMARK.BenchmarkError, OSError, ValueError) as exc:
                result = {
                    "schema_version": 1,
                    "variant": variant,
                    "parent_variant": candidate_parent,
                    "stage": stage_name,
                    "candidate": candidate["name"],
                    "status": "failed",
                    "flags": {
                        "cppflags": cppflags,
                        "cflags": cflags,
                        "ldflags": ldflags,
                        "strip_final": strip_final,
                    },
                    "build": {"directory": _display_path(build_root / variant)},
                    "error": str(exc),
                }
                variant_run_dir.mkdir(parents=True, exist_ok=True)
                cached_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"    FAILED: {exc}", file=sys.stderr)
            hashes = {
                item["input_id"]: item["embedding_sha256"]
                for item in result.get("per_input", [])
            }
            retained_hashes = {
                item["input_id"]: item["payload_sha256"]
                for item in result.get("retained_tensors", [])
            }
            if reference_hashes is None and result.get("status") == "passed":
                reference_hashes = hashes
                reference_retained_hashes = retained_hashes
            result["all_output_hashes_match_reference"] = (
                result.get("status") == "passed"
                and reference_hashes is not None
                and hashes == reference_hashes
            )
            result["all_retained_tensor_hashes_match_reference"] = (
                result.get("status") == "passed"
                and reference_retained_hashes is not None
                and retained_hashes == reference_retained_hashes
            )
            cached_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if baseline_variant is None and result.get("status") == "passed":
                baseline_variant = result
            stage_results.append(result)
            all_results.append(result)
        if args.dry_run:
            continue
        try:
            decision = _choose_stage_winner(
                stage_results, matrix["selection"], stage.get("objective", "latency")
            )
        except MatrixError as exc:
            print(f"stage {stage_name} failed: {exc}", file=sys.stderr)
            return 3
        winner = next(
            item for item in stage_results if item["variant"] == decision["winner"]
        )
        decision["stage"] = stage_name
        stage_decisions.append(decision)
        inherited = dict(winner["flags"])
        parent_variant = winner["variant"]
        print(f"    winner: {winner['variant']} ({decision['reason']})")
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "matrix": matrix["name"],
            "expected_suite": matrix["expected_suite"],
            "configuration": {
                "source": _display_path(matrix["_source_path"]),
                "benchmark_config": _display_path(matrix["_benchmark_config_path"]),
                "cc": args.cc,
            },
            "validation_evidence": validation_evidence,
            "reference_output_hashes": reference_hashes,
            "reference_retained_tensor_hashes": reference_retained_hashes,
            "stage_decisions": stage_decisions,
            "variants": [_compact_result(item) for item in all_results],
        }
        (result_root / "matrix_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_csv(result_root / "matrix_summary.csv", all_results)

    if args.dry_run:
        print("Dry run complete; no build or benchmark was executed")
        return 0
    if baseline_variant is None or parent_variant is None:
        print("compiler matrix produced no winner", file=sys.stderr)
        return 3
    final_winner = next(
        item for item in all_results if item["variant"] == parent_variant
    )
    decision_document = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "matrix": matrix["name"],
        "ready": True,
        "baseline": _compact_result(baseline_variant),
        "winner": _compact_result(final_winner),
        "stage_decisions": stage_decisions,
        "next_gate": "baseline/winner Quick operator profile, then official E2E confirmation",
    }
    (result_root / "decision.json").write_text(
        json.dumps(decision_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Compiler matrix complete")
    print(f"  winner: {final_winner['variant']}")
    print(f"  decision: {_display_path(result_root / 'decision.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
