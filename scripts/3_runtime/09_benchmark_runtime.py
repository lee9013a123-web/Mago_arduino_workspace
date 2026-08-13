#!/usr/bin/env python3
"""Benchmark canonical INT8 ONNX and the compiled C Runtime on QRB2210.

This script does not repeat the numerical validation performed by scripts
06-08.  It verifies their evidence hashes, supplies the same frozen FBank
float32 payload to both backends, and records cold/warm latency, RTF and peak
process RSS.  WAV decoding and FBank extraction are deliberately outside the
measured interval.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
BUCKET_TAGS = {98: "1s", 298: "3s", 498: "5s", 998: "10s"}


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkPaths:
    workspace_root: Path
    canonical_model: Path
    arena_bundle: Path
    feature_manifest: Path
    dataset_manifest: Path
    data_root: Path | None
    c_benchmark: Path
    ort_benchmark: Path
    graph_dir: Path
    verification_result: Path
    tensor_arena_result: Path


@dataclass(frozen=True)
class EnvironmentPolicy:
    affinity: tuple[int, ...]
    required_governor: str | None
    required_device_model_substring: str | None
    maximum_start_temperature_c: float | None


@dataclass(frozen=True)
class BenchmarkConfig:
    source: Path
    profile: str
    buckets: dict[int, float]
    threads: int
    warmup: int
    repeat: int
    cold_runs: int
    cv_threshold_pct: float
    verify_source_wavs: bool
    require_all_latency_inputs: bool
    paths: BenchmarkPaths
    environment: EnvironmentPolicy


@dataclass(frozen=True)
class FeatureInput:
    input_id: str
    bucket_frames: int
    audio_seconds: float
    path: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile from an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Sequence[float], *, suffix: str = "ms") -> dict[str, Any]:
    if not values:
        raise ValueError("at least one measurement is required")
    samples = [float(value) for value in values]
    mean = statistics.fmean(samples)
    stddev = statistics.pstdev(samples)
    cv = stddev / mean * 100.0 if mean > 0.0 else math.inf
    return {
        "count": len(samples),
        f"mean_{suffix}": mean,
        f"stddev_{suffix}": stddev,
        "cv_pct": cv,
        f"min_{suffix}": min(samples),
        f"p50_{suffix}": percentile(samples, 50.0),
        f"p95_{suffix}": percentile(samples, 95.0),
        f"p99_{suffix}": percentile(samples, 99.0),
        f"max_{suffix}": max(samples),
    }


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{name} must be a JSON object")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkError(f"{name} must be an integer")
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        relation = "non-negative" if allow_zero else "positive"
        raise BenchmarkError(f"{name} must be {relation}")
    return value


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{name} must be numeric")
    converted = float(value)
    invalid = converted < 0.0 if allow_zero else converted <= 0.0
    if invalid:
        relation = "non-negative" if allow_zero else "positive"
        raise BenchmarkError(f"{name} must be {relation}")
    return converted


def _path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{name} must be a non-empty path")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def load_config(path: Path, repository_root: Path = ROOT) -> BenchmarkConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read config {path}: {exc}") from exc
    document = _object(raw, "config")
    allowed = {
        "schema_version",
        "profile",
        "buckets",
        "threads",
        "warmup",
        "repeat",
        "cold_runs",
        "cv_threshold_pct",
        "verify_source_wavs",
        "require_all_latency_inputs",
        "paths",
        "environment",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise BenchmarkError(f"unknown config keys: {', '.join(unknown)}")
    if document.get("schema_version") != 1:
        raise BenchmarkError("schema_version must be 1")
    profile = document.get("profile")
    if not isinstance(profile, str) or not profile:
        raise BenchmarkError("profile must be a non-empty string")

    raw_buckets = _object(document.get("buckets"), "buckets")
    buckets: dict[int, float] = {}
    for text, seconds in raw_buckets.items():
        try:
            frames = int(text)
        except (TypeError, ValueError) as exc:
            raise BenchmarkError(f"invalid bucket key: {text}") from exc
        if frames not in BUCKET_TAGS:
            raise BenchmarkError(f"unsupported bucket: {frames}")
        buckets[frames] = _positive_float(seconds, f"buckets.{text}")
    if not buckets:
        raise BenchmarkError("at least one bucket is required")

    raw_paths = _object(document.get("paths"), "paths")
    path_keys = {
        "workspace_root",
        "canonical_model",
        "arena_bundle",
        "feature_manifest",
        "dataset_manifest",
        "data_root",
        "c_benchmark",
        "ort_benchmark",
        "graph_dir",
        "verification_result",
        "tensor_arena_result",
    }
    unknown_paths = sorted(set(raw_paths) - path_keys)
    if unknown_paths:
        raise BenchmarkError(f"unknown paths keys: {', '.join(unknown_paths)}")
    required_paths = path_keys - {"data_root"}
    missing_paths = sorted(required_paths - set(raw_paths))
    if missing_paths:
        raise BenchmarkError(f"missing paths keys: {', '.join(missing_paths)}")
    data_root_value = raw_paths.get("data_root")
    paths = BenchmarkPaths(
        workspace_root=_path(
            repository_root, raw_paths["workspace_root"], "paths.workspace_root"
        ),
        canonical_model=_path(
            repository_root, raw_paths["canonical_model"], "paths.canonical_model"
        ),
        arena_bundle=_path(
            repository_root, raw_paths["arena_bundle"], "paths.arena_bundle"
        ),
        feature_manifest=_path(
            repository_root, raw_paths["feature_manifest"], "paths.feature_manifest"
        ),
        dataset_manifest=_path(
            repository_root, raw_paths["dataset_manifest"], "paths.dataset_manifest"
        ),
        data_root=(
            _path(repository_root, data_root_value, "paths.data_root")
            if data_root_value is not None
            else None
        ),
        c_benchmark=_path(
            repository_root, raw_paths["c_benchmark"], "paths.c_benchmark"
        ),
        ort_benchmark=_path(
            repository_root, raw_paths["ort_benchmark"], "paths.ort_benchmark"
        ),
        graph_dir=_path(repository_root, raw_paths["graph_dir"], "paths.graph_dir"),
        verification_result=_path(
            repository_root,
            raw_paths["verification_result"],
            "paths.verification_result",
        ),
        tensor_arena_result=_path(
            repository_root,
            raw_paths["tensor_arena_result"],
            "paths.tensor_arena_result",
        ),
    )

    raw_environment = _object(document.get("environment", {}), "environment")
    environment_keys = {
        "affinity",
        "required_governor",
        "required_device_model_substring",
        "maximum_start_temperature_c",
    }
    unknown_environment = sorted(set(raw_environment) - environment_keys)
    if unknown_environment:
        raise BenchmarkError(
            f"unknown environment keys: {', '.join(unknown_environment)}"
        )
    affinity_raw = raw_environment.get("affinity", [])
    if not isinstance(affinity_raw, list) or any(
        isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
        for cpu in affinity_raw
    ):
        raise BenchmarkError("environment.affinity must be non-negative CPU IDs")
    affinity = tuple(affinity_raw)
    if len(affinity) != len(set(affinity)):
        raise BenchmarkError("environment.affinity contains duplicates")
    required_governor = raw_environment.get("required_governor")
    if required_governor is not None and (
        not isinstance(required_governor, str) or not required_governor
    ):
        raise BenchmarkError("environment.required_governor must be a string or null")
    required_device = raw_environment.get("required_device_model_substring")
    if required_device is not None and (
        not isinstance(required_device, str) or not required_device
    ):
        raise BenchmarkError(
            "environment.required_device_model_substring must be a string or null"
        )
    maximum_temperature = raw_environment.get("maximum_start_temperature_c")
    environment = EnvironmentPolicy(
        affinity=affinity,
        required_governor=required_governor,
        required_device_model_substring=required_device,
        maximum_start_temperature_c=(
            _positive_float(
                maximum_temperature, "environment.maximum_start_temperature_c"
            )
            if maximum_temperature is not None
            else None
        ),
    )

    verify_source_wavs = document.get("verify_source_wavs", True)
    require_all = document.get("require_all_latency_inputs", True)
    if not isinstance(verify_source_wavs, bool) or not isinstance(require_all, bool):
        raise BenchmarkError(
            "verify_source_wavs and require_all_latency_inputs must be booleans"
        )
    return BenchmarkConfig(
        source=path.resolve(),
        profile=profile,
        buckets=dict(sorted(buckets.items())),
        threads=_positive_int(document.get("threads"), "threads"),
        warmup=_positive_int(document.get("warmup"), "warmup", allow_zero=True),
        repeat=_positive_int(document.get("repeat"), "repeat"),
        cold_runs=_positive_int(
            document.get("cold_runs"), "cold_runs", allow_zero=True
        ),
        cv_threshold_pct=_positive_float(
            document.get("cv_threshold_pct"), "cv_threshold_pct"
        ),
        verify_source_wavs=verify_source_wavs,
        require_all_latency_inputs=require_all,
        paths=paths,
        environment=environment,
    )


def selected_dataset_inputs(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise BenchmarkError(f"dataset manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("latency_selected") != "1":
            continue
        input_id = row.get("input_id", "")
        if not input_id or input_id in selected:
            raise BenchmarkError(f"invalid or duplicate dataset input_id: {input_id!r}")
        selected[input_id] = row
    if not selected:
        raise BenchmarkError("dataset manifest has no latency_selected=1 inputs")
    return selected


def load_feature_manifest(
    path: Path,
    repository_root: Path = ROOT,
) -> tuple[list[FeatureInput], dict[str, Any]]:
    if not path.is_file():
        raise BenchmarkError(f"feature manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid feature manifest: {exc}") from exc
    document = _object(raw, "feature manifest")
    if document.get("schema_version") != 1:
        raise BenchmarkError("feature manifest schema_version must be 1")
    unknown_top = sorted(set(document) - {"schema_version", "preprocessing", "features"})
    if unknown_top:
        raise BenchmarkError(
            f"unknown feature manifest keys: {', '.join(unknown_top)}"
        )
    preprocessing = _object(document.get("preprocessing"), "preprocessing")
    if not preprocessing:
        raise BenchmarkError("preprocessing must record the frozen FBank policy")
    raw_features = document.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise BenchmarkError("feature manifest features must be a non-empty array")
    features: list[FeatureInput] = []
    seen: set[tuple[str, int]] = set()
    for index, raw_feature in enumerate(raw_features):
        item = _object(raw_feature, f"features[{index}]")
        unknown_item = sorted(
            set(item)
            - {"input_id", "bucket_frames", "audio_seconds", "path", "sha256"}
        )
        if unknown_item:
            raise BenchmarkError(
                f"unknown features[{index}] keys: {', '.join(unknown_item)}"
            )
        input_id = item.get("input_id")
        if not isinstance(input_id, str) or not input_id:
            raise BenchmarkError(f"features[{index}].input_id is invalid")
        frames = _positive_int(
            item.get("bucket_frames"), f"features[{index}].bucket_frames"
        )
        seconds = _positive_float(
            item.get("audio_seconds"), f"features[{index}].audio_seconds"
        )
        feature_path = _path(
            repository_root, item.get("path"), f"features[{index}].path"
        )
        checksum = item.get("sha256")
        if not isinstance(checksum, str) or not re.fullmatch(
            r"[0-9a-f]{64}", checksum
        ):
            raise BenchmarkError(f"features[{index}].sha256 is invalid")
        key = (input_id, frames)
        if key in seen:
            raise BenchmarkError(f"duplicate feature entry: {input_id}/{frames}")
        seen.add(key)
        features.append(
            FeatureInput(
                input_id=input_id,
                bucket_frames=frames,
                audio_seconds=seconds,
                path=feature_path,
                sha256=checksum,
            )
        )
    return features, document


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise BenchmarkError(f"{name} not found: {path}")


def validate_features(
    config: BenchmarkConfig,
    selected: dict[str, dict[str, str]],
    features: Iterable[FeatureInput],
) -> list[FeatureInput]:
    chosen = [
        feature
        for feature in features
        if feature.bucket_frames in config.buckets and feature.input_id in selected
    ]
    expected = (
        {(input_id, frames) for input_id in selected for frames in config.buckets}
        if config.require_all_latency_inputs
        else set()
    )
    actual = {(item.input_id, item.bucket_frames) for item in chosen}
    missing = sorted(expected - actual)
    if missing:
        preview = ", ".join(f"{input_id}/{frames}" for input_id, frames in missing[:8])
        raise BenchmarkError(
            f"feature manifest is missing {len(missing)} required entries: {preview}"
        )
    for item in chosen:
        _require_file(item.path, "feature")
        expected_size = item.bucket_frames * 80 * 4
        actual_size = item.path.stat().st_size
        if actual_size != expected_size:
            raise BenchmarkError(
                f"feature size mismatch for {item.input_id}/{item.bucket_frames}: "
                f"{actual_size} != {expected_size}"
            )
        actual_hash = sha256_file(item.path)
        if actual_hash != item.sha256:
            raise BenchmarkError(
                f"feature hash mismatch for {item.input_id}/{item.bucket_frames}"
            )
        expected_seconds = config.buckets[item.bucket_frames]
        if not math.isclose(item.audio_seconds, expected_seconds, abs_tol=1e-9):
            raise BenchmarkError(
                f"audio_seconds mismatch for {item.input_id}/{item.bucket_frames}: "
                f"{item.audio_seconds} != {expected_seconds}"
            )
    if not chosen:
        raise BenchmarkError("no feature inputs matched the selected dataset and buckets")
    return sorted(chosen, key=lambda item: (item.bucket_frames, item.input_id))


def verify_source_wavs(
    data_root: Path | None, selected: dict[str, dict[str, str]]
) -> dict[str, Any]:
    if data_root is None:
        raise BenchmarkError("paths.data_root is required when verify_source_wavs=true")
    verified: list[dict[str, str]] = []
    for input_id, row in sorted(selected.items()):
        source = data_root / row["relative_path"]
        _require_file(source, f"source WAV for {input_id}")
        checksum = sha256_file(source)
        if checksum != row["sha256"]:
            raise BenchmarkError(f"source WAV hash mismatch: {input_id}")
        verified.append(
            {
                "input_id": input_id,
                "relative_path": row["relative_path"],
                "sha256": checksum,
            }
        )
    return {"data_root": str(data_root), "verified": verified}


def load_bundle_provenance(
    bundle_dir: Path, canonical_model: Path, buckets: Iterable[int]
) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    weights_path = bundle_dir / "weights.bin"
    _require_file(manifest_path, "bundle manifest")
    _require_file(weights_path, "bundle weights")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_sha = sha256_file(canonical_model)
    recorded_canonical = _object(
        manifest.get("canonical_model"), "bundle canonical_model"
    ).get("sha256")
    if recorded_canonical != canonical_sha:
        raise BenchmarkError(
            "bundle canonical model hash does not match the ORT baseline model"
        )
    recorded_weights = _object(manifest.get("weights"), "bundle weights").get("sha256")
    weights_sha = sha256_file(weights_path)
    if recorded_weights != weights_sha:
        raise BenchmarkError("bundle weights hash mismatch")
    plan_entries = manifest.get("plans")
    if not isinstance(plan_entries, list):
        raise BenchmarkError("bundle manifest plans must be an array")
    by_bucket = {
        int(item["bucket_frames"]): item
        for item in plan_entries
        if isinstance(item, dict) and "bucket_frames" in item
    }
    plans: dict[str, Any] = {}
    for frames in buckets:
        if frames not in by_bucket:
            raise BenchmarkError(f"bundle has no plan for {frames} frames")
        entry = by_bucket[frames]
        plan_path = bundle_dir / "execution_plans" / f"plan_{frames}.bin"
        _require_file(plan_path, f"plan {frames}")
        checksum = sha256_file(plan_path)
        if entry.get("sha256") != checksum:
            raise BenchmarkError(f"plan hash mismatch for {frames} frames")
        plans[str(frames)] = {
            "path": str(plan_path),
            "sha256": checksum,
            "size_bytes": plan_path.stat().st_size,
            "arena_size_bytes": entry.get("arena_size_bytes"),
        }
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "canonical_model": {
            "path": str(canonical_model),
            "sha256": canonical_sha,
            "size_bytes": canonical_model.stat().st_size,
        },
        "weights": {
            "path": str(weights_path),
            "sha256": weights_sha,
            "size_bytes": weights_path.stat().st_size,
        },
        "plans": plans,
    }


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return None


def thermal_snapshot() -> list[dict[str, Any]]:
    root = Path("/sys/class/thermal")
    if not root.is_dir():
        return []
    values: list[dict[str, Any]] = []
    for zone in sorted(root.glob("thermal_zone*")):
        raw = read_text(zone / "temp")
        if raw is None:
            continue
        try:
            temperature = float(raw)
        except ValueError:
            continue
        if abs(temperature) > 1000.0:
            temperature /= 1000.0
        values.append(
            {
                "zone": zone.name,
                "type": read_text(zone / "type"),
                "temperature_c": temperature,
            }
        )
    return values


def governor_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
    ):
        value = read_text(path)
        if value is not None:
            result[path.parts[-3]] = value
    return result


def git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "sha": git("rev-parse", "HEAD"),
        "status_porcelain": (git("status", "--short") or "").splitlines(),
    }


def environment_snapshot() -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "device_tree_model": read_text(Path("/proc/device-tree/model")),
        "governors": governor_snapshot(),
        "thermal": thermal_snapshot(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "ORT_NUM_THREADS",
            )
        },
    }


def apply_and_validate_environment(
    policy: EnvironmentPolicy, *, allow_mismatch: bool
) -> tuple[dict[str, Any], list[str]]:
    mismatches: list[str] = []
    if policy.affinity:
        try:
            os.sched_setaffinity(0, set(policy.affinity))  # type: ignore[attr-defined]
        except (AttributeError, OSError) as exc:
            mismatches.append(f"cannot set CPU affinity {list(policy.affinity)}: {exc}")
    snapshot = environment_snapshot()
    if policy.affinity and snapshot["cpu_affinity"] != list(policy.affinity):
        mismatches.append(
            f"CPU affinity is {snapshot['cpu_affinity']}, expected {list(policy.affinity)}"
        )
    if policy.required_governor is not None:
        governors = snapshot["governors"]
        if not governors:
            mismatches.append("CPU governor is unavailable")
        wrong = {
            cpu: value
            for cpu, value in governors.items()
            if value != policy.required_governor
        }
        if wrong:
            mismatches.append(
                f"CPU governor mismatch: {wrong}, expected {policy.required_governor}"
            )
    if policy.required_device_model_substring is not None:
        model = snapshot["device_tree_model"] or ""
        if policy.required_device_model_substring not in model:
            mismatches.append(
                f"device model {model!r} does not contain "
                f"{policy.required_device_model_substring!r}"
            )
    if policy.maximum_start_temperature_c is not None:
        temperatures = [
            float(item["temperature_c"]) for item in snapshot["thermal"]
        ]
        if not temperatures:
            mismatches.append("thermal temperature is unavailable")
        elif max(temperatures) > policy.maximum_start_temperature_c:
            mismatches.append(
                f"start temperature {max(temperatures):.1f} C exceeds "
                f"{policy.maximum_start_temperature_c:.1f} C"
            )
    if mismatches and not allow_mismatch:
        raise BenchmarkError("; ".join(mismatches))
    return snapshot, mismatches


def wait_for_thermal_start(
    policy: EnvironmentPolicy, *, timeout_seconds: float = 300.0
) -> None:
    maximum = policy.maximum_start_temperature_c
    if maximum is None:
        return
    deadline = time.monotonic() + timeout_seconds
    while True:
        values = thermal_snapshot()
        if not values:
            raise BenchmarkError("thermal temperature is unavailable")
        current = max(float(item["temperature_c"]) for item in values)
        if current <= maximum:
            return
        if time.monotonic() >= deadline:
            raise BenchmarkError(
                f"temperature stayed at {current:.1f} C; required <= {maximum:.1f} C"
            )
        print(
            f"  cooling: {current:.1f} C > {maximum:.1f} C",
            flush=True,
        )
        time.sleep(5.0)


def evidence(path: Path) -> dict[str, Any]:
    _require_file(path, "validation evidence")
    document = json.loads(path.read_text(encoding="utf-8"))
    selected = {
        name: document.get(name)
        for name in (
            "baseline_id",
            "phase4_ready",
            "all_buckets_numerically_passed",
            "all_bitwise_identical",
        )
        if name in document
    }
    return {"path": str(path), "sha256": sha256_file(path), "status": selected}


def run_json_command(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    cwd: Path = ROOT,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter_ns()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    if completed.returncode != 0:
        raise BenchmarkError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"command returned invalid JSON: {' '.join(command)}\n"
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        ) from exc
    return payload, wall_ms


def c_command(
    binary: Path,
    *,
    plan: Path,
    weights: Path,
    feature: FeatureInput,
    threads: int,
    warmup: int,
    repeat: int,
    embedding_output: Path | None = None,
) -> list[str]:
    command = [
        str(binary),
        "--plan",
        str(plan),
        "--weights",
        str(weights),
        "--input",
        str(feature.path),
        "--audio-seconds",
        str(feature.audio_seconds),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--threads",
        str(threads),
    ]
    if embedding_output is not None:
        command.extend(["--embedding-output", str(embedding_output)])
    return command


def run_c_benchmark(
    config: BenchmarkConfig,
    feature: FeatureInput,
    output_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    plan = (
        config.paths.arena_bundle
        / "execution_plans"
        / f"plan_{feature.bucket_frames}.bin"
    )
    weights = config.paths.arena_bundle / "weights.bin"
    output_dir.mkdir(parents=True, exist_ok=True)
    cold_samples: list[dict[str, Any]] = []
    for run in range(1, config.cold_runs + 1):
        payload, wall_ms = run_json_command(
            c_command(
                config.paths.c_benchmark,
                plan=plan,
                weights=weights,
                feature=feature,
                threads=config.threads,
                warmup=0,
                repeat=1,
            ),
            environment=environment,
        )
        payload["run"] = run
        payload["process_wall_ms"] = wall_ms
        cold_samples.append(payload)
        (output_dir / f"cold_{run:02d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    embedding_path = output_dir / "embedding.f32"
    warm, process_wall_ms = run_json_command(
        c_command(
            config.paths.c_benchmark,
            plan=plan,
            weights=weights,
            feature=feature,
            threads=config.threads,
            warmup=config.warmup,
            repeat=config.repeat,
            embedding_output=embedding_path,
        ),
        environment=environment,
    )
    requested = warm["configuration"]["requested_threads"]
    effective = warm["configuration"]["effective_threads"]
    if requested != config.threads or effective != config.threads:
        raise BenchmarkError(
            f"C Runtime thread mismatch: requested={requested}, effective={effective}, "
            f"experiment={config.threads}"
        )
    timings = [float(value) for value in warm["warm"]["timings_ms"]]
    latency = summarize(timings)
    rtf_values = [value / 1000.0 / feature.audio_seconds for value in timings]
    warm["warm"]["latency"] = latency
    warm["warm"]["rtf"] = summarize(rtf_values, suffix="rtf")
    warm["warm"]["stability_gate"] = {
        "threshold_cv_pct": config.cv_threshold_pct,
        "passed": float(latency["cv_pct"]) <= config.cv_threshold_pct,
    }
    warm["warm"]["process_wall_ms"] = process_wall_ms
    warm["input"].update(
        {
            "input_id": feature.input_id,
            "sha256": feature.sha256,
            "bucket_frames": feature.bucket_frames,
        }
    )
    warm["embedding"]["sha256"] = sha256_file(embedding_path)

    cold = None
    if cold_samples:
        cold = {
            "definition": "fresh process; OS file page cache is not dropped",
            "samples": cold_samples,
            "process_wall_summary": summarize(
                [float(item["process_wall_ms"]) for item in cold_samples]
            ),
            "model_load_summary": summarize(
                [
                    float(item["lifecycle"]["model_load_ms"])
                    for item in cold_samples
                ]
            ),
            "context_create_summary": summarize(
                [
                    float(item["lifecycle"]["context_create_ms"])
                    for item in cold_samples
                ]
            ),
            "first_inference_summary": summarize(
                [
                    float(item["lifecycle"]["first_inference_ms"])
                    for item in cold_samples
                ]
            ),
            "max_peak_rss_bytes": max(
                (
                    int(item["memory"]["after_measurement"]["peak_rss_bytes"])
                    for item in cold_samples
                    if item["memory"]["after_measurement"]["peak_rss_bytes"] is not None
                ),
                default=None,
            ),
        }
    result = {
        "schema_version": 1,
        "runtime": "campp-c-reference",
        "model": warm["model"],
        "configuration": warm["configuration"],
        "input": warm["input"],
        "lifecycle": warm["lifecycle"],
        "cold": cold,
        "warm": warm["warm"],
        "memory": warm["memory"],
        "embedding": warm["embedding"],
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_ort_benchmark(
    config: BenchmarkConfig,
    feature: FeatureInput,
    output_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    ir_path = config.paths.graph_dir / f"ir_{BUCKET_TAGS[feature.bucket_frames]}.json"
    command = [
        sys.executable,
        str(config.paths.ort_benchmark),
        "--model",
        str(config.paths.canonical_model),
        "--input-f32",
        str(feature.path),
        "--frames",
        str(feature.bucket_frames),
        "--audio-seconds",
        str(feature.audio_seconds),
        "--threads",
        str(config.threads),
        "--warmup",
        str(config.warmup),
        "--repeat",
        str(config.repeat),
        "--cold-runs",
        str(config.cold_runs),
        "--cv-threshold-pct",
        str(config.cv_threshold_pct),
        "--ir",
        str(ir_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise BenchmarkError(
            f"ORT benchmark failed ({completed.returncode}): {' '.join(command)}"
        )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    if int(result["configuration"]["threads"]) != config.threads:
        raise BenchmarkError("ORT thread setting does not match the experiment")
    return result


def warm_peak_rss(result: dict[str, Any]) -> int | None:
    if result["runtime"] == "onnxruntime-cpu":
        value = result.get("warm", {}).get("memory", {}).get("peak_rss_bytes")
    else:
        value = (
            result.get("memory", {})
            .get("after_measurement", {})
            .get("peak_rss_bytes")
        )
    return int(value) if value is not None else None


def aggregate_bucket(
    runtime: str,
    frames: int,
    results: Sequence[dict[str, Any]],
    cv_threshold_pct: float,
) -> dict[str, Any]:
    timings: list[float] = []
    first_inference: list[float] = []
    warm_peaks: list[int] = []
    cold_peaks: list[int] = []
    stable = True
    for result in results:
        timings.extend(float(value) for value in result["warm"]["timings_ms"])
        stable = stable and bool(result["warm"]["stability_gate"]["passed"])
        peak = warm_peak_rss(result)
        if peak is not None:
            warm_peaks.append(peak)
        cold = result.get("cold")
        if cold:
            first = cold.get("first_inference_summary", {}).get("mean_ms")
            if first is not None:
                first_inference.extend(
                    float(item["first_inference_ms"])
                    for item in (
                        sample.get("lifecycle", sample)
                        for sample in cold.get("samples", [])
                    )
                )
            cold_peak = cold.get("max_peak_rss_bytes")
            if cold_peak is not None:
                cold_peaks.append(int(cold_peak))
    latency = summarize(timings)
    audio_seconds = float(
        results[0].get("configuration", {}).get(
            "audio_seconds", BUCKET_TAGS.get(frames)
        )
    ) if runtime == "campp-c-reference" else float(
        results[0]["input"]["audio_seconds"]
    )
    rtf = summarize(
        [value / 1000.0 / audio_seconds for value in timings], suffix="rtf"
    )
    all_peaks = warm_peaks + cold_peaks
    return {
        "runtime": runtime,
        "bucket_frames": frames,
        "input_count": len(results),
        "latency": latency,
        "rtf": rtf,
        "first_inference": summarize(first_inference) if first_inference else None,
        "warm_peak_rss_bytes": max(warm_peaks, default=None),
        "cold_peak_rss_bytes": max(cold_peaks, default=None),
        "peak_rss_bytes": max(all_peaks, default=None),
        "stability_gate": {
            "threshold_cv_pct": cv_threshold_pct,
            "all_inputs_passed": stable,
            "aggregate_passed": float(latency["cv_pct"]) <= cv_threshold_pct,
        },
    }


def compare_buckets(
    ort: dict[str, Any], c_runtime: dict[str, Any]
) -> dict[str, Any]:
    ort_p50 = float(ort["latency"]["p50_ms"])
    c_p50 = float(c_runtime["latency"]["p50_ms"])
    ort_peak = ort.get("peak_rss_bytes")
    c_peak = c_runtime.get("peak_rss_bytes")
    return {
        "bucket_frames": ort["bucket_frames"],
        "c_speedup_over_ort_p50": ort_p50 / c_p50 if c_p50 > 0.0 else None,
        "c_rtf_ratio_to_ort_p50": (
            float(c_runtime["rtf"]["p50_rtf"]) / float(ort["rtf"]["p50_rtf"])
            if float(ort["rtf"]["p50_rtf"]) > 0.0
            else None
        ),
        "peak_rss_reduction_bytes": (
            int(ort_peak) - int(c_peak)
            if ort_peak is not None and c_peak is not None
            else None
        ),
        "peak_rss_reduction_ratio": (
            1.0 - int(c_peak) / int(ort_peak)
            if ort_peak not in (None, 0) and c_peak is not None
            else None
        ),
    }


def default_run_id(profile: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{profile}-{stamp}"


def executable_path(path: Path) -> Path:
    if path.is_file():
        return path
    windows = path.with_suffix(path.suffix + ".exe")
    return windows if windows.is_file() else path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "benchmark" / "runtime_qrb2210.json",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--buckets", type=int, nargs="*")
    parser.add_argument("--input-id", action="append")
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate artifacts, dataset and target environment without measuring",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        config = replace(
            config,
            paths=replace(
                config.paths,
                c_benchmark=executable_path(config.paths.c_benchmark),
            ),
        )
    except (BenchmarkError, OSError) as exc:
        print(f"benchmark configuration error: {exc}", file=sys.stderr)
        return 2

    if args.buckets:
        unknown = sorted(set(args.buckets) - set(config.buckets))
        if unknown:
            print(f"unsupported requested buckets: {unknown}", file=sys.stderr)
            return 2
        selected_buckets = set(args.buckets)
    else:
        selected_buckets = set(config.buckets)
    run_id = args.run_id or default_run_id(config.profile)
    if not RUN_ID_PATTERN.fullmatch(run_id):
        print("invalid run-id", file=sys.stderr)
        return 2

    try:
        for path, name in (
            (config.paths.canonical_model, "canonical model"),
            (config.paths.c_benchmark, "C benchmark executable"),
            (config.paths.ort_benchmark, "ORT benchmark script"),
            (config.paths.verification_result, "runtime validation evidence"),
            (config.paths.tensor_arena_result, "arena validation evidence"),
        ):
            _require_file(path, name)
        c_capabilities, _ = run_json_command(
            [str(config.paths.c_benchmark), "--capabilities"],
            environment=os.environ.copy(),
        )
        effective_threads = int(c_capabilities.get("effective_threads", 0))
        if effective_threads != config.threads:
            raise BenchmarkError(
                f"C Runtime effective_threads={effective_threads}, "
                f"but experiment requests {config.threads}"
            )
        selected_dataset = selected_dataset_inputs(config.paths.dataset_manifest)
        all_features, feature_document = load_feature_manifest(
            config.paths.feature_manifest
        )
        filtered_features = [
            item
            for item in all_features
            if item.bucket_frames in selected_buckets
            and (not args.input_id or item.input_id in set(args.input_id))
        ]
        filtered_dataset = {
            key: value
            for key, value in selected_dataset.items()
            if not args.input_id or key in set(args.input_id)
        }
        if args.input_id:
            unknown_inputs = sorted(set(args.input_id) - set(selected_dataset))
            if unknown_inputs:
                raise BenchmarkError(f"unknown input IDs: {unknown_inputs}")
        selected_config = BenchmarkConfig(
            source=config.source,
            profile=config.profile,
            buckets={
                frames: seconds
                for frames, seconds in config.buckets.items()
                if frames in selected_buckets
            },
            threads=config.threads,
            warmup=config.warmup,
            repeat=config.repeat,
            cold_runs=config.cold_runs,
            cv_threshold_pct=config.cv_threshold_pct,
            verify_source_wavs=config.verify_source_wavs,
            require_all_latency_inputs=config.require_all_latency_inputs,
            paths=config.paths,
            environment=config.environment,
        )
        features = validate_features(
            selected_config, filtered_dataset, filtered_features
        )
        source_wavs = (
            verify_source_wavs(config.paths.data_root, filtered_dataset)
            if config.verify_source_wavs
            else {"skipped": True}
        )
        provenance = load_bundle_provenance(
            config.paths.arena_bundle,
            config.paths.canonical_model,
            selected_config.buckets,
        )
        wait_for_thermal_start(config.environment)
        environment_before, environment_mismatches = apply_and_validate_environment(
            config.environment,
            allow_mismatch=args.allow_environment_mismatch,
        )
        evidence_result = {
            "runtime_validation": evidence(config.paths.verification_result),
            "tensor_arena_validation": evidence(
                config.paths.tensor_arena_result
            ),
        }
    except (BenchmarkError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark preflight failed: {exc}", file=sys.stderr)
        return 2

    if args.preflight_only:
        print("Runtime benchmark preflight: PASS")
        print(f"  inputs: {len(features)}")
        print(f"  threads: {config.threads}")
        print(f"  environment mismatches: {environment_mismatches or 'none'}")
        return 0

    run_dir = config.paths.workspace_root / run_id
    if run_dir.exists():
        print(f"run directory already exists: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    base_environment = os.environ.copy()
    base_environment.update(
        {
            "OMP_NUM_THREADS": str(config.threads),
            "ORT_NUM_THREADS": str(config.threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    metadata = {
        "schema_version": 1,
        "profile": config.profile,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": {
            "input": "precomputed FBank float32 [1,frames,80]",
            "output": "embedding [1,192]",
            "excluded": ["WAV read", "FBank extraction", "accuracy evaluation"],
            "cold": "fresh process; model/context load plus first inference",
            "warm": "persistent model/context; file I/O excluded",
            "memory": (
                "whole-process VmHWM; ORT includes its Python harness and "
                "C Runtime is a native process"
            ),
        },
        "configuration": {
            "source": str(config.source),
            "threads": config.threads,
            "warmup": config.warmup,
            "repeat": config.repeat,
            "cold_runs": config.cold_runs,
            "cv_threshold_pct": config.cv_threshold_pct,
            "buckets": selected_config.buckets,
        },
        "git": git_metadata(),
        "artifacts": provenance,
        "evidence": evidence_result,
        "dataset": {
            "manifest": str(config.paths.dataset_manifest),
            "manifest_sha256": sha256_file(config.paths.dataset_manifest),
            "feature_manifest": str(config.paths.feature_manifest),
            "feature_manifest_sha256": sha256_file(
                config.paths.feature_manifest
            ),
            "preprocessing": feature_document.get("preprocessing"),
            "source_wavs": source_wavs,
        },
        "environment_before": environment_before,
        "environment_mismatches": environment_mismatches,
        "c_runtime_capabilities": c_capabilities,
    }
    build_metadata = config.paths.c_benchmark.with_suffix(".build.txt")
    metadata["artifacts"]["tools"] = {
        "c_benchmark": {
            "path": str(config.paths.c_benchmark),
            "sha256": sha256_file(config.paths.c_benchmark),
            "build_metadata_path": (
                str(build_metadata) if build_metadata.is_file() else None
            ),
            "build_metadata_sha256": (
                sha256_file(build_metadata) if build_metadata.is_file() else None
            ),
        },
        "ort_benchmark": {
            "path": str(config.paths.ort_benchmark),
            "sha256": sha256_file(config.paths.ort_benchmark),
        },
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    per_input: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    print("CAM++ on-device Runtime benchmark")
    print(f"  run: {run_id}")
    print(f"  inputs: {len(features)}")
    print(f"  threads: {config.threads}")
    for index, feature in enumerate(features):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", feature.input_id)
        bucket_root = run_dir / "raw" / str(feature.bucket_frames) / safe_id
        backend_order = ("ort", "c_runtime") if index % 2 == 0 else (
            "c_runtime",
            "ort",
        )
        results: dict[str, dict[str, Any]] = {}
        print(
            f"[{index + 1}/{len(features)}] {feature.bucket_frames} "
            f"{feature.input_id}"
        )
        for backend in backend_order:
            print(f"  {backend}")
            try:
                wait_for_thermal_start(config.environment)
                if backend == "ort":
                    result = run_ort_benchmark(
                        selected_config,
                        feature,
                        bucket_root / "ort",
                        base_environment,
                    )
                    runtime_name = "onnxruntime-cpu"
                else:
                    result = run_c_benchmark(
                        selected_config,
                        feature,
                        bucket_root / "c_runtime",
                        base_environment,
                    )
                    runtime_name = "campp-c-reference"
            except (BenchmarkError, OSError, ValueError) as exc:
                failure = {
                    "input_id": feature.input_id,
                    "bucket_frames": feature.bucket_frames,
                    "backend": backend,
                    "error": str(exc),
                    "environment": environment_snapshot(),
                }
                (run_dir / "benchmark_failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"benchmark failed: {exc}", file=sys.stderr)
                return 1
            results[backend] = result
            grouped.setdefault((runtime_name, feature.bucket_frames), []).append(
                result
            )
        ort_hash = results["ort"]["warm"]["embedding"]["sha256"]
        c_hash = results["c_runtime"]["embedding"]["sha256"]
        per_input.append(
            {
                "input_id": feature.input_id,
                "bucket_frames": feature.bucket_frames,
                "audio_seconds": feature.audio_seconds,
                "feature_sha256": feature.sha256,
                "ort_result": str(bucket_root / "ort" / "result.json"),
                "c_runtime_result": str(
                    bucket_root / "c_runtime" / "result.json"
                ),
                "embedding_bitwise_identical": ort_hash == c_hash,
                "ort_embedding_sha256": ort_hash,
                "c_runtime_embedding_sha256": c_hash,
            }
        )

    aggregates: dict[str, dict[str, Any]] = {
        "onnxruntime-cpu": {},
        "campp-c-reference": {},
    }
    comparisons: list[dict[str, Any]] = []
    for frames in sorted(selected_config.buckets):
        ort = aggregate_bucket(
            "onnxruntime-cpu",
            frames,
            grouped[("onnxruntime-cpu", frames)],
            config.cv_threshold_pct,
        )
        c_result = aggregate_bucket(
            "campp-c-reference",
            frames,
            grouped[("campp-c-reference", frames)],
            config.cv_threshold_pct,
        )
        aggregates["onnxruntime-cpu"][str(frames)] = ort
        aggregates["campp-c-reference"][str(frames)] = c_result
        comparisons.append(compare_buckets(ort, c_result))

    environment_after = environment_snapshot()
    summary = {
        "schema_version": 1,
        "profile": config.profile,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accuracy_evaluated": False,
        "existing_accuracy_evidence": evidence_result,
        "configuration": metadata["configuration"],
        "artifacts": provenance,
        "environment_before": environment_before,
        "environment_after": environment_after,
        "environment_mismatches": environment_mismatches,
        "per_input": per_input,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "validity": {
            "same_feature_payload": True,
            "same_canonical_model_provenance": True,
            "requested_threads": config.threads,
            "c_effective_threads": config.threads,
            "environment_matched": not environment_mismatches,
            "all_stability_gates_passed": all(
                item["stability_gate"]["all_inputs_passed"]
                for backend in aggregates.values()
                for item in backend.values()
            ),
            "performance_complete": True,
        },
    }
    summary_path = run_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
