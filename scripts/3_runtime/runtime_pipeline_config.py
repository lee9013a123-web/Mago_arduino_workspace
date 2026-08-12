"""Load and validate configuration for the Reference Runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


SUPPORTED_BUCKETS = (98, 298, 498, 998)
SUPPORTED_BACKENDS = {"cpu_reference"}


class RuntimePipelineConfigError(ValueError):
    """The pipeline JSON document is missing or internally inconsistent."""


@dataclass(frozen=True)
class RuntimePaths:
    workspace_root: Path
    canonical_model: Path
    static_dir: Path
    graph_dir: Path


@dataclass(frozen=True)
class RuntimeTolerances:
    float_atol: float
    float_rtol: float
    embedding_cosine_min: float


@dataclass(frozen=True)
class RuntimeDiagnostics:
    enabled: bool
    dump_tensor_ids: str | tuple[int, ...]
    checkpoint_tensor_ids: tuple[int, ...]
    max_failure_report: int


@dataclass(frozen=True)
class RuntimeBuild:
    cc: str
    cflags: str


@dataclass(frozen=True)
class RuntimePipelineConfig:
    source: Path
    profile: str
    buckets: tuple[int, ...]
    seed: int
    backend: str
    paths: RuntimePaths
    include_weight_index: bool
    build: RuntimeBuild
    diagnostics: RuntimeDiagnostics
    tolerances: RuntimeTolerances
    document: dict[str, Any]


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise RuntimePipelineConfigError(f"{key} must be a JSON object")
    return value


def _repository_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimePipelineConfigError(f"{field} must be a non-empty path")
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimePipelineConfigError(
            f"{field} must stay inside the repository: {resolved}"
        ) from exc
    return resolved


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimePipelineConfigError(f"{field} must be a positive integer")
    return value


def _nonnegative_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimePipelineConfigError(f"{field} must be numeric")
    number = float(value)
    if number < 0.0:
        raise RuntimePipelineConfigError(f"{field} must be non-negative")
    return number


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimePipelineConfigError(f"{field} must be a non-empty array")
    result = tuple(_positive_int(item, field) for item in value)
    if len(set(result)) != len(result):
        raise RuntimePipelineConfigError(f"{field} contains duplicate values")
    return result


def load_runtime_pipeline_config(
    path: Path,
    *,
    repository_root: Path,
) -> RuntimePipelineConfig:
    source = path.resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePipelineConfigError(f"cannot read config {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise RuntimePipelineConfigError("config root must be a JSON object")
    if document.get("schema_version") != 1:
        raise RuntimePipelineConfigError("schema_version must be 1")

    profile = document.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise RuntimePipelineConfigError("profile must be a non-empty string")
    buckets = _integer_tuple(document.get("buckets"), "buckets")
    unsupported = sorted(set(buckets) - set(SUPPORTED_BUCKETS))
    if unsupported:
        raise RuntimePipelineConfigError(f"unsupported buckets: {unsupported}")
    seed = document.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RuntimePipelineConfigError("seed must be a non-negative integer")
    backend = document.get("backend")
    if backend not in SUPPORTED_BACKENDS:
        raise RuntimePipelineConfigError(
            f"unsupported backend {backend!r}; supported: {sorted(SUPPORTED_BACKENDS)}"
        )

    paths_doc = _mapping(document, "paths")
    paths = RuntimePaths(
        workspace_root=_repository_path(
            repository_root, paths_doc.get("workspace_root"), "paths.workspace_root"
        ),
        canonical_model=_repository_path(
            repository_root, paths_doc.get("canonical_model"), "paths.canonical_model"
        ),
        static_dir=_repository_path(
            repository_root, paths_doc.get("static_dir"), "paths.static_dir"
        ),
        graph_dir=_repository_path(
            repository_root, paths_doc.get("graph_dir"), "paths.graph_dir"
        ),
    )
    canonical_bundle = (repository_root / "models" / "compiled" / "reference").resolve()
    if paths.workspace_root == canonical_bundle or canonical_bundle in paths.workspace_root.parents:
        raise RuntimePipelineConfigError(
            "workspace_root must not be inside models/compiled/reference"
        )

    bundle_doc = _mapping(document, "bundle")
    include_weight_index = bundle_doc.get("include_weight_index")
    if not isinstance(include_weight_index, bool):
        raise RuntimePipelineConfigError(
            "bundle.include_weight_index must be true or false"
        )

    build_doc = _mapping(document, "build")
    cc = build_doc.get("cc")
    cflags = build_doc.get("cflags")
    if not isinstance(cc, str) or not cc.strip():
        raise RuntimePipelineConfigError("build.cc must be a non-empty string")
    if not isinstance(cflags, str) or not cflags.strip():
        raise RuntimePipelineConfigError("build.cflags must be a non-empty string")

    diagnostics_doc = _mapping(document, "diagnostics")
    enabled = diagnostics_doc.get("enabled")
    if not isinstance(enabled, bool):
        raise RuntimePipelineConfigError("diagnostics.enabled must be true or false")
    if not enabled:
        raise RuntimePipelineConfigError(
            "the correctness pipeline requires diagnostics.enabled=true"
        )
    raw_dump_ids = diagnostics_doc.get("dump_tensor_ids")
    if raw_dump_ids == "all":
        dump_tensor_ids: str | tuple[int, ...] = "all"
    else:
        dump_tensor_ids = _integer_tuple(
            raw_dump_ids, "diagnostics.dump_tensor_ids"
        )
    # The current 05 comparator validates every operator output.  Selective dump
    # is reserved for a future production-only runner and cannot drive this
    # correctness pipeline yet.
    if dump_tensor_ids != "all":
        raise RuntimePipelineConfigError(
            "the reference verification pipeline currently requires "
            "diagnostics.dump_tensor_ids='all'"
        )
    checkpoint_ids = _integer_tuple(
        diagnostics_doc.get("checkpoint_tensor_ids"),
        "diagnostics.checkpoint_tensor_ids",
    )
    max_failure_report = _positive_int(
        diagnostics_doc.get("max_failure_report"),
        "diagnostics.max_failure_report",
    )

    tolerance_doc = _mapping(document, "tolerances")
    tolerances = RuntimeTolerances(
        float_atol=_nonnegative_float(
            tolerance_doc.get("float_atol"), "tolerances.float_atol"
        ),
        float_rtol=_nonnegative_float(
            tolerance_doc.get("float_rtol"), "tolerances.float_rtol"
        ),
        embedding_cosine_min=_nonnegative_float(
            tolerance_doc.get("embedding_cosine_min"),
            "tolerances.embedding_cosine_min",
        ),
    )
    if tolerances.embedding_cosine_min > 1.0:
        raise RuntimePipelineConfigError(
            "tolerances.embedding_cosine_min must not exceed 1"
        )

    return RuntimePipelineConfig(
        source=source,
        profile=profile,
        buckets=buckets,
        seed=seed,
        backend=backend,
        paths=paths,
        include_weight_index=include_weight_index,
        build=RuntimeBuild(cc=cc, cflags=cflags),
        diagnostics=RuntimeDiagnostics(
            enabled=enabled,
            dump_tensor_ids=dump_tensor_ids,
            checkpoint_tensor_ids=checkpoint_ids,
            max_failure_report=max_failure_report,
        ),
        tolerances=tolerances,
        document=document,
    )
