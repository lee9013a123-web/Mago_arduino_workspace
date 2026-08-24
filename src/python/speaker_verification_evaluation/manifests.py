"""Trial and embedding-manifest loading for speaker verification."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


class ManifestError(ValueError):
    """A trial or embedding manifest violates the evaluation contract."""


@dataclass(frozen=True)
class Trial:
    trial_id: str
    enroll_id: str
    test_id: str
    target: int
    trial_type: str


@dataclass(frozen=True)
class ScoredTrial:
    trial: Trial
    score: float


@dataclass(frozen=True)
class EmbeddingSet:
    backend: str
    manifest_path: Path
    embeddings: dict[tuple[int | None, str], np.ndarray]

    @property
    def buckets(self) -> set[int | None]:
        return {key[0] for key in self.embeddings}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_trials(path: Path) -> list[Trial]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            required = {"trial_id", "enroll_id", "test_id", "target", "trial_type"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ManifestError(
                    f"trial manifest requires columns {sorted(required)}"
                )
            trials: list[Trial] = []
            seen: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                trial_id = (row.get("trial_id") or "").strip()
                enroll_id = (row.get("enroll_id") or "").strip()
                test_id = (row.get("test_id") or "").strip()
                trial_type = (row.get("trial_type") or "").strip()
                try:
                    target = int(row.get("target") or "")
                except ValueError as exc:
                    raise ManifestError(
                        f"{path}:{line_number}: target must be 0 or 1"
                    ) from exc
                if not trial_id or not enroll_id or not test_id or not trial_type:
                    raise ManifestError(f"{path}:{line_number}: empty trial field")
                if target not in (0, 1):
                    raise ManifestError(
                        f"{path}:{line_number}: target must be 0 or 1"
                    )
                if trial_id in seen:
                    raise ManifestError(f"duplicate trial_id: {trial_id}")
                seen.add(trial_id)
                trials.append(
                    Trial(trial_id, enroll_id, test_id, target, trial_type)
                )
    except OSError as exc:
        raise ManifestError(f"cannot read trial manifest {path}: {exc}") from exc
    if not trials:
        raise ManifestError("trial manifest is empty")
    if {trial.target for trial in trials} != {0, 1}:
        raise ManifestError("trial manifest requires target and non-target trials")
    return trials


def _embedding_rows(path: Path) -> tuple[str | None, list[dict[str, object]]]:
    try:
        if path.suffix.lower() == ".json":
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ManifestError("embedding JSON must be an object")
            rows = document.get("embeddings")
            if not isinstance(rows, list) or not all(
                isinstance(row, dict) for row in rows
            ):
                raise ManifestError("embedding JSON requires an embeddings array")
            declared_backend = document.get("backend")
            if declared_backend is not None and not isinstance(declared_backend, str):
                raise ManifestError("embedding backend must be a string")
            return declared_backend, rows

        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
        if not rows:
            raise ManifestError("embedding TSV is empty")
        return None, [dict(row) for row in rows]
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read embedding manifest {path}: {exc}") from exc


def _parse_bucket(raw: object, *, row_number: int) -> int | None:
    if raw is None or str(raw).strip() in ("", "native", "none"):
        return None
    try:
        value = int(str(raw))
    except ValueError as exc:
        raise ManifestError(
            f"embedding row {row_number}: bucket_frames must be an integer"
        ) from exc
    if value <= 0:
        raise ManifestError(
            f"embedding row {row_number}: bucket_frames must be positive"
        )
    return value


def _load_vector(path: Path) -> np.ndarray:
    try:
        if path.suffix.lower() == ".npy":
            vector = np.load(path, allow_pickle=False)
        else:
            vector = np.fromfile(path, dtype="<f4")
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot load embedding {path}: {exc}") from exc
    vector = np.asarray(vector)
    if vector.dtype.kind != "f":
        raise ManifestError(f"embedding is not floating point: {path}")
    vector = vector.astype(np.float32, copy=False).reshape(-1)
    if vector.size == 0:
        raise ManifestError(f"embedding is empty: {path}")
    if not np.isfinite(vector).all():
        raise ManifestError(f"embedding contains NaN or Inf: {path}")
    if float(np.linalg.norm(vector.astype(np.float64))) == 0.0:
        raise ManifestError(f"embedding has zero norm: {path}")
    return vector.copy()


def load_embedding_set(
    path: Path,
    *,
    backend: str,
) -> EmbeddingSet:
    declared_backend, rows = _embedding_rows(path)
    if declared_backend is not None and declared_backend != backend:
        raise ManifestError(
            f"manifest backend {declared_backend!r} does not match {backend!r}"
        )
    embeddings: dict[tuple[int | None, str], np.ndarray] = {}
    dimensions: dict[int | None, int] = {}
    for row_number, row in enumerate(rows, start=1):
        input_id = str(row.get("input_id") or "").strip()
        raw_path = row.get("path") or row.get("embedding_path")
        if not input_id or raw_path is None or not str(raw_path).strip():
            raise ManifestError(
                f"embedding row {row_number}: input_id and path are required"
            )
        bucket = _parse_bucket(row.get("bucket_frames"), row_number=row_number)
        key = (bucket, input_id)
        if key in embeddings:
            raise ManifestError(
                f"duplicate embedding for bucket={bucket}, input_id={input_id}"
            )
        embedding_path = Path(str(raw_path))
        if not embedding_path.is_absolute():
            embedding_path = path.parent / embedding_path
        if not embedding_path.is_file():
            raise ManifestError(f"embedding file does not exist: {embedding_path}")
        expected_sha = str(row.get("sha256") or "").strip().lower()
        if expected_sha and file_sha256(embedding_path) != expected_sha:
            raise ManifestError(f"embedding checksum mismatch: {embedding_path}")
        vector = _load_vector(embedding_path)
        previous_dimension = dimensions.setdefault(bucket, int(vector.size))
        if vector.size != previous_dimension:
            raise ManifestError(
                f"bucket {bucket}: embedding dimensions differ "
                f"({vector.size} != {previous_dimension})"
            )
        embeddings[key] = vector
    if not embeddings:
        raise ManifestError("embedding manifest is empty")
    return EmbeddingSet(backend, path, embeddings)


def required_input_ids(trials: Iterable[Trial]) -> set[str]:
    values: set[str] = set()
    for trial in trials:
        values.add(trial.enroll_id)
        values.add(trial.test_id)
    return values


def score_trials(
    trials: Iterable[Trial],
    embedding_set: EmbeddingSet,
    bucket_frames: int | None,
) -> list[ScoredTrial]:
    trial_list = list(trials)
    missing = sorted(
        input_id
        for input_id in required_input_ids(trial_list)
        if (bucket_frames, input_id) not in embedding_set.embeddings
    )
    if missing:
        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise ManifestError(
            f"{embedding_set.backend}/bucket {bucket_frames}: "
            f"missing {len(missing)} trial embeddings: {preview}{suffix}"
        )

    scored: list[ScoredTrial] = []
    for trial in trial_list:
        enroll = embedding_set.embeddings[(bucket_frames, trial.enroll_id)].astype(
            np.float64
        )
        test = embedding_set.embeddings[(bucket_frames, trial.test_id)].astype(
            np.float64
        )
        if enroll.shape != test.shape:
            raise ManifestError(
                f"{trial.trial_id}: embedding dimensions differ "
                f"({enroll.size} != {test.size})"
            )
        denominator = float(np.linalg.norm(enroll) * np.linalg.norm(test))
        score = float(np.dot(enroll, test) / denominator)
        if not math.isfinite(score):
            raise ManifestError(f"{trial.trial_id}: cosine score is not finite")
        scored.append(ScoredTrial(trial, score))
    return scored
