"""Numerically checked speaker embedding scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np


EMBEDDING_DIMENSION = 192


class SimilarityError(RuntimeError):
    """Raised when a template or score is invalid."""


def normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.size != EMBEDDING_DIMENSION:
        raise SimilarityError(
            f"embedding has {value.size} values; expected {EMBEDDING_DIMENSION}"
        )
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise SimilarityError("embedding norm is zero or non-finite")
    return (value / np.float32(norm)).astype(np.float32)


def load_embedding(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SimilarityError(f"embedding file is missing: {path}")
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        if path.stat().st_size != EMBEDDING_DIMENSION * 4:
            raise SimilarityError(
                f"raw embedding must be {EMBEDDING_DIMENSION * 4} bytes: {path}"
            )
        value = np.fromfile(path, dtype="<f4")
    return normalize(value)


def cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = normalize(reference)
    right = normalize(candidate)
    score = float(np.dot(left, right))
    if not np.isfinite(score):
        raise SimilarityError("cosine score is non-finite")
    return min(1.0, max(-1.0, score))


def resolve_speaker_embedding(pipeline_root: Path, value: str) -> Path:
    boundary = pipeline_root.resolve()
    candidate = Path(value)
    relative = candidate if candidate.is_absolute() else pipeline_root / candidate
    if relative.is_file():
        resolved = relative.resolve()
        if not resolved.is_relative_to(boundary):
            raise SimilarityError(
                f"speaker embedding must stay inside {boundary}: {resolved}"
            )
        return resolved
    embedded_file = pipeline_root / "voice/embedded" / value
    if embedded_file.is_file():
        return embedded_file.resolve()
    speaker_template = pipeline_root / "voice/embedded" / value / "mean_embedding.f32"
    if speaker_template.is_file():
        return speaker_template.resolve()
    raise SimilarityError(
        f"cannot resolve speaker embedding {value!r}; pass a file or an "
        "enrolled speaker folder name"
    )
