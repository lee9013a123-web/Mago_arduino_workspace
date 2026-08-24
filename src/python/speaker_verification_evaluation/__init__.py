"""Speaker-verification scoring and detection metrics."""

from .manifests import (
    EmbeddingSet,
    Trial,
    load_embedding_set,
    read_trials,
    score_trials,
)
from .metrics import VerificationMetrics, compute_verification_metrics

__all__ = [
    "EmbeddingSet",
    "Trial",
    "VerificationMetrics",
    "compute_verification_metrics",
    "load_embedding_set",
    "read_trials",
    "score_trials",
]
