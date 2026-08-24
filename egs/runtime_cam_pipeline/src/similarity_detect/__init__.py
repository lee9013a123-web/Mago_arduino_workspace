"""Speaker-template similarity scoring and terminal reporting."""

from .scoring import cosine_similarity, load_embedding

__all__ = ["cosine_similarity", "load_embedding"]
