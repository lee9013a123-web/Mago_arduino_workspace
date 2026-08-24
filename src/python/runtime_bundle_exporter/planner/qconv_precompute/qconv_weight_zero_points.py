"""Weight zero-point predicates for QConv v5 planning."""

from __future__ import annotations

from collections.abc import Iterable


def all_weight_zero_points_are_zero(values: Iterable[int]) -> bool:
    """Return true when every exported weight zero-point is exactly zero."""

    return all(int(value) == 0 for value in values)
