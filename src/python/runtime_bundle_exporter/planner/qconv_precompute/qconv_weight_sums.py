"""Bitwise-safe QConv weight-sum helpers."""

from __future__ import annotations

from collections.abc import Sequence


def weight_sums_by_output(
    weights: Sequence[int],
    output_channels: int,
    inputs_per_output: int,
) -> list[int]:
    """Return per-output int32 sums for logical O-major weights."""

    if output_channels < 0 or inputs_per_output < 0:
        raise ValueError("channel counts must be non-negative")
    expected = output_channels * inputs_per_output
    if len(weights) != expected:
        raise ValueError(
            f"expected {expected} weights, got {len(weights)}"
        )
    return [
        sum(int(value) for value in weights[
            output * inputs_per_output:(output + 1) * inputs_per_output
        ])
        for output in range(output_channels)
    ]
