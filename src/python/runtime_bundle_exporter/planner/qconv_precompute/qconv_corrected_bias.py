"""Corrected-bias helpers for QConv v5 algebraic fast paths."""

from __future__ import annotations

from collections.abc import Sequence


def corrected_bias(
    bias: Sequence[int],
    input_zero_point: int,
    weight_sums: Sequence[int],
) -> list[int]:
    """Return `bias - input_zero_point * sum(weight)` per output channel."""

    if len(bias) != len(weight_sums):
        raise ValueError("bias and weight_sums must have the same length")
    input_zero = int(input_zero_point)
    return [
        int(channel_bias) - input_zero * int(weight_sum)
        for channel_bias, weight_sum in zip(bias, weight_sums, strict=True)
    ]
