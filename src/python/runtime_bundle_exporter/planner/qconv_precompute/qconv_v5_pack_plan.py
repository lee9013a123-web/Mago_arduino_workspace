"""Manifest-only QConv v5 weight-pack planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QConvV5PackPlan:
    output_channels: int
    input_channels: int
    kernel_elements: int
    symmetric_weight_zero_point: bool
    stores_corrected_bias: bool


def make_v5_pack_plan(
    *,
    output_channels: int,
    input_channels: int,
    kernel_elements: int,
    symmetric_weight_zero_point: bool,
    stores_corrected_bias: bool = False,
) -> QConvV5PackPlan:
    """Create a checked v5 pack-plan descriptor without writing a bundle."""

    if output_channels <= 0:
        raise ValueError("output_channels must be positive")
    if input_channels <= 0:
        raise ValueError("input_channels must be positive")
    if kernel_elements <= 0:
        raise ValueError("kernel_elements must be positive")
    return QConvV5PackPlan(
        output_channels=output_channels,
        input_channels=input_channels,
        kernel_elements=kernel_elements,
        symmetric_weight_zero_point=bool(symmetric_weight_zero_point),
        stores_corrected_bias=bool(stores_corrected_bias),
    )
