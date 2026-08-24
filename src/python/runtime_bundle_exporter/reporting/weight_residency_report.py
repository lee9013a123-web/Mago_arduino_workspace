"""Compact cross-bucket reporting for static weight layouts."""

from __future__ import annotations

from typing import Sequence

from ..planner.weight_residency_planner import StaticWeightPlan


def build_weight_residency_summary(
    plans: Sequence[StaticWeightPlan],
) -> dict:
    if not plans:
        raise ValueError("at least one static weight plan is required")
    rows = []
    for plan in sorted(plans, key=lambda item: item.bucket_frames):
        rows.append({
            "bucket_frames": plan.bucket_frames,
            "constant_count": len(plan.entries),
            "used_constant_count": plan.used_constant_count,
            "source_weights_bytes": plan.source_weights_bytes,
            "static_weights_bytes": len(plan.weight_bytes),
            "padding_bytes": plan.output_padding_bytes,
            "saved_bytes": plan.saved_bytes,
            "saved_pct": (
                plan.saved_bytes / plan.source_weights_bytes * 100.0
                if plan.source_weights_bytes else 0.0
            ),
            "execution_plan_sha256": plan.output_plan_sha256,
            "weights_sha256": plan.output_weights_sha256,
            "inference_weight_moves": 0,
        })
    return {
        "schema_version": 1,
        "strategy": "offline_first_use_static_blob",
        "production_invariant": "no weight relocation during inference",
        "buckets": rows,
    }


__all__ = ["build_weight_residency_summary"]
