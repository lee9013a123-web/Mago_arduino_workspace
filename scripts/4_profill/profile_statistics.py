"""Operator profiling sample의 통계와 end-to-end 누적 비중을 계산한다."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence


def percentile(values: Sequence[int | float], percent: float) -> float:
    """기존 Runtime benchmark와 같은 선형 보간 percentile을 계산한다."""

    if not values:
        raise ValueError("cannot compute percentile from empty samples")
    if not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ns(samples_ns: Sequence[int]) -> dict[str, Any]:
    """nanosecond sample을 사람이 읽는 millisecond 통계로 변환한다."""

    if not samples_ns:
        raise ValueError("at least one sample is required")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in samples_ns):
        raise ValueError("samples_ns must contain non-negative integers")
    milliseconds = [value / 1_000_000.0 for value in samples_ns]
    mean_ms = statistics.fmean(milliseconds)
    stddev_ms = statistics.pstdev(milliseconds)
    return {
        "call_count": len(samples_ns),
        "total_exclusive_ns": sum(samples_ns),
        "total_exclusive_ms": sum(milliseconds),
        "mean_ms": mean_ms,
        "stddev_ms": stddev_ms,
        "cv_pct": stddev_ms / mean_ms * 100.0 if mean_ms > 0.0 else None,
        "min_ms": min(milliseconds),
        "p50_ms": percentile(milliseconds, 50.0),
        "p95_ms": percentile(milliseconds, 95.0),
        "max_ms": max(milliseconds),
    }


def rank_operator_samples(
    samples_by_operator: Mapping[int, Sequence[int]],
    end_to_end_samples_ns: Sequence[int],
    *,
    threshold_pct: float = 80.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Operator를 exclusive time으로 정렬하고 end-to-end 누적 비중을 붙인다."""

    if not samples_by_operator:
        raise ValueError("at least one operator is required")
    if not end_to_end_samples_ns:
        raise ValueError("end-to-end samples are required")
    if not 0.0 < threshold_pct <= 100.0:
        raise ValueError("threshold_pct must be in (0, 100]")
    if any(value <= 0 for value in end_to_end_samples_ns):
        raise ValueError("end-to-end samples must be positive")

    total_end_to_end_ns = sum(end_to_end_samples_ns)
    statistics_by_operator = {
        int(operator_id): summarize_ns(tuple(samples))
        for operator_id, samples in samples_by_operator.items()
    }
    total_operator_exclusive_ns = sum(
        int(item["total_exclusive_ns"])
        for item in statistics_by_operator.values()
    )
    ordered = sorted(
        statistics_by_operator.items(),
        key=lambda item: (-int(item[1]["total_exclusive_ns"]), item[0]),
    )

    cumulative_end_to_end_pct = 0.0
    cumulative_operator_pct = 0.0
    top_set_ids: list[int] = []
    threshold_reached = False
    ranked: list[dict[str, Any]] = []
    for rank, (operator_id, item) in enumerate(ordered, start=1):
        exclusive_ns = int(item["total_exclusive_ns"])
        end_to_end_share = exclusive_ns / total_end_to_end_ns * 100.0
        operator_share = (
            exclusive_ns / total_operator_exclusive_ns * 100.0
            if total_operator_exclusive_ns > 0
            else 0.0
        )
        cumulative_end_to_end_pct += end_to_end_share
        cumulative_operator_pct += operator_share
        in_top_set = not threshold_reached
        if in_top_set:
            top_set_ids.append(operator_id)
        if cumulative_end_to_end_pct >= threshold_pct:
            threshold_reached = True
        ranked.append(
            {
                "rank": rank,
                "operator_id": operator_id,
                **item,
                "end_to_end_share_pct": end_to_end_share,
                "cumulative_end_to_end_share_pct": cumulative_end_to_end_pct,
                "operator_exclusive_share_pct": operator_share,
                "cumulative_operator_exclusive_share_pct": (
                    cumulative_operator_pct
                ),
                "in_top_bottleneck_set": in_top_set,
            }
        )

    accounted_pct = total_operator_exclusive_ns / total_end_to_end_ns * 100.0
    summary = {
        "threshold_pct": threshold_pct,
        "complete": threshold_reached,
        "operator_ids": top_set_ids,
        "operator_count": len(top_set_ids),
        "total_end_to_end_ns": total_end_to_end_ns,
        "total_end_to_end_ms": total_end_to_end_ns / 1_000_000.0,
        "total_operator_exclusive_ns": total_operator_exclusive_ns,
        "total_operator_exclusive_ms": (
            total_operator_exclusive_ns / 1_000_000.0
        ),
        "kernel_accounted_end_to_end_pct": accounted_pct,
        "unattributed_runtime_ns": total_end_to_end_ns - total_operator_exclusive_ns,
        "unattributed_runtime_ms": (
            total_end_to_end_ns - total_operator_exclusive_ns
        ) / 1_000_000.0,
    }
    if not threshold_reached:
        summary["reason"] = (
            "kernel exclusive time accounts for less than "
            f"{threshold_pct:g}% of end-to-end latency"
        )
    return ranked, summary
