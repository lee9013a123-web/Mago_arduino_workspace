"""Tie-aware EER and MinDCF calculation for speaker-verification scores."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


class MetricError(ValueError):
    """The supplied scores cannot define speaker-verification metrics."""


@dataclass(frozen=True)
class OperatingPoint:
    """One threshold operating point using the rule ``score >= threshold``.

    ``threshold`` is ``None`` for the initial point above the maximum score,
    where every trial is rejected.
    """

    threshold: float | None
    false_accept_rate: float
    false_reject_rate: float
    false_accept_count: int
    false_reject_count: int

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "threshold": self.threshold,
            "threshold_position": (
                "above_maximum_score" if self.threshold is None else "score"
            ),
            "false_accept_rate": self.false_accept_rate,
            "false_reject_rate": self.false_reject_rate,
            "false_accept_count": self.false_accept_count,
            "false_reject_count": self.false_reject_count,
        }


@dataclass(frozen=True)
class VerificationMetrics:
    """Detection metrics calculated from one common set of trials."""

    trial_count: int
    target_count: int
    non_target_count: int
    eer_rate: float
    eer_threshold: float
    eer_nearest_point: OperatingPoint
    min_dcf: float
    min_dcf_normalized: float
    min_dcf_point: OperatingPoint
    p_target: float
    c_miss: float
    c_false_alarm: float

    def to_dict(self) -> dict[str, object]:
        return {
            "trial_count": self.trial_count,
            "target_count": self.target_count,
            "non_target_count": self.non_target_count,
            "eer": {
                "rate": self.eer_rate,
                "percent": self.eer_rate * 100.0,
                "threshold": self.eer_threshold,
                "threshold_rule": "accept_if_score_greater_than_or_equal",
                "interpolation": "linear_between_adjacent_tie_grouped_points",
                "nearest_measured_operating_point": (
                    self.eer_nearest_point.to_dict()
                ),
            },
            "min_dcf": {
                "value": self.min_dcf,
                "normalized": self.min_dcf_normalized,
                "p_target": self.p_target,
                "c_miss": self.c_miss,
                "c_false_alarm": self.c_false_alarm,
                "operating_point": self.min_dcf_point.to_dict(),
            },
        }


def _validate_inputs(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[list[float], list[int], int, int]:
    if len(scores) != len(labels):
        raise MetricError("scores and labels must have the same length")
    if not scores:
        raise MetricError("at least one score is required")

    checked_scores: list[float] = []
    checked_labels: list[int] = []
    for index, (score, label) in enumerate(zip(scores, labels)):
        value = float(score)
        if not math.isfinite(value):
            raise MetricError(f"score {index} is not finite")
        if isinstance(label, bool):
            label_value = int(label)
        elif isinstance(label, int) and label in (0, 1):
            label_value = label
        else:
            raise MetricError(f"label {index} must be 0 or 1")
        checked_scores.append(value)
        checked_labels.append(label_value)

    target_count = sum(checked_labels)
    non_target_count = len(checked_labels) - target_count
    if target_count == 0 or non_target_count == 0:
        raise MetricError("both target and non-target trials are required")
    return checked_scores, checked_labels, target_count, non_target_count


def _operating_points(
    scores: Sequence[float],
    labels: Sequence[int],
    target_count: int,
    non_target_count: int,
) -> list[OperatingPoint]:
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    points = [
        OperatingPoint(
            threshold=None,
            false_accept_rate=0.0,
            false_reject_rate=1.0,
            false_accept_count=0,
            false_reject_count=target_count,
        )
    ]
    accepted_targets = 0
    accepted_non_targets = 0
    cursor = 0
    while cursor < len(ordered):
        threshold = ordered[cursor][0]
        next_cursor = cursor
        while next_cursor < len(ordered) and ordered[next_cursor][0] == threshold:
            if ordered[next_cursor][1] == 1:
                accepted_targets += 1
            else:
                accepted_non_targets += 1
            next_cursor += 1
        false_rejects = target_count - accepted_targets
        points.append(
            OperatingPoint(
                threshold=threshold,
                false_accept_rate=accepted_non_targets / non_target_count,
                false_reject_rate=false_rejects / target_count,
                false_accept_count=accepted_non_targets,
                false_reject_count=false_rejects,
            )
        )
        cursor = next_cursor
    return points


def _equal_error_rate(
    points: Sequence[OperatingPoint],
) -> tuple[float, float, OperatingPoint]:
    nearest = min(
        points,
        key=lambda point: (
            abs(point.false_reject_rate - point.false_accept_rate),
            point.false_reject_rate + point.false_accept_rate,
        ),
    )
    previous = points[0]
    for current in points[1:]:
        previous_difference = (
            previous.false_reject_rate - previous.false_accept_rate
        )
        current_difference = current.false_reject_rate - current.false_accept_rate
        if current_difference > 0.0:
            previous = current
            continue
        if current_difference == 0.0:
            assert current.threshold is not None
            return current.false_accept_rate, current.threshold, nearest

        denominator = previous_difference - current_difference
        fraction = previous_difference / denominator if denominator else 0.0
        false_accept_rate = previous.false_accept_rate + fraction * (
            current.false_accept_rate - previous.false_accept_rate
        )
        false_reject_rate = previous.false_reject_rate + fraction * (
            current.false_reject_rate - previous.false_reject_rate
        )
        eer_rate = (false_accept_rate + false_reject_rate) / 2.0
        if previous.threshold is None:
            assert current.threshold is not None
            threshold = current.threshold
        else:
            assert current.threshold is not None
            threshold = previous.threshold + fraction * (
                current.threshold - previous.threshold
            )
        return eer_rate, threshold, nearest
    raise MetricError("failed to locate the EER crossing")


def compute_verification_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_false_alarm: float = 1.0,
) -> VerificationMetrics:
    """Calculate tie-aware EER and normalized MinDCF.

    A score equal to the threshold is accepted. EER is linearly interpolated
    between adjacent operating points after equal scores have been grouped, so
    the result never depends on the order of tied trials.
    """

    if not 0.0 < p_target < 1.0:
        raise MetricError("p_target must be between 0 and 1")
    if c_miss <= 0.0 or c_false_alarm <= 0.0:
        raise MetricError("DCF costs must be positive")
    checked_scores, checked_labels, target_count, non_target_count = (
        _validate_inputs(scores, labels)
    )
    points = _operating_points(
        checked_scores, checked_labels, target_count, non_target_count
    )
    eer_rate, eer_threshold, nearest = _equal_error_rate(points)

    normalization = min(
        c_miss * p_target, c_false_alarm * (1.0 - p_target)
    )
    dcf_values = [
        c_miss * p_target * point.false_reject_rate
        + c_false_alarm * (1.0 - p_target) * point.false_accept_rate
        for point in points
    ]
    best_index = min(range(len(points)), key=lambda index: dcf_values[index])
    min_dcf = dcf_values[best_index]
    return VerificationMetrics(
        trial_count=len(checked_scores),
        target_count=target_count,
        non_target_count=non_target_count,
        eer_rate=eer_rate,
        eer_threshold=eer_threshold,
        eer_nearest_point=nearest,
        min_dcf=min_dcf,
        min_dcf_normalized=min_dcf / normalization,
        min_dcf_point=points[best_index],
        p_target=p_target,
        c_miss=c_miss,
        c_false_alarm=c_false_alarm,
    )
