"""Stable terminal report required by the microphone demo."""

from __future__ import annotations

from voice_embedding.runtime import RuntimeMetrics


def _mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def _share(part: int, total: int) -> float:
    return (part / total * 100.0) if total > 0 else 0.0


def format_terminal_report(score: float, metrics: RuntimeMetrics) -> str:
    return "\n".join([
        "=================================",
        "=================================",
        "=====                         ======",
        f"=====      final score: {score:.6f}      ======",
        "=====                         ======",
        "=================================",
        "=================================",
        "[report]",
        "Peak RAM",
        f"- total RAM: {_mib(metrics.peak_rss_bytes):.2f} MiB",
        (
            f"- weight 점유율: {_mib(metrics.weight_bytes):.2f} MiB "
            f"({_share(metrics.weight_bytes, metrics.peak_rss_bytes):.2f}%)"
        ),
        (
            f"- activation 점유율: {_mib(metrics.activation_bytes):.2f} MiB "
            f"({_share(metrics.activation_bytes, metrics.peak_rss_bytes):.2f}%)"
        ),
        f"RTF: {metrics.rtf:.6f}",
    ])
